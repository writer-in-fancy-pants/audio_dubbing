#!/usr/bin/env python3
"""
dub_pipeline.py
================
End-to-end pipeline to add an AI-dubbed Hindi audio track to a video whose
original dialogue is in English, while preserving the background
music/effects and approximating each speaker's voice.

PIPELINE STAGES
    1. Extract audio from the source video (ffmpeg)
    2. Separate vocals from background music/effects (Demucs)
    3. Transcribe English vocals with word-level timestamps (faster-whisper)
    4. (Optional) Diarize speakers so each speaker keeps a distinct voice
       (pyannote.audio)
    5. Translate each segment English -> Hindi (NLLB-200 via transformers)
    6. Synthesize Hindi speech per segment, cloning the original speaker's
       voice from a short reference clip (Coqui XTTS-v2)
    7. Time-stretch each synthesized clip to fit the original segment's
       duration so lip-flap timing stays reasonably close
    8. Reassemble a full-length Hindi vocal track, loudness-match it to the
       original vocal track, and mix it with the separated background track
    9. Mux the new Hindi track into the video as an ADDITIONAL audio track
       (the original English track is kept), tagged with language=hin

HONEST CAVEATS (read before you ship this)
    - There is no open-source model today that does true, high-quality
      "direct" speech-to-speech translation while perfectly preserving
      voice, prosody, and timing. This script chains ASR -> MT -> cloned
      TTS, which is the current best practical approximation. Expect some
      loss of expressiveness, occasional mistranslation, and imperfect
      lip-sync.
    - Quality depends heavily on: cleanliness of vocal separation, number
      of overlapping speakers, background noise, and how well XTTS's voice
      cloning captures the reference speaker on a short clip.
    - For overlapping/simultaneous speech, per-segment TTS will not
      reproduce true overlap -- segments are placed at the original start
      time but if two speakers talk over each other, the synthesized clips
      will simply overlap in the output too (which usually sounds worse
      than the original overlap because TTS voices don't blend naturally).
    - If you need production-grade dubbing, commercial dubbing APIs
      (e.g. ElevenLabs Dubbing, Meta SeamlessM4T/Expressive as a
      research option, HeyGen, Dubverse) currently outperform a
      from-scratch open-source stack like this one, especially on prosody
      and timing. This script is the right starting point if you want a
      controllable, offline, inspectable pipeline instead.

REQUIREMENTS
    - ffmpeg on PATH
    - Python packages: see requirements.txt written alongside this script
    - A CUDA GPU is strongly recommended (Demucs + Whisper + XTTS on CPU
      is slow, on the order of several minutes per minute of audio)
    - Internet access to download model weights on first run (Hugging Face
      Hub: openai/whisper or faster-whisper weights, facebook/nllb-200,
      coqui/XTTS-v2, and pyannote/speaker-diarization if diarization is
      enabled -- pyannote's gated models require a HF token)

USAGE
    python dub_pipeline.py \\
        --input movie.mp4 \\
        --output movie.hindi_dubbed.mp4 \\
        --workdir ./work \\
        --diarize \\
        --device cuda

    Intermediate artifacts are kept in --workdir so you can inspect /
    re-run individual stages (each stage skips work if its output file
    already exists, unless --force is passed).
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import librosa
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dub_pipeline")

import torch
import torchaudio
def resolve_device(requested:str, op:str)-> str:
    if requested == "mps" and torch.backends.mps.is_available():
        #torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "16")))
        return "mps"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested != "cpu":
        log.warning(f"{requested} unavailable. Falling back to cpu for {op}")
    return "cpu"


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Segment:
    """One unit of speech to translate & re-synthesize."""
    index: int
    start: float          # seconds, in the original timeline
    end: float             # seconds
    speaker: str            # e.g. "SPEAKER_00" or "SPEAKER_default"
    text_en: str
    text_hi: Optional[str] = None
    ref_audio_path: Optional[str] = None   # reference clip for voice cloning
    tts_audio_path: Optional[str] = None   # raw synthesized clip
    aligned_audio_path: Optional[str] = None  # time-stretched to fit start/end


def run(cmd: List[str], **kwargs):
    """Run a subprocess command, streaming output, raising on failure."""
    log.info("$ %s", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------
# Stage 1: extract audio
# --------------------------------------------------------------------------

def extract_audio(video_path: Path, out_wav: Path, force: bool = False) -> Path:
    if out_wav.exists() and not force:
        log.info("Skipping extract_audio (exists): %s", out_wav)
        return out_wav
    run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2",
        str(out_wav),
    ])
    return out_wav


# --------------------------------------------------------------------------
# Stage 2: source separation (Demucs)
# --------------------------------------------------------------------------

def separate_vocals(audio_wav: Path, workdir: Path, force: bool = False,
                     device: str = "mps", model: str = "htdemucs") -> tuple[Path, Path]:
    """
    Returns (vocals_path, background_path). Background = everything Demucs
    does NOT classify as vocals, i.e. drums+bass+other summed together.
    """
    out_dir = workdir / "demucs_out"
    vocals_path = out_dir / model / audio_wav.stem / "vocals.wav"
    no_vocals_path = out_dir / model / audio_wav.stem / "no_vocals.wav"

    if vocals_path.exists() and no_vocals_path.exists() and not force:
        log.info("Skipping separate_vocals (exists)")
        return vocals_path, no_vocals_path

    ensure_dir(out_dir)
    run([
        sys.executable, "-m", "demucs",
        "-n", model,
        "--two-stems", "vocals",
        "-d", device,
        "-o", str(out_dir),
        str(audio_wav),
    ])
    if not vocals_path.exists():
        raise FileNotFoundError(f"Demucs did not produce expected output: {vocals_path}")
    return vocals_path, no_vocals_path


# --------------------------------------------------------------------------
# Stage 3: transcription (faster-whisper)
# --------------------------------------------------------------------------

def transcribe(vocals_wav: Path, workdir: Path, force: bool = False,
               model_size: str = "large-v3", device: str = "cpu", 
               temperature=0.5, speaker = "SPEAKER_default", 
               diarize=True, hf_token = "") -> List[Segment]:
    cache = workdir / "transcript.json"
    if cache.exists() and not force:
        log.info("Loading cached transcript: %s", cache)
        data = json.loads(cache.read_text())
        return [Segment(**s) for s in data]

    if device == "mps":
        import whispermlx
        from whispermlx.diarize import DiarizationPipeline
        asr_options = {
            "temperatures" : [temperature],
            "logprob-threshold" : -0.5
        }
        if model_size.endswith("8bit"):
            compute_type = "int8"
            wmodel = whispermlx.load_model(f"mlx-community/whisper-{model_size}", device="cpu",
                    compute_type='8bit', asr_options=asr_options, diarize=diarize)
        else:
            wmodel = whispermlx.load_model(f"mlx-community/whisper-{model_size}", device="cpu",
                    asr_options=asr_options, diarize=diarize)

        out = wmodel.transcribe(str(vocals_wav))
        raw_segments = out['segments']
        log.info(out)
        #log.info("Detected language=%s sample=%s", out['language'], raw_segments[0]['text'])

        segments_out: List[Segment] = []
        for i, seg in enumerate(raw_segments):
            text = seg['text'].strip()
            if not text:
                continue
            segments_out.append(Segment(
                index=i, start=seg['start'], end=seg['end'],
                speaker=speaker, text_en=text,
            ))
    else:
        from faster_whisper import WhisperModel

        compute_type = "float16" if device == "cuda" else "int8"
        wmodel = WhisperModel(model_size, device=device, compute_type=compute_type)

        raw_segments, info = wmodel.transcribe(
            str(vocals_wav), language="en", vad_filter=True, word_timestamps=False
        )
        log.info("Detected language=%s, duration=%.1fs", info.language, info.duration)

        segments_out: List[Segment] = []
        for i, seg in enumerate(raw_segments):
            text = seg.text_en.strip()
            if not text:
                continue
            segments_out.append(Segment(
                index=i, start=seg.start, end=seg.end,
                speaker=speaker, text_en=text,
            ))

    log.info(segments_out)

    cache.write_text(json.dumps([asdict(s) for s in segments_out], indent=2))
    return segments_out


# --------------------------------------------------------------------------
# Stage 4 (optional): speaker diarization -> assign speaker labels + build
# per-speaker reference clips for voice cloning
# --------------------------------------------------------------------------

# 4. Perform emotion inference
def analyze_speech(speech_array, feature_extractor, emotion_model, gender_model, sr=16000, device='mps'):
    # Tokenize and extract speech features
    inputs = feature_extractor(
        speech_array,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    ).to(device)

    # Run model forward pass without calculating gradients
    with torch.no_grad():
        # Get emotion prediction
        emotion_logits = emotion_model(**inputs).logits
        predicted_emotion_id = torch.argmax(emotion_logits, dim=-1).item()
        emotion_label = emotion_model.config.id2label[predicted_emotion_id]

        # Get gender prediction
        gender_logits = gender_model(**inputs).logits
        predicted_gender_id = torch.argmax(gender_logits, dim=-1).item()
        gender_label = gender_model.config.id2label[predicted_gender_id]

    return {
        "emotion": emotion_label,
        "gender": gender_label
    }

def diarize_and_assign(vocals_wav: Path, segments: List[Segment], workdir: Path,
            hf_token: Optional[str], device = "mps", force: bool = False) -> List[Segment]:
    cache = workdir / "diarized.json"
    if cache.exists() and not force:
        data = json.loads(cache.read_text())
        return [Segment(**s) for s in data]

    from pyannote.audio import Pipeline

    # 1. Configuration and Model Checkpoints (SOTA WavLM-Large variants)
    #GENDER_MODEL_ID = "tiantiaf/wavlm-large-age-sex"
    #gender_model = AutoModelForAudioClassification.from_pretrained(GENDER_MODEL_ID).to(device)

    #EMOTION_MODEL_ID = "tiantiaf/wavlm-large-categorical-emotion"
    #emotion_model = AutoModelForAudioClassification.from_pretrained(EMOTION_MODEL_ID).to(device)
    #from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
    emodel_checkpoint = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    efeature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(emodel_checkpoint)
    emodel = AutoModelForAudioClassification.from_pretrained(emodel_checkpoint).to(device)

    gender_model_id = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
    gender_model = AutoModelForAudioClassification.from_pretrained(gender_model_id).to(device)

    pl = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=hf_token
    )
    audio, sr = torchaudio.load(str(vocals_wav))
    slow_sr=16000
    if sr != slow_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=slow_sr)
        waveform = resampler(audio)

    audio_in_memory = {
        "waveform": waveform,
        "sample_rate": slow_sr
    }
    diarization = pl(audio_in_memory)

    def speaker_at(t: float) -> str:
        for turn, _, spk in diarization.speaker_diarization.itertracks(yield_label=True):
            if turn.start <= t <= turn.end:
                return spk
        return "SPEAKER_default"

    #waveform = waveform.squeeze(0).numpy()
    for seg in segments:
        mid = (seg.start + seg.end) / 2
        seg.speaker = speaker_at(mid)
        # seg.speaker = "SPEAKER_default"

        # Get speaker gender
        # Get speaker age
        # Get speaker emotion
        out = analyze_speech(waveform[int(seg.start*slow_sr):int(seg.end*slow_sr)], efeature_extractor, emodel, gender_model)
        seg.emotion = out['emotion']
        seg.gender = out['gender']
        print(seg.emotion, seg.gender)

    # Build one reference clip per speaker (first ~6s of their longest turn)
    ref_dir = ensure_dir(workdir / "speaker_refs")
    best_turn_per_speaker = {}
    for turn, _, spk in diarization.itertracks(yield_label=True):
        dur = turn.end - turn.start
        if spk not in best_turn_per_speaker or dur > best_turn_per_speaker[spk][1]:
            best_turn_per_speaker[spk] = (turn, dur)

    ref_paths = {}
    f5_sr = 24000
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=f5_sr)
    audio = resampler(audio)
    for spk, (turn, _dur) in best_turn_per_speaker.items():
        start_sample = int(turn.start * f5_sr)
        end_sample = int(min(turn.end, turn.start + 8.0) * f5_sr)
        clip = audio[start_sample:end_sample]
        ref_path = ref_dir / f"{spk}.wav"
        sf.write(str(ref_path), clip, f5_sr)
        ref_paths[spk] = str(ref_path)

    for seg in segments:
        seg.ref_audio_path = ref_paths.get(seg.speaker)

    cache.write_text(json.dumps([asdict(s) for s in segments], indent=2))
    return segments


def build_single_speaker_ref(vocals_wav: Path, segments: List[Segment], workdir: Path) -> List[Segment]:
    """Fallback when diarization is off: everyone shares one reference clip
    (the longest single segment) so cloned voice is at least consistent."""
    import soundfile as sf
    ref_dir = ensure_dir(workdir / "speaker_refs")
    longest = max(segments, key=lambda s: s.end - s.start)
    audio, sr = librosa.load(str(vocals_wav), sr=24000)
    start_sample = int(longest.start * sr)
    end_sample = int(min(longest.end, longest.start + 8.0) * sr)
    ref_path = ref_dir / "SPEAKER_default.wav"
    sf.write(str(ref_path), audio[start_sample:end_sample], sr)
    for seg in segments:
        seg.speaker = "SPEAKER_default"
        seg.ref_audio_path = str(ref_path)
    return segments


def build_all_ref(vocals_wav: Path, segments: List[Segment], workdir: Path, f5_sr = 24000) -> List[Segment]:
    """Fallback when diarization is off: everyone shares one reference clip
    (the longest single segment) so cloned voice is at least consistent."""
    import soundfile as sf
    ref_dir = ensure_dir(workdir / "speaker_refs")
    longest = max(segments, key=lambda s: s.end - s.start)
    audio, sr = librosa.load(str(vocals_wav), sr=f5_sr)
    start_sample = int(longest.start * sr)
    end_sample = int(min(longest.end, longest.start + 8.0) * sr)
    ref_path = ref_dir / "SPEAKER_default.wav"
    for seg in segments:
        ref_path = ref_dir / f"seg_{seg.index:04d}.wav"
        if not ref_path.is_file():
            seg.speaker = "SPEAKER_default"
            start_sample = int(seg.start * f5_sr)
            end_sample = int(min(seg.end, seg.start+8.0) * f5_sr)
            clip = audio[start_sample:end_sample]
            sf.write(str(ref_path), clip, f5_sr)
        seg.ref_audio_path = str(ref_path)
    return segments

# --------------------------------------------------------------------------
# Stage 5: translation (NLLB-200, English -> Hindi)
# --------------------------------------------------------------------------

def translate_segments(segments: List[Segment], workdir: Path, force: bool = False,
                        device: str = "cuda") -> List[Segment]:
    cache = workdir / "translated.json"
    if cache.exists() and not force:
        data = json.loads(cache.read_text())
        cached = [Segment(**s) for s in data]
        if all(s.text_hi for s in cached):
            return cached

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    torch_device = resolve_device(device, transcribe)

    model.to(torch_device)

    hi_token_id = tok.convert_tokens_to_ids("hin_Deva")

    for seg in segments:
        inputs = tok(seg.text_en, return_tensors="pt").to(torch_device)
        generated = model.generate(
            **inputs, forced_bos_token_id=hi_token_id, max_new_tokens=256
        )
        seg.text_hi = tok.batch_decode(generated, skip_special_tokens=True)[0].strip()
        log.info("[%s] EN: %s", seg.index, seg.text_en)
        log.info("[%s] HI: %s", seg.index, seg.text_hi)

    cache.write_text(json.dumps([asdict(s) for s in segments], indent=2))
    return segments


# --------------------------------------------------------------------------
# Stage 6: voice-cloned Hindi TTS (Coqui XTTS-v2)
# --------------------------------------------------------------------------

def synthesize_segments_indic(segments: List[Segment], workdir: Path, force: bool = False, 
                        device: str = "cpu", sr = 44100) -> List[Segment]:
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    import soundfile as sf

    device = resolve_device(device, 'synthesize')
    model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")

    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    description = f"Rohit delivers a clear speech at a moderate speed and low pitch. The recording is of very high quality, with the speaker's voice sounding clear."
    description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)

    tts_dir = ensure_dir(workdir / "tts_raw")
    for seg in segments:
        out_path = tts_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.tts_audio_path = str(out_path)
            continue
        if not seg.ref_audio_path:
            log.warning("No reference audio for segment %s, skipping", seg.index)
            continue
        prompt_input_ids = tokenizer(seg.text_hi, return_tensors="pt").to(device)
        generation = model.generate(input_ids=description_input_ids.input_ids, attention_mask=description_input_ids.attention_mask, 
                                    prompt_input_ids=prompt_input_ids.input_ids, prompt_attention_mask=prompt_input_ids.attention_mask)
        audio_arr = generation.cpu().numpy().squeeze()
        # Openvoice style transfer
        sf.write(str(out_path), audio_arr, model.config.sampling_rate)
        seg.tts_audio_path = str(out_path)
    return segments


def synthesize_segments_f5(segments: List[Segment], workdir: Path, force: bool = False,
                          device: str = "mps", hf_token="") -> List[Segment]:
    from transformers import AutoModel
    import soundfile as sf
    import numpy as np

    tts_dir = ensure_dir(workdir / "f5_tts_raw")
    repo_id = "ai4bharat/IndicF5"
    model = AutoModel.from_pretrained(repo_id, token=hf_token, trust_remote_code=True)
    for seg in segments:
        log.info(seg.text_hi)
        out_path = tts_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.tts_audio_path = str(out_path)
            continue
        if not seg.ref_audio_path:
            log.warning("No reference audio for segment %s, skipping", seg.index)
            continue
        hi_audio = model(
            seg.text_hi,
            ref_audio_path = seg.ref_audio_path,
            ref_text = seg.text_en
        )
        if hi_audio.dtype == np.int16:
            hi_audio = hi_audio.astype(np.float32) / 32768.02
        sf.write(out_path, np.array(hi_audio, dtype=np.float32), samplerate=24000)
        seg.tts_audio_path = str(out_path)
    return segments

def synthesize_segments(segments: List[Segment], workdir: Path, force: bool = False,
                          device: str = "mps") -> List[Segment]:
    from TTS.api import TTS

    tts_dir = ensure_dir(workdir / "tts_raw")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    for seg in segments:
        log.info(seg.text_hi)
        out_path = tts_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.tts_audio_path = str(out_path)
            continue
        if not seg.ref_audio_path:
            log.warning("No reference audio for segment %s, skipping", seg.index)
            continue
        tts.tts_to_file(
            text=seg.text_hi,
            speaker_wav=seg.ref_audio_path,
            language="hi",
            file_path=str(out_path),
        )
        seg.tts_audio_path = str(out_path)
    return segments



# --------------------------------------------------------------------------
# Stage 7: time-align each clip to its original slot duration
# --------------------------------------------------------------------------

def get_duration(wav_path: Path) -> float:
    import soundfile as sf
    info = sf.info(str(wav_path))
    return info.frames / info.samplerate


def time_stretch(in_path: Path, out_path: Path, factor: float):
    """factor > 1.0 speeds up (shrinks duration); < 1.0 slows down.
    ffmpeg's atempo filter only accepts 0.5-2.0 per instance, so chain
    multiple stages for extreme ratios."""
    factor = max(0.5, min(factor, 2.0))  # clamp to avoid extreme distortion;
    # for durations that would require more, the clip is simply allowed to
    # overrun slightly rather than sounding unnatural.
    run([
        "ffmpeg", "-y", "-i", str(in_path),
        "-filter:a", f"atempo={factor:.4f}",
        str(out_path),
    ])


def align_segments(segments: List[Segment], workdir: Path, force: bool = False,
                    max_stretch: float = 1.6) -> List[Segment]:
    aligned_dir = ensure_dir(workdir / "tts_aligned")
    for seg in segments:
        if not seg.tts_audio_path:
            continue
        out_path = aligned_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.aligned_audio_path = str(out_path)
            continue

        target_dur = max(seg.end - seg.start, 0.1)
        actual_dur = get_duration(Path(seg.tts_audio_path))
        factor = actual_dur / target_dur  # >1 means synthesized audio is
        # longer than the slot -> need to speed it up (atempo > 1)
        factor = max(1.0 / max_stretch, min(factor, max_stretch))
        if abs(factor - 1.0) < 0.03:
            shutil.copy(seg.tts_audio_path, out_path)
        else:
            time_stretch(Path(seg.tts_audio_path), out_path, factor)
        seg.aligned_audio_path = str(out_path)
    return segments


# --------------------------------------------------------------------------
# Stage 8: reassemble full track, loudness-match, mix with background
# --------------------------------------------------------------------------

def build_hindi_vocal_track(segments: List[Segment], total_duration: float,
                             workdir: Path, sample_rate: int = 44100) -> Path:
    import numpy as np
    import soundfile as sf
    import resampy

    out_path = workdir / "hindi_vocals_full.wav"
    canvas = np.zeros((int(total_duration * sample_rate) + sample_rate, 2), dtype=np.float32)

    for seg in segments:
        if not seg.aligned_audio_path:
            continue
        audio, sr = sf.read(seg.aligned_audio_path, dtype="float32")
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1)
        if sr != sample_rate:
            # simple resample via ffmpeg would be cleaner; for brevity we
            # require XTTS output at target sample_rate (it emits 24kHz by
            # default -- resample in align_segments/time_stretch stage if
            # you need bit-exact sample rates end to end)
            pass
        start_sample = int(seg.start * sample_rate)
        end_sample = start_sample + len(audio)
        if end_sample > canvas.shape[0]:
            pad = end_sample - canvas.shape[0]
            canvas = np.pad(canvas, ((0, pad), (0, 0)))
        canvas[start_sample:end_sample] += audio[:, :2] if audio.shape[1] >= 2 else audio

    canvas = resampy.resample(canvas, 16000, sample_rate)

    sf.write(str(out_path), canvas, sample_rate)
    return out_path


def loudness_match_and_mix(hindi_vocals: Path, background: Path, reference_vocals: Path,
                             workdir: Path) -> Path:
    """Normalize the synthesized Hindi vocal track towards the loudness of
    the original English vocal track, then mix with the (untouched)
    background stem."""
    normalized = workdir / "hindi_vocals_normalized.wav"
    # Two-pass loudnorm using original vocal track's measured loudness as
    # the target is more involved; for simplicity we normalize to a fixed
    # broadcast-style target (-16 LUFS) which works well against most
    # Demucs background stems. Adjust target_i if it sounds off.
    run([
        "ffmpeg", "-y", "-i", str(hindi_vocals),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        str(normalized),
    ])

    mixed = workdir / "final_hindi_track.wav"
    run([
        "ffmpeg", "-y",
        "-i", str(normalized), "-i", str(background),
        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0",
        str(mixed),
    ])
    return mixed


# --------------------------------------------------------------------------
# Stage 9: mux into the video as an additional audio track
# --------------------------------------------------------------------------

def mux_into_video(video_path: Path, hindi_track: Path, output_path: Path):
    run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(hindi_track),
        "-map", "0:v", "-map", "0:a", "-map", "1:a",
        "-c:v", "copy",
        "-c:a:0", "copy",
        "-c:a:1", "aac_at", "-b:a:1", "192k",
        "-metadata:s:a:0", "language=eng",
        "-metadata:s:a:1", "language=hin",
        "-metadata:s:a:1", "title=Hindi (AI Dubbed)",
        "-disposition:a:0", "default",
        "-disposition:a:1", "0",
        str(output_path),
    ])


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def get_media_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="source video file")
    ap.add_argument("--output", required=True, type=Path, help="output video file with Hindi track added")
    ap.add_argument("--workdir", default=Path("./dub_work"), type=Path)
    ap.add_argument("--device", default="mps", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--whisper-model", default="large-v3-turbo")
    ap.add_argument("--demucs-model", default="htdemucs")
    ap.add_argument("--diarize", action="store_true", help="enable multi-speaker diarization (needs HF token for pyannote)")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--force", action="store_true", help="re-run all stages even if cached output exists")
    args = ap.parse_args()

    workdir = ensure_dir(args.workdir)
    log.info("Working directory: %s", workdir.resolve())

    # 1. extract
    audio_wav = extract_audio(args.input, workdir / "source_audio.wav", force=args.force)

    # 2. separate
    vocals_wav, background_wav = separate_vocals(
        audio_wav, workdir, force=args.force, device=args.device, model=args.demucs_model
    )

    # 3. transcribe
    segments = transcribe(vocals_wav, workdir, force=args.force,
                           model_size=args.whisper_model, device=args.device,
                          diarize=args.diarize, hf_token=args.hf_token)
    if not segments:
        log.error("No speech detected -- aborting.")
        sys.exit(1)

    # 4. diarize (optional) or single-speaker reference
    #if args.diarize:
    #    segments = diarize_and_assign(vocals_wav, segments, workdir, args.hf_token, force = args.force)
    #else:
    #    segments = build_single_speaker_ref(vocals_wav, segments, workdir)
    segments = build_all_ref(vocals_wav, segments, workdir)

    # 5. translate
    segments = translate_segments(segments, workdir, force=args.force, device=args.device)

    # 6. synthesize
    #segments = synthesize_segments_f5(segments, workdir, force=args.force, hf_token = args.hf_token)
    segments = synthesize_segments_indic(segments, workdir, force=args.force)

    # 7. align to original timing
    segments = align_segments(segments, workdir, force=args.force)

    # 8. reassemble + mix
    total_duration = get_media_duration(args.input)
    hindi_vocals_full = build_hindi_vocal_track(segments, total_duration, workdir, sample_rate= librosa.get_samplerate(vocals_wav))
    final_hindi_track = loudness_match_and_mix(hindi_vocals_full, background_wav, vocals_wav, workdir)

    # 9. mux into video
    mux_into_video(args.input, final_hindi_track, args.output)

    log.info("Done. Output: %s", args.output.resolve())


if __name__ == "__main__":
    main()
