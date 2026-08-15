#!/usr/bin/env python3
"""
dub_pipeline_direct_s2st.py
============================
Variant of the dubbing pipeline that performs DIRECT speech-to-speech
translation (English audio in -> Hindi audio out) with NO transcription
step anywhere in the pipeline, then applies a separate voice style
transfer stage to make the translated speech sound like the original
speaker.

WHY THIS IS DIFFERENT FROM dub_pipeline.py
    dub_pipeline.py chains ASR (Whisper) -> MT (NLLB text translation) ->
    cloned TTS. That pipeline produces a text transcript as an
    intermediate artifact and relies on TTS voice-cloning to approximate
    the speaker.

    This script instead:
      - Segments speech using pure Voice Activity Detection (Silero VAD),
        which finds speech/silence boundaries acoustically -- it does NOT
        recognize or output any words.
      - Feeds each speech segment's raw audio directly into Meta's
        SeamlessM4T-v2 model in speech-to-speech (S2ST) mode, which maps
        source-language speech to target-language speech end-to-end. No
        intermediate transcript is generated or used anywhere in this
        script.
      - Because SeamlessM4T-v2's S2ST output uses a model-internal voice
        (not the original speaker's), it runs a dedicated VOICE STYLE
        TRANSFER stage (OpenVoice tone-color conversion) that re-times
        the translated audio's timbre to match a short reference clip of
        the original speaker, while keeping the translated content and
        prosody from the S2ST step.

PIPELINE STAGES
    1. Extract audio from the source video (ffmpeg)
    2. Separate vocals from background music/effects (Demucs)
    3. Voice Activity Detection -> speech segment boundaries (Silero VAD)
       -- acoustic only, no transcription
    4. Direct speech-to-speech translation per segment, English -> Hindi
       (SeamlessM4T-v2, generate_speech=True)
    5. Voice style transfer: convert the translated segment's timbre to
       match the original speaker (OpenVoice ToneColorConverter), using a
       short reference clip pulled from that speaker's own audio
    6. Time-stretch each converted clip to fit its original segment
       duration
    7. Reassemble a full-length Hindi vocal track, loudness-match it, and
       mix with the separated background track
    8. Mux the new Hindi track into the video as an ADDITIONAL audio
       track (the original English track is kept), tagged language=hin

HONEST CAVEATS
    - SeamlessM4T-v2 was trained using text as an auxiliary/intermediate
      training signal internally, but its public inference API for S2ST
      takes audio in and returns audio out -- no transcript is produced
      or consumed by this script. That is what "direct" means here; it is
      the closest practical thing to true direct S2ST available
      open-source today.
    - Direct S2ST models generally translate shorter utterances more
      reliably than long, complex sentences. Very long segments may be
      truncated or mistranslated; consider tightening VAD segmentation
      (shorter max_segment_len) if you see this.
    - The voice style transfer stage transfers timbre (voice "color"),
      not necessarily full prosody/emotion -- expressiveness is limited
      by what SeamlessM4T-v2's S2ST output already carries.
    - Overlapping speech, heavy accents, and noisy vocal-separation
      artifacts will degrade both the S2ST and voice-conversion stages.
    - For production-grade dubbing, SeamlessExpressive (Meta's research
      model with better prosody preservation) or commercial dubbing APIs
      currently outperform this from-scratch open-source stack.

REQUIREMENTS
    - ffmpeg on PATH
    - Python packages: see requirements_direct_s2st.txt
    - A GPU (CUDA or Apple Silicon MPS) is strongly recommended --
      SeamlessM4T-v2-large is a large multilingual model
    - Internet access to download model weights on first run
      (facebook/seamless-m4t-v2-large, Silero VAD via torch.hub, and
      OpenVoice checkpoints)

USAGE
    python dub_pipeline_direct_s2st.py \\
        --input movie.mp4 \\
        --output movie.hindi_dubbed.mp4 \\
        --workdir ./work_direct \\
        --device cuda
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import soundfile as sf
import librosa
import numpy as np
import torch
from collections import Counter, defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dub_pipeline_direct_s2st")

# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Segment:
    """One VAD-detected speech region. No text is ever stored here --
    only audio file paths and timing."""
    index: int
    start: float
    end: float
    speaker: str = "SPEAKER_default"
    text_en: Optional[str] = ""
    gender: Optional[str] = None
    pitch_hz: Optional[float] = None
    speed_wps: Optional[float] = None 

    ref_audio_path: Optional[str] = None       # clip of original speaker, for voice style transfer
    s2st_audio_path: Optional[str] = None       # raw SeamlessM4T-v2 Hindi output (generic voice)
    styled_audio_path: Optional[str] = None     # after voice style transfer
    aligned_audio_path: Optional[str] = None    # after time-stretch to fit start/end


def run(cmd: List[str], **kwargs):
    log.info("$ %s", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available, falling back to CPU")
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        log.warning("MPS requested but not available, falling back to CPU")
        return "cpu"
    return requested

def get_audio_files_in_dir(loc:Path)-> List[Path]:
    audio_extensions = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"}
    # Get all audio files recursively
    return sorted([
        file for file in loc.rglob("*") 
        if file.is_file() and file.suffix.lower() in audio_extensions
    ])

def get_speaker_mapping(speakers_dir = Path("./speakers"), use_from_source = False):
    speaker_mapping = {}
    if not use_from_source:
        for spk_cls in speakers_dir.iterdir():
            if spk_cls.is_dir():
                speaker_mapping = [(spk_cls.name.lower(), id) for id in get_audio_files_in_dir(spk_cls)]
    return speaker_mapping

# --------------------------------------------------------------------------
# Stage 1: extract audio
# --------------------------------------------------------------------------

def extract_audio(video_path: Path, out_wav: Path, force: bool = False) -> Path:
    if out_wav.exists() and not force:
        log.info("Skipping extract_audio (exists): %s", out_wav)
        return out_wav
    run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(out_wav),
    ])
    return out_wav


# --------------------------------------------------------------------------
# Stage 2: source separation (Demucs)
# --------------------------------------------------------------------------

def separate_vocals(stereo_wav: Path, workdir: Path, force: bool = False,
                     device: str = "cpu", model: str = "htdemucs", demucs: str = "demucs"):
    out_dir = workdir / "demucs_out"
    vocals = out_dir / model / stereo_wav.stem / "vocals.wav"
    background = out_dir / model / stereo_wav.stem / "no_vocals.wav"
    vocals_16k = workdir / "vocals_16k_mono.wav"
    if not(vocals.exists() and background.exists() and vocals_16k.exists() and not force):
        ensure_dir(out_dir)
        run([sys.executable, "-m", demucs, "-n", model, "--two-stems", "vocals",
             "-d", device, "-o", str(out_dir), str(stereo_wav)])
        if not vocals.exists():
            raise FileNotFoundError(f"Demucs did not produce expected output: {vocals}")
        # 16kHz mono copy of the *clean* vocal stem -- used for ASR, classifiers,
        # pitch extraction, and as the cloning reference source (cleaner than
        # the original mixed audio, since music/fx are stripped out)
        vocals_16k = workdir / "vocals_16k_mono.wav"
        if not vocals_16k.exists() or force:
            run(["ffmpeg", "-y", "-i", str(vocals), "-ac", "1", "-ar", "16000", str(vocals_16k)])
    # Adjust loudness
    from pydub import AudioSegment
    audio = AudioSegment.from_file(vocals_16k)
    normalized_audio = audio.normalize(headroom=0.3)
    normalized_audio.export(vocals_16k, format="wav")
    return vocals, background, vocals_16k, audio.max

# --------------------------------------------------------------------------
# Stage 3: Voice Activity Detection (Silero VAD) -- NO transcription
# --------------------------------------------------------------------------

def vad_segment(vocals_wav: Path, workdir: Path, force: bool = False,
                 max_segment_len: float = 12.0, min_silence_ms: int = 300) -> List[Segment]:
    """Finds speech regions purely acoustically. Produces (start, end)
    timestamps only -- no words are recognized at this stage. Long
    detected regions are chunked to max_segment_len since S2ST models
    handle short utterances more reliably."""
    cache = workdir / "vad_segments.json"
    if cache.exists() and not force:
        data = json.loads(cache.read_text())
        return [Segment(**s) for s in data]

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    (get_speech_timestamps, _, read_audio, *_rest) = utils

    audio = read_audio(str(vocals_wav), sampling_rate=16000)
    raw_timestamps = get_speech_timestamps(
        audio, model, sampling_rate=16000, min_silence_duration_ms=min_silence_ms
    )

    segments: List[Segment] = []
    idx = 0
    for ts in raw_timestamps:
        start = ts["start"] / 16000.0
        end = ts["end"] / 16000.0
        # chunk long regions so S2ST sees manageable utterance lengths
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + max_segment_len, end)
            segments.append(Segment(index=idx, start=cursor, end=chunk_end))
            idx += 1
            cursor = chunk_end

    log.info("VAD found %d speech segments (no words recognized, timing only)", len(segments))
    cache.write_text(json.dumps([asdict(s) for s in segments], indent=2))
    return segments

def transcribe_and_diarize(vocals_16k: Path, workdir: Path, force: bool = False,
                 model_size: str = "large-v3", device: str = "cpu",
                 diarize: bool = True, hf_token: Optional[str] = None) -> List[Segment]:
    cache = workdir / "transcript_diarized.json"
    ref_dir = ensure_dir(workdir / "speaker_refs")
    if cache.exists() and not force:
        shutil.copytree(ref_dir, f"./raw_audio/{workdir.name}")
        data = json.loads(cache.read_text())
        return [Segment(**s) for s in data]

    import whispermlx
    asr_options = {
        "temperatures" : [0.4],
        "logprob-threshold" : -0.25,
        #"condition_on_previous_text": False
    }
    model = whispermlx.load_model(model_size, device=device, asr_options=asr_options)
    result = model.transcribe(str(vocals_16k))
    log.info("Transcribed %d raw segments (language=%s)",
              len(result.get("segments", [])), result.get("language", "en"))

    # Word-level alignment improves diarization speaker-assignment accuracy
    try:
        model_a, metadata = whispermlx.load_align_model(
            language_code=result.get("language", "en"), device=device
        )
        result = whispermlx.align(result["segments"], model_a, metadata, str(vocals_16k), device=device)
    except Exception as e:
        log.warning("Word alignment failed (%s); continuing with segment-level timestamps", e)

    if diarize:
        from whispermlx.diarize import DiarizationPipeline
        diarize_model = DiarizationPipeline(token=hf_token, device=device)
        diarize_segments = diarize_model(str(vocals_16k))
        result = whispermlx.assign_word_speakers(diarize_segments, result)

    from transformers import pipeline
    gender_model = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
    gender_classifier = pipeline("audio-classification", model=gender_model) 

    segments: List[Segment] = []
    
    audio, sr = sf.read(str(vocals_16k))
    for i, seg in enumerate(result["segments"]):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        ref_path = ref_dir / f"{speaker}_{i}.wav"
        if not speaker and seg.get("words"):
            # fall back to majority vote over this segment's words
            spk_counts = Counter(w.get("speaker") for w in seg["words"] if w.get("speaker"))
            speaker = spk_counts.most_common(1)[0][0] if spk_counts else "SPEAKER_00"
        
        # Audio clip
        start_sample = int(seg["start"] * sr)
        #end_sample = int(min(seg["end"], seg["start"] + 15.0) * sr)
        end_sample = int(seg["end"] * sr) # needed for complete s2st
        clip = audio[start_sample:end_sample]

        # Get gender
        results = gender_classifier(clip)
        gender = results[0]['label'].lower()

        # Reference voice id
        try:
            speaker_id = int(speaker.split('_')[-1])
        except:
            speaker_id = -1

        # Write audio
        sf.write(str(ref_path), clip, sr)
        # Finalized segment
        segments.append(Segment(
            index=i, start=float(seg["start"]), end=float(seg["end"]),
            speaker=speaker_id, text_en=text,
            ref_audio_path = str(ref_path), gender=gender
        ))

    cache.write_text(json.dumps([asdict(s) for s in segments], indent=2))
    shutil.copytree(ref_dir, f"./raw_audio/{workdir.name}")
    log.info("Final: %d segments across %d speakers", len(segments),
              len({s.speaker for s in segments}))
    return segments


def build_speaker_reference(segments: List[Segment], spk_map = {}, use_originals = False, 
                            min_audio_duration = 3.0, max_audios = 6) -> List[Segment]:
    """Speaker references."""
    if use_originals:
        from heapq import heappush, heappop
        import random
        speaker_thresh = defaultdict(min_audio_duration)
        speakers = defaultdict([])
        for seg in segments:
            dur = seg.end - seg.start
            if speaker_thresh[seg.speaker][0] < dur:
                heappush(speakers[seg.speaker], (dur, seg.ref_audio_path))
                if len(speakers[seg.speaker]) >max_audios:
                    heappop(speakers[seg.speaker])
        # Returning randomly chosen the list of longest phrases
        spk_map = {k:(seg.gender, random.choice(list(zip(*v))[1])) for k, v in speakers.items()}

    for seg in segments:
        if spk_map != {}:
            # If speaker examples provided, loops over the available voices for different speakers
            try:
                seg.ref_audio_path = spk_map[(seg.gender, seg.speaker % len(spk_map[seg.gender]))]
            except:
                pass
    return segments


# --------------------------------------------------------------------------
# Stage 4: DIRECT speech-to-speech translation (SeamlessM4T-v2)
# --------------------------------------------------------------------------

def direct_s2st(vocals_wav: Path, segments: List[Segment], workdir: Path,
                 force: bool = False, device: str = "cpu") -> List[Segment]:
    """Feeds raw audio chunks directly into SeamlessM4T-v2's S2ST head and
    gets raw audio back. No text is generated or consumed anywhere in this
    function."""
    import torchaudio
    from transformers import AutoProcessor, SeamlessM4Tv2Model

    device = resolve_device(device)
    out_dir = ensure_dir(workdir / "s2st_raw")

    model_name = "facebook/seamless-m4t-v2-large"
    processor = AutoProcessor.from_pretrained(model_name)
    model = SeamlessM4Tv2Model.from_pretrained(model_name).to(device)
    generation_config = {
        "tgt_lang": "hin",
        "generate_speech":True,
        # Reduce repetition, improve quality
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 4,
        "length_penalty": 1.0,
        "temperature" : 0.6,
    }

    new_sr = 16000
    audio, sr = torchaudio.load(vocals_wav)
    if sr != new_sr:
        resampler = torchaudio.transforms.Resample(sr, new_sr)
        audio = resampler(audio)

    np_audio = audio[0].numpy()
    for seg in segments:
        out_path = out_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.s2st_audio_path = str(out_path)
            continue

        if seg.end - seg.start < 0.05:  # skip slivers under ~50ms
            continue
        elif seg.end - seg.start < 0.4 or len(seg.text_en) < 10: # Keep originals for short clips
            seg.s2st_audio_path =  seg.ref_audio_path
        else:
            start_sample = int(seg.start * new_sr)
            end_sample = int(seg.end * new_sr)
            chunk = np_audio[start_sample:end_sample]

            inputs = processor(audio=chunk, sampling_rate=new_sr, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(**inputs, speaker_id = seg.speaker, 
                                        max_new_tokens= min(len(seg.text_en), 50),
                                        **generation_config) 

            waveform = output[0][0].cpu().numpy().squeeze()
            sf.write(str(out_path), waveform, model.config.sampling_rate)
            seg.s2st_audio_path = str(out_path)

            log.info("Segment %d: %.2fs -> %.2fs translated (direct S2ST, no text)",
                    seg.index, seg.start, seg.end)
    return segments


# --------------------------------------------------------------------------
# Stage 5: Voice style transfer (OpenVoice tone-color conversion)
# --------------------------------------------------------------------------

def voice_style_transfer(segments: List[Segment], workdir: Path, force: bool = False,
                           device: str = "cpu", checkpoint_dir: str = "checkpoints_v2") -> List[Segment]:
    """Re-times each S2ST segment's timbre to match the original speaker's
    reference clip, keeping the translated Hindi content/prosody from
    stage 4 and only transferring voice color."""
    from openvoice.api import ToneColorConverter
    from openvoice import se_extractor

    out_dir = ensure_dir(workdir / "voice_styled")
    converter = ToneColorConverter(f"{checkpoint_dir}/config.json", device=device)
    converter.load_ckpt(f"{checkpoint_dir}/checkpoint.pth")

    # cache target-speaker embeddings per unique reference clip
    target_se_cache = {}
    for seg in segments:
        if not seg.s2st_audio_path:
            continue
        out_path = out_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.styled_audio_path = str(out_path)
            continue
        if not seg.ref_audio_path:
            log.warning("No speaker reference for segment %d, skipping voice style transfer", seg.index)
            seg.styled_audio_path = seg.s2st_audio_path
            continue

        if seg.ref_audio_path not in target_se_cache:
            target_se, _ = se_extractor.get_se(seg.ref_audio_path, converter, vad=False)
            target_se_cache[seg.ref_audio_path] = target_se
        target_se = target_se_cache[seg.ref_audio_path]

        source_se, _ = se_extractor.get_se(seg.s2st_audio_path, converter, vad=False)

        converter.convert(
            audio_src_path=seg.s2st_audio_path,
            src_se=source_se,
            tgt_se=seg.ref_audio_path,
            output_path=str(out_path),
        )
        seg.styled_audio_path = str(out_path)

    return segments

def voice_style_transfer_chatterbox(segments: List[Segment], workdir: Path, 
                force: bool = False, device: str = "cpu") -> List[Segment]:
    from chatterbox.vc import ChatterboxVC
    import torchaudio
    out_dir = ensure_dir(workdir / "voice_styled")
    voice_model = ChatterboxVC.from_pretrained(
        device=device,
    )

    for seg in segments:
        if not seg.s2st_audio_path:
            continue
        out_path = out_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.styled_audio_path = str(out_path)
            continue
        if not seg.ref_audio_path:
            log.warning("No speaker reference for segment %d, skipping voice style transfer", seg.index)
            seg.styled_audio_path = seg.s2st_audio_path
            continue

        arr = voice_model.generate(
            seg.s2st_audio_path,
            seg.ref_audio_path
        )

        log.info(f"{voice_model.sr} {arr.shape}")
        torchaudio.save(out_path, arr, voice_model.sr)
        #sf.write(out_path, data=arr.numpy(), samplerate=voice_model.sr)
        seg.styled_audio_path = str(out_path)
    return segments

# --------------------------------------------------------------------------
# Stage 6: time-align each clip to its original slot duration
# --------------------------------------------------------------------------

def get_duration(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    return info.frames / info.samplerate


def time_stretch(in_path: Path, out_path: Path, factor: float):
    factor = max(0.5, min(factor, 2.0))
    run([
        "ffmpeg", "-y", "-i", str(in_path),
        "-filter:a", f"atempo={factor:.4f}",
        str(out_path),
    ])

def align_segments(segments: List[Segment], workdir: Path, force: bool = False,
                    max_stretch: float = 1.6) -> List[Segment]:
    aligned_dir = ensure_dir(workdir / "aligned")
    for seg in segments:
        source_path = seg.styled_audio_path or seg.s2st_audio_path
        if not source_path:
            continue
        out_path = aligned_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.aligned_audio_path = str(out_path)
            continue
        
        target_dur = max(seg.end - seg.start, 0.1)
        actual_dur = get_duration(Path(source_path))
        factor = actual_dur / target_dur
        factor = max(1.0 / max_stretch, min(factor, max_stretch))
        if abs(factor - 1.0) < 0.03:
            shutil.copy(source_path, out_path)
        else:
            time_stretch(Path(source_path), out_path, factor)
        seg.aligned_audio_path = str(out_path)
    return segments


# --------------------------------------------------------------------------
# Stage 7a: reassemble full track
# --------------------------------------------------------------------------
def load_in_stereo(wav, sample_rate):
    audio, sr = sf.read(wav, dtype="float64")
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    if sr != sample_rate:
        # resample if needed (XTTS/IndicF5/Parler emit at their own
        # native rates -- 24kHz for IndicF5, model-specific for Parler)
        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=sample_rate).T
    return audio

def build_hindi_vocal_track(segments: List[Segment], total_duration: float,
                             workdir: Path, sample_rate: int = 48000) -> Path:
    out_path = workdir / "hindi_vocals_full.wav"
    canvas = np.zeros((int(total_duration * sample_rate) + sample_rate, 2), dtype=np.float32)

    for seg in segments:
        if not seg.aligned_audio_path:
            continue
        audio = load_in_stereo(seg.aligned_audio_path, sample_rate)
        start_sample = int(seg.start * sample_rate)
        end_sample = start_sample + len(audio)
        if end_sample > canvas.shape[0]:
            pad = end_sample - canvas.shape[0]
            canvas = np.pad(canvas, ((0, pad), (0, 0)))
        canvas[start_sample:end_sample] += audio[:, :2] if audio.shape[1] >= 2 else audio

    sf.write(str(out_path), canvas, sample_rate)
    return out_path

# --------------------------------------------------------------------------
# Stage 7b: Vocal output cleaning and volume matching
# --------------------------------------------------------------------------

def compute_rms_envelope(x, sr, window_ms=50.0, hop_ms=10.0):
    """Windowed RMS envelope of x (mono or multi-channel), computed WITHOUT
    downmixing to mono. For multi-channel input this uses the mean power
    across channels per sample (sqrt(mean(x**2)) over window*channels),
    which avoids the phase-cancellation issues a straight mono average can
    cause, while still giving one loudness value per window.

    Returns (times_seconds, rms_values) where times are window centers.
    """
    win = max(1, int(round(sr * window_ms / 1000)))
    hop = max(1, int(round(sr * hop_ms / 1000)))

    power = np.mean(x.astype(np.float64) ** 2, axis=1) if x.ndim > 1 else x.astype(np.float64) ** 2
    n = len(power)

    if n < win:
        rms = np.sqrt(np.mean(power) + 1e-12)
        return np.array([n / (2 * sr)]), np.array([rms])

    starts = np.arange(0, n - win + 1, hop)
    csum = np.cumsum(np.concatenate(([0.0], power)))
    sums = csum[starts + win] - csum[starts]
    rms = np.sqrt(sums / win)
    times = (starts + win / 2) / sr
    return times, rms


def match_envelope(src, ref, sr_src, sr_ref, window_ms=50.0, hop_ms=10.0,
                    max_gain=8.0, noise_floor_db=-60.0):
    """Return src scaled so its windowed RMS envelope follows ref's envelope.

    Both src and ref keep their original channel layout throughout -- no
    mono downmix is performed anywhere, including in the envelope analysis.
    A single gain curve (derived from combined channel power) is applied
    uniformly across all channels, so stereo/multichannel imaging is
    preserved.
    """
    src = np.asarray(src, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)

    t_src, env_src = compute_rms_envelope(src, sr_src, window_ms, hop_ms)
    t_ref, env_ref = compute_rms_envelope(ref, sr_ref, window_ms, hop_ms)

    # Map the reference envelope onto the source's timeline proportionally,
    # so differing durations / sample rates both work.
    src_dur = src.shape[0] / sr_src
    ref_dur = ref.shape[0] / sr_ref
    src_pos_norm = t_src / src_dur if src_dur > 0 else t_src
    ref_pos_norm = t_ref / ref_dur if ref_dur > 0 else t_ref

    env_ref_on_src = np.interp(
        src_pos_norm, ref_pos_norm, env_ref, left=env_ref[0], right=env_ref[-1]
    )

    noise_floor = 10 ** (noise_floor_db / 20)
    safe_src = np.maximum(env_src, noise_floor)

    gain_frames = env_ref_on_src / safe_src
    gain_frames = np.clip(gain_frames, 1.0 / max_gain, max_gain)

    # Don't boost source silence up to full volume just because the
    # reference is loud there -- cap gain to 1x (i.e. leave it quiet).
    silent = env_src < noise_floor
    gain_frames[silent] = np.minimum(gain_frames[silent], 1.0)

    # Interpolate the gain curve to per-sample resolution: this is what
    # makes the result click-free instead of stepping between windows.
    n_samples = src.shape[0]
    sample_times = np.arange(n_samples) / sr_src
    gain_curve = np.interp(
        sample_times, t_src, gain_frames, left=gain_frames[0], right=gain_frames[-1]
    )

    if src.ndim > 1:
        gain_curve = gain_curve[:, None]

    out = src * gain_curve

    # Peak-safe scaling instead of hard clipping: preserves the relative
    # envelope shape rather than distorting individual samples.
    peak = np.max(np.abs(out)) if out.size else 0.0
    if peak > 0.999:
        out = out * (0.999 / peak)
    return out

def normalize_new_vocals(hindi_vocals_wav: Path, vocals_wav: Path, workdir: Path, 
                        demucs = "demucs", demucs_model="htdemucs", device="cpu", sample_rate: int = 48000) -> Path:
    # Denoise
    out_dir = workdir / "hindi_demucs_out"
    ensure_dir(out_dir)
    run([demucs, "-n", demucs_model, "--two-stems=vocals", "-o", str(out_dir), str(hindi_vocals_wav)])

    # Adjust volume to match original vocals by windowing and scaling
    new_vocals_wav = out_dir / "htdemucs" / "hindi_vocals_full" / "vocals.wav"
    src = load_in_stereo(str(new_vocals_wav), sample_rate)
    ref = load_in_stereo(vocals_wav, sample_rate)
    out = match_envelope(src, ref, sample_rate, sample_rate)

    normalized_wav = workdir / "hindi_vocals_normalized.wav"
    sf.write(str(normalized_wav), out, sample_rate)

    return normalized_wav

# --------------------------------------------------------------------------
# Stage 7c: reassemble full track, loudness-match, mix with background
# --------------------------------------------------------------------------
def loudness_match_and_mix(hindi_vocals: Path, vocals_wav:Path, background: Path, workdir: Path,
                           device = 'cpu') -> Path:
    normalized = normalize_new_vocals(hindi_vocals, vocals_wav, workdir, device = device)
    # run([
    #     "ffmpeg", "-y", "-i", str(hindi_vocals),
    #     "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
    #     str(normalized),
    # ])
    mixed = workdir / "final_hindi_track.wav"
    run([
        "ffmpeg", "-y",
        "-i", str(normalized), "-i", str(background),
        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0",
        str(mixed),
    ])
    return mixed


# --------------------------------------------------------------------------
# Stage 8: mux into the video as an additional audio track
# --------------------------------------------------------------------------

def mux_into_video(video_path: Path, hindi_track: Path, output_path: Path):
    run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(hindi_track),
        "-map", "0:v", "-map", "0:a", "-map", "1:a",
        "-c:v", "copy",
        "-c:a:0", "copy",
        "-c:a:1", "aac", "-b:a:1", "192k",
        "-metadata:s:a:0", "language=eng",
        "-metadata:s:a:1", "language=hin",
        "-metadata:s:a:1", "title=Hindi (AI Dubbed, direct S2ST)",
        "-disposition:a:0", "default",
        "-disposition:a:1", "0",
        str(output_path),
    ])


def get_media_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--workdir", default=Path("./dub_work_direct"), type=Path)
    ap.add_argument("--device", default="mps", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--whisper-model", default="large-v3-turbo")
    ap.add_argument("--hf-token", default=None, help="required for whispermlx diarization (pyannote gated model)")
    ap.add_argument("--demucs-model", default="htdemucs")
    ap.add_argument("--diarize", action="store_true")
    ap.add_argument("--no-voice-cloning", action="store_true")
    ap.add_argument("--openvoice-checkpoints", default="checkpoints_v2",
                     help="path to downloaded OpenVoice V2 checkpoint directory")
    ap.add_argument("--max-segment-len", type=float, default=12.0,
                     help="max seconds per VAD-detected chunk fed to S2ST")
    ap.add_argument("--vc", default="chatterbox", choices=["openvoice", "chatterbox"])
    ap.add_argument("--speakers-dir", default=Path("./speakers"), help="Directory for dubbing voices")
    ap.add_argument("--original-speakers", action='store_true', help="Use voices from the speaker in the video")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    workdir = ensure_dir(args.workdir)
    log.info("Working directory: %s", workdir.resolve())

    # 0. Speaker mapping -> Dict {(speaker_category, speaker_id) : speaker_reference_filepath}
    spk_map = get_speaker_mapping(Path(args.speakers_dir), args.original_speakers)

    # 1. extract
    audio_wav = extract_audio(args.input, workdir / "source_audio.wav", force=args.force)

    # 2. separate
    vocals_wav, background_wav, vocals_16k, vol = separate_vocals(
        audio_wav, workdir, force=args.force, device=args.device, model=args.demucs_model
    )

    # 3b : Transcribe and diarize
    if args.diarize:
        segments = transcribe_and_diarize(
            vocals_16k, workdir, force=args.force, model_size=args.whisper_model,
            device=args.device, diarize=args.diarize, hf_token=args.hf_token,
        )
    else:
        # 3. VAD segmentation (acoustic only -- no transcription)
        segments = vad_segment(vocals_wav, workdir, force=args.force,
                            max_segment_len=args.max_segment_len)
    if not segments:
        log.error("No speech detected -- aborting.")
        sys.exit(1)

    # 4. direct speech-to-speech translation
    segments = direct_s2st(vocals_wav, segments, workdir, force=args.force, device=args.device)

    # 5. voice style transfer onto the original speaker's timbre
    if not args.no_voice_cloning:
        segments = build_speaker_reference(segments, spk_map)

        if args.vc == "chatterbox":
            segments = voice_style_transfer_chatterbox(segments, workdir, force=args.force, device=args.device)
        else:
            segments = voice_style_transfer(segments, workdir, force=args.force, device=args.device,
                                                checkpoint_dir=args.openvoice_checkpoints)
    else:
        for seg in segments:
            seg.styled_audio_path = seg.s2st_audio_path

    # 6. align to original timing
    segments = align_segments(segments, workdir, force=args.force)

    # 7. reassemble + mix
    total_duration = get_media_duration(args.input)
    hindi_vocals_full = build_hindi_vocal_track(segments, total_duration, workdir)
    final_hindi_track = loudness_match_and_mix(hindi_vocals_full, vocals_wav, background_wav, workdir, args.device)

    # 8. mux into video
    mux_into_video(args.input, final_hindi_track, args.output)

    log.info("Done. Output: %s", args.output.resolve())


if __name__ == "__main__":
    main()
