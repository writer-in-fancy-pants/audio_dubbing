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
    if cache.exists() and not force:
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
    ref_dir = ensure_dir(workdir / "speaker_refs")
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
        end_sample = int(min(seg["end"], seg["start"] + 8.0) * sr)
        clip = audio[start_sample:end_sample]

        # Get gender
        results = gender_classifier(clip)

        # Write audio
        sf.write(str(ref_path), clip, sr)
        # Finalized segment
        segments.append(Segment(
            index=i, start=float(seg["start"]), end=float(seg["end"]),
            speaker=speaker or "SPEAKER_00", text_en=text,
            ref_audio_path = str(ref_path), gender=results[0]['label']
        ))


    cache.write_text(json.dumps([asdict(s) for s in segments], indent=2))
    log.info("Final: %d segments across %d speakers", len(segments),
              len({s.speaker for s in segments}))
    return segments


def build_speaker_reference(vocals_wav: Path, segments: List[Segment], workdir: Path) -> List[Segment]:
    """All segments share one reference clip (the longest segment) so the
    voice style transfer stage has consistent timbre to target. Swap this
    for per-speaker diarization + per-speaker refs if you need multiple
    distinct cloned voices."""
    import soundfile as sf

    ref_dir = ensure_dir(workdir / "speaker_refs")
    longest = max(segments, key=lambda s: s.end - s.start)
    audio, sr = sf.read(str(vocals_wav))
    start_sample = int(longest.start * sr)
    end_sample = int(min(longest.end, longest.start + 8.0) * sr)
    ref_path = ref_dir / "SPEAKER_default.wav"
    sf.write(str(ref_path), audio[start_sample:end_sample], sr)
    for seg in segments:
        seg.ref_audio_path = str(ref_path)
    return segments


# --------------------------------------------------------------------------
# Stage 4: DIRECT speech-to-speech translation (SeamlessM4T-v2)
# --------------------------------------------------------------------------

SPEAKER_MAPPING={
    "male": [58, 46, 33, 1, 51, 4, 5, 6, 8, 12, 19, 24, 25, 26, 27, 29, 30],
    "female": [44, 175, 43, 39, 2, 3, 9, 13, 14, 15, 16, 17, 18]
}

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

        start_sample = int(seg.start * new_sr)
        end_sample = int(seg.end * new_sr)
        chunk = np_audio[start_sample:end_sample]
        if len(chunk) < new_sr * 0.5:  # skip slivers under ~500ms
            continue

        inputs = processor(audio=chunk, sampling_rate=new_sr, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, tgt_lang="hin", generate_speech=True, 
                        speaker_id = SPEAKER_MAPPING[seg.gender.lower()][int(seg.speaker.split('_')[-1])])

        waveform = output[0][0].cpu().numpy().squeeze()
        sf.write(str(out_path), waveform, model.config.sampling_rate)
        seg.s2st_audio_path = str(out_path)
        log.info("Segment %d: %.2fs -> %.2fs translated (direct S2ST, no text)",
                  seg.index, seg.start, seg.end)

    return segments


# --------------------------------------------------------------------------
# Stage 5: Voice style transfer (OpenVoice tone-color conversion)
# --------------------------------------------------------------------------

#def voice_style_transfer(segments: List[Segment], workdir: Path, force: bool = False,
#                           device: str = "cpu", checkpoint_dir: str = "checkpoints_v2") -> List[Segment]:
#    from voxcpm import VoxCPM
#    model = VoxCPM.from_pretrained("openbmb/VoxCPM2")

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
    # target_se_cache = {}
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

        # if seg.ref_audio_path not in target_se_cache:
        #     target_se, _ = se_extractor.get_se(seg.ref_audio_path, converter, vad=False)
        #     target_se_cache[seg.ref_audio_path] = target_se
        # target_se = target_se_cache[seg.ref_audio_path]

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

        speaker_id = SPEAKER_MAPPING[seg.gender.lower()][int(seg.speaker.split('_')[-1])]

        arr = voice_model.generate(
            seg.s2st_audio_path,
            Path("./seamless_outputs") / f"seg_0348_spk_{speaker_id}_{seg.gender}.wav"
            #seg.ref_audio_path,
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
# Stage 7: reassemble full track, loudness-match, mix with background
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


def loudness_match_and_mix(hindi_vocals: Path, background: Path, workdir: Path) -> Path:
    normalized = workdir / "hindi_vocals_normalized.wav"
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
    ap.add_argument("--openvoice-checkpoints", default="checkpoints_v2",
                     help="path to downloaded OpenVoice V2 checkpoint directory")
    ap.add_argument("--max-segment-len", type=float, default=12.0,
                     help="max seconds per VAD-detected chunk fed to S2ST")
    ap.add_argument("--vc", default="chatterbox", choices=["openvoice", "chatterbox"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    workdir = ensure_dir(args.workdir)
    log.info("Working directory: %s", workdir.resolve())

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
        segments = build_speaker_reference(vocals_wav, segments, workdir)
    else:
        # 3. VAD segmentation (acoustic only -- no transcription)
        segments = vad_segment(vocals_wav, workdir, force=args.force,
                            max_segment_len=args.max_segment_len)
    if not segments:
        log.error("No speech detected -- aborting.")
        sys.exit(1)

    # 4. direct speech-to-speech translation
    segments = direct_s2st(vocals_wav, segments, workdir, force=args.force, device=args.device)

    log.info(segments)
    # 5. voice style transfer onto the original speaker's timbre
    if args.vc == "chatterbox":
        segments = voice_style_transfer_chatterbox(segments, workdir, force=args.force, device=args.device)
    else:
        segments = voice_style_transfer(segments, workdir, force=args.force, device=args.device,
                                              checkpoint_dir=args.openvoice_checkpoints)

    # 6. align to original timing
    segments = align_segments(segments, workdir, force=args.force)

    # 7. reassemble + mix
    total_duration = get_media_duration(args.input)
    hindi_vocals_full = build_hindi_vocal_track(segments, total_duration, workdir)
    final_hindi_track = loudness_match_and_mix(hindi_vocals_full, background_wav, workdir)

    # 8. mux into video
    mux_into_video(args.input, final_hindi_track, args.output)

    log.info("Done. Output: %s", args.output.resolve())


if __name__ == "__main__":
    main()
