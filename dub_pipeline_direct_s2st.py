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
        --output movie.target_dubbed.mp4 \\
        --workdir ./work_direct \\
        --device cuda
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import soundfile as sf
import torch
import re

from utils import (
    Segment, log, seamless_speakers,  
    run, ensure_dir, resolve_device,
    get_speaker_mapping, build_speaker_ref_profiles,
    separate_vocals,
    transcribe_and_diarize, get_media_duration,
    build_clone_references, extract_pitch_and_speed,
    align_segments, build_target_vocal_track, 
    loudness_match_and_mix, mux_into_video
)

def get_seamless_speaker(spk, gender):
    if type(spk) == str:
        spk = int(re.sub("[^0-9]", "", spk))
    return seamless_speakers[gender][spk]

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

# --------------------------------------------------------------------------
# Stage 3.5: Identifying speakers, getting reference clips
# --------------------------------------------------------------------------
def build_speaker_reference(segments: List[Segment], spk_map = {}) -> List[Segment]:
    """Speaker references."""
    for seg in segments:
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
            seg.gen_audio_path = str(out_path)
            continue

        if seg.end - seg.start < 0.4 or len(seg.text_in) < 10 or not seg.speaker:
             # Keep originals for short clips, ambient sounds
            seg.gen_audio_path = seg.audio_path
        else:
            start_sample = int(seg.start * new_sr)
            end_sample = int(seg.end * new_sr)
            chunk = np_audio[start_sample:end_sample]

            inputs = processor(audio=chunk, sampling_rate=new_sr, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output = model.generate(**inputs, speaker_id = get_seamless_speaker(seg.speaker, seg.gender), 
                                        max_new_tokens= min(len(seg.text_in), 50),
                                        **generation_config) 

            waveform = output[0][0].cpu().numpy().squeeze()
            sf.write(str(out_path), waveform, model.config.sampling_rate)
            seg.gen_audio_path = str(out_path)

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
        if not seg.gen_audio_path:
            continue
        out_path = out_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.styled_audio_path = str(out_path)
            continue
        if not seg.ref_audio_path:
            log.warning("No speaker reference for segment %d, skipping voice style transfer", seg.index)
            seg.styled_audio_path = seg.gen_audio_path
            continue

        if seg.ref_audio_path not in target_se_cache:
            target_se, _ = se_extractor.get_se(seg.ref_audio_path, converter, vad=False)
            target_se_cache[seg.ref_audio_path] = target_se
        target_se = target_se_cache[seg.ref_audio_path]

        source_se, _ = se_extractor.get_se(seg.gen_audio_path, converter, vad=False)

        converter.convert(
            audio_src_path=seg.gen_audio_path,
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
        out_path = out_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.styled_audio_path = str(out_path)
            continue
        # No reference
        if not seg.ref_audio_path:
            log.warning("No speaker reference for segment %d, skipping voice style transfer", seg.index)
            seg.styled_audio_path = seg.gen_audio_path
            continue
        # Original audio, no style transfer needed
        if not seg.gen_audio_path or seg.gen_audio_path == seg.audio_path:
            seg.styled_audio_path = seg.audio_path
            continue

        arr = voice_model.generate(
            seg.gen_audio_path,
            seg.ref_audio_path
        )

        log.info(f"{voice_model.sr} {arr.shape}")
        torchaudio.save(out_path, arr, voice_model.sr)
        seg.styled_audio_path = str(out_path)
    return segments


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
    ap.add_argument("--use-stock-speakers", action='store_true', help="Use voices from the speakers directory")
    ap.add_argument("--speakers-dir", default=Path("./speakers"), help="Directory for dubbing voices")
    ap.add_argument("--openvoice-checkpoints", default="checkpoints_v2",
                     help="path to downloaded OpenVoice V2 checkpoint directory")
    ap.add_argument("--max-segment-len", type=float, default=12.0,
                     help="max seconds per VAD-detected chunk fed to S2ST")
    ap.add_argument("--vc", default="chatterbox", choices=["openvoice", "chatterbox"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    workdir = ensure_dir(args.workdir)
    log.info("Working directory: %s", workdir.resolve())

    # 0. Speaker mapping -> Dict {(speaker_category, speaker_id) : speaker_reference_filepath}
    spk_map = get_speaker_mapping(Path(args.speakers_dir))

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
    segments = extract_pitch_and_speed(segments, vocals_16k)
    if not args.no_voice_cloning:
        if not args.use_stock_speakers or spk_map == {}:
            # Alternatively
            # return build_clone_references(segments, vocals_16k, workdir)[0]
            segments = build_speaker_ref_profiles(segments, workdir)[0]
        else:
            segments = build_speaker_reference(segments, spk_map)

        if args.vc == "chatterbox":
            segments = voice_style_transfer_chatterbox(segments, workdir, force=args.force, device=args.device)
        else:
            segments = voice_style_transfer(segments, workdir, force=args.force, device=args.device,
                                                checkpoint_dir=args.openvoice_checkpoints)
    # else:
    #     for seg in segments:
    #         seg.styled_audio_path = seg.gen_audio_path

    # 6. align to original timing
    segments = align_segments(segments, workdir, force=args.force)

    # 7. reassemble + mix
    total_duration = get_media_duration(args.input)
    target_vocals_full = build_target_vocal_track(segments, total_duration, workdir)
    final_target_track = loudness_match_and_mix(target_vocals_full, background_wav, workdir)

    # 8. mux into video
    mux_into_video(args.input, final_target_track, args.output)

    log.info("Done. Output: %s", args.output.resolve())


if __name__ == "__main__":
    main()
