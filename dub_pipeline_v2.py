#!/usr/bin/env python3
"""
dub_pipeline_s2t2s.py
======================
English -> Hindi dubbing via speech-to-text-to-speech, built around:

  ASR + diarization : whispermlx  (pip install whispermlx)
                       https://github.com/KalebJS/whispermlx
                       A WhisperX fork with mlx-whisper as the inference
                       backend (Apple Silicon native), retaining
                       word-level alignment, VAD, and pyannote diarization.
  Paralinguistics   : wav2vec2-based classifiers (optional, per segment)
                        - age/gender: audeering/wav2vec2-large-robust-24-ft-age-gender
                        - emotion:    superb/wav2vec2-base-superb-er (categorical)
                                      or audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
                                      (continuous arousal/valence/dominance)
                        - pitch (f0) and speaking rate are always computed
                          directly from audio/text (no model needed)
  Translation       : ai4bharat/indictrans2-en-indic-1B (default), or NLLB-200
  TTS -- Method A   : ai4bharat/indic-parler-tts (description-conditioned).
                       Diarized speakers are mapped to the model's named
                       Hindi voices (Rohit, Divya, Aman, Rani), and a
                       natural-language caption is built per segment from
                       that speaker's gender/age/emotion/pitch/speed.
  TTS -- Method B   : ai4bharat/IndicF5 (F5-TTS-based voice cloning). Each
                       segment's OWN English audio is used as the
                       reference prompt (+ its own English transcript as
                       ref_text, which is how IndicF5 expects the
                       reference to be described) to clone that speaker's
                       voice while generating the translated Hindi text.

PIPELINE STAGES
    1. Extract audio (stereo, for mixing) and a 16kHz mono copy (for
       ASR/diarization/classifiers) via ffmpeg
    2. Separate vocals from background music/fx (Demucs)
    3. Transcribe + diarize with whispermlx -> per-segment English text,
       timing, and speaker label
    4. (Optional, on by default) classify gender/age and emotion per
       segment; always compute pitch (f0) and speaking rate per segment
    5. Translate each segment's English text to Hindi (IndicTrans2 or NLLB)
    6. Build per-speaker voice profiles (aggregated gender/age/emotion/
       pitch/speed) and map each diarized speaker to a Parler-TTS preset
       voice
    7. Synthesize Hindi audio per segment using EITHER:
         (A) Indic Parler-TTS, conditioned on a generated description, or
         (B) IndicF5, cloning each segment's own English audio
    8. Time-stretch each synthesized segment to fit its original slot,
       crossfade at boundaries, and place it on the timeline
    9. Loudness-match the reassembled Hindi vocal track and mix with the
       separated background stem
   10. Mux the result into the video as an ADDITIONAL audio track (the
       original English track is kept), tagged language=hin

WHY TWO TTS METHODS, AND HOW TO CHOOSE
    Method A (Parler, --tts-method parler) does NOT need any reference
    audio -- it generates a consistent, stable voice per named preset,
    which is best when you want polish and don't mind the dubbed voice
    not being an exact timbre match. Hindi currently only has 4 named
    presets (Rohit, Divya, Aman, Rani), so more than 2 speakers per
    gender in your source video will share a base timbre; segments are
    still differentiated by their generated per-segment description
    (pitch/rate/emotion adjectives).

    Method B (IndicF5 cloning, --tts-method clone) tries to match each
    individual speaker's actual timbre using their own voice as the
    reference, which is what "match individual speakers" ultimately
    calls for -- but per-segment cloning from noisy/short movie dialogue
    is less stable than Method A, and quality depends heavily on how
    clean vocal separation was for that speaker.

    RECOMMENDATION: for a movie/show with many distinct speakers where
    speaker identity matters, prefer --tts-method clone. For a small
    cast or when stability/quality matters more than exact timbre match,
    --tts-method parler is more reliable. You can also run both and pick
    per-segment.

HONEST CAVEATS AND FURTHER RECOMMENDATIONS
    - whispermlx is a small third-party fork (not an official Whisper or
      WhisperX release). Pin a version and sanity-check its output before
      relying on it for anything production-critical.
    - Hindi is not one of Indic Parler-TTS's officially validated
      "emotion prompt" languages (only Assamese, Bengali, Bodo, Dogri,
      Kannada, Malayalam, Marathi, Sanskrit, Nepali, Tamil are). Emotion
      adjectives are still passed in the caption since pitch/rate/
      expressivity ARE officially supported for all languages and give
      most of the emotional coloring, but treat explicit "angry"/"sad"
      style tags for Hindi as best-effort, not validated.
    - The wav2vec2 emotion/age/gender classifiers here were trained on
      English (mostly Western) speech corpora (MSP-Podcast, aGender,
      Common Voice, Voxceleb, IEMOCAP-style data). Predictions on movie
      dialogue -- especially non-English-accented English, shouting,
      whispering, or emotionally stylized delivery -- will be noisier
      than on clean read speech. Treat outputs as coarse signals, not
      ground truth.
    - IndicF5's ref_text is meant to be the transcript of ref_audio in
      the language actually spoken in ref_audio -- AI4Bharat's own demo
      demonstrates cross-lingual use (Punjabi reference audio + Punjabi
      ref_text -> Hindi output text), so English ref_audio + English
      ref_text -> Hindi gen_text (as this script does) follows the same
      documented pattern, but it is still comparatively out-of-domain
      versus same-language cloning. Keep reference clips SHORT (a few
      seconds) -- this is the main lever against the clone "overfitting"
      to English phonetic/prosodic patterns while still carrying the
      speaker's pitch, cadence, and timbre.
    - Overlapping speech will not be reconstructed correctly by either
      TTS method; segments are placed independently at their diarized
      start time.
    - BETTER MODELS TO CONSIDER IF AVAILABLE TO YOU:
        * Diarization: pyannote's newer community diarization pipeline,
          or NVIDIA NeMo's diarizer, if whispermlx's bundled pyannote
          version underperforms on your audio (crosstalk-heavy movie
          audio is harder than podcasts).
        * Emotion: emotion2vec (Alibaba/FunASR) is multilingual and
          generally more robust than English-only wav2vec2 SER models --
          worth swapping in if available in your environment.
        * Voice cloning: if IndicF5 struggles with movie-quality audio,
          XTTS-v2 (Coqui) is a solid non-Indic-specialized fallback that
          also does cross-lingual cloning and tends to be more forgiving
          of noisy references.
        * If you need tighter lip-sync than time-stretching alone can
          give you, consider a video re-timing/lip-sync pass (e.g.
          Wav2Lip) as a separate post-process -- out of scope here.

REQUIREMENTS
    ffmpeg on PATH; see requirements_s2t2s.txt for Python packages. A
    Mac with Apple Silicon (for whispermlx/MLX) is assumed for stage 3;
    everything else runs on CPU/CUDA/MPS via --device.

USAGE
    python dub_pipeline_s2t2s.py \\
        --input movie.mp4 --output movie.hindi_dubbed.mp4 \\
        --workdir ./work --tts-method clone --hf-token hf_xxx

    python dub_pipeline_s2t2s.py \\
        --input movie.mp4 --output movie.hindi_dubbed.mp4 \\
        --workdir ./work --tts-method parler --hf-token hf_xxx
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Dict
import soundfile as sf
import torch
import numpy as np
import librosa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dub_pipeline_s2t2s")


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Segment:
    index: int
    start: float
    end: float
    speaker: str = "SPEAKER_00"
    text_en: str = ""
    text_hi: Optional[str] = None

    gender: Optional[str] = None          # "male" | "female" | "child"
    age_years: Optional[float] = None
    emotion: Optional[str] = None          # coarse categorical label
    arousal: Optional[float] = None        # 0..1, if using the dimensional model
    valence: Optional[float] = None
    dominance: Optional[float] = None

    pitch_hz: Optional[float] = None
    speed_wps: Optional[float] = None      # words per second

    ref_audio_path: Optional[str] = None   # this segment's own clean EN clip (Method B)
    tts_audio_path: Optional[str] = None
    aligned_audio_path: Optional[str] = None


@dataclass
class SpeakerProfile:
    speaker_id: str
    gender: str = "male"
    age_years: float = 30.0
    dominant_emotion: str = "neutral"
    mean_pitch_hz: float = 130.0
    mean_speed_wps: float = 2.5
    parler_preset: str = "Rohit"


def run(cmd: List[str], **kwargs):
    log.info("$ %s", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable, falling back to CPU")
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        log.warning("MPS requested but unavailable, falling back to CPU")
        return "cpu"
    return requested


# --------------------------------------------------------------------------
# Stage 1: extract audio
# --------------------------------------------------------------------------

def extract_audio(video_path: Path, workdir: Path, force: bool = False):
    stereo = workdir / "audio_48k_stereo.wav"
    mono16k = workdir / "audio_16k_mono.wav"
    if not stereo.exists() or force:
        run(["ffmpeg", "-y", "-i", str(video_path), "-vn",
             "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2", str(stereo)])
    if not mono16k.exists() or force:
        run(["ffmpeg", "-y", "-i", str(video_path), "-vn",
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(mono16k)])
    return stereo, mono16k


# --------------------------------------------------------------------------
# Stage 2: source separation (Demucs)
# --------------------------------------------------------------------------

def separate_vocals(stereo_wav: Path, workdir: Path, force: bool = False,
                     device: str = "cpu", model: str = "htdemucs"):
    out_dir = workdir / "demucs_out"
    vocals = out_dir / model / stereo_wav.stem / "vocals.wav"
    background = out_dir / model / stereo_wav.stem / "no_vocals.wav"
    vocals_16k = workdir / "vocals_16k_mono.wav"
    if not(vocals.exists() and background.exists() and vocals_16k.exists() and not force):
        ensure_dir(out_dir)
        run([sys.executable, "-m", "demucs", "-n", model, "--two-stems", "vocals",
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
# Stage 2.5: Optional subtitle processing to avoid transcription
# --------------------------------------------------------------------------

def get_subtitles(video_path: str, lang='hi') -> bool:
    """Checks if a video file contains embedded English subtitle streams.

    Args:
        video_path: The file path to the video.

    Returns:
        bool: True if English subtitles are detected, False otherwise.
    """
    import ffmpeg
    out = []
    lang_ref ={
        'en':['eng', 'en', 'english'],
        'hi':['hin', 'hi', 'hindi']
    }
    try:
        # Probe the video file to extract its metadata
        probe = ffmpeg.probe(video_path)
    except ffmpeg.Error as e:
        print(f"Error probing video file: {e.stderr.decode()}")
        return out

    # Iterate through all available streams in the video file
    for stream in probe.get("streams", []):
        # Look specifically for subtitle streams
        if stream.get("codec_type") == "subtitle":
            # Extract tags which contain metadata like language
            tags = stream.get("tags", {})

            # Check if the language tag exists and is flagged as English
            language = tags.get("language", "").lower()
            if language in lang_ref[lang]:
                out.append(stream)
    return out

def extract_embedded_subtitles(video_path, output_srt_path, stream_index=0, lang='hi'):
    subs = get_subtitles(video_path, lang = lang)
    if len(subs)>0:
        stream_index = subs[0]['index'] -2
        log.info(f"Sub stream : {stream_index}")
    else:
        return None
    # stream_index 's:0' refers to the first subtitle track found in the video
    command = [
        'ffmpeg', 
        '-i', video_path, 
        '-map', f'0:s:{stream_index}', 
        output_srt_path,
        '-y' # Overwrite output file if it exists
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Success! Subtitles extracted to {output_srt_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Could not extract subtitles. {e.stderr.decode()}")
    return output_srt_path


def read_subs_as_whisper_segments(sub_file, lang='hi'):
    try:
        import pysrt
        subtitles = pysrt.open(sub_file)
        segments = []

        for sub in subtitles:
            # Convert pysrt time into total seconds (float)
            start_seconds = (
                sub.start.hours * 3600 +
                sub.start.minutes * 60 +
                sub.start.seconds +
                sub.start.milliseconds / 1000.0
            )
            end_seconds = (
                sub.end.hours * 3600 +
                sub.end.minutes * 60 +
                sub.end.seconds +
                sub.end.milliseconds / 1000.0
            )

            segment = {
                "start": start_seconds,
                "end": end_seconds,
                f"text_{lang}": sub.text.replace("\n", " ")
            }
            segments.append(segment)
        return segments
    except:
        return None


# --------------------------------------------------------------------------
# Stage 3: transcription + diarization (whispermlx)
# --------------------------------------------------------------------------

def subs_and_diarize(vocals_16k: Path, workdir: Path, segments, force: bool = False,
                 model_size: str = "large-v3", device: str = "cpu", lang='en',
                 diarize: bool = True, hf_token: Optional[str] = None) -> List[Segment]:
    cache = workdir / "transcript_diarized.json"
    if cache.exists() and not force:
        data = json.loads(cache.read_text())
        return [Segment(**s) for s in data]

    from pyannote.audio import Pipeline
    pl = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=hf_token
    )
    diarization = pl(str(vocals_16k))

    # Gender classifer
    from transformers import pipeline
    gender_model = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
    gender_classifier = pipeline("audio-classification", model=gender_model)

    ref_dir = ensure_dir(workdir / "speaker_refs")
    audio, sr = sf.read(str(vocals_16k))

    new_segments: List[Segment] = []
    i = 0
    j = 0
    for turn, _, spk in diarization.speaker_diarization.itertracks(yield_label=True):
        log.info(f"{i},{j}, {turn.start}, {turn.end}")
        # Match segment
        ref_path = ref_dir / f"{spk}_{j}.wav"
        new_seg = Segment(
            index=j, start=float(turn.start), end=float(turn.end),
            speaker=spk or "SPEAKER_00", 
            text_en=segments[i].get("text_end", ""),
            text_hi=segments[i].get("text_hi", ""),
            ref_audio_path = str(ref_path), gender="Male"
        )

        # Write audio
        start_sample = int(turn.start * sr)
        end_sample = int(min(turn.end, turn.start + 8.0) * sr)
        if end_sample > len(audio):
            new_seg.end = len(audio)/sr
            new_segments.append(new_seg)
        clip = audio[start_sample:end_sample]

        # Get gender
        new_seg.gender = gender_classifier(clip)[0]['label']

        sf.write(str(ref_path), clip, sr)

        # Build text from sub segments
        try:
            while True:
                i+=1
                log.info(f"{i},{j}, {segments[i]["start"]}, {segments[i]["end"]}")
                #mid = (segments[i]["start"] + segments[i]["end"]) / 2
                if turn.start <= segments[i]["end"]  <= turn.end:
                    try:
                        new_seg.text_en += f" {segments[i]["text_en"]}"
                    except:
                        pass

                    try:
                        new_seg.text_hi += f" {segments[i]["text_hi"]}"
                    except:
                        pass
                else:
                    log.info(new_seg)
                    if len(new_seg.text_en) > 1 or len(new_seg.text_hi) > 1:
                        new_segments.append(new_seg)
                        j+=1
                    break
        except Exception as e:
            log.info(e)
            break

    if new_segments:
        cache.write_text(json.dumps([asdict(s) for s in new_segments], indent=2))
    return new_segments


def transcribe_and_diarize(vocals_16k: Path, workdir: Path, force: bool = False,
                 model_size: str = "large-v3", device: str = "cpu",
                 diarize: bool = True, hf_token: Optional[str] = None) -> List[Segment]:
    cache = workdir / "transcript_diarized.json"
    if cache.exists() and not force:
        data = json.loads(cache.read_text())
        return [Segment(**s) for s in data]

    import whispermlx
    asr_options = {
        "temperatures" : [0.6],
        "logprob-threshold" : -1.25,
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


# --------------------------------------------------------------------------
# Stage 4a: optional gender/age classification (wav2vec2, audeering)
# --------------------------------------------------------------------------

def _load_age_gender_model(device: str):
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class ModelHead(nn.Module):
        def __init__(self, config, num_labels):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, num_labels)

        def forward(self, features):
            x = self.dropout(features)
            x = torch.tanh(self.dense(x))
            x = self.dropout(x)
            return self.out_proj(x)

    class AgeGenderModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.age = ModelHead(config, 1)
            self.gender = ModelHead(config, 3)
            self.init_weights()
            self.post_init()

        def forward(self, input_values):
            hidden_states = self.wav2vec2(input_values)[0]
            pooled = torch.mean(hidden_states, dim=1)
            logits_age = self.age(pooled)
            logits_gender = torch.softmax(self.gender(pooled), dim=1)
            return pooled, logits_age, logits_gender

    name = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    processor = Wav2Vec2Processor.from_pretrained(name)
    model = AgeGenderModel.from_pretrained(name).to(device).eval()
    return processor, model

def classify_gender(segments):
    # 1. Load the audio file and automatically resample to 16kHz
    speech, sample_rate = librosa.load(audio_path, sr=16000)
    
    # 2. Initialize the audio classification pipeline
    # We use a specialized fine-tuned wav2vec2 model for gender classification
    
    return predictions

def classify_age_gender(segments: List[Segment], vocals_16k: Path, workdir: Path,
                          force: bool = False, device: str = "cpu") -> List[Segment]:
    import numpy as np

    cache = workdir / "age_gender.json"
    if cache.exists() and not force:
        cached = {s["index"]: s for s in json.loads(cache.read_text())}
        for seg in segments:
            if seg.index in cached:
                seg.gender = cached[seg.index]["gender"]
                seg.age_years = cached[seg.index]["age_years"]
        return segments

    processor, model = _load_age_gender_model(device)
    audio, sr = sf.read(str(vocals_16k))
    gender_labels = ["child", "female", "male"]

    results = []
    for seg in segments:
        chunk = audio[int(seg.start * sr):int(seg.end * sr)]
        if len(chunk) < sr * 0.3:
            continue
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
        input_values = inputs["input_values"].to(device)
        with torch.no_grad():
            _, logits_age, logits_gender = model(input_values)
        seg.age_years = int(logits_age.cpu().numpy().squeeze()*10) * 10.0
        seg.gender = gender_labels[int(np.argmax(logits_gender.cpu().numpy()))]
        log.info(f"{seg.gender} : {seg.text_en}")
        results.append({"index": seg.index, "gender": seg.gender, "age_years": seg.age_years})

    cache.write_text(json.dumps(results, indent=2))
    return segments


# --------------------------------------------------------------------------
# Stage 4b: optional emotion classification (wav2vec2)
# --------------------------------------------------------------------------

def classify_emotion(segments: List[Segment], vocals_16k: Path, workdir: Path,
                       force: bool = False, device: str = "cpu",
                       backend: str = "categorical") -> List[Segment]:
    """backend='categorical' uses superb/wav2vec2-base-superb-er (returns a
    label like hap/ang/sad/neu). backend='dimensional' uses audeering's
    arousal/dominance/valence regression model and maps it to a coarse
    label for description-building, while also storing the raw scores."""
    cache = workdir / "emotion.json"
    if cache.exists() and not force:
        cached = {s["index"]: s for s in json.loads(cache.read_text())}
        for seg in segments:
            if seg.index in cached:
                seg.emotion = cached[seg.index].get("emotion")
                seg.arousal = cached[seg.index].get("arousal")
                seg.valence = cached[seg.index].get("valence")
                seg.dominance = cached[seg.index].get("dominance")
        return segments

    audio, sr = sf.read(str(vocals_16k))
    results = []

    if backend == "categorical":
        from transformers import pipeline
        #clf = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=device if device != "mps" else -1)
        #label_map = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}
        clf = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", device=device if device != "mps" else -1)
        label_map = {v:v for v in ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']}
        for seg in segments:
            chunk = audio[int(seg.start * sr):int(seg.end * sr)]
            if len(chunk) < sr * 0.3:
                continue
            preds = clf({"array": chunk, "sampling_rate": sr}, top_k=1)
            raw_label = preds[0]["label"].lower()
            seg.emotion = label_map.get(raw_label, raw_label)
            results.append({"index": seg.index, "emotion": seg.emotion})
    else:
        import torch.nn as nn
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

        class RegressionHead(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, features):
                x = self.dropout(features)
                x = torch.tanh(self.dense(x))
                x = self.dropout(x)
                return self.out_proj(x)

        class EmotionModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.classifier = RegressionHead(config)
                self.init_weights()

            def forward(self, input_values):
                hidden_states = self.wav2vec2(input_values)[0]
                pooled = torch.mean(hidden_states, dim=1)
                return pooled, self.classifier(pooled)

        name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        processor = Wav2Vec2Processor.from_pretrained(name)
        model = EmotionModel.from_pretrained(name).to(device).eval()

        for seg in segments:
            chunk = audio[int(seg.start * sr):int(seg.end * sr)]
            if len(chunk) < sr * 0.3:
                continue
            inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
            input_values = inputs["input_values"].to(device)
            with torch.no_grad():
                _, logits = model(input_values)
            arousal, dominance, valence = logits.cpu().numpy().squeeze().tolist()
            seg.arousal, seg.dominance, seg.valence = arousal, dominance, valence
            # crude heuristic mapping to a word -- treat as a rough signal
            if valence > 0.6 and arousal > 0.55:
                seg.emotion = "happy"
            elif valence < 0.4 and arousal > 0.55:
                seg.emotion = "angry"
            elif valence < 0.45 and arousal < 0.45:
                seg.emotion = "sad"
            else:
                seg.emotion = "neutral"
            results.append({"index": seg.index, "emotion": seg.emotion,
                             "arousal": arousal, "valence": valence, "dominance": dominance})

    cache.write_text(json.dumps(results, indent=2))
    return segments


# --------------------------------------------------------------------------
# Stage 4c: pitch + speaking rate (always computed, no model needed)
# --------------------------------------------------------------------------

def extract_pitch_and_speed(segments: List[Segment], vocals_16k: Path) -> List[Segment]:

    audio, sr = sf.read(str(vocals_16k))
    for seg in segments:
        chunk = audio[int(seg.start * sr):int(seg.end * sr)]
        duration = max(seg.end - seg.start, 0.01)
        seg.speed_wps = len(seg.text_en.split()) / duration

        if len(chunk) < sr * 0.2:
            seg.pitch_hz = 130.0
            continue
        try:
            f0, voiced_flag, _ = librosa.pyin(
                chunk.astype(np.float32), fmin=60, fmax=400, sr=sr
            )
            voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
            voiced_f0 = voiced_f0[~np.isnan(voiced_f0)] if voiced_f0 is not None else []
            seg.pitch_hz = float(np.median(voiced_f0)) if len(voiced_f0) else 130.0
        except Exception as e:
            log.warning("pyin failed on segment %d (%s); defaulting pitch", seg.index, e)
            seg.pitch_hz = 130.0

        # gender heuristic fallback, only used if gender wasn't classified
        if seg.gender is None:
            seg.gender = "female" if seg.pitch_hz > 175 else "male"
    return segments


# --------------------------------------------------------------------------
# Stage 5: translation
# --------------------------------------------------------------------------

def translate_indictrans2(segments: List[Segment], workdir: Path, force: bool = False,
                            device: str = "cpu") -> List[Segment]:
    cache = workdir / "translated.json"
    if cache.exists() and not force:
        cached = {s["index"]: s["text_hi"] for s in json.loads(cache.read_text())}
        if all(seg.index in cached for seg in segments):
            for seg in segments:
                seg.text_hi = cached[seg.index]
            return segments

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit.processor import IndicProcessor

    model_name = "ai4bharat/indictrans2-en-indic-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    ip = IndicProcessor(inference=True)

    src_lang, tgt_lang = "eng_Latn", "hin_Deva"
    texts = [seg.text_en for seg in segments]
    batch_size = 8
    translations: List[str] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        prepped = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(prepped, truncation=True, padding="longest", return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, use_cache=True, min_length=0,
                                        max_length=256, num_beams=5, num_return_sequences=1)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=True)
        translations.extend(ip.postprocess_batch(decoded, lang=tgt_lang))

    for seg, hi in zip(segments, translations):
        seg.text_hi = hi
        log.info("[%d] EN: %s", seg.index, seg.text_en)
        log.info("[%d] HI: %s", seg.index, seg.text_hi)

    cache.write_text(json.dumps([{"index": s.index, "text_hi": s.text_hi} for s in segments], indent=2))
    return segments


def translate_nllb(segments: List[Segment], workdir: Path, force: bool = False,
                     device: str = "cpu") -> List[Segment]:
    """Fallback translator if IndicTransToolkit isn't installed."""
    cache = workdir / "translated.json"
    if cache.exists() and not force:
        cached = {s["index"]: s["text_hi"] for s in json.loads(cache.read_text())}
        if all(seg.index in cached for seg in segments):
            for seg in segments:
                seg.text_hi = cached[seg.index]
            return segments

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    hi_id = tok.convert_tokens_to_ids("hin_Deva")

    for seg in segments:
        inputs = tok(seg.text_en, return_tensors="pt").to(device)
        generated = model.generate(**inputs, forced_bos_token_id=hi_id, max_new_tokens=256)
        seg.text_hi = tok.batch_decode(generated, skip_special_tokens=True)[0].strip()

    cache.write_text(json.dumps([{"index": s.index, "text_hi": s.text_hi} for s in segments], indent=2))
    return segments

def translate_sarvam(segments: List[Segment], workdir: Path, force: bool = False,
                     device: str = "cpu", tgt_lang = "Hindi", model_name = "sarvamai/sarvam-translate") -> List[Segment]:
    """Fallback translator if IndicTransToolkit isn't installed."""
    cache = workdir / "translated.json"
    if cache.exists() and not force:
        cached = {s["index"]: (s["text_hi"], s["start"], s["end"]) for s in json.loads(cache.read_text())}
        if all(seg.index in cached for seg in segments):
            for seg in segments:
                seg.text_hi = cached[seg.index][0]
                seg.start = cached[seg.index][1]
                seg.end = cached[seg.index][2]
            return segments

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import re

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Generate the output
    texts = []
    out = []
    thresh = 30
    i = 0
    text = ''
    for seg in segments:
        text += f'<{seg.gender}>{seg.text_en}|'
        i+=1
        if i%thresh == 0:
            texts.append(text[:-1])
            text = f'<{seg.gender}>{seg.text_en}|'
            i = 0
            continue
    if i!= 0:
        texts.append(text[:-1])
    for text in texts:
        messages = [
            {"role": "system", "content": f"Translate the text below to {tgt_lang}. Each text section is preceded by |(Male/Female). Use surrounding text as context."},
            {"role": "user", "content": text}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.01,
            num_return_sequences=1
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        log.info(f"Pairs : {text}, {output_text}")
        output_text = re.sub(r'<[^>]*>', '', output_text) # remove model output tags
        output_text = re.sub(r'([^)]*)', '', output_text) # remove model output tags
        log.info(f"Pairs : {text}, {output_text}")
        temp = output_text.split('|')
        out.extend(temp[:thresh-1])
    if len(temp) > thresh :
        out.append(temp[thresh]) # last element
    log.info(f"Length {len(segments)}, {len(out)}")
    for i, t in enumerate(out[:len(segments)]):
        segments[i].text_hi = t

    cache.write_text(json.dumps([{"index": s.index, "text_hi": s.text_hi, 
                              "start":s.start, "end":s.end} for s in segments], indent=2))
    return segments


# --------------------------------------------------------------------------
# Stage 6: speaker profiles + Parler preset mapping
# --------------------------------------------------------------------------

# Verified against the ai4bharat/indic-parler-tts model card's Hindi row.
# Only 4 named voices exist for Hindi; if you have more than 2 speakers of
# the same detected gender, they will share a base timbre under Method A --
# differentiate them further via speaker_overrides.json (see below) or use
# Method B (voice cloning) instead.
HINDI_PARLER_PRESETS = {
    "male": ["Rohit", "Aman"],          # Rohit is AI4Bharat's recommended male preset
    "female": ["Divya", "Rani"],        # Divya is AI4Bharat's recommended female preset
    "child": ['Aman',"Divya"],          # no dedicated child voice for Hindi; closest available
}


def build_speaker_profiles(segments: List[Segment], workdir: Path,
                             override_path: Optional[Path] = None) -> Dict[str, SpeakerProfile]:
    by_speaker: Dict[str, List[Segment]] = defaultdict(list)
    samples: Dict[str, List[str]] = defaultdict(list)
    for seg in segments:
        by_speaker[seg.speaker].append(seg)
        samples[seg.speaker].append(seg.ref_audio_path)

    overrides = {}
    if override_path and override_path.exists():
        overrides = json.loads(override_path.read_text())

    profiles: Dict[str, SpeakerProfile] = {}
    preset_cursor = {"male": 0, "female": 0, "child": 0}

    for speaker_id, segs in sorted(by_speaker.items()):
        genders = Counter(s.gender for s in segs if s.gender)
        gender = genders.most_common(1)[0][0] if genders else "male"
        emotions = Counter(s.emotion for s in segs if s.emotion)
        dominant_emotion = emotions.most_common(1)[0][0] if emotions else "neutral"
        ages = [s.age_years for s in segs if s.age_years]
        mean_age = sum(ages) / len(ages) if ages else 30.0
        pitches = [s.pitch_hz for s in segs if s.pitch_hz]
        mean_pitch = sum(pitches) / len(pitches) if pitches else 130.0
        speeds = [s.speed_wps for s in segs if s.speed_wps]
        mean_speed = sum(speeds) / len(speeds) if speeds else 2.5

        if speaker_id in overrides and "parler_preset" in overrides[speaker_id]:
            preset = overrides[speaker_id]["parler_preset"]
        else:
            pool = HINDI_PARLER_PRESETS.get(gender, HINDI_PARLER_PRESETS["male"])
            preset = pool[preset_cursor[gender] % len(pool)]
            preset_cursor[gender] += 1

        profiles[speaker_id] = SpeakerProfile(
            speaker_id=speaker_id, gender=gender, age_years=mean_age,
            dominant_emotion=dominant_emotion, mean_pitch_hz=mean_pitch,
            mean_speed_wps=mean_speed, parler_preset=preset,
        )
        log.info("Speaker %s -> gender=%s age=%.0f emotion=%s preset=%s (%d segments)",
                  speaker_id, gender, mean_age, dominant_emotion, preset, len(segs))

    (workdir / "speaker_profiles.json").write_text(
        json.dumps({k: asdict(v) for k, v in profiles.items()}, indent=2, ensure_ascii=False)
    )
    return profiles


def build_parler_description(seg: Segment, profile: SpeakerProfile) -> str:
    """Composes a natural-language caption per Indic Parler-TTS's documented
    controls: named speaker + pitch + speaking rate + expressivity/emotion
    + recording quality. Pitch/rate thresholds are simple heuristics --
    tune them by ear against your source material."""
    pitch = seg.pitch_hz or profile.mean_pitch_hz
    speed = seg.speed_wps or profile.mean_speed_wps
    emotion = seg.emotion or profile.dominant_emotion

    pitch_adj = "low-pitched" if pitch < 110 else ("high-pitched" if pitch > 200 else "moderately pitched")
    rate_adj = "slow-paced" if speed < 2.0 else ("moderately fast-paced" if speed > 3.2 else "moderately paced")
    emotion_adj = {
        "happy": "cheerful and animated", "angry": "tense and forceful",
        "sad": "subdued and downcast", "neutral": "calm and even",
    }.get(emotion, "conversational")

    return (
        f"{profile.parler_preset}'s voice is {pitch_adj} and {rate_adj}, delivered in a "
        f"{emotion_adj} tone. The recording is very clear and close-sounding, with no "
        f"background noise."
    )

# Voice clone profiles
def build_speaker_ref_profiles(segments: List[Segment], workdir: Path,
                             override_path: Optional[Path] = None) -> Dict[str, SpeakerProfile]:
    import librosa
    profiles: Dict[str, List[str]] = defaultdict(list)
    for seg in segments:
        if librosa.get_duration(path = seg.ref_audio_path) > 5.0:
            profiles[seg.speaker].append(str(seg.ref_audio_path))

    (workdir / "speaker_profiles.json").write_text(
        json.dumps({k: v for k, v in profiles.items()}, indent=2, ensure_ascii=False)
    )
    return profiles

# --------------------------------------------------------------------------
# Stage 7a: TTS Method A -- Indic Parler-TTS
# --------------------------------------------------------------------------

def synth_parler(segments: List[Segment], profiles: Dict[str, SpeakerProfile],
                   workdir: Path, force: bool = False, device: str = "cpu") -> List[Segment]:
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    out_dir = ensure_dir(workdir / "tts_parler")
    model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

    for seg in segments:
        out_path = out_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.tts_audio_path = str(out_path)
            continue
        profile = profiles[seg.speaker]
        description = build_parler_description(seg, profile)

        desc_ids = description_tokenizer(description, return_tensors="pt").to(device)
        prompt_ids = tokenizer(seg.text_hi, return_tensors="pt").to(device)
        with torch.no_grad():
            generation = model.generate(
                input_ids=desc_ids.input_ids, attention_mask=desc_ids.attention_mask,
                prompt_input_ids=prompt_ids.input_ids, prompt_attention_mask=prompt_ids.attention_mask,
            )
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(str(out_path), audio_arr, model.config.sampling_rate)
        seg.tts_audio_path = str(out_path)
        log.info("[%d] Parler synth via %s: %s", seg.index, profile.parler_preset, description)

    return segments


# --------------------------------------------------------------------------
# Stage 7b: TTS Method B -- IndicF5 voice cloning (self-reference)
# --------------------------------------------------------------------------

def build_clone_references(segments: List[Segment], vocals_16k: Path, workdir: Path,
                             ref_max_seconds: float = 6.0) -> List[Segment]:
    """Trims each segment's own English audio into a short, silence-trimmed
    reference clip. Kept SHORT deliberately: longer reference clips make
    the clone lean more heavily on English phonetic/prosodic patterns;
    short clips (~3-6s) still carry pitch/timbre/cadence without as much
    English-specific articulation for the model to imitate."""
    ref_dir = ensure_dir(workdir / "clone_refs")
    audio, sr = sf.read(str(vocals_16k))

    profiles = {}

    for seg in segments:
        out_path = ref_dir / f"seg_{seg.index:04d}_ref.wav"
        if out_path.exists():
            seg.ref_audio_path = str(out_path)
            continue
        start_sample = int(seg.start * sr)
        end_sample = int(min(seg.end, seg.start + ref_max_seconds) * sr)
        chunk = audio[start_sample:end_sample]
        if len(chunk) < sr * 0.3:
            # too short to be a useful reference; borrow from a wider window
            end_sample = int(min(seg.end + 2.0, seg.start + ref_max_seconds + 2.0) * sr)
            chunk = audio[start_sample:end_sample]
        trimmed, _ = librosa.effects.trim(chunk, top_db=30)
        if len(trimmed) < sr * 0.3:
            trimmed = chunk  # fall back to untrimmed if trim over-cut
        sf.write(str(out_path), trimmed, sr)
        seg.ref_audio_path = str(out_path)
    return segments


def synth_clone_indicf5(segments: List[Segment], workdir: Path, force: bool = False) -> List[Segment]:
    from transformers import AutoModel
    out_dir = ensure_dir(workdir / "tts_clone")
    model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)

    for seg in segments:
        out_path = out_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.tts_audio_path = str(out_path)
            continue
        if not seg.ref_audio_path or not seg.text_hi:
            log.warning("Segment %d missing reference audio or translation; skipping", seg.index)
            continue

        audio = model(
            seg.text_hi,
            ref_audio_path=seg.ref_audio_path,
            ref_text=seg.text_en,   # transcript of the reference clip, per IndicF5's documented API
        )
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        sf.write(str(out_path), np.array(audio, dtype=np.float32), samplerate=24000)
        seg.tts_audio_path = str(out_path)
        log.info("[%d] IndicF5 clone synth (ref=%s)", seg.index, Path(seg.ref_audio_path).name)

    return segments


# --------------------------------------------------------------------------
# Stage 7b: TTS Method B -- XTTS voice cloning (few shot)
# --------------------------------------------------------------------------
def synth_coqui(segments: List[Segment], profiles: Dict[str, List[str]],
                workdir: Path, gpt_cond_len = 30,
                force: bool = False, device: str = "cpu") -> List[Segment]:
    import shutil
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts_dir = ensure_dir(workdir / "tts_raw")
    for seg in segments:
        out_path = tts_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.tts_audio_path = str(out_path)
            continue
        if not seg.ref_audio_path:
            log.warning("No reference audio for segment %s, skipping", seg.index)
            continue
        if len(seg.text_en) < 11:# or len(profiles[seg.speaker]) == 0:
            shutil.copy(seg.ref_audio_path, out_path)
        else:
            if profiles[seg.speaker] and len(profiles[seg.speaker]) > 0:
                refs = [seg.ref_audio_path] + list(np.random.choice(profiles[seg.speaker], size=3, replace=True))
            else:
                refs = seg.ref_audio_path
            try:
                tts.tts_to_file(text=seg.text_hi,
                        file_path=out_path,
                        speaker_wav=refs,
                        language="hi",
                        gpt_cond_len = gpt_cond_len)
            except:
                shutil.copy(seg.ref_audio_path, out_path)
                #tts.tts_to_file(text=seg.text_hi[:20],
                #        file_path=out_path,
                #        speaker_wav=seg.ref_audio_path,
                #        language="hi",
                #        gpt_cond_len = gpt_cond_len)
        seg.tts_audio_path = str(out_path)
    return segments

# --------------------------------------------------------------------------
# Stage 8: time-align each clip to its original slot, with crossfades
# --------------------------------------------------------------------------

def get_duration(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    return info.frames / info.samplerate


def time_stretch(in_path: Path, out_path: Path, factor: float):
    factor = max(0.5, min(factor, 2.0))
    run(["ffmpeg", "-y", "-i", str(in_path), "-filter:a", f"atempo={factor:.4f}", str(out_path)])


def align_segments(segments: List[Segment], workdir: Path, force: bool = False,
                    max_stretch: float = 1.6) -> List[Segment]:
    aligned_dir = ensure_dir(workdir / "aligned")
    for seg in segments:
        if not seg.tts_audio_path:
            continue
        out_path = aligned_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.aligned_audio_path = str(out_path)
            continue

        target_dur = max(seg.end - seg.start, 0.1)
        actual_dur = get_duration(Path(seg.tts_audio_path))
        factor = actual_dur / target_dur
        factor = max(1.0 / max_stretch, min(factor, max_stretch))
        if abs(factor - 1.0) < 0.03:
            shutil.copy(seg.tts_audio_path, out_path)
        else:
            time_stretch(Path(seg.tts_audio_path), out_path, factor)
        seg.aligned_audio_path = str(out_path)
    return segments


# --------------------------------------------------------------------------
# Stage 9: reassemble (with short crossfades), loudness-match, mix
# --------------------------------------------------------------------------

def build_hindi_vocal_track(segments: List[Segment], total_duration: float, workdir: Path,
                              sample_rate: int = 48000, fade_ms: float = 15.0) -> Path:
    out_path = workdir / "hindi_vocals_full.wav"
    canvas = np.zeros((int(total_duration * sample_rate) + sample_rate, 2), dtype=np.float32)
    fade_samples = int(sample_rate * fade_ms / 1000.0)

    for seg in sorted(segments, key=lambda s: s.start):
        if not seg.aligned_audio_path:
            continue
        audio, sr = sf.read(seg.aligned_audio_path, dtype="float32")
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1)
        if sr != sample_rate:
            # resample if needed (XTTS/IndicF5/Parler emit at their own
            # native rates -- 24kHz for IndicF5, model-specific for Parler)
            import librosa
            audio = librosa.resample(audio.T, orig_sr=sr, target_sr=sample_rate).T

        if len(audio) > 2 * fade_samples:
            ramp = np.linspace(0, 1, fade_samples)[:, None]
            audio[:fade_samples] *= ramp
            audio[-fade_samples:] *= ramp[::-1]

        start_sample = int(seg.start * sample_rate)
        end_sample = start_sample + len(audio)
        if end_sample > canvas.shape[0]:
            canvas = np.pad(canvas, ((0, end_sample - canvas.shape[0]), (0, 0)))
        canvas[start_sample:end_sample] += audio[:, :2] if audio.shape[1] >= 2 else audio

    sf.write(str(out_path), canvas, sample_rate)
    return out_path

def int16_to_dbfs(sample_value):
    # Handle absolute silence to avoid math domain error
    if sample_value == 0:
        return float('-inf')
    # 32768 is the maximum absolute value for a signed 16-bit integer
    max_val = 32768.0
    # Calculate decibels relative to full scale (dBFS)
    dbfs = 20 * math.log10(abs(sample_value) / max_val)
    return dbfs

def loudness_match_and_mix(hindi_vocals: Path, background: Path, workdir: Path, volume) -> Path:
    normalized = workdir / "hindi_vocals_normalized.wav"
    run(["ffmpeg", "-y", "-i", str(hindi_vocals), "-af", f"loudnorm=I=-16:TP={int16_to_dbfs(volume)}:LRA=11", str(normalized)])
    #from pydub import AudioSegment
    #audio = AudioSegment.from_file(normalized)
    #change_in_dBFS = volume - audio.dBFS
    #normalized_audio = audio.apply_gain(change_in_dbFS)
    #normalized_audio.export(normalized, format="wav")
    mixed = workdir / "final_hindi_track.wav"
    run(["ffmpeg", "-y", "-i", str(normalized), "-i", str(background),
         "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0", str(mixed)])
    return mixed


# --------------------------------------------------------------------------
# Stage 10: mux into video
# --------------------------------------------------------------------------

def mux_into_video(video_path: Path, hindi_track: Path, output_path: Path):
    run(["ffmpeg", "-y", "-i", str(video_path), "-i", str(hindi_track),
         "-map", "0:v", "-map", "0:a", "-map", "1:a",
         "-c:v", "copy", "-c:a:0", "copy", "-c:a:1", "aac", "-b:a:1", "192k",
         "-metadata:s:a:0", "language=eng",
         "-metadata:s:a:1", "language=hin",
         "-metadata:s:a:1", "title=Hindi (AI Dubbed)",
         "-disposition:a:0", "default", "-disposition:a:1", "0",
         str(output_path)])


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
    ap.add_argument("--workdir", default=Path("./dub_work_s2t2s"), type=Path)
    ap.add_argument("--device", default="cpu", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--demucs-model", default="htdemucs")

    ap.add_argument("--whisper-model", default="large-v3-turbo")
    ap.add_argument("--hf-token", default=None, help="required for whispermlx diarization (pyannote gated model)")
    # Subtitles and language
    ap.add_argument("--source-lang", default='en', type=str, help="Video language")
    ap.add_argument("--target-lang", default='hi', type=str, help="Dubbing language")
    ap.add_argument("--subtitle-file", default=None, type=str, help="Use the subtitle file")
    ap.add_argument("--no-use-subs", action="store_true", help="Do not use subtitles embedded in the video")
    ap.add_argument("--no-diarize", action="store_true")

    ap.add_argument("--no-detect-age-gender", action="store_true",
                     help="skip wav2vec2 age/gender classification; falls back to a pitch-threshold gender guess")
    ap.add_argument("--no-detect-emotion", action="store_true",
                     help="skip emotion classification; defaults every segment to 'neutral'")
    ap.add_argument("--emotion-backend", default="categorical", choices=["categorical", "dimensional"])

    ap.add_argument("--translator", default="indictrans2", choices=["indictrans2", "nllb", "gemma", "sarvam"])

    ap.add_argument("--tts-method", required=True, choices=["parler", "clone", 'coqui'])
    ap.add_argument("--speaker-overrides", type=Path, default=None,
                     help="JSON: {\"SPEAKER_00\": {\"parler_preset\": \"Rohit\"}} to manually pin presets")
    ap.add_argument("--f5-ref-max-seconds", type=float, default=6.0)
    ap.add_argument("--max-stretch", type=float, default=1.5)

    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    workdir = ensure_dir(args.workdir)
    device = resolve_device(args.device)
    log.info("Working directory: %s", workdir.resolve())


    sub = None
    segments=None
    need_translate = True
    lang = args.target_lang
    if args.subtitle_file:
        segments = read_subs_as_whisper_segments(sub, lang = lang)
    if sub is None and not args.no_use_subs:
        sub_file = extract_embedded_subtitles(args.input, workdir / f"{lang}_subs.srt", lang=lang)
        if sub_file is None:
            lang = args.source_lang
            sub_file = extract_embedded_subtitles(args.input, workdir / f"{lang}_subs.srt", lang=lang)
        else:
            need_translate = False
        segments = read_subs_as_whisper_segments(sub_file, lang)

    stereo_wav, _mono_unused = extract_audio(args.input, workdir, force=args.force)
    vocals_wav, background_wav, vocals_16k, vol = separate_vocals(
        stereo_wav, workdir, force=args.force, device=device, model=args.demucs_model
    )

    # Subtitle created segments
    if segments:
        segments = subs_and_diarize(
            vocals_16k, workdir, segments, force=args.force, model_size=args.whisper_model,
            device=device, diarize=not args.no_diarize, hf_token=args.hf_token,
        )

    if not segments:
        # Subtitle pipeline failed, transcribe instead
        segments = transcribe_and_diarize(
            vocals_16k, workdir, force=args.force, model_size=args.whisper_model,
            device=device, diarize=not args.no_diarize, hf_token=args.hf_token,
        )

    if not segments:
        log.error("No speech detected -- aborting.")
        sys.exit(1)

    #if not args.no_detect_age_gender:
    #    segments = classify_age_gender(segments, vocals_16k, workdir, force=args.force, device=device)

    if need_translate:
        if args.translator == "indictrans2":
            try:
                segments = translate_indictrans2(segments, workdir, force=args.force, device=device)
            except ImportError as e:
                log.warning("IndicTransToolkit unavailable (%s); falling back to NLLB", e)
                segments = translate_nllb(segments, workdir, force=args.force, device=device)
        elif args.translator == "nllb":
            segments = translate_nllb(segments, workdir, force=args.force, device=device)
        elif args.translator == "gemma":
            segments = translate_sarvam(segments, workdir, force=args.force, device=device, model_name="google/translategemma-4b-it")
        else:
            segments = translate_sarvam(segments, workdir, force=args.force, device=device)

        for seg in segments:
            print(seg.text_en, seg.text_hi)

    if args.tts_method == "parler":
        if not args.no_detect_emotion:
            segments = classify_emotion(segments, vocals_16k, workdir, force=args.force,
                                          device=device, backend=args.emotion_backend)
        segments = extract_pitch_and_speed(segments, vocals_16k)  # always runs; also fills gender fallback
        profiles = build_speaker_profiles(segments, workdir, override_path=args.speaker_overrides)

        for seg in segments:
            if seg.emotion is None:
                seg.emotion = "neutral"
        segments = synth_parler(segments, profiles, workdir, force=args.force, device=device)
    else:
        # Collect coice samples
        profiles = build_speaker_ref_profiles(segments, workdir, override_path=args.speaker_overrides)

        if args.tts_method == "clone":
            segments = build_clone_references(segments, vocals_16k, workdir,
                                                ref_max_seconds=args.f5_ref_max_seconds)
            segments = synth_clone_indicf5(segments, workdir, force=args.force)
        elif args.tts_method == "coqui":
            segments = synth_coqui(segments, profiles, workdir, force=args.force, device=device)

    segments = align_segments(segments, workdir, force=args.force, max_stretch=args.max_stretch)

    total_duration = get_media_duration(args.input)
    hindi_vocals_full = build_hindi_vocal_track(segments, total_duration, workdir)
    final_hindi_track = loudness_match_and_mix(hindi_vocals_full, background_wav, workdir, vol)

    mux_into_video(args.input, final_hindi_track, args.output)
    log.info("Done. Output: %s", args.output.resolve())

if __name__ == "__main__":
    main()
