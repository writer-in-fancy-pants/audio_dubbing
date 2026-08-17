import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Dict
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import torch
import json
import librosa
import soundfile as sf
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dub_pipeline")

# --------------------------------------------------------------------------
# Constants, maps
# --------------------------------------------------------------------------

SPEECH_CATEGORIES = {
    "age": {
        "labels": ["child", "young adult", "middle-aged", "old"],
        "template": "a recording of a {} speaker's voice",
    },
    "gender": {
        "labels": ["male", "female"],
        "template": "a recording of a {} voice",
    },
    "quality": {
        "labels": ["smooth", "rough", "breathy", "raspy", "nasal", "shaky", "monotone", "warm"],
        "template": "a voice with a {} quality",
    },
    "emotion": {
        "labels": ["happy", "sad", "angry", "neutral", "excited", "calm", "fearful", "surprised"],
        "template": "speech delivered in a {} and expressive manner",
    },
    "speed": {
        "labels": ["slow", "moderate", "fast"],
        "template": "speech spoken at a {} speed",
    },
    "pitch": {
        "labels": ["low", "medium", "high"],
        "template": "a voice with a {} pitch",
    },
    "clarity": {
        "labels": ["very clear", "muffled"],
        "template": "a recording where the speaker's voice sounds {}",
    },
    "distance": {
        "labels": ["close", "distant"],
        "template": "a recording where the speaker sounds {}",
    },
}

lang_ref ={
    'en':['eng', "eng_Latn", 'en', 'english'],
    'hi':['hin', "hin_Deva", 'hi', 'hindi']
}

# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Segment:
    index: int
    start: float
    end: float
    speaker: str = "SPEAKER_00"
    text_in: str = ""
    text_out: Optional[str] = None

    # CLAP based
    description: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None  
    quality: Optional[str] = None
    emotion: Optional[str] = None    
    speed: Optional[str] = None 
    pitch: Optional[str] = None 
    clarity: Optional[str] = None 
    distance: Optional[str] = None 

    # coarse categorical label
    arousal: Optional[float] = None        # 0..1, if using the dimensional model
    valence: Optional[float] = None
    dominance: Optional[float] = None

    age_years: Optional[int] = None
    pitch_hz: Optional[float] = None
    speed_wps: Optional[float] = None      # words per second

    audio_path: Optional[str] = None   # this segment's own clean EN clip (Method B)
    ref_audio_path: Optional[str] = None   # in case multiple tts engines are used before voice cloning
    gen_audio_path: Optional[str] = None
    styled_audio_path: Optional[str] = None     # after voice style transfer
    aligned_audio_path: Optional[str] = None


# --------------------------------------------------------------------------
# Basic utils
# --------------------------------------------------------------------------

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
    if requested == "mps":
        if not torch.backends.mps.is_available():
            log.warning("MPS requested but unavailable, falling back to CPU")
            return "cpu"
    return requested


def get_audio_files_in_dir(loc:Path)-> List[Path]:
    audio_extensions = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"}
    # Get all audio files recursively
    return sorted([
        file for file in loc.rglob("*") 
        if file.is_file() and file.suffix.lower() in audio_extensions
    ])


def get_media_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def check_dont_generate(seg, time = 1.0, chars = 2):
    return (seg.end - seg.start < time or not seg.text_out  or len(seg.text_out) < chars) or \
            (seg.gen_audio_path and Path(seg.gen_audio_path).exists())


def find_closest(sorted_tuples, target, idx = 0):
    return min(sorted_tuples, key=lambda x: abs(x[idx] - target))


#--------------------------
# Audio utilities
#--------------------------

def get_mfcc(audio_file):
    y, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs


def load_in_stereo(wav, sample_rate):
    audio, sr = sf.read(wav, dtype="float64")
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    if sr != sample_rate:
        # resample if needed (XTTS/IndicF5/Parler emit at their own
        # native rates -- 24kHz for IndicF5, model-specific for Parler)
        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=sample_rate).T
    return audio


def slowdown(input, speed, workdir, force=False):
    outpath = workdir / f"temp.{input.suffix}"
    # No need to slowdown
    if abs(speed - round(speed)) <= 0.001:
        return input
    if not outpath.exists() or force:
        run(['ffmpeg', '-i', input, '-filter_complex', f"[0:v]setpts=1/{speed}*PTS[v];[0:a]atempo={speed}[a]",
            '-map', "[v]", '-map', "[a]", outpath])
    return outpath


def normalize_peak(in_array, ceiling=0.999):
    """Apply the maximum possible gain to the whole track without clipping.
    Returns (normalized_out, applied_gain_linear).
    """
    out = np.asarray(in_array, dtype=np.float64)
    peak = np.max(np.abs(out)) if out.size else 0.0

    if peak == 0:
        return out, 1.0  # silent track, nothing to normalize

    applied_gain = ceiling / peak
    return out * applied_gain, applied_gain


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


def normalize_new_vocals(target_vocals_wav: Path, vocals_wav: Path, workdir: Path, 
                        demucs = "demucs", demucs_model="htdemucs", device="cpu", sample_rate: int = 48000) -> Path:
    # Denoise
    out_dir = workdir / "target_demucs_out"
    ensure_dir(out_dir)
    run([demucs, "-n", demucs_model, "--two-stems=vocals", "-o", str(out_dir), str(target_vocals_wav)])

    # Adjust volume to match original vocals by windowing and scaling
    new_vocals_wav = out_dir / "htdemucs" / "target_vocals_full" / "vocals.wav"
    src = load_in_stereo(str(new_vocals_wav), sample_rate)
    ref = load_in_stereo(vocals_wav, sample_rate)
    out = match_envelope(src, ref, sample_rate, sample_rate)

    normalized_wav = workdir / "target_vocals_normalized.wav"
    sf.write(str(normalized_wav), out, sample_rate)

    return normalized_wav


def solve_coefficients(arr1, arr2, arr3):
    """
    Finds scalars x and y that minimize ||x*arr1 + y*arr2 - arr3||_2
    Works for arrays of any shape, including multi-channel layouts.
    """
    # 1. Stack as columns to form the design matrix A of shape (N,4)
    x = []
    y =[]

    log.info(f"arr1={np.linalg.norm(arr1):0.2f}, arr2={np.linalg.norm(arr2):.2f}, arr3={np.linalg.norm(arr3):.2f}")
    for i in range(arr3.shape[1]):
        A = np.column_stack([arr1[:, i], arr2[:, i]])
        (x1, x2), _, _, _ = np.linalg.lstsq(A, arr3[:, i], rcond=None)
        residual = np.sqrt(np.mean((x1 * arr1[:,i] + x2 * arr2[:,i] - arr3[:,i]) ** 2))
        log.info(f"x1={x1:.4f}, x2={x2:.4f}, fit RMS error={residual:.6f}")
        x.append(x1)
        y.append(x2)
    # 4. Return optimized x and y
    return np.array(x), np.array(y)


def adjust_array(arr, max_len):
    if max_len > arr.shape[0]:
        return np.pad(arr, ((0, max_len - arr.shape[0]), (0,0)), mode='constant')
    elif max_len < arr.shape[0]:
        return arr[:max_len, :]
    return arr


def scale_vocal_background(vocals_wav, background_wav, stereo_wav, target_vocals, workdir, duration, sr=48000):
    # May need to stack audios
    arr1 = load_in_stereo(vocals_wav, sr)
    arr2  = load_in_stereo(background_wav, sr)
    arr3 = load_in_stereo(stereo_wav, sr)

    l = min(arr1.shape[0], arr2.shape[0], arr3.shape[0])
    x, y = solve_coefficients(arr1[:l,:], arr2[:l,:], arr3[:l,:])

    log.info(f"{x},{y}")

    new_arr1 = load_in_stereo(target_vocals, sr)
    max_len = int(duration*sr)

    log.info(f"{new_arr1.shape}, {arr2.shape}, {duration}")
    new_arr1 = x[np.newaxis,:]*adjust_array(new_arr1, max_len)
    arr2 = y[np.newaxis,:]*adjust_array(arr2, max_len)

    final, gain = normalize_peak(new_arr1+arr2)
    log.info(f"{final.shape}, {gain}")
    output_path = workdir / "target_final_track.wav"
    sf.write(output_path, final, sr)
    return output_path

# --------------------------------------------------------------------------
# Stage 2: source separation (Demucs)
# --------------------------------------------------------------------------

def separate_vocals(stereo_wav: Path, workdir: Path, force: bool = False,
                     device: str = "cpu", model: str = "htdemucs", demucs: str = "demucs"):
    import sys
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

#--------------------------
# Speaker utilities
#--------------------------

def get_speaker_mapping(speakers_dir = Path("./speakers"), use_from_source = False):
    speaker_mapping = {}
    if not use_from_source:
        for spk_cls in speakers_dir.iterdir():
            if spk_cls.is_dir():
                speaker_mapping = [(spk_cls.name.lower(), id) for id in get_audio_files_in_dir(spk_cls)]
    return speaker_mapping


def get_closest_long_clip(segments:List[Segment], speech_time = 3.0):
    from heapq import heappush
    speakers = defaultdict(list)
    for i, seg in enumerate(segments):
        log.info(f"{seg.gender} {seg.pitch_hz} {seg.text_in}")
        if seg.end - seg.start > speech_time:
            heappush(speakers[seg.gender], (seg.pitch_hz, i))
        if not seg.ref_audio_path:
            seg.ref_audio_path = seg.audio_path

    for seg in segments:
        try:
            if (seg.end - seg.start < speech_time):
                _, i = find_closest(speakers[seg.gender], seg.pitch_hz)
            seg.ref_audio_path = segments[i].ref_audio_path
        except:
            if not seg.ref_audio_path:
                seg.ref_audio_path = seg.audio_path
            
    return segments, speakers


# Voice clone profiles
def build_speaker_ref_profiles(segments: List[Segment], workdir: Path, 
                thresh = 4.0, max_audios =6, max_thresh = 8.0):
    from heapq import heappush, heappop
    speakers = defaultdict(list)
    for seg in segments:
        dur = seg.end - seg.start
        if thresh < dur:
            heappush(speakers[seg.speaker], (dur, seg.pitch_hz, seg.ref_audio_path))
            if len(speakers[seg.speaker]) >max_audios:
                heappop(speakers[seg.speaker])

    for seg in segments:
        if seg.end - seg.start > thresh:
            seg.ref_audio_path = seg.audio_path
        elif len(speakers[seg.speaker]) > 0:
            seg.ref_audio_path = find_closest(speakers[seg.speaker], seg.pitch_hz, idx=1)[-1]
        else:
            # don't generate this person whose voice is not available at all
            seg.gen_audio_path = seg.audio_path

    (workdir / "speaker_profiles.json").write_text(
        json.dumps({k: v for k, v in speakers.items()}, indent=2, ensure_ascii=False)
    )
    return segments, speakers


def build_clone_references(segments: List[Segment], vocals_16k: Path, workdir: Path,
                             ref_max_seconds: float = 6.0) -> List[Segment]:
    """Trims each segment's own English audio into a short, silence-trimmed
    reference clip. Kept SHORT deliberately: longer reference clips make
    the clone lean more heavily on English phonetic/prosodic patterns;
    short clips (~3-6s) still carry pitch/timbre/cadence without as much
    English-specific articulation for the model to imitate."""
    ref_dir = ensure_dir(workdir / "clone_refs")
    audio, sr = sf.read(str(vocals_16k))

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


def read_subs_as_whisper_segments(sub_file, stype='out'):
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
                f"text_{stype}": sub.text.replace("\n", " ")
            }
            segments.append(segment)
        return segments
    except:
        return None

# --------------------------------------------------------------------------
# CLAP based classification, descriptions
# --------------------------------------------------------------------------

def extract_tensor(output) -> torch.Tensor:
    """get_audio_features / get_text_features should return a tensor directly,
    but some versions/paths can return a model output object instead.
    This normalizes both cases into a plain tensor."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]
    raise TypeError(f"Unexpected output type from CLAP feature extractor: {type(output)}")


@torch.no_grad()
def get_audio_embedding(audio, model, processor, device) -> torch.Tensor:
    inputs = processor(audio=audio, sampling_rate=48000, return_tensors="pt").to(device)
    audio_embed = extract_tensor(model.get_audio_features(**inputs))
    return audio_embed / audio_embed.norm(dim=-1, keepdim=True)


def get_clap_features(segment, audio, model, processor, device='mps', language='hindi'):
    audio_embed = get_audio_embedding(audio, model, processor, device)

    @torch.no_grad()
    def best_clap_label(category: str) -> str:
        labels = SPEECH_CATEGORIES[category]["labels"]
        template = SPEECH_CATEGORIES[category]["template"]
        texts = [template.format(l) for l in labels]

        text_inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        text_embed = extract_tensor(model.get_text_features(**text_inputs))
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        sims = (audio_embed @ text_embed.T).squeeze(0)
        best_idx = sims.argmax().item()
        return labels[best_idx]
    
    # segment.age = best_clap_label("age")
    # segment.gender = best_clap_label("gender") # Better gender model used
    segment.quality = best_clap_label("quality")
    segment.emotion = best_clap_label("emotion")
    segment.speed = best_clap_label("speed")
    #segment.pitch = best_clap_label("pitch")
    #segment.clarity = best_clap_label("clarity")
    segment.distance = best_clap_label("distance")

    return segment

def build_clap_profiles(segments, vocals_wav, workdir, clap_model:str, device = 'mps', 
                        tgt_lang='hindi', force=False):
    cache = workdir / "transcript_clap.json"
    if cache.exists() and not force:
        data = json.loads(cache.read_text())
        return [Segment(**s) for s in data]
    # CLAP descriptions
    from transformers import ClapModel, ClapProcessor
    desc_model = ClapModel.from_pretrained(clap_model).to(device).eval()
    desc_processor = ClapProcessor.from_pretrained(clap_model)
    audio_48k, _ = librosa.load(str(vocals_wav), sr = 48000)

    for new_seg in segments:
        # Get description
        clip_48k = audio_48k[int(new_seg.start * 48000):int(new_seg.end*48000)]
        new_seg = get_clap_features(new_seg, clip_48k, desc_model, desc_processor, device, tgt_lang)

    cache.write_text(json.dumps([asdict(s) for s in segments], indent=2))
    return segments


# --------------------------------------------------------------------------
# Stage 3: transcription + diarization (whispermlx)
# --------------------------------------------------------------------------
def subs_only_transcription(vocals_16k: Path, workdir: Path, segments, force: bool = False, stype='in') -> List[Segment]:
    cache = workdir / "transcript_subs.json"
    if cache.exists() and not force:
        data = json.loads(cache.read_text())
        return [Segment(**s) for s in data]

    # Gender classifer
    from transformers import pipeline
    gender_model = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
    gender_classifier = pipeline("audio-classification", model=gender_model)

    ref_dir = ensure_dir(workdir / "speaker_refs")
    audio, sr = sf.read(str(vocals_16k))

    new_segments: List[Segment] = []
    for j, seg in enumerate(segments):
        ref_path = ref_dir / f"seg_{j}.wav"
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        if end_sample > len(audio):
            seg["end"] = len(audio)/sr
        clip = audio[start_sample:end_sample]

        # Use the sound within the subtitle as the generator
        new_seg = Segment(
            index=j, start=float(seg["start"]), end=float(seg["end"]),
            speaker=None, 
            text_in=seg.get("text_in", ""),
            text_out=seg.get("text_out", None),
            audio_path = str(ref_path), ref_audio_path = str(ref_path),
            gender=gender_classifier(clip)[0]['label']
        )
        sf.write(new_seg.audio_path, clip, sr)
        new_segments.append(new_seg)

    if new_segments:
        cache.write_text(json.dumps([asdict(s) for s in new_segments], indent=2))
    return new_segments


def subs_and_diarize(vocals_16k: Path, workdir: Path, segments, force: bool = False,
                 device: str = "cpu", stype='in', hf_token: Optional[str] = None) -> List[Segment]:
    cache = workdir / "transcript_subs.json"
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
        # Match segment
        ref_path = ref_dir / f"{spk}_{j}.wav"
        new_seg = Segment(
            index=j, start=float(turn.start), end=float(turn.end),
            speaker=spk or "SPEAKER_00", 
            text_in=segments[i].get("text_in", ""),
            text_out=segments[i].get("text_out", ""),
            audio_path = str(ref_path), gender="Male"
        )

        start_sample = int(turn.start * sr)
        end_sample = int(min(turn.end, turn.start + 8.0) * sr)
        if end_sample > len(audio):
            new_seg.end = len(audio)/sr
            # new_segments.append(new_seg)
        clip = audio[start_sample:end_sample]

        new_seg.gender = gender_classifier(clip)[0]['label']

        sf.write(str(ref_path), clip, sr)
        # Build text from sub segments
        try:
            while True:
                i+=1
                log.info(f'{i},{j}, {segments[i]["start"]}, {segments[i]["end"]}')
                mid = (segments[i]["start"] + segments[i]["end"]) / 2
                if turn.start <= segments[i]["end"]  <= turn.end:
                    if stype == 'in':
                        try:
                            new_seg.text_in += f' {segments[i]["text_in"]}'
                        except:
                            pass
                    else:
                        try:
                            new_seg.text_out += f' {segments[i]["text_out"]}'
                        except:
                            pass
                else:
                    log.info(new_seg)
                    if len(new_seg.text_in) > 0 or len(new_seg.text_out) > 0:
                        new_segments.append(new_seg)
                        j+=1
                    break
        except Exception as e:
            log.info(e)
            break # pass?

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
        "temperatures" : [0.1, 0.3],
        "logprob-threshold" : -0.5,
        "best_of":2,
        "condition_on_previous_text": True
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

    segments: List[Segment] = []
    ref_dir = ensure_dir(workdir / "speaker_refs")
    audio, sr = sf.read(str(vocals_16k))

    # Basic Gender classifer
    from transformers import pipeline
    gender_model = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
    gender_classifier = pipeline("audio-classification", model=gender_model) 
    
    for i, seg in enumerate(result["segments"]):
        text = (seg.get("text") or "").strip()
        if not text or len(text)<0:
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

        new_seg = Segment(
            index=i, start=float(seg["start"]), end=float(seg["end"]),
            speaker=speaker or "SPEAKER_00", text_in=text,
            audio_path = str(ref_path)
        )
        new_seg.gender = gender_classifier(clip)[0]['label']
        # Write audio
        sf.write(str(ref_path), clip, sr)
        segments.append(new_seg)

    cache.write_text(json.dumps([asdict(s) for s in segments], indent=2))
    log.info("Final: %d segments across %d speakers", len(segments),
              len({s.speaker for s in segments}))
    return segments


# --------------------------------------------------------------------------
# Audio features
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

def extract_pitch_and_speed(segments: List[Segment], vocals_16k: Path) -> List[Segment]:

    audio, sr = sf.read(str(vocals_16k))
    for seg in segments:
        chunk = audio[int(seg.start * sr):int(seg.end * sr)]
        duration = max(seg.end - seg.start, 0.01)
        seg.speed_wps = len(seg.text_in.split()) / duration

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
# Stage 8: time-align each clip to its original slot, with crossfades
# --------------------------------------------------------------------------

def get_duration(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    return info.frames / info.samplerate

def time_stretch(in_path: Path, out_path: Path, factor: float, min_stretch:float=0.7):
    factor = max(min_stretch, factor)
    run(["ffmpeg", "-y", "-i", str(in_path), "-filter:a", f"atempo={factor:.4f}", str(out_path)])


def align_segments(segments: List[Segment], workdir: Path, force: bool = False,
                    max_stretch: float = 1.5) -> List[Segment]:
    aligned_dir = ensure_dir(workdir / "aligned")
    for seg in segments:
        log.info(f"Creating {seg.index} : {seg.styled_audio_path}, {seg.aligned_audio_path}")
        if not seg.styled_audio_path:
            if seg.gen_audio_path:
                seg.styled_audio_path = seg.gen_audio_path
            elif seg.audio_path:
                seg.styled_audio_path = seg.audio_path
            else:
                log.info(f"No audio for alignment {seg.index}, {seg.text_out}")
                continue

        out_path = aligned_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.aligned_audio_path = str(out_path)
            continue

        target_dur = max(seg.end - seg.start, 0.1)
        actual_dur = get_duration(Path(seg.styled_audio_path))
        factor = actual_dur / target_dur
        factor = max(1.0 / max_stretch, min(factor, max_stretch))
        if abs(factor - 1.0) < 0.03:
            seg.aligned_audio_path = seg.styled_audio_path
        else:
            # Speed up or slow down a little, not completely
            time_stretch(Path(seg.styled_audio_path), out_path, np.sqrt(factor)) 
            seg.aligned_audio_path = str(out_path)
    return segments


# --------------------------------------------------------------------------
# Stage 9: reassemble (with short crossfades), loudness-match, mix
# --------------------------------------------------------------------------

def build_target_vocal_track(segments: List[Segment], vocals:Path, total_duration: float, workdir: Path,
                              sample_rate: int = 48000, fade_ms: float = 10.0) -> Path:
    out_path = workdir / "target_vocals_full.wav"
    canvas = np.zeros((int(total_duration * sample_rate) + sample_rate, 2), dtype=np.float64)
    fade_samples = int(sample_rate * fade_ms / 1000.0)

    # Do not overlap speech
    prev_end = -1
    fade = int(fade_ms*sample_rate/1000)
    for seg in sorted(segments, key=lambda s: s.start):
        if not seg.aligned_audio_path:
            continue
        audio = load_in_stereo(seg.aligned_audio_path, sample_rate)

        # Start, end
        start_sample = int(seg.start * sample_rate)
        if start_sample < prev_end:
            start_sample = prev_end+fade
        end_sample = int(start_sample + len(audio))
        #prev_end = end_sample

        if len(audio) > 2 * fade_samples:
            ramp = np.linspace(0, 1, fade_samples)[:, None]
            audio[:fade_samples] *= ramp
            audio[-fade_samples:] *= ramp[::-1]

        if end_sample > canvas.shape[0]:
            canvas = np.pad(canvas, ((0, end_sample - canvas.shape[0]), (0, 0)))
        canvas[start_sample:end_sample] += audio[:, :2] if audio.shape[1] >= 2 else audio
    canvas, _ = normalize_peak(canvas, ceiling=0.999)

    sf.write(str(out_path), canvas, sample_rate,subtype='PCM_16')
    return out_path


def loudness_match_and_mix(target_vocals: Path, background: Path, workdir: Path) -> Path:
    # normalized = workdir / "target_vocals_final.wav"
    # run(["ffmpeg", "-y", "-i", str(target_vocals), "-af", f"loudnorm=I=-16:TP=-1.5:LRA=11", str(normalized)])
    mixed = workdir / "target_final_track.wav"
    run(["ffmpeg", "-y", "-i", str(target_vocals), "-i", str(background),
         "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0", str(mixed)])
    return mixed


# --------------------------------------------------------------------------
# Stage 10: mux into video
# --------------------------------------------------------------------------

def mux_into_video(video_path: Path, hindi_track: Path, output_path: Path,
            src_lang='eng', tgt_lang='hin'):
    run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(hindi_track),
        "-map", "0:v", "-map", "0:a", "-map", "1:a",
        "-c:v", "copy",
        "-c:a:0", "copy",
        "-c:a:1", "aac", "-b:a:1", "192k",
        "-metadata:s:a:0", f"language={src_lang}",
        "-metadata:s:a:1", f"language={tgt_lang}",
        "-metadata:s:a:1", "title=Hindi (AI Dubbed)",
        "-disposition:a:0", "default",
        "-disposition:a:1", "0",
        str(output_path),
    ])