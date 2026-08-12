# Experimental Modular Voice Dubbing for Shows & Movies
An End-to-end voice dubbing tool. For a chosen video, the tool translat


**PLEASE NOTE** : I will not be uploading any speech clips, for privacy and as a good practice.
Please acquire speech clips and place them in speakers/male and speakers/female directories. 
I will soon add support for generating these reference clips. For best results, it is better
to use voices of the speakers already in the voice models used.

The requirements are intentionally incomplete. What's provided works, while the other options need
understanding what you are doing. You can also add finetuned models yourselves, for your hardware.

## Current Features
- VAD, whisper mlx, diarization
- Speech to speech translation - Seamless M4T + ChatterboxVC
- Text translation (nllb, gemma, sarvam, indictrans2)
- Emotion, gender detection
- F5-voice cloning using ai4bharat models
- Voice cloning Coqui XTTS v2 (preferred)
- Removing voice artefacts, normalizing
- integrate audio track into the video
- Autoconfiguring speaker references
- English to Hindi dubbing

## In Progress
- Voice LORA support
- Speaker classification
- Alignment improvement
- Prosody transfer (open problem)
- Scripts for model/library setup
- Multilingual dubbing


# Video Dubbing Pipeline

Adds an AI-generated Hindi audio track to a video, alongside the original
English track, while keeping the background music/effects intact and
attempting to preserve each speaker's voice via voice cloning.

## How it works

### Transcription + TTS pipeline

```
video.mp4
   │  ffmpeg
   ▼
audio.wav ──────────────► Demucs ──► vocals.wav + background.wav
                                          │
                                          ▼
                  Transcription : whisper (ASR + timestamps) / Using subtitles
                                          │
                                          ▼
                  (optional but recommended) pyannote diarization → per-speaker
                                  reference clips
                                          │
                                          ▼
                  Translate - NLLB-200 / Sarvam / Gemma  (EN text → HI text)
                                          │
                                          ▼
                TTS with Voice cloning - Coqui XTTS-v2 (voice-cloned HI speech)
                          many options - ai4Bharat/IndicF5, etc
                                          │
                                          ▼
                     time-stretch each clip to fit original slot
                                          │
                                          ▼
                    place clips on timeline → loudness-normalize
                                          │
                                          ▼
                    mix with background.wav → hindi_final_track.wav
                                          │
                                          ▼
                ffmpeg mux: video + original EN track + new HI track
```

### Speech to speech pipeline

```
video.mp4
   │  ffmpeg
   ▼
audio.wav ──────────────► Demucs ──► vocals.wav + background.wav
                                          │
                                          ▼
                  (optional but recommended) transcription + diarization → per-speaker
                                  reference clips
                                          │
                                          ▼
                      Translate - Seamless M4T  (EN speech → HI speech)
                                          │
                                          ▼
                          Voice cloning - ChatterboxVC
                                          │
                                          ▼
                     time-stretch each clip to fit original slot
                                          │
                                          ▼
                    place clips on timeline → loudness-normalize
                                          │
                                          ▼
                    mix with background.wav → hindi_final_track.wav
                                          │
                                          ▼
                ffmpeg mux: video + original EN track + new HI track
```


## Setup

```bash
# system dependency

# Download tool
git clone https://github.com/writer-in-fancy-pants/audio_dubbing
cd audio_dubbing

# ffmpeg on linus
# sudo apt-get install ffmpeg   # Linux
#brew install ffmpeg            # Mac

# python dependencies
uv venv --python 3.13
source .venv/bin/activate

# if using --device cuda, install build of torch matching your cuda version
# pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
```

Model weights (Demucs, whisper, NLLB-200, XTTS-v2, pyannote, etc) download automatically 
from Hugging Face on first run. Models like `pyannote/speaker-diarization-3.1` are gated — 
always check that you have access on a model's Hugging Face page and pass `--hf-token <your token>` 
or set the `HF_TOKEN` environment variable.

## Usage
I have only tested on M1 mac for English to Hindi dubbing. I will be uploading scripts for cuda devices later.

### Whisper (with diarization) -> Transcription -> Translation -> TTS + Voice Cloning
Also slows down the video, suitable for shows with a lot of dialog since Hindi has more words on an average, and takes longer time.
```bash
python dub_pipeline_v2.py \
  --input ./output.mkv\
  --output ./output_dubbed.mkv \
  --workdir ./work_output \
  --device mps \
  --hf-token $HF_TOKEN \
  --tts-method coqui \
  --no-use-subs \
  --translator sarvam \
  --slowdown 0.9 --force
```

### Whisper (with diarization) -> SeamlessM4T -> ChatterboxVC
Diarization is highly recommended, though it takes some time. The output without diarization has been found monotonous.
```
python ./dub_pipeline_direct_s2st.py \
  --input ./output.mkv \
  --output ./output_dubbed.mkv \
  --workdir work_s2st \
  --diarize
```

The result is `output_dubbed.mkv` with **two audio tracks**: the
original English (default) and the new Hindi track (needs to be manually selected
in the media player that supports audio-track switching, e.g. VLC, mpv, most streaming
platforms).

Every stage caches its output inside `--workdir`, so if a later stage
fails (e.g. TTS runs out of memory) you can fix it and re-run — completed
stages are skipped. Use `--force` to redo everything from scratch.


## Honest limitations

This is a practical ASR → MT → cloned-TTS pipeline, not a true end-to-end
"direct" speech-to-speech translation model — no open-source model today
that I know of. Expect:

- Flattening of emotion/emphasis compared to the original delivery
- Translation errors, especially with idioms or sarcasm
- Overlapping / missing dialogue
- Lip-sync drift on segments that need heavy time-stretching
- Arbitrary speed changes (specially in the TTS pipeline)
- Extra work needed for scenes with many simultaneous speakers

This project is meant as an experiment in controllable, fully offline dubbing
with modular intermediate steps you can swap components in and out of.
