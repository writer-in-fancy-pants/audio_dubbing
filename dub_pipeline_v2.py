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
        --input movie.mp4 --output movie.target_dubbed.mp4 \\
        --workdir ./work --tts-method clone --hf-token hf_xxx

    python dub_pipeline_s2t2s.py \\
        --input movie.mp4 --output movie.target_dubbed.mp4 \\
        --workdir ./work --tts-method parler --hf-token hf_xxx
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Dict
import soundfile as sf
import torch
import numpy as np
import re

from utils import (
    Segment, lang_ref, log,
    run, ensure_dir, resolve_device,
    extract_embedded_subtitles, read_subs_as_whisper_segments,
    separate_vocals, slowdown,
    transcribe_and_diarize, subs_and_diarize, subs_only_transcription,
    build_clap_profiles,
    check_dont_generate, classify_emotion,
    extract_pitch_and_speed, get_closest_long_clip, build_speaker_ref_profiles,
    align_segments, get_media_duration,
    build_target_vocal_track, loudness_match_and_mix, mux_into_video
)

HINDI_PARLER_PRESETS = {
    "male": ["Rohit", "Aman"],          # Rohit is AI4Bharat's recommended male preset
    "female": ["Divya", "Rani"],        # Divya is AI4Bharat's recommended female preset
    "child": ['Aman',"Divya"],          # no dedicated child voice for Hindi; closest available
}

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
# Stage 5: translation
# --------------------------------------------------------------------------

def translate_indictrans2(segments: List[Segment], workdir: Path, force: bool = False,
                        src_lang:str='eng_Latn', tgt_lang:str='hin_Deva',
                        device: str = "cpu") -> List[Segment]:
    cache = workdir / "translated.json"
    if cache.exists() and not force:
        data = json.loads(cache.read_text())
        return [Segment(**s) for s in data]

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit.processor import IndicProcessor

    model_name = "ai4bharat/indictrans2-en-indic-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    ip = IndicProcessor(inference=True)

    texts = [seg.text_in for seg in segments]
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

    for seg, out in zip(segments, translations):
        seg.text_out = out
        log.info("[%d] IN: %s", seg.index, seg.text_in)
        log.info("[%d] OUT: %s", seg.index, seg.text_out)

    cache.write_text(json.dumps([asdict(s) for s in segments], indent=2))
    return segments


def translate_nllb(segments: List[Segment], workdir: Path, force: bool = False,
                    src_lang:str='eng_Latn', tgt_lang:str='hin_Deva',
                    device: str = "cpu") -> List[Segment]:
    """Fallback translator if IndicTransToolkit isn't installed."""
    cache = workdir / "translated.json"
    if cache.exists() and not force:
        data = json.loads(cache.read_text())
        return [Segment(**s) for s in data]

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    tgt_id = tok.convert_tokens_to_ids(tgt_lang)

    for seg in segments:
        inputs = tok(seg.text_in, return_tensors="pt").to(device)
        generated = model.generate(**inputs, forced_bos_token_id=tgt_id, max_new_tokens=256)
        seg.text_out = tok.batch_decode(generated, skip_special_tokens=True)[0].strip()

    cache.write_text(json.dumps([asdict(s) for s in segments], indent=2))
    return segments


def build_texts_for_translate(segments, sym, thresh=20):
    # Generate the output
    remove = r'[\[\]-_]'
    texts = []
    i = 0
    text = ''
    gender = segments[i].gender

    def append_text(seg, t, g, i):
        if len(t) > 0:
            texts.append((t, g, i))
        return f'{re.sub(remove, '', seg.text_in)} {sym} ', seg.gender, 1

    for seg in segments:
        if gender != seg.gender:
            text, gender, i = append_text(seg, text, gender, i)
        else:
            text += f'{re.sub(remove, '', seg.text_in)} {sym} '
            i+=1
        if i%thresh == 0:
            text, gender,i = append_text(seg, text, gender, i)

    if i!= 0:
        texts.append((text, gender, i))
    return texts

def translate_mlx(segments: List[Segment], workdir: Path, force: bool = False,
                  src_lang:str='english', tgt_lang:str='hindi', sym=';',
                  model_name = "lmstudio-community/gemma-4-E4B-it-MLX-4bit") -> List[Segment]:
    cache = workdir / "translated.json"
    if cache.exists() and not force:
        data = json.loads(cache.read_text())
        return [Segment(**s) for s in data]

    import re
    from mlx_lm import generate, load

    # Load the model and tokenizer
    model, tokenizer = load(model_name)
    thresh = 30
    texts = build_texts_for_translate(segments, sym, thresh)
    out = []

    for t,g,i in texts:
        prompt = f"""<bos><start_of_turn>user
Translate the {src_lang} text below to {tgt_lang}. Speaker gender {g}. Do not translate, change, or remove {sym}, keep {sym} in the exact same spot in the text. 

{t}<end_of_turn>"""
        output_text = generate(model, tokenizer, prompt=prompt, max_tokens = min(4*len(t),200))
        output_text = re.sub(r'<[^>]*>', '', output_text) # remove model output tags
        temp = output_text.strip().strip(sym).split(sym)
        out.extend(temp[:thresh])

    for i, t in enumerate(out[:len(segments)]):
        segments[i].text_out = t
        if len(t)<1:
            segments[i].gen_audio_path = segments[i].audio_path

    cache.write_text(json.dumps([asdict(s) for s in segments], indent=2))
    return segments

def translate_sarvam(segments: List[Segment], workdir: Path, force: bool = False,
                     device: str = "cpu", src_lang = 'English', tgt_lang = "Hindi", sym=';',
                     model_name = "sarvamai/sarvam-translate") -> List[Segment]:
    """Fallback translator if IndicTransToolkit isn't installed."""
    cache = workdir / "translated.json"
    if cache.exists() and not force:
        data = json.loads(cache.read_text())
        return [Segment(**s) for s in data]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    thresh = 20
    texts = build_texts_for_translate(segments, sym, thresh)
    offset = 0
    for t,g,i in texts:
        messages = [
            {"role": "system", "content": f"Translate the {src_lang} text to {tgt_lang}. {g} speaker."
             f"Keep {sym} in the exact same spot in the text"},
            {"role": "user", "content": t}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        temp = []
        #while (i > len(temp)):
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.01,
            top_p = 0.5,
            top_k = 20,
            num_return_sequences=1
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        output_text = re.sub(r'<[^>]*>', '', output_text)
        log.info(f"{t}, {output_text}")
        temp = output_text.strip().strip(sym).split(sym)
        
        # Adds translated text - TODO : sometimes missing a few in the middle 
        for j, ot in enumerate(temp):
            k = offset + j
            if len(segments) > k:
                log.info(f"{j} -> {k} - {segments[k].text_in} : {ot}")
                segments[k].text_out = ot
        offset+=i

    cache.write_text(json.dumps([asdict(s) for s in segments], indent=2))
    return segments


# --------------------------------------------------------------------------
# Stage 6: speaker profiles + Parler preset mapping
# --------------------------------------------------------------------------

# Verified against the ai4bharat/indic-parler-tts model card's Hindi row.
# Only 4 named voices exist for Hindi; if you have more than 2 speakers of
# the same detected gender, they will share a base timbre under Method A --
# differentiate them further via speaker_overrides.json (see below) or use
# Method B (voice cloning) instead.

def build_parler_description(seg: Segment) -> str:
    """Composes a natural-language caption per Indic Parler-TTS's documented
    controls: named speaker + pitch + speaking rate + expressivity/emotion
    + recording quality. Pitch/rate thresholds are simple heuristics --
    tune them by ear against your source material."""
    pitch = seg.pitch_hz
    speed = seg.speed_wps
    emotion = seg.emotion

    pitch_adj = "low-pitched" if pitch < 110 else ("high-pitched" if pitch > 200 else "moderately pitched")
    rate_adj = "slow-paced" if speed < 2.0 else ("moderately fast-paced" if speed > 3.2 else "moderately paced")
    emotion_adj = {
        "happy": "cheerful and animated", "angry": "tense and forceful",
        "sad": "subdued and downcast", "neutral": "calm and even",
    }.get(emotion, "conversational")
    #clarity = seg.clarity or "very clear"
    distance = seg.distance or "close"

    return (
        f"{HINDI_PARLER_PRESETS[seg.gender][0]}'s voice is {pitch_adj} and {rate_adj}, delivered in a "
        f"{emotion_adj} tone. The recording is very clear and {distance}-sounding, with no "
        f"background noise."
    )


# --------------------------------------------------------------------------
# Stage 7a: TTS Method A -- Indic Parler-TTS
# Mostly use for creating styled audio since it is good at copying voice accent,
# but really bad at actually generating TTS
# --------------------------------------------------------------------------
def synth_parler(segments: List[Segment], workdir: Path, force: bool = False, 
                 model_name = "ai4bharat/indic-parler-tts", device: str = "cpu",
                 clap_model = None) -> List[Segment]:
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    out_dir = ensure_dir(workdir / "tts_parler")
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_name,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

    for seg in segments:
        out_path = out_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.gen_audio_path = str(out_path)
            continue

        if check_dont_generate(seg):
            seg.gen_audio_path = seg.audio_path
            continue

        seg.description = build_parler_description(seg)

        desc_ids = description_tokenizer(seg.description, return_tensors="pt").to(device)
        prompt_ids = tokenizer(seg.text_out, return_tensors="pt").to(device)
        with torch.no_grad():
            generation = model.generate(
                input_ids=desc_ids.input_ids, attention_mask=desc_ids.attention_mask,
                prompt_input_ids=prompt_ids.input_ids, prompt_attention_mask=prompt_ids.attention_mask,
            )
        audio_arr = generation.cpu().numpy().squeeze()
        if audio_arr.ndim == 1:
            audio_arr = audio_arr[:, np.newaxis]
        else:
            print(audio_arr.shape)
        sf.write(str(out_path), audio_arr, model.config.sampling_rate)
        seg.gen_audio_path = str(out_path)
        log.info("[%d] Parler synth : %s", seg.index, seg.description)
    return segments

# --------------------------------------------------------------------------
# Stage 7b: TTS Method B -- IndicF5 voice cloning (self-reference)
# --------------------------------------------------------------------------

# Use only with indic-parler-tts to fix accent issues
def synth_clone_f5(segments: List[Segment], workdir: Path,
                    model_name:str = "ai4bharat/IndicF5", source = 'ref',
                    force: bool = False) -> List[Segment]:
    from transformers import AutoModel
    out_dir = ensure_dir(workdir / "tts_clone")
    model = AutoModel.from_pretrained(model_name)#, trust_remote_code=True)

    for seg in segments:
        out_path = out_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.gen_audio_path = str(out_path)
            continue
        if not seg.audio_path or not seg.text_out:
            log.warning("Segment %d missing reference audio or translation; skipping", seg.index)
            continue

        if check_dont_generate(seg):
            if not seg.gen_audio_path:
                seg.gen_audio_path = seg.audio_path
        else:
            if source == 'ref':
                ref = seg.ref_audio_path
            else:
                ref = seg.audio_path
            audio = model(
                seg.text_out,
                audio_path=ref,
                ref_text=seg.text_in,   # transcript of the reference clip, per IndicF5's documented API
            )
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            sf.write(str(out_path), np.array(audio, dtype=np.float32), samplerate=24000)
            seg.gen_audio_path = str(out_path)
        log.info("[%d] IndicF5 clone synth (ref=%s)", seg.index, Path(seg.audio_path).name)
    return segments


# --------------------------------------------------------------------------
# Stage 7c: TTS Method c -- XTTS voice cloning (few shot)
# --------------------------------------------------------------------------
def synth_coqui(segments: List[Segment], workdir: Path, profiles: Dict[str, List[str]]={}, 
                tgt_lang = 'hi', gpt_cond_len = 30, force: bool = False ) -> List[Segment]:
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts_dir = ensure_dir(workdir / "tts_raw")
    for seg in segments:
        out_path = tts_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.gen_audio_path = str(out_path)
            continue
        if not seg.audio_path:
            log.warning("No reference audio for segment %s, skipping", seg.index)
            continue
        if check_dont_generate(seg, 0.3, 1):
            if not seg.gen_audio_path:
                seg.gen_audio_path = seg.audio_path
        else:
            if seg.ref_audio_path:
                refs = [seg.audio_path, seg.ref_audio_path]
            else:
                refs = seg.audio_path
                try:
                    tts.tts_to_file(text=seg.text_out,
                            file_path=out_path,
                            speaker_wav=refs,
                            language=tgt_lang,
                            gpt_cond_len = gpt_cond_len)
                    seg.gen_audio_path = str(out_path)
                except:
                    seg.gen_audio_path = seg.audio_path
    return segments

# --------------------------------------------------------------------------
# Stage 7e: TTS Method D -- Chatterbox Multilingual + voice cloning
# --------------------------------------------------------------------------
def synth_chatterbox(segments:List[Segment], workdir:Path,
            tgt_lang = 'hi', force: bool = False,  device: str = "cpu" ) -> List[Segment]:
    # 24000 hz
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    import torchaudio

    out_dir = ensure_dir(workdir / "tts")
    multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
    
    for seg in segments:
        out_path = out_dir / f"seg_{seg.index:04d}_temp.wav"
        if out_path.exists() and not force:
            seg.gen_audio_path = str(out_path)
            continue

        if check_dont_generate(seg, 0.5, 2):
            if not seg.gen_audio_path:
                seg.gen_audio_path = seg.audio_path
            continue

        seg.gen_audio_path = str(out_path)
        arr = multilingual_model.generate(seg.text_out, language_id=tgt_lang)
        torchaudio.save(seg.gen_audio_path, arr, multilingual_model.sr)
    return segments


def voice_style_transfer_chatterbox(segments: List[Segment], workdir: Path,
                tgt_lang:str = 'hi', force: bool = False, device: str = "cpu") -> List[Segment]:
    # 24000 hz
    from chatterbox.vc import ChatterboxVC
    import torchaudio
    out_dir = ensure_dir(workdir / "voice_styled")
    voice_model = ChatterboxVC.from_pretrained(
        device=device,
    )

    for seg in segments:
        out_path = out_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.gen_audio_path = str(out_path)
            continue

        if check_dont_generate(seg, 0.7, 2):
            if not seg.gen_audio_path:
                seg.gen_audio_path = seg.audio_path
            continue

        # Using english audio to voice clone if no references
        if not seg.ref_audio_path:
            continue
        print(seg.text_in, seg.text_out, seg.gen_audio_path)
        arr = voice_model.generate(
            seg.gen_audio_path,
            seg.ref_audio_path
        )
        
        log.info(f"{voice_model.sr} {arr.shape}")
        torchaudio.save(str(out_path), arr, voice_model.sr)
        # Note variable reset to styled wav
        seg.styled_audio_path = str(out_path)
        
    return segments

# --------------------------------------------------------------------------
# Stage 7e: TTS Method D -- Voxtral (Zero shot)
# --------------------------------------------------------------------------

def synth_voxtral(segments: List[Segment], workdir: Path, lang = 'hi', force = False) -> List[Segment]:
    from mlx_audio.tts.utils import load
    model = load("mlx-community/Voxtral-4B-TTS-2603-mlx-bf16")
    tts_dir = ensure_dir(workdir / "tts_voxtral")
    for seg in segments:
        out_path = tts_dir / f"seg_{seg.index:04d}.wav"
        if out_path.exists() and not force:
            seg.gen_audio_path = str(out_path)
            continue
        if not seg.audio_path:
            log.warning("No reference audio for segment %s, skipping", seg.index)
            continue
        if len(seg.text_in) < 12 or seg.end - seg.start < 0.3:
            seg.gen_audio_path = seg.audio_path
        else:
            try:
                model.generate(text=seg.text_out, voice=f"{lang}_{seg.gender}")
                seg.gen_audio_path = str(out_path)
            except:
                seg.gen_audio_path = seg.audio_path
    return segments


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
    ap.add_argument("--no-voice-styling", action="store_true")
    ap.add_argument("--no-detect-age-gender", action="store_true",
                     help="skip wav2vec2 age/gender classification; falls back to a pitch-threshold gender guess")
    ap.add_argument("--no-detect-emotion", action="store_true",
                     help="skip emotion classification; defaults every segment to 'neutral'")
    ap.add_argument("--emotion-backend", default="categorical", choices=["categorical", "dimensional"])

    ap.add_argument("--translator", default="indictrans2", choices=["indictrans2", "nllb", "mlx", "sarvam"])

    ap.add_argument("--tts-method", required=True, choices=["parler", "voxtral", 'coqui', 'chatterbox'])
    ap.add_argument("--clap-model", action='store_true')
    ap.add_argument("--speaker-overrides", type=Path, default=None,
                     help="JSON: {\"SPEAKER_00\": {\"parler_preset\": \"Rohit\"}} to manually pin presets")
    ap.add_argument("--ref-max-seconds", type=float, default=20.0)
    ap.add_argument("--max-stretch", type=float, default=1.3)
    ap.add_argument("--slowdown", type=float, default=1.0)

    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    workdir = ensure_dir(args.workdir)
    device = resolve_device(args.device)
    log.info("Working directory: %s", workdir.resolve())

    # Slow down video a bit, helps with language transition - hindi is slower
    args.input = slowdown(args.input, args.slowdown, workdir)

    # Subtitles - mostly not used right now
    sub = None
    segments=None
    need_translate = True
    lang = args.target_lang
    if args.subtitle_file:
        segments = read_subs_as_whisper_segments(sub, stype = 'out')
    if sub is None and not args.no_use_subs:
        sub_file = extract_embedded_subtitles(args.input, workdir / f"subs.srt", lang=lang)
        if sub_file is None:
            lang = args.source_lang
            sub_file = extract_embedded_subtitles(args.input, workdir / f"subs.srt", lang=lang)
        else:
            need_translate = False
        segments = read_subs_as_whisper_segments(sub_file, stype='in')

    # Separate audio
    stereo_wav, _mono_unused = extract_audio(args.input, workdir, force=args.force)
    vocals_wav, background_wav, vocals_16k, vol = separate_vocals(
        stereo_wav, workdir, force=args.force, device=device, model=args.demucs_model
    )

    # Subtitle created segments
    if segments:
        if args.no_diarize:
            segments = subs_only_transcription(vocals_16k, workdir, segments, force=args.force)
        else:
            segments = subs_and_diarize(
                vocals_16k, workdir, segments, force=args.force,
                device=device, hf_token=args.hf_token
            )

    if not segments:
        # Subtitle pipeline failed, transcribe instead
        segments = transcribe_and_diarize(
            vocals_16k, workdir, force=args.force, model_size=args.whisper_model,
            device=device, diarize=not args.no_diarize, hf_token=args.hf_token
        )

    if not segments:
        log.error("No speech detected -- aborting.")
        sys.exit(1)

    # Get speech features
    segments = extract_pitch_and_speed(segments, vocals_16k)
    # Need to limit the size of sample audio
    if args.no_diarize:
        segments, profiles = get_closest_long_clip(segments)
    else:
        segments, profiles = build_speaker_ref_profiles(segments, workdir)


    if need_translate:
        if args.translator == "indictrans2":
            try:
                segments = translate_indictrans2(segments, workdir, force=args.force, device=device, 
                                src_lang=lang_ref[args.source_lang][1], tgt_lang=lang_ref[args.target_lang][1])
            except ImportError as e:
                log.warning("IndicTransToolkit unavailable (%s); falling back to NLLB", e)
                segments = translate_nllb(segments, workdir, force=args.force, device=device,
                                src_lang=lang_ref[args.source_lang][1], tgt_lang=lang_ref[args.target_lang][1])
        elif args.translator == "nllb":
            segments = translate_nllb(segments, workdir, force=args.force, device=device,
                                src_lang=lang_ref[args.source_lang][1], tgt_lang=lang_ref[args.target_lang][1])
        elif args.translator == "mlx":
            segments = translate_mlx(segments, workdir, force=args.force, device=device, model_name="mlx-community/sarvam-translate-mlx-4bit",
                                src_lang=lang_ref[args.source_lang][-1], tgt_lang=lang_ref[args.target_lang][-1])
        else:
            segments = translate_sarvam(segments, workdir, force=args.force, device=device,
                                src_lang=lang_ref[args.source_lang][-1], tgt_lang=lang_ref[args.target_lang][-1])

        for seg in segments[-10:]:
            print(seg.text_in, seg.text_out)


    if args.tts_method == "parler":
        if args.clap_model:
            segments = build_clap_profiles(segments, vocals_wav, workdir, clap_model= 'laion/larger_clap_general', 
                            device = device, tgt_lang=lang_ref[lang][-1], force = args.force)
        elif not args.no_detect_emotion:
            segments = classify_emotion(segments, vocals_16k, workdir, force=args.force,
                                          device=device, backend=args.emotion_backend)
            for seg in segments:
                if seg.emotion is None:
                    seg.emotion = "neutral"

        segments = synth_parler(segments, workdir, force=args.force, device=device)
    else:
        # Cluster speaker clips, get number of speakers
        if args.tts_method == "coqui":
            segments = synth_coqui(segments, profiles, workdir, force=args.force, device=device)
        elif args.tts_method == "voxtral":
            # Voxtral + indicf5 clone
            segments = synth_voxtral(segments, workdir, lang=args.target_lang)
        else:
            segments = synth_chatterbox(segments, workdir,tgt_lang=args.target_lang, force=args.force, device=device)

    if not args.no_voice_styling:
        segments = voice_style_transfer_chatterbox(segments, workdir, tgt_lang=args.target_lang,  force=args.force, device="cpu")

    segments = align_segments(segments, workdir, force=args.force, max_stretch=args.max_stretch)

    total_duration = get_media_duration(args.input)
    target_vocals_full = build_target_vocal_track(segments, total_duration, workdir)
    # target_vocals_normalized = normalize_new_vocals(target_vocals_full, vocals_wav, workdir)
    target_final_track = loudness_match_and_mix(target_vocals_full, background_wav, workdir)

    mux_into_video(args.input, target_final_track, args.output)
    log.info("Done. Output: %s", args.output.resolve())

if __name__ == "__main__":
    main()
