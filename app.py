import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "hf_cache")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR
import tempfile
tempfile.tempdir = TEMP_DIR

import gradio as gr
import subprocess
import sys
import shutil
from pydub import AudioSegment

SAVED_VOICES_DIR = os.path.join(BASE_DIR, "saved_voices")
os.makedirs(SAVED_VOICES_DIR, exist_ok=True)

EDGE_TTS_EXE = shutil.which("edge-tts") or os.path.join(
    BASE_DIR, "venv", "Scripts", "edge-tts.exe"
)

F5_TTS_EXE = shutil.which("f5-tts_infer-cli") or os.path.join(
    BASE_DIR, "venv", "Scripts", "f5-tts_infer-cli.exe"
)
RVC_MODELS_DIR = os.path.join(BASE_DIR, "rvc_models")
RVC_PYTHON_EXE = os.path.join(BASE_DIR, "rvc_venv", "Scripts", "python.exe")
RVC_INFER_SCRIPT = os.path.join(BASE_DIR, "rvc_infer.py")
os.makedirs(RVC_MODELS_DIR, exist_ok=True)

# ─── Utility: Run edge-tts via subprocess (avoids asyncio conflicts with Gradio) ───
def run_edge_tts(text, voice, output_path, rate=None, pitch=None):
    """Generate TTS audio using edge-tts CLI. Returns True on success."""
    cmd = [EDGE_TTS_EXE, "--voice", voice, "--text", text, "--write-media", output_path]
    if rate:
        cmd += ["--rate", rate]
    if pitch:
        cmd += ["--pitch", pitch]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    return result.returncode == 0, result.stderr

# ─── Voice Library ───
def get_saved_voices():
    voices = []
    if os.path.exists(SAVED_VOICES_DIR):
        for d in sorted(os.listdir(SAVED_VOICES_DIR)):
            if os.path.isdir(os.path.join(SAVED_VOICES_DIR, d)):
                voices.append(d)
    return voices

def load_voice(name):
    if not name:
        return None, ""
    audio_path = os.path.join(SAVED_VOICES_DIR, name, "audio.wav")
    text_path = os.path.join(SAVED_VOICES_DIR, name, "text.txt")
    text = ""
    if os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
    if not os.path.exists(audio_path):
        return None, text
    return audio_path, text

def save_voice(name, audio_path, text):
    if not name or not audio_path:
        return " Provide a name AND audio file.", gr.update(), gr.update(), gr.update(), gr.update()
    name = name.strip().replace(" ", "_")
    voice_dir = os.path.join(SAVED_VOICES_DIR, name)
    os.makedirs(voice_dir, exist_ok=True)
    shutil.copy(audio_path, os.path.join(voice_dir, "audio.wav"))
    with open(os.path.join(voice_dir, "text.txt"), "w", encoding="utf-8") as f:
        f.write(text or "")
    choices = get_saved_voices()
    return f" Voice '{name}' saved!", gr.update(choices=choices, value=name), gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices)

def delete_voice(name):
    if not name:
        return "Select a voice first.", gr.update(), gr.update(), gr.update(), gr.update()
    voice_dir = os.path.join(SAVED_VOICES_DIR, name)
    if os.path.exists(voice_dir):
        shutil.rmtree(voice_dir)
    choices = get_saved_voices()
    return f" Deleted '{name}'", gr.update(choices=choices, value=None), gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices)

# ─── RVC Backend ───
def get_rvc_models():
    models = []
    if os.path.exists(RVC_MODELS_DIR):
        for f in os.listdir(RVC_MODELS_DIR):
            if f.endswith(".pth"):
                models.append(f)
    return models

def run_rvc_conversion(input_audio, model_name, pitch):
    if not input_audio: return None, "Please upload a reference audio."
    if not model_name: return None, "Please select an RVC model (.pth)."
    
    model_path = os.path.join(RVC_MODELS_DIR, model_name)
    output_path = os.path.join(BASE_DIR, "rvc_output.wav")
    
    cmd = [
        RVC_PYTHON_EXE, RVC_INFER_SCRIPT,
        "--model", model_path,
        "--input", input_audio,
        "--output", output_path,
        "--pitch", str(int(pitch)),
        "--method", "rmvpe"
    ]
    
    # Try finding an index file with the same name
    index_path = model_path.replace(".pth", ".index")
    if os.path.exists(index_path):
        cmd += ["--index", index_path]
        
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode == 0 and os.path.exists(output_path):
        return output_path, " Voice converted successfully!"
    else:
        return None, f" RVC Error:\n{result.stdout}\n{result.stderr}"

# ─── F5-TTS Core Engine ───
def run_f5tts(
    text,
    ref_audio_path,
    ref_text,
    output_name="output_cloned.wav"
):
    if not text or not text.strip():
        return None, "Enter text to generate."

    if not ref_audio_path or not os.path.exists(ref_audio_path):
        return None, "Reference audio was not found."

    output_path = os.path.join(BASE_DIR, output_name)
    trimmed_path = os.path.join(TEMP_DIR, "trimmed_ref_gen.wav")
    config_path = os.path.join(BASE_DIR, "inference_config.toml")

    try:
        reference_audio = AudioSegment.from_file(ref_audio_path)

        if len(reference_audio) == 0:
            return None, "Reference audio is empty."

        # Keep the reference clip short for F5-TTS.
        if len(reference_audio) > 8000:
            reference_audio = reference_audio[:8000]

        reference_audio = (
            reference_audio
            .set_channels(1)
            .set_frame_rate(24000)
        )

        reference_audio.export(
            trimmed_path,
            format="wav"
        )

        # Delete any previous output so stale audio is never reused.
        if os.path.exists(output_path):
            os.remove(output_path)

        cleaned_text = text.strip()
        cleaned_ref_text = (ref_text or "").strip()

        # Add punctuation when the user did not provide any.
        if cleaned_text[-1] not in ".!?":
            cleaned_text += "."

        # Automatically transcribe the reference when no transcript is given.
        if not cleaned_ref_text:
            class DummyProgress:
                def __call__(self, *args, **kwargs):
                    pass

            cleaned_ref_text = extract_text_fn(
                trimmed_path,
                progress=DummyProgress()
            )

            if (
                not cleaned_ref_text
                or cleaned_ref_text.startswith("Error")
            ):
                return None, (
                    "Failed to transcribe reference audio: "
                    f"{cleaned_ref_text}"
                )

        import tomli_w

        config_data = {
            "model": "F5TTS_Base",
            "ref_audio": trimmed_path,
            "ref_text": cleaned_ref_text,
            "gen_text": cleaned_text,
            "speed": 1.0,
            "nfe_step": 16,
            "output_dir": BASE_DIR,
            "output_file": output_name,
            "voices": {}
        }

        with open(config_path, "wb") as config_file:
            tomli_w.dump(config_data, config_file)

        environment = os.environ.copy()

        environment.update(
            {
                "TEMP": TEMP_DIR,
                "TMP": TEMP_DIR,
                "NUMBA_DISABLE_JIT": "1",
                "HF_HOME": os.environ["HF_HOME"],
                "PYTHONIOENCODING": "utf-8"
            }
        )

        result = subprocess.run(
            [F5_TTS_EXE, "-c", config_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=environment
        )

        if result.returncode != 0:
            error_message = (
                result.stderr[-1000:]
                or result.stdout[-1000:]
            )

            return None, f"F5-TTS error: {error_message}"

        if not os.path.exists(output_path):
            return None, "Output file was not created."

        import soundfile as sf
        import numpy as np

        audio_data, sample_rate = sf.read(output_path)

        if len(audio_data) == 0:
            return None, "Generated output is empty."

        if np.std(audio_data) < 0.001:
            return None, (
                "Generated output is silent. "
                "Try a clearer reference recording."
            )

        duration = len(audio_data) / sample_rate

        return (
            output_path,
            f"Generated {duration:.1f} seconds of audio."
        )

    except Exception as error:
        return None, f"F5-TTS error: {error}"
                    
# ─── Tab 1: Standard Clone ───
def clone_voice_tab1(text, ref_text, audio_ref, progress=gr.Progress()):
    if not text: return None, "Enter text to generate."
    if not audio_ref: return None, "Upload a reference audio."
    progress(0.2, desc="Processing reference...")
    progress(0.4, desc="Running F5-TTS (1-3 min)...")
    path, log = run_f5tts(text, audio_ref, ref_text)
    progress(1.0)
    return path, log

# ─── Tab 2: Dramatic Story Mode ───
NARRATOR_VOICES = {
    "Guy (Passionate Male)": "en-US-GuyNeural",
    "Christopher (Authority Male)": "en-US-ChristopherNeural",
    "Andrew (Confident Male)": "en-US-AndrewNeural",
    "Eric (Rational Male)": "en-US-EricNeural",
    "Brian (Casual Male)": "en-US-BrianNeural",
    "Jenny (Friendly Female)": "en-US-JennyNeural",
    "Aria (Confident Female)": "en-US-AriaNeural",
    "Ava (Expressive Female)": "en-US-AvaNeural",
    "Ryan (British Male)": "en-GB-RyanNeural",
    "Sonia (British Female)": "en-GB-SoniaNeural",
}

def dramatic_clone(text, saved_voice_name, narrator_style, progress=gr.Progress()):
    if not text:
        return None, None, "Enter a story script."
    if not saved_voice_name:
        return None, None, "Select a saved voice from your library first."

    log_lines = []

    # Step 1: Generate emotional narration via edge-tts
    progress(0.1, desc="Step 1: Generating dramatic narration...")
    voice_id = NARRATOR_VOICES.get(narrator_style, "en-US-GuyNeural")
    emotion_path = os.path.join(TEMP_DIR, "emotion_base.mp3")
    ok, err = run_edge_tts(text, voice_id, emotion_path)
    if not ok:
        return None, None, f" Edge-TTS failed: {err}"
    log_lines.append(f"Step 1:  Emotional narration generated ({narrator_style})")

    # Step 2: Clone into anime voice using F5-TTS
    progress(0.4, desc="Step 2: Cloning into anime voice (1-3 min)...")
    voice_audio, voice_text = load_voice(saved_voice_name)
    if not voice_audio:
        log_lines.append(f"Step 2:  Voice '{saved_voice_name}' audio not found. Showing emotion base only.")
        return emotion_path, None, "\n".join(log_lines)

    clone_path, clone_log = run_f5tts(text, voice_audio, voice_text, "dramatic_clone.wav")
    log_lines.append(f"Step 2: {clone_log}")
    progress(1.0)
    return emotion_path, clone_path, "\n".join(log_lines)

# ─── Tab 3: Hindi/Urdu ───
def generate_hindi(text, voice_id, use_transliteration, speed, pitch, progress=gr.Progress()):
    if not text:
        return None, "Enter some text."

    status = []
    final_text = text

    if use_transliteration:
        has_devanagari = any('\u0900' <= c <= '\u097F' for c in text)
        if not has_devanagari:
            from transliterate import roman_to_devanagari
            final_text = roman_to_devanagari(text)
            status.append(f" Transliterated to: {final_text}")
        else:
            status.append("Text already in Devanagari.")

    output_path = os.path.join(TEMP_DIR, "hindi_output.mp3")

    rate_arg = f"{speed:+d}%" if speed != 0 else None
    pitch_arg = f"{pitch:+d}Hz" if pitch != 0 else None

    progress(0.5, desc="Generating voice...")
    ok, err = run_edge_tts(final_text, voice_id, output_path, rate=rate_arg, pitch=pitch_arg)
    if not ok:
        return None, f" Error: {err}"

    status.append(" Generated successfully!")
    progress(1.0)
    return output_path, "\n".join(status)

# ─── Extract Text (Whisper) ───
def extract_text_fn(audio_path, progress=gr.Progress()):
    if not audio_path: return "Upload an audio file first!"
    try:
        trimmed = os.path.join(TEMP_DIR, "extract_temp.wav")
        audio = AudioSegment.from_file(audio_path)
        if len(audio) > 8000: audio = audio[:8000]
        audio.export(trimmed, format="wav")
        progress(0.4, desc="Loading Whisper...")
        import torch
        from transformers import pipeline
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base",
                        device=device, torch_dtype=torch.float16)
        progress(0.7, desc="Transcribing...")
        result = pipe(trimmed, chunk_length_s=30, generate_kwargs={"task": "transcribe"})
        text = result['text'].strip()
        del pipe
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return text
    except Exception as e:
        return f"Error: {str(e)}"

# ─── Tab 4: Multi-Voice Podcast ───
import re

def normalize_speaker_name(name):
    """Normalize speaker names so spaces, hyphens, underscores, and case all match."""
    normalized = re.sub(r"[\s\-_]+", "_", name.strip())
    return normalized.strip("_").casefold()


def parse_podcast_script(script_text):
    """
    Parse podcast lines safely.

    Accepted examples:
    NARUTO: Hello
    Naruto : Hello
    Naruto Uzumaki: Hello
    Naruto-Uzumaki : Hello

    Returns:
        parsed_lines: list of (speaker, dialogue)
        warnings: list of friendly validation messages
    """
    parsed_lines = []
    warnings = []

    for line_number, raw_line in enumerate((script_text or "").splitlines(), start=1):
        line = raw_line.strip()

        if not line:
            continue

        if ":" not in line:
            warnings.append(
                f"Line {line_number}: skipped because ':' is missing."
            )
            continue

        speaker, dialogue = line.split(":", 1)
        speaker = speaker.strip()
        dialogue = dialogue.strip()

        if not speaker:
            warnings.append(
                f"Line {line_number}: skipped because the speaker name is missing."
            )
            continue

        if not dialogue:
            warnings.append(
                f"Line {line_number}: skipped because '{speaker}' has no dialogue."
            )
            continue

        if len(speaker) > 80:
            warnings.append(
                f"Line {line_number}: skipped because the speaker name is too long."
            )
            continue

        parsed_lines.append((speaker, dialogue))

    return parsed_lines, warnings

def detect_podcast_language(text):
    """
    Detect whether dialogue contains Hindi or Urdu script.

    Returns:
        "hi" for Hindi/Devanagari
        "ur" for Urdu/Arabic script
        "en" for other text
    """
    if any("\u0900" <= character <= "\u097F" for character in text):
        return "hi"

    if any(
        "\u0600" <= character <= "\u06FF"
        or "\u0750" <= character <= "\u077F"
        for character in text
    ):
        return "ur"

    roman_words = {
        "kya", "hai", "hain", "aap", "tum", "mera", "meri",
        "hum", "kaise", "acha", "accha", "bhai", "yaar",
        "nahi", "nahin", "kyun", "kahan", "aaj", "kal",
        "bahut", "bohat", "shukriya", "salam", "salaam"
    }

    words = {
        re.sub(r"[^a-z]", "", word.casefold())
        for word in text.split()
    }

    matches = words.intersection(roman_words)

    if len(matches) >= 2:
        return "hi"

    return "en"


def find_matching_rvc_model(character_name):
    """
    Find an RVC model whose filename matches the podcast character name.

    Examples:
        ARIA -> ARIA.pth
        Naruto Uzumaki -> Naruto_Uzumaki.pth
    """
    normalized_character = normalize_speaker_name(character_name)

    for model_name in get_rvc_models():
        model_stem = os.path.splitext(model_name)[0]

        if normalize_speaker_name(model_stem) == normalized_character:
            return model_name

    return None


def generate_neural_base(
    dialogue,
    language,
    output_name,
    character_name=None
):
    """
    Generate native Hindi or Urdu pronunciation using Microsoft Neural TTS.

    Uses a female Neural voice for female characters and a male voice
    otherwise. An RVC model is still required to reproduce the exact
    character voice.
    """

    female_characters = {
        "aria",
        "real",
        "swara",
        "uzma"
    }

    normalized_character = normalize_speaker_name(
        character_name or ""
    )

    use_female_voice = (
        normalized_character in female_characters
    )

    if language == "ur":
        voice_id = (
            "ur-PK-UzmaNeural"
            if use_female_voice
            else "ur-PK-AsadNeural"
        )
    else:
        voice_id = (
            "hi-IN-SwaraNeural"
            if use_female_voice
            else "hi-IN-MadhurNeural"
        )

    final_text = dialogue

    if language == "hi":
        has_devanagari = any(
            "\u0900" <= character <= "\u097F"
            for character in dialogue
        )

        if not has_devanagari:
            try:
                from transliterate import roman_to_devanagari
                final_text = roman_to_devanagari(dialogue)
            except Exception:
                final_text = dialogue

    output_path = os.path.join(
        TEMP_DIR,
        output_name
    )

    success, error = run_edge_tts(
        final_text,
        voice_id,
        output_path
    )

    if not success:
        return None, error

    return output_path, ""

def generate_podcast(script_text, pause_ms, progress=gr.Progress()):
    if not script_text or not script_text.strip():
        return None, "TEST RELOAD..."

    parsed, parser_warnings = parse_podcast_script(script_text)

    if not parsed:
        details = "\n".join(
            f"- {warning}"
            for warning in parser_warnings
        )

        return None, (
            "No valid podcast lines were found.\n\n"
            f"{details}\n\n"
            "Use this format:\n"
            "NARUTO: Hey Luffy!\n"
            "LUFFY : Hey Naruto!"
        )

    characters = list(
        dict.fromkeys(
            name for name, _ in parsed
        )
    )

    saved = get_saved_voices()

    saved_lookup = {
        normalize_speaker_name(voice_name): voice_name
        for voice_name in saved
    }

    voice_map = {}
    missing = []

    for character in characters:
        normalized_name = normalize_speaker_name(character)
        matched_voice = saved_lookup.get(normalized_name)

        if matched_voice:
            voice_map[character] = matched_voice
        else:
            missing.append(character)

    if missing:
        parsed = [
            (character, dialogue)
            for character, dialogue in parsed
            if character not in missing
        ]

        if not parsed:
            return None, (
                "None of the speakers in the script have matching saved voices.\n\n"
                f"Unknown speakers: {', '.join(missing)}\n"
                f"Available saved voices: {', '.join(saved) or 'None'}"
            )

    valid_characters = list(
        dict.fromkeys(
            character for character, _ in parsed
        )
    )

    log_lines = []

    if parser_warnings:
        log_lines.append("Some malformed lines were skipped:")

        for warning in parser_warnings:
            log_lines.append(f"  - {warning}")

        log_lines.append("")

    if missing:
        log_lines.append("Unknown speakers were skipped:")

        for speaker in missing:
            log_lines.append(f"  - {speaker}")

        log_lines.append("")

    log_lines.append(
        f"Processing {len(parsed)} valid lines "
        f"from {len(valid_characters)} matching characters"
    )

    for character in valid_characters:
        log_lines.append(
            f"  {character} -> voice '{voice_map[character]}'"
        )

    audio_segments = []

    pause_duration = max(
        0,
        int(pause_ms)
    )

    pause = AudioSegment.silent(
        duration=pause_duration
    )

    fade_ms = 40

    for index, (character, dialogue) in enumerate(parsed):
        progress(
            (index + 1) / len(parsed),
            desc=(
                f"Generating line {index + 1}/{len(parsed)}: "
                f"{character}..."
            )
        )

        dialogue_preview = dialogue[:50]

        if len(dialogue) > 50:
            dialogue_preview += "..."

        log_lines.append(
            f"\n[{index + 1}/{len(parsed)}] "
            f'{character}: "{dialogue_preview}"'
        )

        voice_name = voice_map[character]
        voice_audio, voice_text = load_voice(voice_name)

        if not voice_audio:
            log_lines.append(
                f"Audio file missing for '{voice_name}'. "
                "This line was skipped."
            )
            continue

        output_name = f"podcast_line_{index}.wav"
        language = detect_podcast_language(dialogue)

        try:
            if language in {"hi", "ur"}:
                language_name = "Urdu" if language == "ur" else "Hindi"

                log_lines.append(
                    f"Detected {language_name}. "
                    "Generating Microsoft Neural base audio first."
                )

                neural_path, neural_error = generate_neural_base(
                dialogue, language,
                        f"podcast_neural_{index}.mp3",
                        character_name=character
)

                if not neural_path:
                    log_lines.append(
                        f"Neural pronunciation generation failed: "
                        f"{neural_error}"
                    )
                    continue

                rvc_model = find_matching_rvc_model(character)

                if rvc_model:
                    log_lines.append(
                        f"Converting Neural audio using RVC model "
                        f"'{rvc_model}'."
                    )

                    path, generation_log = run_rvc_conversion(
                        neural_path,
                        rvc_model,
                        0
                    )
                else:
                    path = neural_path
                    generation_log = (
                        f"No matching RVC model was found for "
                        f"'{character}'. Native Neural audio was used."
                    )
                    log_lines.append(generation_log)

            else:
                path, generation_log = run_f5tts(
                    dialogue,
                    voice_audio,
                    voice_text,
                    output_name=output_name
                )

        except Exception as error:
            log_lines.append(
                f"Generation error: {error}"
            )
            continue

        if path and os.path.exists(path):
            try:
                segment = AudioSegment.from_file(path)

                segment = (
                    segment
                    .set_channels(1)
                    .set_frame_rate(24000)
                )

                safe_fade = min(
                    fade_ms,
                    max(0, len(segment) // 4)
                )

                if safe_fade:
                    segment = (
                        segment
                        .fade_in(safe_fade)
                        .fade_out(safe_fade)
                    )

                audio_segments.append(segment)

                log_lines.append(
                    f"{len(segment) / 1000:.1f}s generated"
                )

            except Exception as error:
                log_lines.append(
                    f"Could not process generated audio: {error}"
                )

        else:
            log_lines.append(
                f"Failed: {generation_log}"
            )

    if not audio_segments:
        return None, (
            "\n".join(log_lines)
            + "\n\nNo audio was generated."
        )

    log_lines.append(
        f"\nCombining {len(audio_segments)}  audio segments..."
    )

    final = audio_segments[0]

    for segment in audio_segments[1:]:
        final = final + pause + segment

    output_path = os.path.join(
        BASE_DIR,
        "podcast_output.wav"
    )

    final.export(
        output_path,
        format="wav"
    )

    log_lines.append(
        f"Final podcast: {len(final) / 1000:.1f}s total "
        f"with {pause_duration}ms pauses and short fades"
    )

    progress(1.0)

    return output_path, "\n".join(log_lines)


# ─── Audio Editor Functions ───
def edit_audio_trim(audio_path, start_s, end_s):
    if not audio_path: return None, "Upload audio first."
    try:
        audio = AudioSegment.from_file(audio_path)
        start_ms, end_ms = int(start_s * 1000), int(end_s * 1000)
        trimmed = audio[start_ms:end_ms]
        out = os.path.join(BASE_DIR, "edited_audio.wav")
        trimmed.export(out, format="wav")
        return out, f" Trimmed: Kept {start_s}s to {end_s}s"
    except Exception as e:
        return None, f" Error: {e}"

def edit_audio_cut(audio_path, start_s, end_s):
    if not audio_path: return None, "Upload audio first."
    try:
        audio = AudioSegment.from_file(audio_path)
        start_ms, end_ms = int(start_s * 1000), int(end_s * 1000)
        cut = audio[:start_ms] + audio[end_ms:]
        out = os.path.join(BASE_DIR, "edited_audio.wav")
        cut.export(out, format="wav")
        return out, f" Cut: Removed {start_s}s to {end_s}s"
    except Exception as e:
        return None, f" Error: {e}"

def edit_audio_replace(
    audio_path,
    start_s,
    end_s,
    text,
    voice_name,
    progress=gr.Progress()
):
    if not audio_path:
        return None, "Upload audio first."

    if not text:
        return None, "Enter text to generate."

    if not voice_name:
        return None, "Select a voice."

    try:
        audio = AudioSegment.from_file(audio_path)

        start_ms = int(float(start_s) * 1000)
        end_ms = int(float(end_s) * 1000)

        if start_ms < 0 or end_ms < 0:
            return None, "Start and end times cannot be negative."

        if start_ms >= end_ms:
            return None, "End time must be greater than start time."

        if start_ms > len(audio):
            return None, "Start time is beyond the audio duration."

        end_ms = min(end_ms, len(audio))

        voice_audio, voice_text = load_voice(voice_name)

        if not voice_audio:
            return None, f"Audio file missing for '{voice_name}'."

        progress(0.3, desc="Generating replacement segment...")

        new_path, generation_log = run_f5tts(
            text,
            voice_audio,
            voice_text,
            output_name="replacement.wav"
        )

        if not new_path or not os.path.exists(new_path):
            return None, f"Generation failed: {generation_log}"

        new_segment = AudioSegment.from_file(new_path)

        final_audio = (
            audio[:start_ms]
            + new_segment
            + audio[end_ms:]
        )

        output_path = os.path.join(
            BASE_DIR,
            "edited_audio.wav"
        )

        final_audio.export(
            output_path,
            format="wav"
        )

        progress(1.0, desc="Replacement completed.")

        return (
            output_path,
            f"Replaced audio from {start_s}s to {end_s}s "
            "with the newly generated segment."
        )

    except Exception as error:
        return None, f"Error: {error}"
    
# ─── ML FEATURE: Audio Dataset Preprocessing ───
TRAINING_DIR = os.path.join(BASE_DIR, "training_data")
os.makedirs(TRAINING_DIR, exist_ok=True)

def preprocess_training_audio(
    audio_path,
    chunk_seconds=10,
    normalize_db=-20.0,
    progress=gr.Progress()
):
    """
    Clean and prepare uploaded audio for voice-model training.

    Pipeline:
    1. Load the audio.
    2. Convert it to 16 kHz mono.
    3. Reduce background noise.
    4. Remove long silent sections.
    5. Normalize the volume.
    6. Split the cleaned audio into chunks.
    7. Save the cleaned full recording and all chunks.
    """

    if not audio_path:
        return None, "Upload an audio file first."

    try:
        import math
        import numpy as np
        import noisereduce as nr
        from datetime import datetime
        from pydub.silence import split_on_silence

        progress(0.05, desc="Loading raw audio...")

        audio = AudioSegment.from_file(audio_path)

        if len(audio) == 0:
            return None, "The uploaded audio file is empty."

        original_duration = len(audio) / 1000.0

        progress(0.15, desc="Converting audio to 16 kHz mono...")

        audio = (
            audio
            .set_channels(1)
            .set_frame_rate(16000)
            .set_sample_width(2)
        )

        samples = np.array(
            audio.get_array_of_samples(),
            dtype=np.float32
        )

        samples = samples / 32768.0

        progress(0.30, desc="Removing background noise...")

        reduced_samples = nr.reduce_noise(
            y=samples,
            sr=16000,
            stationary=False,
            prop_decrease=0.80
        )

        reduced_samples = np.clip(
            reduced_samples,
            -1.0,
            1.0
        )

        reduced_int16 = (
            reduced_samples * 32767
        ).astype(np.int16)

        cleaned_audio = AudioSegment(
            data=reduced_int16.tobytes(),
            sample_width=2,
            frame_rate=16000,
            channels=1
        )

        if math.isfinite(cleaned_audio.dBFS):
            silence_threshold = cleaned_audio.dBFS - 16
        else:
            silence_threshold = -40

        progress(0.45, desc="Removing silent sections...")

        speech_sections = split_on_silence(
            cleaned_audio,
            min_silence_len=500,
            silence_thresh=silence_threshold,
            keep_silence=150,
            seek_step=10
        )

        if not speech_sections:
            return None, (
                "No usable speech was detected after silence removal. "
                "Try using a clearer or louder recording."
            )

        separator = AudioSegment.silent(
            duration=120,
            frame_rate=16000
        )

        silence_removed_audio = speech_sections[0]

        for section in speech_sections[1:]:
            silence_removed_audio += separator + section

        silence_removed_duration = (
            len(silence_removed_audio) / 1000.0
        )

        removed_silence_seconds = max(
            0,
            original_duration - silence_removed_duration
        )

        progress(0.60, desc="Normalizing volume levels...")

        if (
            math.isfinite(silence_removed_audio.dBFS)
            and silence_removed_audio.dBFS != float("-inf")
        ):
            gain_change = (
                float(normalize_db)
                - silence_removed_audio.dBFS
            )

            silence_removed_audio = (
                silence_removed_audio.apply_gain(gain_change)
            )

        progress(0.72, desc="Creating training chunks...")

        chunk_ms = max(
            1000,
            int(float(chunk_seconds) * 1000)
        )

        chunks = [
            silence_removed_audio[start:start + chunk_ms]
            for start in range(
                0,
                len(silence_removed_audio),
                chunk_ms
            )
        ]

        chunks = [
            chunk
            for chunk in chunks
            if len(chunk) >= 2000
        ]

        if not chunks:
            return None, (
                "The cleaned audio is too short to create "
                "a training chunk of at least two seconds."
            )

        session_name = datetime.now().strftime(
            "session_%Y%m%d_%H%M%S"
        )

        session_dir = os.path.join(
            TRAINING_DIR,
            session_name
        )

        os.makedirs(
            session_dir,
            exist_ok=True
        )

        cleaned_output_path = os.path.join(
            session_dir,
            "cleaned_full_audio.wav"
        )

        silence_removed_audio.export(
            cleaned_output_path,
            format="wav"
        )

        progress(0.85, desc="Exporting training chunks...")

        for index, chunk in enumerate(chunks):
            chunk_path = os.path.join(
                session_dir,
                f"chunk_{index:03d}.wav"
            )

            chunk.export(
                chunk_path,
                format="wav"
            )

        final_duration = len(silence_removed_audio) / 1000.0

        log = (
            "Audio dataset preprocessing completed.\n\n"
            f"Original duration: {original_duration:.1f} seconds\n"
            f"Cleaned duration: {final_duration:.1f} seconds\n"
            f"Silence removed: {removed_silence_seconds:.1f} seconds\n"
            "Noise reduction: Applied\n"
            f"Silence threshold: {silence_threshold:.1f} dBFS\n"
            f"Normalized volume: {normalize_db} dBFS\n"
            "Audio format: 16 kHz mono\n"
            f"Training chunks created: {len(chunks)}\n"
            f"Chunk size: {chunk_seconds} seconds\n"
            f"Output directory: {session_dir}\n"
            f"Cleaned recording: {cleaned_output_path}"
        )

        progress(1.0, desc="Preprocessing completed.")

        return session_dir, log

    except Exception as error:
        return None, f"Preprocessing error: {error}"

def analyze_voice_similarity(audio_a, audio_b, progress=gr.Progress()):
    """Real ML: Compare two audio files using Whisper encoder embeddings + cosine similarity."""
    if not audio_a or not audio_b:
        return "Upload both audio files to compare."
    try:
        progress(0.2, desc="Loading Whisper encoder...")
        import torch
        import numpy as np
        from transformers import WhisperProcessor, WhisperModel

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        model = WhisperModel.from_pretrained("openai/whisper-base").to(device).to(dtype)

        def get_embedding(path):
            audio = AudioSegment.from_file(path).set_channels(1).set_frame_rate(16000)
            if len(audio) > 15000:
                audio = audio[:15000]
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device).to(dtype)
            with torch.no_grad():
                encoder_out = model.encoder(input_features)
                embedding = encoder_out.last_hidden_state.mean(dim=1).squeeze()
            return embedding

        progress(0.5, desc="Extracting voice embeddings...")
        emb_a = get_embedding(audio_a)
        progress(0.7, desc="Comparing voice signatures...")
        emb_b = get_embedding(audio_b)

        # Cosine Similarity
        cos_sim = torch.nn.functional.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
        similarity_pct = max(0, min(100, cos_sim * 100))

        # Cleanup GPU
        del model, processor, emb_a, emb_b
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        grade = " Excellent" if similarity_pct > 85 else " Good" if similarity_pct > 70 else " Poor"
        progress(1.0)
        return (
            f" Voice Similarity Analysis\n"
            f"{'='*40}\n"
            f"Cosine Similarity Score: {similarity_pct:.1f}%\n"
            f"Quality Grade: {grade}\n\n"
            f"{'='*40}\n"
            f"If the score is below 70%, consider:\n"
            f"  • Using a longer/cleaner reference audio\n"
            f"  • Fine-tuning the model with more training data\n"
            f"  • Adjusting the pitch shift parameter"
        )
    except Exception as e:
        return f" Analysis Error: {e}"

# ═══════════════════════════════════════
#  GRADIO UI
# ═══════════════════════════════════════
import base64
logo_path = os.path.join(BASE_DIR, "LOGO.jpg")
logo_b64 = ""
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode("utf-8")

custom_css = """
footer {display: none !important;}
.zenvyro-header {text-align: center; padding: 20px 0; border-bottom: 2px solid #eee; margin-bottom: 20px;}
.zenvyro-logo {font-size: 2.5em; font-weight: 800; color: #2563eb; letter-spacing: 2px;}
.zenvyro-subtitle {font-size: 1.1em; color: #64748b; margin-top: 5px;}
"""

header_html = f"""
    <div class="zenvyro-header">
        <img src="data:image/jpeg;base64,{logo_b64}" alt="Zenvyrolabs Logo" style="height: 80px; margin-bottom: 10px; display: inline-block;">
        <div class="zenvyro-logo">ZENVYROLABS</div>
        <div class="zenvyro-subtitle">Internal Advanced Voice Studio • Clone anime voices • Dramatic storytelling • Multi-voice podcasts</div>
    </div>
"""

with gr.Blocks(title=" Zenvyrolabs Voice Studio") as interface:
    gr.HTML(header_html)

    with gr.Tabs():
        # ─── TAB 1: Voice Cloner ───
        with gr.TabItem(" Voice Cloner"):
            gr.Markdown("Upload any voice clip → the AI clones it and speaks your text in that voice.")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("###  Voice Library")
                    saved_dd = gr.Dropdown(choices=get_saved_voices(), label="Saved Voices", interactive=True)
                    with gr.Row():
                        load_btn = gr.Button(" Load", size="sm")
                        del_btn = gr.Button(" Delete", size="sm", variant="stop")
                    gr.Markdown("---")
                    gr.Markdown("###  Save Voice")
                    voice_name = gr.Textbox(label="Name", placeholder="e.g. Gojo_Dramatic")
                    save_btn = gr.Button(" Save to Library", variant="primary")
                    lib_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=2):
                    gen_text1 = gr.Textbox(label="Script to Speak", lines=6, placeholder="Type your story here...")
                    ref_audio1 = gr.Audio(type="filepath", label="Reference Voice (auto-trims to 8s)")
                    with gr.Row():
                        ref_text1 = gr.Textbox(label="Reference Text", lines=2, scale=4,
                            placeholder="Type exact words from the reference audio...")
                        extract_btn1 = gr.Button(" Auto-Extract", variant="secondary", scale=1)
                    clone_btn1 = gr.Button(" Generate Clone", variant="primary", size="lg")

            with gr.Row():
                out_audio1 = gr.Audio(label="Generated Audio")
                out_log1 = gr.Textbox(label="Log")

            load_btn.click(fn=load_voice, inputs=[saved_dd], outputs=[ref_audio1, ref_text1])
            extract_btn1.click(fn=extract_text_fn, inputs=[ref_audio1], outputs=[ref_text1])
            clone_btn1.click(fn=clone_voice_tab1, inputs=[gen_text1, ref_text1, ref_audio1], outputs=[out_audio1, out_log1])

        # ─── TAB 2: Dramatic Story Mode ───
        with gr.TabItem(" Dramatic Story Mode", visible=False):
            gr.Markdown("""### How it works:
1. **Step 1:** Microsoft Neural AI creates a dramatic, emotional narration (perfect pronunciation & emotions).
2. **Step 2:** F5-TTS re-generates the same script using your saved anime voice (Gojo, Naruto, etc).
3. You get **two outputs** — pick whichever sounds better!

**Pro tip:** The emotion base alone sounds incredible for YouTube. The anime clone adds character flavor.""")

            with gr.Row():
                with gr.Column():
                    saved_dd2 = gr.Dropdown(choices=get_saved_voices(), label="Select Saved Anime Voice", interactive=True)
                    narrator_style = gr.Dropdown(
                        choices=list(NARRATOR_VOICES.keys()),
                        label="Emotion Narrator Style", value="Guy (Passionate Male)"
                    )
                    story_text = gr.Textbox(label="Your Story Script", lines=10,
                        placeholder="My daughter went missing five years ago...")
                    dramatic_btn = gr.Button(" Generate Dramatic Voiceover", variant="primary", size="lg")

                with gr.Column():
                    gr.Markdown("### Step 1: Emotional Narration (Microsoft Neural)")
                    emotion_audio = gr.Audio(label="Emotion Base")
                    gr.Markdown("### Step 2: Anime Voice Clone (F5-TTS)")
                    clone_audio = gr.Audio(label="Anime Voice Version")
                    dramatic_log = gr.Textbox(label="Generation Log")

            dramatic_btn.click(fn=dramatic_clone,
                inputs=[story_text, saved_dd2, narrator_style],
                outputs=[emotion_audio, clone_audio, dramatic_log])

        # ─── TAB 3: Multi-Voice Podcast ───
        with gr.TabItem(" Multi-Voice Podcast"):
            gr.Markdown("""### Create Podcasts with Multiple Anime Voices
Write a script with character names that **match your saved voices**. Each line is generated with the correct voice and stitched into one seamless audio.

**Script Format:**
```
NARUTO: Hey Luffy, what's up man!
LUFFY: Yo Naruto! Just finished eating, I'm pumped!
NARUTO: Wanna go train together?
LUFFY: Let's gooo!
```
 Character names must **exactly match** your saved voice names (case-insensitive).""")

            with gr.Row():
                with gr.Column():
                    podcast_voices_dd = gr.Dropdown(choices=get_saved_voices(), multiselect=True, label="Your Saved Voices", info="Select the characters you want to use in your podcast script", interactive=True)
                    podcast_script = gr.Textbox(label="Podcast Script", lines=14,
                        placeholder="NARUTO: Hey Luffy, what's going on?\nLUFFY: Hey Naruto! Just had the best meat ever!\nNARUTO: That sounds awesome, want to spar?\nLUFFY: You're on!")
                    pause_slider = gr.Slider(100, 2000, value=500, step=50,
                        label="Pause Between Lines (ms)", info="How long to pause between each character's line")
                    podcast_btn = gr.Button(" Generate Full Podcast", variant="primary", size="lg")

                with gr.Column():
                    podcast_audio = gr.Audio(label="Final Podcast Audio")
                    podcast_log = gr.Textbox(label="Generation Log", lines=15)

            podcast_btn.click(fn=generate_podcast,
                inputs=[podcast_script, pause_slider],
                outputs=[podcast_audio, podcast_log])

        # ─── TAB 4: Hindi / Urdu ───
        with gr.TabItem(" Hindi / Urdu"):
            gr.Markdown("""### Perfect Hindi & Urdu Pronunciation
**Fix:** Auto-converts Roman Hindi/Urdu → Devanagari script before generating, so pronunciation is accurate.
- Type **Roman** (kya haal hai) → auto-converts to **Devanagari** (क्या हाल है)
- Or type directly in **Devanagari** for best quality""")

            with gr.Row():
                with gr.Column():
                    hindi_text = gr.Textbox(label="Hindi / Urdu Text", lines=6,
                        placeholder="Hello bhai, kya haal hai? Aaj hum ek bahut hi dilchasp kahani sunenge...")
                    transliterate_toggle = gr.Checkbox(label=" Auto-convert Roman → Devanagari (Recommended!)", value=True)
                    hindi_voice = gr.Dropdown(
                        choices=["hi-IN-MadhurNeural", "hi-IN-SwaraNeural",
                                 "ur-PK-AsadNeural", "ur-PK-UzmaNeural",
                                 "ur-IN-SalmanNeural", "ur-IN-GulNeural"],
                        label="Voice", value="hi-IN-MadhurNeural",
                        info="Madhur=Hindi Male, Swara=Hindi Female, Asad=Urdu Male, Uzma=Urdu Female"
                    )
                    with gr.Row():
                        hindi_speed = gr.Slider(-30, 30, value=0, step=5, label="Speed (%)")
                        hindi_pitch = gr.Slider(-20, 20, value=0, step=2, label="Pitch (Hz)")
                    hindi_btn = gr.Button(" Generate Hindi/Urdu Voice", variant="primary", size="lg")

                with gr.Column():
                    hindi_audio = gr.Audio(label="Generated Audio")
                    hindi_log = gr.Textbox(label="Status")

            hindi_btn.click(fn=generate_hindi,
                inputs=[hindi_text, hindi_voice, transliterate_toggle, hindi_speed, hindi_pitch],
                outputs=[hindi_audio, hindi_log])

        # ─── TAB 5: Audio Editor ───
        with gr.TabItem(" Audio Editor"):
            gr.Markdown("Upload an audio file (or download a generated one and upload here) to trim, cut, or completely replace a bad segment with a newly generated voice!")
            
            with gr.Row():
                with gr.Column(scale=1):
                    edit_audio_in = gr.Audio(type="filepath", label="Source Audio", interactive=True)
                    start_s = gr.Number(label="Start Time (seconds)", value=0.0)
                    end_s = gr.Number(label="End Time (seconds)", value=5.0)
                    
                    with gr.Row():
                        trim_btn = gr.Button(" Trim (Keep Only Selection)", variant="secondary")
                        cut_btn = gr.Button(" Cut (Remove Selection)", variant="secondary")
                        
                    gr.Markdown("### Replace Segment")
                    replace_text = gr.Textbox(label="New Text for Segment", lines=2)
                    replace_voice = gr.Dropdown(choices=get_saved_voices(), label="Select Voice for New Segment", interactive=True)
                    replace_btn = gr.Button(" Replace Segment", variant="primary")
                
                with gr.Column(scale=1):
                    edit_audio_out = gr.Audio(label="Edited Audio")
                    edit_log = gr.Textbox(label="Status Log")
                    
            trim_btn.click(fn=edit_audio_trim, inputs=[edit_audio_in, start_s, end_s], outputs=[edit_audio_out, edit_log])
            cut_btn.click(fn=edit_audio_cut, inputs=[edit_audio_in, start_s, end_s], outputs=[edit_audio_out, edit_log])
            replace_btn.click(fn=edit_audio_replace, inputs=[edit_audio_in, start_s, end_s, replace_text, replace_voice], outputs=[edit_audio_out, edit_log])

        # ─── TAB 6: Voice-to-Voice (RVC) ───
        with gr.TabItem(" Voice-to-Voice (RVC)", visible=False):
            gr.Markdown("""### True Emotional Voice Cloning (Speech-to-Speech)
Upload an audio of **you acting out a line**, select a downloaded `.pth` anime character model, and the AI will convert your voice while preserving exactly the timing, emotion, and breath.
*(Models must be placed in `e:\project\searching\anime_voice_cloner\\rvc_models`)*""")
            with gr.Row():
                with gr.Column():
                    rvc_in = gr.Audio(type="filepath", label="Input Audio (Your acting/reference)")
                    rvc_model = gr.Dropdown(choices=get_rvc_models(), label="RVC Model (.pth)", interactive=True)
                    rvc_refresh = gr.Button(" Refresh Models List", size="sm")
                    rvc_pitch = gr.Slider(-24, 24, value=0, step=1, label="Pitch Shift (Semitones)", info="Use +12 for Male->Female, -12 for Female->Male. Leave 0 if same gender.")
                    rvc_btn = gr.Button(" Convert Voice", variant="primary", size="lg")
                with gr.Column():
                    rvc_out = gr.Audio(label="Converted Audio")
                    rvc_log = gr.Textbox(label="Status Log", lines=10)
                    
            rvc_btn.click(fn=run_rvc_conversion, inputs=[rvc_in, rvc_model, rvc_pitch], outputs=[rvc_out, rvc_log])
            rvc_refresh.click(fn=lambda: gr.update(choices=get_rvc_models()), outputs=[rvc_model])

        # ─── TAB 7: Perfect Pronunciation Clone ───
        with gr.TabItem(" Perfect Pronunciation Clone", visible=False):
            gr.Markdown("""### Get Anime Voices with PERFECT Pronunciation
F5-TTS sometimes struggles with pronunciation. This tab fixes that! 
It uses **Edge-TTS (Eric, Guy, etc.)** to generate perfect, native pronunciation, and then uses **RVC** to seamlessly morph that audio into your Anime character's voice.
*(Requires an RVC `.pth` model in `rvc_models/`)*""")
            with gr.Row():
                with gr.Column():
                    perf_text = gr.Textbox(label="Script", lines=6, placeholder="Type perfectly pronounced English here...")
                    perf_neural = gr.Dropdown(choices=list(NARRATOR_VOICES.keys()), label="Base Neural Voice (for acting/pronunciation)", value="Eric (Rational Male)")
                    perf_rvc = gr.Dropdown(choices=get_rvc_models(), label="Target Anime Voice (RVC Model)", interactive=True)
                    perf_pitch = gr.Slider(-24, 24, value=0, step=1, label="Pitch Shift", info="Match Neural gender to Anime gender. e.g. Male to Female: +12")
                    perf_btn = gr.Button(" Generate Perfect Clone", variant="primary", size="lg")
                with gr.Column():
                    perf_audio = gr.Audio(label="Final Perfect Audio")
                    perf_log = gr.Textbox(label="Status Log")

            def run_perfect_clone(text, neural_voice, rvc_model, pitch, progress=gr.Progress()):
                if not text: return None, "Please enter text."
                if not rvc_model: return None, "Please select an RVC model."
                
                progress(0.2, desc="Generating perfect pronunciation...")
                voice_id = NARRATOR_VOICES.get(neural_voice, "en-US-EricNeural")
                temp_audio = os.path.join(TEMP_DIR, "perf_base.mp3")
                ok, err = run_edge_tts(text, voice_id, temp_audio)
                if not ok:
                    return None, f" Edge-TTS failed: {err}"
                
                progress(0.6, desc="Morphing into Anime Voice (RVC)...")
                final_path, log = run_rvc_conversion(temp_audio, rvc_model, pitch)
                progress(1.0)
                return final_path, log
            
            perf_btn.click(fn=run_perfect_clone, inputs=[perf_text, perf_neural, perf_rvc, perf_pitch], outputs=[perf_audio, perf_log])

        # ─── TAB 8: Voice Training Studio (Real ML) ───
        with gr.TabItem(" Voice Training Studio"):
            gr.Markdown("""###  AI Model Training Pipeline
This is the **core Machine Learning** feature of the application. Instead of relying on zero-shot cloning (which can sound robotic), you can **train a custom voice model** by feeding it high-quality audio data.

**How it works (Real ML Pipeline):**
1. **Upload** a long audio recording of your target voice (5-10 minutes recommended).
2. **Preprocess** — Our pipeline will automatically normalize volume levels, resample to 16kHz mono (the standard for speech ML models), remove silence, and chunk the audio into clean 10-second training segments.
3. **Analyze** — Use the Voice Quality Analyzer to compare your cloned output vs the original and get a real ML similarity score using Whisper neural embeddings.

*This is the exact same data preprocessing pipeline used in production ML systems at companies like ElevenLabs and OpenAI.*""")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Step 1: Upload Raw Training Audio")
                    train_audio = gr.File(
    type="filepath",
    file_types=[
        ".wav",
        ".mp3",
        ".mp4",
        ".m4a",
        ".aac",
        ".ogg",
        ".flac",
        ".webm",
        ".wma",
        ".aiff"
    ],
    label="Raw Training Audio (5-10 min recommended)"
)
                    chunk_size = gr.Slider(5, 30, value=10, step=1, label="Chunk Size (seconds)", info="Each chunk becomes one training sample")
                    norm_db = gr.Slider(-30, -10, value=-20, step=1, label="Target Volume (dBFS)", info="Normalizes all chunks to this volume level for consistent training")
                    preprocess_btn = gr.Button(" Preprocess Dataset", variant="primary", size="lg")

                with gr.Column():
                    gr.Markdown("### Preprocessing Results")
                    train_output_dir = gr.Textbox(label="Output Directory", interactive=False)
                    train_log = gr.Textbox(label="Pipeline Log", lines=10)

            preprocess_btn.click(fn=preprocess_training_audio,
                inputs=[train_audio, chunk_size, norm_db],
                outputs=[train_output_dir, train_log])

            gr.Markdown("---")
            gr.Markdown("""### Step 2: Voice Quality Analyzer (Cosine Similarity)
Upload the **original voice** and your **cloned output** to measure how accurate the clone is using real ML metrics.
The system uses **OpenAI Whisper's neural encoder** to extract voice embeddings and computes **cosine similarity** — the same technique used in speaker verification systems.""")

            with gr.Row():
                with gr.Column():
                    sim_audio_a = gr.Audio(type="filepath", label="Audio A: Original Voice")
                    sim_audio_b = gr.Audio(type="filepath", label="Audio B: Cloned Voice")
                    sim_btn = gr.Button(" Analyze Similarity", variant="primary", size="lg")
                with gr.Column():
                    sim_result = gr.Textbox(label="ML Analysis Results", lines=12)

            sim_btn.click(fn=analyze_voice_similarity,
                inputs=[sim_audio_a, sim_audio_b],
                outputs=[sim_result])

    # Global event bindings
    save_btn.click(fn=save_voice, inputs=[voice_name, ref_audio1, ref_text1], outputs=[lib_status, saved_dd, saved_dd2, podcast_voices_dd, replace_voice])
    del_btn.click(fn=delete_voice, inputs=[saved_dd], outputs=[lib_status, saved_dd, saved_dd2, podcast_voices_dd, replace_voice])

if __name__ == "__main__":
    print("Launching Advanced Voice Studio...")
    print(f"Saved Voices: {get_saved_voices()}")
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=False,
        css=custom_css
    )
