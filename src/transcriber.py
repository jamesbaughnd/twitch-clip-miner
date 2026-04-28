

"""
Speech-to-text transcription using faster-whisper
    Caching - transcription takes time, we avoid repeating work
    Word Timestamps - you'll need these to pinpoint interesting phrases
    Model Size - "tiny" (fastest, less accurate), "base", and "small" (good value). For use on
                 minimal hardware, "base" works on CPU in reasonable time
    vad_filter=True - automatically skips long silences (great for VODs with long silences/quiet gameplay)

    Remember to install the ROCm-enabled CTranslate2 (for faster-whisper) if you have an AMD GPU!
"""
import json
import logging
import os
import ctranslate2
from pathlib import Path
from faster_whisper import WhisperModel
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

# Quiet warning from HuggingFace Hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def detect_best_model_size() -> str:
    """
    Return 'small' if CUDA/ROCm GPU is available, else 'base'.
    """
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            logger.info("GPU detected (CUDA/ROCm) - using 'small' whisper transcription model.")
            return "small"
    except Exception:
        pass
    logger.info("No GPU found - using 'base' model.")
    return "base"

def detect_device() -> tuple[str, str]:
    """
    Return (device, compute_type) based on available hardware.
    - If a CUDA/ROCm GPU is found -> ('cuda', 'float16')
    - Otherwise -> ('cpu', 'int8')
    """
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            logger.info("GPU detected via CTranslate2 - using CUDA/HIP backend.")
            return "cuda", "float16"
    except Exception:
        pass
    logger.info("No GPU - using CPU.")
    return "cpu", "int8"

def transcribe_audio(
        audio_path: Path,
        model_size: str | None = None,
        device: str | None = None,              # 'auto', 'cpu', 'cuda'
        compute_type: str | None = None,        # 'auto', 'int8', 'float16', etc.
        output_dir: Path = Path("data/transcripts"),
        force: bool = False,
        language: str = "en"
) -> list[dict]:
    """
    Transcribe an audio file and return word-level data.
    Each word dict : {"start": float, "end": float, "word": str, "confidence": float}
    Results are cached as JSON next to the audio file.
    """
    # --- Auto-detection for model size ---
    if model_size is None or model_size.lower() == "auto":
        model_size = detect_best_model_size()

    # --- Auto-detection for device and compute type ---
    if device is None or device.lower() == "auto":
        device, auto_compute = detect_device()
        if compute_type is None or compute_type.lower() == "auto":
            compute_type = auto_compute
    elif compute_type is None or compute_type.lower() == "auto":
        # User specified device but not compute_type, reasonable default
        compute_type = "float16" if device == "cuda" else "int8"

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"{audio_path.stem}_words.json"

    if cache_path.exists() and not force:
        logger.info(f"Loading cached transcript from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    console.log(f"Transcribing {audio_path.name} with device={device}, compute={compute_type}...")
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        word_timestamps=True,
        language=language,          
        vad_filter=False,           # set True to skip silence automatically 
    )

    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append({
                    "start": w.start,
                    "end": w.end,
                    "word": w.word.strip(),
                    "confidence": w.probability,
                })

    # Cache the result
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(words, f, indent=2, ensure_ascii=False)
    logger.info(f"Transcription complete: {len(words)} words, cached to {cache_path}")
    return words

def transcribe_audio_auto(
        audio_path: Path,
        config: dict | None = None,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        output_dir: Path = Path("data/transcripts"),
        force: bool = False,
        language: str = "en",
) -> list[dict]:
    """
    Pick the transcription backend based on config or auto-detection.
    """
    if config is None:
        from src.detector import load_config
        config = load_config()

    trans_cfg = config.get("transcription", {})
    backend = trans_cfg.get("backend", "auto").lower()

    # Check for whisper.cpp availability
    cpp_cfg = trans_cfg.get("whisper_cpp", {})
    cpp_exe = Path(cpp_cfg.get("executable", "tools/whisper-cli.exe"))
    cpp_model = Path(cpp_cfg.get("model", "models/ggml-base.en.bin"))
    cpp_available = cpp_exe.exists() and cpp_model.exists()

    if backend == "auto":
        # Use whisper.cpp if available, otherwise faster-whisper
        if cpp_available:
            logger.info("Auto backend: using whisper.cpp (Vulkan)")
            from src.transcriber_whisper_cpp import transcribe_with_whisper_cpp
            return transcribe_with_whisper_cpp(
                audio_path, model_path=cpp_model, whisper_cli=cpp_exe,
                output_dir=output_dir, force=force, language=language
            )
        else:
            logger.info("Auto backend: falling back to faster-whisper")
            return transcribe_audio(
                audio_path, model_size=model_size, device=device,
                compute_type=compute_type, output_dir=output_dir, force=force, language=language
            )
    elif backend == "whisper-cpp":
        if not cpp_available:
            raise FileNotFoundError(
                "whisper-cpp backend configured but executable or model not found. "
                "Check config.yaml -> transcription.whisper_cpp"
            )
        from src.transcriber_whisper_cpp import transcribe_with_whisper_cpp
        return transcribe_with_whisper_cpp(
            audio_path, model_path=cpp_model, whisper_cli=cpp_exe,
            output_dir=output_dir, force=force, language=language
        )
    else:  #faster-whisper
        return transcribe_audio(
            audio_path, model_size=model_size, device=device,
            compute_type=compute_type, output_dir=output_dir, force=force, language=language
        )
