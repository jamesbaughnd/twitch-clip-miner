"""
Whisper.cpp transcription backend (with Vulkan GPU acceleration).
"""
import subprocess
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# A separate logger for whisper.cpp progress
#whisper_logger = logging.getLogger("whisper_cpp.progress")

def transcribe_with_whisper_cpp(
        audio_path: Path,
        model_path: Path = Path("models/ggml-small.en.bin"),
        whisper_cli: Path = Path("C:/Programming/Projects/Python/twitch-clipper/tools/whisper-cli.exe"),
        output_dir: Path = Path("data/transcripts"),
        force: bool = False,
        language: str = "en",
) -> list[dict]:
    """
    Run whisper.cpp CLI and return word-level data.
    Uses a GPU if available (Vulkan), else falls back to CPU
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"{audio_path.stem}_words_cpp.json"

    if cache_path.exists() and not force:
        logger.info(f"Loading cached whisper.cpp transcript from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_words = json.load(f)
        if cached_words:
            return cached_words
        logger.warning("Cached transcript is empty - re-transcribing...")
    
    # Generate a temporary JSON output file. Whisper-cli can output JSON with: --output-json
    # The file will be named based on the input: <audio_path>.json
    # We'll redirect stdout to a file or use --ofile
    output_json = audio_path.with_suffix(".json")

    cmd = [
        str(whisper_cli),
        "-m", str(model_path),
        "-f", str(audio_path),
        "-oj",                              # output JSON
        "-of", 
        str(audio_path.with_suffix("")),    # Output filename prefix (no extension)
        "--print-progress",                 # show progress, redirected to log
        "--word-thold", "0.01",
        "--split-on-word",
        "--max-len", "0",
        "--best-of", "5",
        "--language", language,
    ]

    logger.info(f"Transcription in progress. Running {' '.join(cmd)}") #{' '.join(cmd)}

    proc = subprocess.run(
        cmd,
        #capture_output=True,
        shell=True,
        #text=True,
        check=False     # handle errors manually
    )

    # Log progress lines line by line
    if proc.stderr:
        for line in proc.stderr.splitlines():
            # Log each line to the progress logger at DEBUG level
            logger.debug(line.strip())

    # Log final result
    if proc.stdout:
        logger.debug(proc.stdout)

    # check for errors
    if proc.returncode != 0:
        logger.error(f"whisper.cpp failed with code {proc.returncode}")
        logger.error(proc.stderr[-1000:])   # last 1000 chars for context
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)

    if not output_json.exists():
        raise RuntimeError(f"whisper.cpp did not produce {output_json}")
    
    # Parse the JSON output
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = []
    # New formatting necessary: each segment is a single word balls bro
    if "transcription" in data:
        for seg in data["transcription"]:
            # Skip empty for silent segments
            if not seg.get("text", "").strip():
                continue
            # Use offsets (milliseconds) -> convert to seconds
            off = seg.get("offsets", {})
            start_ms = off.get("from", 0)
            end_ms = off.get("to", 0)
            words.append({
                "start": start_ms / 1000.0,
                "end": end_ms / 1000.0,
                "word": seg["text"].strip(),
                "confidence": 0.0,
            })
    elif "words" in data:
        # Old format (if ever used again)
        for w in data["words"]:
            words.append({
                "start": w["start"],
                "end": w["end"],
                "word": w["word"].strip(),
                "confidence": w.get("p", 0.0),
            })
    else:
        logger.warning(f"Unexpected JSON format in {output_json}")
    
    # Cache the parsed result
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(words, f, indent=2)
    logger.info(f"whisper.cpp transcription complete - {len(words)} words")
    return words