
# -----------------------------------------------
# Extract audio, compute loudness/envelope
# - 16 Hz is the sweet spot for speech transcription + keeps file sizes small while
#   still preserving all audible frequencies (<8 kHz). Perfect for loudness. scream
#   for me papi
# -----------------------------------------------
import subprocess
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def extract_audio(video_path: Path, output_dir: Path = Path("data/audio")) -> Path:
    """
    Extract mono, 16khz audio from a video file as WAV.
    Returns the path to the extracted audio file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{video_path.stem}.wav"

    if audio_path.exists():
        logger.info(f"Audio already exists: {audio_path}")
        return audio_path
    
    # Using ffmpeg via command line (robust (busty) and well-tested)
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",                  # no video
        "-ac", "1",             # mono
        "-ar", "16000",         # 16 kHz sample rate
        "-sample_fmt", "s16",   # 16-bit PCM
        "-y",                   # overwrite
        str(audio_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    logger.info(f"Extracted audio to {audio_path}")
    return audio_path

def compute_loudness(
        audio_path: Path,
        hop_length: int = 512,
        frame_length: int = 2048,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load audio and return (times, rms, sr).
    - times: array of time stamps (seconds) for each frame
    - rms:   root-mean-square energy per frame (linear)
    - sr:    sample rate (Hz)
    """
    y, sr = librosa.load(str(audio_path), sr=None) # Keep original 16kHz
    # Compute short-time RMS energy
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    logger.info(f"Computed loudness: {len(rms)} frames, duration={times[-1]:.1f}s")
    return times, rms, sr

def plot_loudness(
        times: np.ndarray,
        rms: np.ndarray,
        audio_path: Path,
        output_dir: Path = Path("data/plots")
) -> Path:
    """
    Save a plot of the raw loudness curve to help visualize peaks (robusty).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 3))
    plt.plot(times, rms, linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("RMS Energy")
    plt.title(f"Loudness curve: {audio_path.stem}")
    plt.tight_layout()
    plot_path = output_dir / f"{audio_path.stem}_loudness.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved loudness plot to {plot_path}")
    return plot_path

# Uses ffmpeg to cut a mono 16 kHz WAV slice from the video
# without needing to re-encode the whole file
def extract_audio_segment(
        video_path: Path,
        start_time: float,
        duration: float,
        output_dir: Path = Path("data/audio_chunks")
) -> Path:
    """
    Extract a mono 16kHz WAV segment from the video for chunked processing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    seg_path = output_dir / f"{video_path.stem}_{int(start_time)}s_{int(duration)}s.wav"

    if seg_path.exists():
        logger.info(f"Audio segment already exists: {seg_path}")
        return seg_path
    
    cmd = [
        "ffmpeg",
        "-ss", str(start_time),
        "-i", str(video_path),
        "-t", str(duration),
        "-vn",               # no video
        "-ac", "1",          # mono
        "-ar", "16000",      # 16 kHz
        "-sample_fmt", "s16",
        "-y",
        str(seg_path)       
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    logger.info(f"Extracted audio segment: {seg_path}")
    return seg_path