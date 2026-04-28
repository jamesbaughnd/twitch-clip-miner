

"""
Detects clip-worthy moments by combining loudness, transcript, chat, and visual signals
"""
from src.vision import VISUAL_AVAILABLE, analyze_visual_engagement
import logging
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import yaml
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------
def load_config(config_path: Path = Path("config.yaml")) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
def _normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Convert to zero-mean, unit-variance (z-score) to make thresholds generic."""
    std = np.std(signal)
    if std == 0:
        return np.zeros_like(signal)
    return (signal - np.mean(signal)) / std

def _avg_chat_vel(chat_times, chat_vel, start, end):
    """Mean normalized chat velocity inside a time window."""
    if chat_times is None or chat_vel is None or len(chat_times) == 0:
        return 0.0
    mask = (chat_times >= start) & (chat_times <= end)
    if not mask.any():
        return 0.0
    return float(np.mean(chat_vel[mask]))


# ---------------------------------------------------------------------------
# Peak detection & window creation
# ---------------------------------------------------------------------------
def _find_audio_peaks(
        audio_times: np.ndarray, rms: np.ndarray, det_cfg: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Return (peak_times, prominences) for the loudness envelope."""

    # Normalize and smooth loudness
    rms_norm = _normalize_signal(rms)
    # light smoothing to merge micro-peaks
    smoothed = gaussian_filter1d(rms_norm, sigma=1.0)

    # sample interval for distance conversion
    dt = audio_times[1] - audio_times[0] if len(audio_times) > 1 else 0.1
    # Find peaks
    peaks, properties = find_peaks(
        smoothed,
        prominence=det_cfg["peak_prominence"],
        distance=int(det_cfg["peak_distance"] / dt),
    )
    if len(peaks) == 0:
        return np.array([]), np.array([])
    
    # Convert peak indices to times
    peak_times = audio_times[peaks]
    prominence_values = properties["prominences"]
    return peak_times, prominence_values


def _build_clip_windows(
        peak_times: np.ndarray, padding: float
) -> list[tuple[float, float]]:
    """Convert peak times to (start, end) windows."""
    return [(max(0, t - padding), t + padding) for t in peak_times]


# ---------------------------------------------------------------------------
# Transcript scoring
# ---------------------------------------------------------------------------
def _compute_transcript_score(
        words: list[dict], start_t: float, end_t: float, config: dict
) -> float:
    """
    Score a time window based on presence of hype words and laughter.
    Returns a value that grows with excitement.
    """
    hype_words = [w.lower() for w in config["detector"]["hype_words"]]
    laugh_patterns = [w.lower() for w in config["detector"]["laughter_patterns"]]

    score = 0.0
    word_count = 0
    for w in words:
        if start_t <= w["start"] <= end_t or start_t <= w["end"] <= end_t:
            word_count += 1
            word_lower = w["word"].lower()
            # Check for exact matches or substring presence
            for hw in hype_words:
                if hw in word_lower:
                    score += 1.5    # strong signal
                    break
            for lp in laugh_patterns:
                if lp in word_lower:
                    score += 1.0
                    break
    
    # Add small bonus for speech density (words per second) - active talking often good
    duration = end_t - start_t
    if duration > 0:
        score += 0.2 * (word_count / duration)
    return score

# ---------------------------------------------------------------------------
# Single‑candidate scorer (pulls in all signals)
# ---------------------------------------------------------------------------

def _score_candidate(
        start: float,
        end: float,
        loudness_score: float,
        transcript_words: list[dict],
        det_cfg: dict,
        config: dict,
        chat_times: np.ndarray | None,
        chat_vel: np.ndarray | None,
        video_path: Path | None,
        use_visual: bool,
) -> dict:
    """Returns a dict with combined score, component scores, and the window."""
    trans_score = _compute_transcript_score(transcript_words, start, end, config)
    chat_score = _avg_chat_vel(chat_times, chat_vel, start, end)

    # Visual Score (if enabled)
    visual_score = 0.0
    if use_visual and video_path is not None:
        from src.vision import analyze_visual_engagement
        visual_score = analyze_visual_engagement(video_path, start, end, config=config)
    
    # Combined score
    combined = (
        det_cfg["weight_loudness"] * loudness_score
        + det_cfg["weight_transcript"] * trans_score
        + det_cfg.get("weight_chat", 0.0) * chat_score
        + det_cfg.get("weight_visual", 0.0) * visual_score
    )

    return{
        "start": start,
        "end": end,
        "score": combined,
        "loudness_score": loudness_score,
        "transcript_score": trans_score,
        "chat_score": chat_score,
        "visual_score": visual_score,
        "peak_time": (start + end) / 2,     # compatibility
    }


# ---------------------------------------------------------------------------
# Clip merging (NMS)
# ---------------------------------------------------------------------------
def _merge_clips(clips: list[dict], min_distance: float = 10.0) -> list[dict]:
    """Keep highest-scoring clips, discarding those within min_distance seconds."""
    sorted_clips = sorted(clips, key=lambda x: x["score"], reverse=True)
    kept = []
    used_starts = []
    for clip in sorted_clips:
        if not any(abs(clip["start"] - s) < min_distance for s in used_starts):
            kept.append(clip)
            used_starts.append(clip["start"])
    return kept

# ---------------------------------------------------------------------------
# Public API – clean orchestrator
# ---------------------------------------------------------------------------
def detect_clips(
        audio_times: np.ndarray,
        rms: np.ndarray,
        transcript_words: list[dict],
        config: dict | None = None,
        chat_times: np.ndarray | None = None,
        chat_vel: np.ndarray | None = None,
        video_path: Path | None = None,
) -> list[dict]:
    """
    Fing and rank clip windows by fusing multiple signals.

    Parameters
    ----------
    audio_times, rms: time/loudness arrays
    transcript_words: word-level transcript list
    config: optional config dict (loaded from config.yaml if missing)
    chat_times, chat_vel : optional chat velocity signal
    video_path : optional video file for visual scoring
    Main detection routine. Returns list of dicts:

    Returns
    -------
    lost of clip dicts with keys: start, end, score, component scores
    """
    if config is None:
        config = load_config()
    det_cfg = config["detector"]

    # Find loudness peaks & build windows
    peak_times, prominences = _find_audio_peaks(audio_times, rms, det_cfg)
    if len(peak_times) == 0:
        logger.warning("No loudness peaks found. Try lowering peak_prominence.")
        return []
    
    windows = _build_clip_windows(peak_times, det_cfg["clip_padding"])
    use_visual = (
        det_cfg.get("visual_enabled", False) 
        and video_path is not None
        and VISUAL_AVAILABLE
    )
    
    # Progress bar for visual (optional but looks frozen without - gross)
    progress = None
    task_id = None
    if use_visual:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )
        task_id = progress.add_task("[cyan]Visual Scoring", total=len(windows))
        progress.start()

    # Score every window
    clips = []
    for i, ((start, end), prom) in enumerate(zip(windows, prominences)):
        clip = _score_candidate(
            start, end, prom, transcript_words, det_cfg, config,
            chat_times, chat_vel, video_path, use_visual,
        )
        if use_visual and progress is not None:
            assert progress is not None and task_id is not None
            progress.update(task_id, advance=1)

        if clip["score"] >= det_cfg["min_score"]:
            clips.append(clip)
    
    if use_visual and progress is not None:
        progress.stop()

    # Merge overlapping clips
    clips = _merge_clips(clips, min_distance=10.0)
    logger.info(f"Found {len(clips)} candidate clips above minimum score (min_score)!")
    return clips

 