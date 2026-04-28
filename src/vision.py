

"""
Lightweight visual engagement scoring using facial expression recognition (FER).
Features:
  * Auto‑detects fer library (supports both 'from fer import FER' and
    'from fer.fer import FER').
  * Falls back if fer is not installed (VISUAL_AVAILABLE = False).
  * Configurable speed / accuracy via config.yaml → detector.visual.
  * Caches scores per time window to 'data/visual_scores/'.
  * Suppresses FFmpeg [h264] noise during frame capture (unless you turned it back on, freak).
  * Provides a Rich spinner so users know analysis is progressing (uggo without).

Usage (inside the pipeline):
    from src.vision import analyze_visual_engagement, VISUAL_AVAILABLE
    score = analyze_visual_engagement(video_path, start, end, config=config)
"""

import  os
import sys
import io
import contextlib
import json
import logging
from pathlib import Path

import numpy as np
import cv2
from rich.console import Console

# ---------------------------------------------------------------------------
# Optional FER dependency - try both known import paths 
# (I fought fight this battle with fer so you don't have to)
# ---------------------------------------------------------------------------
FER = None
try:
    from fer import FER          #type: ignore <- (for me) this works in fer==22.0.1 and earlier 
except ImportError:
    try:
        from fer.fer import FER  # works in fer>=25.10
    except ImportError:
        pass


FER_AVAILABLE = FER is not None   # Public flag for other modules
VISUAL_AVAILABLE = FER_AVAILABLE

# ---------------------------------------------------------------------------
# Logging & Console
# ---------------------------------------------------------------------------
console = Console()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FFmpeg noise suppression (the [h264] spam comes from here). mmm spam
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def suppress_stderr_fd():
    """
    Temporarily redirect file descriptor 2 (stderr) to os.devnull.
    This silences the dumb [h264] messages that OpenCV’s FFmpeg spams
    directly to stderr—without affecting Python's sys.stderr!
    """
    null_fd = os.open(os.devnull, os.O_WRONLY)
    old_fd = os.dup(2)          # save a copy of the current stderr fd
    os.dup2(null_fd, 2)                # replace stderr with null
    os.close(null_fd)
    try:
        yield
    finally:
        os.dup2(old_fd, 2)      # restore the original stderr
        os.close(old_fd)


# ---------------------------------------------------------------------------
# Global detector cache (respects MTCNN settings)
# ---------------------------------------------------------------------------
_detector = None
_current_use_mtcnn = None

def _get_detector(use_mtcnn: bool = True):
    """
    Return a cached FER detector instance. If 'use_mtcnn' changes, the detector is re‑created.
    Raises RuntimeError if fer is not installed. 800th god***n check for fer
    """
    global _detector, _current_use_mtcnn

    if not FER_AVAILABLE:
        raise RuntimeError(
            "The 'fer' library is not installed (or did you forget to do 'import fer.fer?')\n"
            "Visual scoring is disabled. Install fer with: pip install==22.0.1\n"
            "Then set visual_enabled: true in config.yaml."
        )
    if _detector is None or _current_use_mtcnn != use_mtcnn:
        logger.info("Loading face emotion detector (FER) with MTCNN={use_mtcnn}...")
        assert FER is not None # narrow da type for Pylance
        _detector = FER(mtcnn=use_mtcnn)
        _current_use_mtcnn = use_mtcnn
    return _detector


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------
def _cache_path(video_path: Path, start_s: float, end_s: float,
                output_dir: Path = Path("data/visual_scores")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{video_path.stem}_{start_s:.1f}_{end_s:.1f}_visual.json"

def _load_cached_score(video_path: Path, start_s: float, end_s: float) -> float | None:
        cp = _cache_path(video_path, start_s, end_s)
        if cp.exists():
            with open(cp, "r") as f:
                return json.load(f).get("score")
        return None

def _save_cached_score(video_path: Path, start_s: float, end_s: float, score: float):
    cp = _cache_path(video_path, start_s, end_s)
    with open(cp, "w") as f:
        json.dump({"start": start_s, "end": end_s, "score": score}, f)


# ---------------------------------------------------------------------------
# Main visual engagement analysis (scorer)
# ---------------------------------------------------------------------------
def analyze_visual_engagement(
        video_path: Path,
        start_s: float,
        end_s: float,
        sample_rate: float = 1.0,       # sample every N seconds
        max_samples: int = 30,
        use_mtcnn: bool = True,
        config: dict | None = None
) -> float:
    """
    Return a visual engagemenet score (0-1) for a video segment.
    Combines the average intensity of 'happy' and 'surprise' emotions across sampled frames.
    if 'config' is provided, the 'detector.visual' section can override 'sample_rate,
    'max_samples', and 'use_mtcnn'.

    Results are cached to 'data/visual_scores/'.
    """
    # ANOTHER CHECK FOR FER !*&#*@#&
    if not VISUAL_AVAILABLE:
        return 0.0

    # --- Config overrides ---
    if config:
        vis_cfg = config.get("detector", {}).get("visual", {})
        sample_rate = vis_cfg.get("sample_rate", sample_rate)
        max_samples = vis_cfg.get("max_samples", max_samples)
        use_mtcnn = vis_cfg.get("use_mtcnn", use_mtcnn)

    # --- Check cache ---
    cached = _load_cached_score(video_path, start_s, end_s)
    if cached is not None:
        return cached
    
    if not video_path.exists():
        logger.warning(f"Video file not found: {video_path}")
        return 0.0
    
    detector = _get_detector(use_mtcnn)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return 0.0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    duration = end_s - start_s
    num_samples = min(max_samples, max(1, int(duration / sample_rate)))
    timestamps = np.linspace(start_s, end_s, num_samples)

    emotions_sum = 0.0
    frames_evaluated = 0

    # --- Per‑frame analysis with Rich spinner & stderr suppression ---
    with console.status(f"[bold green]Analyzing visual engagement … (0 frames)[/bold green]") as status:
        with suppress_stderr_fd():
            for t in timestamps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
                ret, frame = cap.read()
                if not ret:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    result = detector.detect_emotions(rgb)
                except Exception:
                    continue
                if result:
                    e = result[0]["emotions"]
                    reaction = e.get("happy", 0.0) + e.get("surprise", 0.0)
                    reaction = min(reaction, 1.0)
                    emotions_sum += reaction
                    frames_evaluated += 1
                status.update(f"[bold green]Analyzing visual engagement … ({frames_evaluated} frames)[/bold green]")

    cap.release()
    score = (emotions_sum / frames_evaluated) if frames_evaluated > 0 else 0.0

    # --- Cache and return ---
    _save_cached_score(video_path, start_s, end_s, score)
    return score


