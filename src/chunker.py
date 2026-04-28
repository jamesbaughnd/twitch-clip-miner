"""
Process a long VOD in overlapping time chunks.
"""
import logging
from pathlib import Path
import numpy as np
import json


from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
from rich.live import Live 

from src.audio import extract_audio_segment, compute_loudness
from src.transcriber import transcribe_audio_auto
from src.detector import detect_clips, load_config
from src.clipper import get_video_duration

logger = logging.getLogger(__name__)
import sys

def _load_cached_transcript(cache_path: Path) -> list[dict] | None:
    """Load a cached word-level transcript if it exists."""
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.warning(f"Corrupt transcript cache: {cache_path}")
            return None  
    return None

def process_video_in_chunks(
        video_path: Path,
        chunk_duration: float = 900,        # 15-minutes
        overlap: float = 30.0,              # overlap to catch events on boundaries
        config: dict | None = None,         # Python 3.10+ syntax, shutup type checker
        chat_times: np.ndarray | None = None,
        chat_vel: np.ndarray | None = None,
        force: bool = False,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        language: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Split video into time windows, detect clips in each, then merge.
    returns list of clip dicts with absolute timestamps (score-sorted, deduplicated).
    """
    total_duration = get_video_duration(video_path)
    if config is None:
        config = load_config()

    # Generate chunk start times (with overlap)
    chunk_starts = []
    current = 0.0
    while current < total_duration:
        chunk_starts.append(current)
        current += chunk_duration - overlap

    audio_chunk_dir = Path("data/audio_chunks")
    transcript_dir = Path("data/transcripts")    

    #console = Console()
    """
    progress_bar = Progress(
        SpinnerColumn(),
        TextColumn("Processing chunks"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    )
    task_id = progress_bar.add_task("Chunks", total=len(chunk_starts))
    """
    all_clips = []
    all_words = []

    for i, start in enumerate(chunk_starts):
        end = min(start + chunk_duration, total_duration)
        length = end - start

        #live.console.log(f"Chunk {i+1}/{len(chunk_starts)}: {start:.0f}s–{end:.0f}s", justify="right", _stack_offset=1,)
        logger.info(f"Processing Chunk {i+1}/{len(chunk_starts)}: {start:.0f}s–{end:.0f}s")

        # File names for this chunk
        seg_name = f"{video_path.stem}_{int(start)}s_{int(length)}s.wav"
        audio_seg_path = audio_chunk_dir / seg_name
        trans_cache_path = transcript_dir / f"{video_path.stem}_{int(start)}s_{int(length)}s_words.json"
        
        # Audio segment (skip extraction if file exists and not forced)
        if force or not audio_seg_path.exists():
            audio_seg = extract_audio_segment(video_path, start, length)
        else:
            logger.debug("Using cached audio segment.")
            audio_seg = audio_seg_path
        
        # Loudness (always run - fast)
        times_local, rms, sr = compute_loudness(audio_seg)

        # Transcription (skip if cached, unless forced)
        words = None
        if not force:
            words = _load_cached_transcript(trans_cache_path)
        if words is None: 
            words = transcribe_audio_auto(
                audio_seg, config=config, model_size=model_size, 
                device=device, compute_type=compute_type, force=force, language=language or "en"
            )

        # Add words to the global list with shifted timestamps
        if words:
            for w in words:
                w_global = dict(w)
                w_global["start"] += start
                w_global["end"] += start
                all_words.append(w_global)

        # Slice chat signal for this chunk
        if chat_times is not None and chat_vel is not None:
            mask = (chat_times >= start) & (chat_times < end)
            chunk_chat_times = chat_times[mask]
            chunk_chat_vel = chat_vel[mask]
        else:
            chunk_chat_times = None
            chunk_chat_vel = None

        # Detect clips (always run - fast, uses timestamps relative to chunk)
        clips_chunk = detect_clips(
            audio_times=times_local,
            rms=rms,
            transcript_words=words,
            config=config,
            chat_times=chunk_chat_times,
            chat_vel=chunk_chat_vel,
            video_path=video_path,
        )
        
        # Shift timestamps to global VOD time
        for clip in clips_chunk:
            clip["start"] += start
            clip["end"] += start
            clip["peak_time"] += start
            clip["chunk_index"] = i
        all_clips.extend(clips_chunk)
        
        logger.info(f"✅ Chunk {i+1}/{len(chunk_starts)} complete.")
        print(" ")

        sys.stderr.flush()
        sys.stdout.flush()

    # Merge and deduplicate clips  
    all_clips.sort(key=lambda x: x["score"], reverse=True)
    final_clips = _nms(all_clips, min_distance=10.0)
    logger.info(f"Merged {len(all_clips)} candidates -> {len(final_clips)} final clips")

    # Sort combined words by start time
    all_words.sort(key=lambda w: w["start"])
    logger.info(f"Combined transcript: {len(all_words)} words total.")

    return final_clips, all_words

def _nms(clips: list[dict], min_distance: float = 10.0) -> list[dict]:
    """
    Keep highest-scoring clip, discard any that start within 'min_distance' seconds.
    """
    kept = []
    used_starts = []
    for clip in clips:
        too_close = any(abs(clip["start"] - s) < min_distance for s in used_starts)
        if not too_close:
            kept.append(clip)
            used_starts.append(clip["start"])
    return kept