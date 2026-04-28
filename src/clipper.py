

# Cut export clips using ffmpeg/moviepy

import subprocess
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Blur has been removed
def build_video_filter(fade_duration: float, dur: float, vertical: bool) -> str:
    """
    Build the ffmpeg video filter chain.
    vertical: if True, crop centered to 9:16 (and optionally add blurred background.)
    blur:     if vertical, fill empty space with a blurred copy of the original.
    """
    filters = []

    if vertical:
        filters.append("crop=ih*9/16:ih,scale=1080:1920")

    # Fade in / out
    fade_in = f"fade=t=in:d={fade_duration}"
    fade_out_start = max(0, dur - fade_duration)
    fade_out = f"fade=t=out:st={fade_out_start}:d={fade_duration}"
    filters.append(f"{fade_in},{fade_out}")

    return ",".join(filters)


def get_video_duration(video_path: Path) -> float:
    """Get duration in seconds using ffprobe"""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format_duration",
            "-of", "json",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning(f"ffprobe failed. Trying ffmpeg fallback.")

    try:
        cmd = ["ffmpeg", "-i", str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        # ffmpeg prints info to stderr, exit code is non-zero because no output file specified
        stderr = result.stderr
        match = re.search(r" Duration: (\d+):(\d+):(\d+)\.(\d+)", stderr)
        if match:
            h, m, s, frac = match.groups()
            duration = int(h) * 3600 + int(m) * 60 + int(s) + float(f"0.{frac}")
            logger.info(f"Got duration from ffmpeg: {duration}s")
            return duration
        else:
            raise RuntimeError(" Could not parse duration from ffmpeg output.")
    except FileNotFoundError:
        raise RuntimeError(
            "Neither ffprobe nor ffmpeg found. Make sure ffmpeg is installed and in your PATH.\n"
            "Download from https://ffmpeg.org/download.html"
        )

def clip_segments(
        video_path: Path,
        clips: list[dict],
        output_dir: Path = Path("data/clips"),
        fade_duration: float = 0.5,
        crf: int = 23,
        max_clips: int | None = None,       # Py 3.10+ syntax for type error fix
        vertical: bool = False,
        #vertical_blur: bool = True
) -> list[Path]:
    """
    Cut and export clips.
    clips: list of dicts with 'start', 'end', 'score' (from detector)
    Returns list of output file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    video_duration = get_video_duration(video_path)

    # Sort by score descending and take top N
    clips = sorted(clips,  key=lambda x: x["score"], reverse=True)
    if max_clips is not None:
        clips = clips[:max_clips]

    if not clips:
        logger.warning("No clips to export.")
        return []
    
    output_paths = []
    for idx, clip in enumerate(clips, start=1):
        start = max(0, clip["start"])
        end = min(clip["end"], video_duration)
        if end - start < 1.0:
            logger.info(f" Skipping clip {idx}: duration too short ({end-start:.1f}s)")
            continue

        score = clip["score"]
        out_name = f"clip_{idx:02d}_score_{score:.2f}.mp4"
        out_path = output_dir / out_name

        # Build ffmpeg command. We'll use the fade filter: fade in at start, fade out at end
        dur = end - start
        fade_out_start = max(0, dur - fade_duration)
        vf_filter = build_video_filter(fade_duration, dur, vertical)

        cmd = [
            "ffmpeg",
            "-ss", str(start),          # fast seek before input (accurate enough)
            "-i", str(video_path),
            "-t", str(dur),
            "-vf", vf_filter,
            "-af", f"afade=t=in:d={fade_duration},afade=t=out:st={fade_out_start}:d={fade_duration}",
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", "medium", 
            "-c:a", "aac",
            "-b:a", "128k", 
            "-y", 
            str(out_path)
        ]

        logger.info(f"Exporting clip {idx}: {start:.1f}s - {end:.1f}s {score:.2f})")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg failed: {result.stderr}")
            result.check_returncode()

        output_paths.append(out_path)

    logger.info(f"Exported {len(output_paths)} clips to {output_dir}")
    return output_paths