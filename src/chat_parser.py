

"""
Twtich chat replay downloader and velocity computation.
"""
import subprocess
import json
import re
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def extract_vod_id(url_or_path: str) -> str | None:
    """Extract numeric VOD ID from a Twitch URL or file name."""
    # Try to find pattern like videos/12353456.mp4
    match = re.search(r"videos/(\d+)", str(url_or_path))
    if match:
        return match.group(1)
    
    # Otherwise maybe it's a file name like 123456789.mp4
    # assume the basename without extension is the ID if it's all digits
    stem = Path(url_or_path).stem
    if stem.isdigit():
        return stem
    # Potential fallback here: prompt user?
    return None

def download_chat(vod_id: str, output_dir: Path = Path("data/chat")) -> Path:
    """
    Download the full chat replay of a VOD using TwitchDownloaderCLI.
    Returns path to the resulting JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{vod_id}_chat.json"

    if output_path.exists():
        logger.info(f"Chat log already exists: {output_path}")
        return output_path
    
    exe_name = "TwitchDownloaderCLI.exe"
    local_exe = Path("tools") / exe_name
    if local_exe.exists():
        exe_path = str(local_exe)
    else:
        exe_path = exe_name     # rely on Path
    
    # TwitchDownloaderCLI command
    cmd = [
        exe_path,
        "chatdownload",
        "-u", f"https://www.twitch.tv/videos/{vod_id}",
        "-o", str(output_path),
    ]
    logger.info(f"Downloading chat for VOD {vod_id}...")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    logger.info(f"Chat saved to {output_path}")
    return output_path

def load_chat_dataframe(chat_json_path: Path) -> pd.DataFrame:
    """
    Load the downloaded chat JSON into a DataFrame with columns 'time' (seconds) and 'message'
    """
    with open(chat_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    comments = data.get("comments", [])
    rows = []
    for c in comments:
        # Time offset in seconds
        t = c.get("content_offset_seconds", 0.0)
        # Message body might be nested
        msg = c.get("message", {}).get("body", "")
        rows.append({"time": t, "message": msg})

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No chat messages found.")
        # Return empty with expected columns
        return pd.DataFrame(columns=["time", "message"])
    return df

def compute_chat_velocity(
        df: pd.DataFrame,
        time_range: tuple[float, float] | None = None,
        bin_width: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert chat message timestamps into a per-second message count signal.
    If time_range is provided, restrict to [start, end].
    Returns (times, velocity) arrays of same length covering the range.
    """
    if df.empty:
        if time_range:
            t0, t1 = time_range
            times = np.arange(t0, t1, bin_width)
            return times, np.zeros_like(times)
        return np.array([]), np.array([])
        
    if time_range:
        df = df[(df["time"] >= time_range[0]) & (df["time"] <= time_range[1])]

    if df.empty:
        t0, t1 = time_range if time_range else (0, 0)
        times = np.arange(t0, t1, bin_width)
        return times, np.zeros_like(times)
    
    # Define time grid from min to max time rounded up
    t_min = df["time"].min() if time_range else df["time"].min()
    t_max = df["time"].min() if time_range else df["time"].max()
    bins = np.arange(t_min, t_max + bin_width, bin_width)
    # Count messages per bin
    counts, _ = np.histogram(df["time"], bins=bins)
    # Times for the centers of bins
    times = (bins[:-1] + bins[1:]) / 2
    return times, counts.astype(float)


