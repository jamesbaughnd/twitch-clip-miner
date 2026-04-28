
# --------------------------------------------------------------
# Fetch VOD (yt-dlp wrapper)
# --------------------------------------------------------------
import yt_dlp
from pathlib import Path

def download_vod(vod_url: str, output_dir: str = "data/vods") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        'outtmpl' : str(out_dir / '%(id)s.%(ext)s'),
        'format': 'best[height<=1080]',
        'quiet': False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:             # type: ignore[arg-type] (dict works fine here, flagger just being strict!)
        info = ydl.extract_info(vod_url, download=True)
        filename = ydl.prepare_filename(info)

    return Path(filename)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python src/downloader.py <twitch_vod_url>")
        sys.exit(1)
    url = sys.argv[1]
    path = download_vod(url)
    print(f"Downloaded to {path}")
