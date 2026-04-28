

# ============================================================================
#  🎬 Twitch Clip Miner  –  main.py
#  AI that watches your VODs so you don't have to.
# ============================================================================
import os
import sys
import subprocess
import logging
from pathlib import Path
import json
import argparse

# ---------------------------------------------------------------------------
#  Environment setup – MUST run before any library that touches TensorFlow,
#  OpenCV, or FFmpeg. This silences the startup & [h264] noise.
#  If you're a masochist, feel free to delete/comment this out. freak
# ---------------------------------------------------------------------------
def _configure_environment():
    """Set environment variables to suppress those dumbass TensorFlow / OpenCV /FFmpeg warnings."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")      # ERROR only
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")     # oneDNN quiet
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")      # suppress < ERROR
    os.environ.setdefault("ABSL_LOG_SUPPRESS_ALL", "1")     # absl quiet

_configure_environment()

# ---------------------------------------------------------------------------
#  Standard + third‑party imports (safe to run)
# ---------------------------------------------------------------------------
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.align import Align

# ---------------------------------------------------------------------------
#  Project modules
# ---------------------------------------------------------------------------
from src.downloader import download_vod
from src.audio import extract_audio, compute_loudness, plot_loudness
from src.transcriber import transcribe_audio_auto
from src.detector import detect_clips, load_config
from src.clipper import clip_segments, get_video_duration
from src.chunker import process_video_in_chunks
from src.chat_parser import extract_vod_id, download_chat, load_chat_dataframe, compute_chat_velocity
from src.vision import VISUAL_AVAILABLE   # True if fer library is installed


# ============================================================================
#  Main pipeline
# ============================================================================
def main():
    # ------------------------------------------------------------------
    #  Logging – Rich handler for sexy, hot, handsome-squidward output
    # ------------------------------------------------------------------
    
    #log_console = Console(stderr=True)
    #main_console = Console()
    console = Console(stderr=True)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    #console = Console(stderr=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False,
                              show_level=True, show_path=False)]
    )
    logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    #  Centered banner (the only centered element)
    # ------------------------------------------------------------------
    console.print("")
    banner = Panel.fit( "[bold magenta]🎬 Twitch AI Clip Miner[/bold magenta]",
                       border_style="bright_blue")
    console.print(Align.center(banner))
    console.print("")

    # ------------------------------------------------------------------
    #  Argument parsing  (run with --help to see all options)
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description= "🎬 Twitch Clip Miner – AI that watches your VODs so you don't have to."
    )
    parser.add_argument(
        "vod_url", nargs="?",
        help="Twitch VOD URL or local video file path. If omitted, you will be prompted."
    )
    parser.add_argument("--force", action="store_true", 
                        help="Re‑process all audio/transcript files (ignore caches).")
    # for the F U T U R E
    # parser.add_argument("--review", action="store_true", 
    #                    help="Open the clip review GUI after processing.")
    parser.add_argument("--max-clips", type=int, default=None, 
                        help="Override max number of clips to export (otherwise load from config.yaml).")
    parser.add_argument("--min-score", type=float, default=None,
                        help="Override min clip score threshold.")
    parser.add_argument("--vertical", action="store_true", default=None,
                        help="Force vertical (9:16) output. Overrides config.yaml setting.")
    parser.add_argument("--language", type=str, default=None,
                        help="Language for transcription (e.g., en, de, fr). Overrides config.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Base directory for clips (default: data/clips/<vod_id>).")
    args = parser.parse_args()

    force_reprocess = args.force
    if force_reprocess:
        logger.info("Force mode - will re-create all audio/transcript files.")

    # ------------------------------------------------------------------
    # Input - URL, local file, or interactive input
    # ------------------------------------------------------------------
    console.rule("[bold blue]Download / Input[/bold blue]", align="left")
    console.print("")
    if args.vod_url:
        USER_INPUT = args.vod_url
    else:
        USER_INPUT = input("Enter a Twitch VOD URL or local video path: ").strip()

    input_path = Path(USER_INPUT)
    if input_path.exists():
        video_path = input_path.resolve()
        logger.info(f"Using local video: {video_path}")
    else:
        # Assume it's a URL and download
        logger.info(f"Downloading VOD from URL: {USER_INPUT}")
        video_path = download_vod(USER_INPUT)


    # ------------------------------------------------------------------
    #  Load configuration (config.yaml)
    # ------------------------------------------------------------------
    config = load_config(Path("config.yaml"))
    det_cfg = config["detector"]

    # Apply any CLI Overrides (or use the config, nerd)
    if args.max_clips is not None:
        max_clips = args.max_clips
    else:
        max_clips = config["output"]["max_clips"]

    if args.min_score is not None:
        det_cfg["min_score"] = args.min_score
    
    if args.vertical is not None:
        vertical_mode = args.vertical
    else:
        vertical_mode = config.get("output", {}).get("vertical", False)

    if args.language is not None:
        config.setdefault("transcription", {})["language"] = args.language
        logger.info(f"Language forced to: {args.language}")

    if args.output is not None:
        base_output_dir = args.output
    else:
        base_output_dir = Path("data/clips")


    trans_cfg = config.get("transcription", {})
    model_size = trans_cfg.get("model_size", "auto")    
    device = trans_cfg.get("device", "auto")
    compute_type = trans_cfg.get("compute_type", "auto")
    language = trans_cfg.get("language", "en")


    # ------------------------------------------------------------------
    #  Chat download (optional)
    # ------------------------------------------------------------------
    vod_id = extract_vod_id(str(USER_INPUT))
    chat_times_global, chat_vel_global = None, None
    chat_df = None

    if vod_id:
        try:
            chat_json = download_chat(vod_id)
            chat_df = load_chat_dataframe(chat_json)
            chat_times_global, chat_vel_global = compute_chat_velocity(chat_df)
            # Normalize chat velocity (z-score) so the weight makes more sense across VODs
            if chat_vel_global is not None and chat_vel_global.std() > 0:
                chat_vel_global = (chat_vel_global- chat_vel_global.mean()) / chat_vel_global.std()
            logger.info(f"Chat velocity signal ready, ({len(chat_df)} messages.)")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Chat download skipped: {e}")
    else:
        logger.warning("Chat not extracted. No VOD ID could be found.")

    # ------------------------------------------------------------------
    #  Check visual scoring availability (extra lil check for fer)
    # ------------------------------------------------------------------
    if det_cfg.get("visual_enabled", False) and not VISUAL_AVAILABLE:
        logger.warning(
            "Visual scoring is enabled in config but 'fer' library is not installed!\n"
            "Run: pip install fer==22.0.1   (and setuptools==68.2.2)"
        )

    # ------------------------------------------------------------------
    #  Processing path – full VOD or chunked (CHONKY)
    # ------------------------------------------------------------------
    video_duration = get_video_duration(video_path)
    logger.info(f"VOD duration: {video_duration/60:.1f} minutes")

    LONG_THRESHOLD = 1800   # 30 minutes
    show_insights = False
    words = None

    if video_duration > LONG_THRESHOLD:
        console.print("\n")
        console.rule("[bold blue]Audio & Chat (chunked)[/bold blue]", align="left")
        clips, words = process_video_in_chunks(
            video_path,
            chunk_duration=900,
            overlap=30,
            config=config,
            chat_times=chat_times_global,
            chat_vel=chat_vel_global,
            force=force_reprocess,
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
        )
        show_insights = True
    else:
        console.print("")
        console.rule("[bold blue]Audio Extraction & Loudness[/bold blue]", align="left")
        audio_path = extract_audio(video_path)
        times, rms, sr = compute_loudness(audio_path)
        plot_loudness(times, rms, audio_path)

        console.print("")
        console.rule("[bold blue]Transcription[/bold blue]", align="left")
        words = transcribe_audio_auto(
            audio_path,
            config=config, 
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            force=force_reprocess,
            language=language,
        )

        console.print("")
        console.rule("[bold blue]Detection[/bold blue]", align="left")
        clips = detect_clips(
            audio_times=times, 
            rms=rms, 
            transcript_words=words, 
            config=config,
            chat_times=chat_times_global,
            chat_vel=chat_vel_global,
            video_path=video_path,
        )

    # ------------------------------------------------------------------
    #  Show results table & export clips
    # ------------------------------------------------------------------
    top_clips = clips[:max_clips]
    console.print("")
    console.rule("[bold green]Clipping & Results[/bold green]", align="left")

    if top_clips:
        table = Table(title="Top Clip Candidates", style="cyan")
        table.add_column("#", justify="right", style="magenta")
        table.add_column("Start", justify="right")
        table.add_column("End", justify="right")
        table.add_column("Dur", justify="right")
        table.add_column("Score", justify="right", style="green")

        for i, c in enumerate(top_clips, 1):
            dur = c['end'] - c['start']
            table.add_row(
                str(i),
                f"{c['start']:6.1f}s",
                f"{c['end']:6.1f}s", 
                f"{dur:.1f}s",
                f"{c['score']:.2f}"
            )
        console.print(table)

        # Per-VOD output folder
        clip_folder = vod_id if vod_id else video_path.stem
        clips_output_dir = base_output_dir / clip_folder

        # Extract clips
        clip_paths = clip_segments(
            video_path, 
            top_clips,
            output_dir=clips_output_dir, 
            max_clips=max_clips,
            vertical=vertical_mode,
        )
        console.print(f"\n✅ Created {len(clip_paths)} clip(s) in {clips_output_dir}/", style="green")
        

        # ------------------------------------------------------------------
        #  Clip insights (for full transcript available paths)
        # ------------------------------------------------------------------
        if show_insights and words is not None:
            console.print("")
            console.rule("[bold magenta]Clip Insights[/bold magenta]", align="left")
            console.print("")
            from src.summarizer import generate_clip_insights

            for i, c in enumerate(top_clips, 1):
                insights = generate_clip_insights(
                    c,
                    all_words=words,
                    chat_df=chat_df,
                    config=config,
                )
                # Display panel
                console.print(
                    Panel(
                        f"[bold]#{i}  {c['start']:.1f}s – {c['end']:.1f}s[/bold]\n"
                        f"[bold magenta]Category:[/bold magenta] {insights['viral_category']}\n"
                        f"[bold]Dominant signal:[/bold] {insights['dominant_signal']}\n"
                        f"[bold]Keywords:[/bold] {', '.join(insights['keywords']) if insights['keywords'] else 'none'}\n"
                        f"[bold]Chat:[/bold] {insights['chat_peak']}\n"
                        f"[bold]Face:[/bold] {insights['visual_mood']}\n"
                        f"[bold]Editing tips:[/bold]\n" +
                        "\n".join(f" • {tip}" for tip in insights['editing_tips']) +
                        f"\n\n[dim]Transcript snippet:[/dim] \"{insights['full_text_sample']}\"",
                        title=f"Clip {i} Insights",
                        border_style="green"
                    )
                )

                # Enrich insights with clip metadata for the F U T U R E
                insights["start"] = c["start"]
                insights["end"] = c["end"]
                insights["score"] = c["score"]
                insights["source_video"] = str(video_path)

                insights_path = clips_output_dir / f"clip_{i:02d}_insights.json"
                with open(insights_path, "w", encoding="utf-8") as f:
                    json.dump(insights, f, indent=2, ensure_ascii=False)

            console.print(f"[dim]Insights saved to {clips_output_dir}[/dim]")
        else:
            console.print("[yellow]ℹ️ Insights not available (empty transcript)[/yellow]")
    else:
        console.print("\n❌ No clips found - try lowering thresholds in config.yaml.", style="yellow")

if __name__ == "__main__":
    main()