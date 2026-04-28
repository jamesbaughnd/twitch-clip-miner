# 🎬 Twitch AI Clip Miner

AI that watches your Twitch VODs so you can have a f**kin' yank—uhh, i mean, so you don't have to!

*Automagically* (BOOOO) turn hours of Twitch VODs into bite‑sized, viral‑ready clips — with multi-signal detection, automatic GPU acceleration, and drop-dead, sexy af, synth-wave-vibe-ahh command-line experience.

## ✨ Features
- **Multi‑signal detection** – combines audio energy, transcribed speech, chat velocity, and facial reactions for picking the best clips.
- **GPU auto‑detection** – NVIDIA CUDA, AMD Vulkan, or CPU fallback.
- **Resumable chunking** – handles 10‑hour VODs without memory issues.
- **Smart insights** – tells you *why* each clip is worth sharing, with editing tips (limited for now, but helpful still).
- **TikTok / Shorts ready** – one toggle for vertical 9:16 output.
- **Offline & private** – runs entirely on your own hardware!

## 🚀 Quickstart (for my impatient guys/gals/enbys)
### ~ Prerequisites ~
- **Python 3.10–3.12** - (3.12 recommended)
- **FFmpeg** - (must be in your PATH (details in [Noob Guide](#Noob_Guide.md) if you're just a babby)
- **TwitchDownloaderCLI** (optional, for downloading chat) – download the latest release and place `TwitchDownloaderCLI.exe` into the `tools/` folder.
- **whisper.cpp** - optional, but HIGHLY recommended for AMD GPU transcription (unless you have a small penar, of course) – see [Advanced GPU Setup](#advanced-gpu-setup).

### 1. Clone the repository
```bash
git clone https://github.com/jamesbaughnd/twitch-clip-miner.git
cd twitch-clip-miner
```

### 2. Create a virtual environment and install
```bash
python -m venv venv
venv\Scripts\activate         # env\Scripts\Activate.ps1 in Windows PowerShell
source venv/bin/activate      # Linux/macOS

pip install -r requirements.txt
```
### 3. Run it!
```bash
python main.py "https://www.twitch.tv/videos/yourvidhere"
```
>(or simply `python main.py` to be prompted for a URL or local video location! Also cool guys just use *py*)


## ⚙️ Configuration
All settings live in config.yaml. The most important:
```yaml
detector:
  clip_padding: 5.0          # seconds before/after a peak (15 = 30 second clip)
  peak_prominence: 0.8       # lower = more peaks
  min_score: 0.5             # absolute threshold (not a percentage)
  weight_loudness: 0.35
  weight_transcript: 0.25
  weight_chat: 0.2
  weight_visual: 0.2

output:
  max_clips: 10
  vertical: false            # set to true for TikTok/Shorts output

transcription:
  backend: auto              # auto, faster-whisper, whisper-cpp
  model_size: auto
  device: auto
  compute_type: auto
  ```
See the [Cheatsheet](#cheatsheet.md) for a full explanation of scoring and tuning. BUT if you don't want to mess with the settings, the default settings will already get you 10 usable clips!


## 📂 Output
All clips are saved in `data/clips/<vod_id>/` along with an insights JSON per clip.
You'll also find cached transcripts, audio segments, and visual scores in `data/` - all reusable across runs.

## 🧪 Command‑line options
```text
python main.py [vod_url] [--force] [--max-clips N] [--min-score X] [--vertical] [--language LANG] [--output DIR]
```

| Flag             |	       What it does             |
| :--------        | :--------------------------------------------------------|
|`--force`         |	Re‑process all audio/transcript (ignore caches) |
|`--max-clips N`   |	Override the number of clips to export |
|`--min-score X`   |	Set a custom minimum clip score |
|`--vertical`      |	Force vertical 9:16 output |
|`--language LANG` |	Transcription language - en, de, fr, etc. (that's not one dummy)|
|`--output DIR`    |	Custom base directory for clips|

---
## 🔧 Advanced GPU Setup

### AMD GPU (Vulkan) via whisper.cpp
1. Download a whisper.cpp model (e.g. ggml-small.en.bin) into models/.
2. Build or download whisper-cli.exe with Vulkan enabled and place it in tools/.
3. In config.yaml, set:

```yaml
transcription:
  backend: whisper-cpp
  whisper_cpp:
    executable: tools/whisper-cli.exe
    model: models/ggml-base.en.bin
```
For detailed build instructions, see the [Noob’s Guide]().


### 🤔 Troubleshooting
#### **"fer" library not installed / visual scoring won’t start** 
- Install fer==22.0.1 and setuptools==68.2.2 (already in requirements). If you get pkg_resources errors, make sure you’re using Python 3.12 and those exact versions.

#### **Chat download fails**
- Ensure TwitchDownloaderCLI.exe is in tools/ or on your PATH.
#### **Whisper progress lines flood the terminal**
 - They’re redirected to a debug logger – you won’t see them in normal use. If you need them, check whisper_progress.log.

#### **GPU not detected for transcription**
 - Run the manual test in the Noob’s Guide. Usually you need to install the Vulkan Runtime or update your AMD drivers.
---
---
## 🧭 Roadmap
### v1.1+ – Clip Review & Tweaks
- [ ] Built‑in clip review GUI (VLC‑powered playback with audio) 
- [ ] Trim controls for fine‑tuning start/end times
- [ ] Batch export of all adjusted clips

### v2.0 – Smart Creator Studio
- [ ] Full‑featured GUI wrapping the entire pipeline
- [ ] Platform‑aware export presets and their requirements (TikTok, Shorts, Twitter, etc.)
- [ ] AI‑powered hashtag suggestions based on clip content
- [ ] Save and reuse custom hashtag sets, fine-tuned for per-platform performance

### Beyond v2.0 - Multi-Media Manager God?
- [ ] Batch‑process multiple VODs (queuing, scheduling, posting)
- [ ] One‑click upload to TikTok, YouTube, Instagram, and Twitter
- [ ] Auto‑chapter detection for long streams
- [ ] Plugins / signals SDK for community contributions
- [ ] Essentially a full-blown social media manager

--- 
---
# If you think this is a neat tool, consider supporting me by [buying me a coffee!! ☕]()
---
>
### 📜 License
MIT – see [LICENSE]().

---
---
# Nerd Section

## 🤖 For Developers: How the Scoring Works

### The Big Picture
The detector listens for **loudness peaks** – sudden jumps in audio energy. Each peak becomes a candidate clip window (peak ± `clip_padding`). Then every window is scored by up to four independent signals:

| Signal        | Raw output               | What it measures |
|---------------|--------------------------|------------------|
| Loudness      | z‑score prominence       | How much louder the audio got at that moment |
| Transcript    | sum of bonus points      | Hype words, laughter, speech density |
| Chat          | mean normalised velocity | Messages‑per‑second spike |
| Visual        | 0–1 emotion intensity    | Happy + surprise from the streamer’s face |

Each signal returns a **float** that can be arbitrarily large (no fixed maximum). Those floats are multiplied by configurable weights and added together:

```
final_score = w_loud · L  +  w_trans · T  +  w_chat · C  +  w_vis · V
```

- **'min_score' is NOT a percentage** – it’s an absolute threshold on this sum.  
  - Example: a timeframe with a huge loudness spike (prominence 4.0), one hype word (1.5), and a normal chat (0.0) might get  
  `0.35*4.0 + 0.25*1.5 + 0.2*0.0 + 0.2*0.8 = 2.335`. 
 So, `min_score: 2.0` would keep this clip, `min_score: 2.5` wouldn't, while `min_score: 0.5` would keep almost everything.

### Adding a New Signal
The detector’s loop calls `_score_candidate()` for every window. To add a new signal:

1. **Create a new module** (e.g., `src/audio_classifier.py`) with a function that takes a video path and a time window, and returns a float score.
2. **Add a weight** to `config.yaml` under `detector:` (e.g., `weight_audio_class: 0.15`).
3. **Call your function inside `_score_candidate()`** in `src/detector.py`, multiply by the weight, and add it to `combined`.
4. **Optionally add a cache** – follow the pattern in `src/vision.py`.

No other code changes needed – the rest of the pipeline (chunking, merging, exporting) uses the final score automatically.

### Code Map
- `detector.py::_find_audio_peaks()` → where clip-candidate windows come from
- `detector.py::_score_candidate()` → where all signals are combined
- `summarizer.py` → builds the human‑readable insights after scoring
- `chunker.py` → splits long VODs, feeds each chunk to `detect_clips`
- `config.yaml` → all thresholds, weights, and model settings

### Contributing
Pull requests are welcome! Please keep new signal modules **optional** (graceful fallback if a dependency is missing) and add a flag in `config.yaml` so users can enable/disable them. The existing signals are great templates – especially `vision.py` for caching and `chat_parser.py` for external tool integration.

---

[**📜MIT License**](LICENSE.txt)
– do whatever you like, just plz your god king lord credit (and the other tool creators you use!)
