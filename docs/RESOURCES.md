# 📚 Master Resource List

Everything that went into building **Twitch AI Clip Miner**, organized by category so you can find exactly what you need – whether you're setting up the tool for the first time or dinking around with the source code.

---

## 🐍 Python Libraries (PyPI)

All of these appear in `requirements.txt`. The pinned versions are the ones we confirmed work together with Python 3.12.

| Package | Version | What it does in the project | Link |
|---------|---------|-----------------------------|------|
| **faster-whisper** | ≥1.0,<2 | CPU / NVIDIA CUDA transcription backend – uses CTranslate2 under the hood | [PyPI](https://pypi.org/project/faster-whisper/) |
| **fer** | 22.0.1 | Facial expression recognition (happy, surprise) for visual scoring. Depends on TensorFlow & OpenCV | [PyPI](https://pypi.org/project/fer/) |
| **setuptools** | 68.2.2 | Provides `pkg_resources` which `fer` needs internally. Newer versions removed it | [PyPI](https://pypi.org/project/setuptools/) |
| **yt-dlp** | latest | VOD downloader – handles Twitch URLs and local files. Uses ffmpeg internally | [GitHub](https://github.com/yt-dlp/yt-dlp) / [PyPI](https://pypi.org/project/yt-dlp/) |
| **rich** | latest | Beautiful terminal output: coloured sections, tables, panels, spinners, progress bars | [PyPI](https://pypi.org/project/rich/) |
| **tqdm** | latest | Alternative progress bars (used in some paths) | [PyPI](https://pypi.org/project/tqdm/) |
| **librosa** | latest | Audio loading + RMS loudness computation | [PyPI](https://pypi.org/project/librosa/) |
| **soundfile** | latest | WAV file read/write (used by librosa) | [PyPI](https://pypi.org/project/soundfile/) |
| **scipy** | latest | Peak finding (`find_peaks`) + Gaussian smoothing in detector | [PyPI](https://pypi.org/project/scipy/) |
| **numpy** | latest | Array math – used everywhere | [PyPI](https://pypi.org/project/numpy/) |
| **pandas** | latest | Chat DataFrame loading & velocity computation | [PyPI](https://pypi.org/project/pandas/) |
| **opencv-python** | latest | Frame capture for visual scoring + thumbnail generation in GUI | [PyPI](https://pypi.org/project/opencv-python/) |
| **Pillow** | latest | Image resizing for GUI thumbnails (`Image.Resampling.LANCZOS`) | [PyPI](https://pypi.org/project/Pillow/) |
| **pyyaml** | latest | Config file loading (`config.yaml`) | [PyPI](https://pypi.org/project/pyyaml/) |
| **moviepy** | latest | Lightweight video utilities (optional) | [PyPI](https://pypi.org/project/moviepy/) |
| **sv-ttk** | latest | Sun Valley theme – makes the tkinter GUI look like Windows 11 | [PyPI](https://pypi.org/project/sv-ttk/) |
| **python-vlc** | *(optional)* | VLC-powered GUI (future release) – embeds VLC player in tkinter | [PyPI](https://pypi.org/project/python-vlc/) |

---

## 🧰 External CLI Tools

These are executables users place in `tools/` (or in their system PATH).

| Tool | What it does | Notes | Link |
|------|--------------|-------|------|
| **FFmpeg** | Audio extraction, clip trimming, vertical output, codec encoding. The Swiss Army knife of the project | Must be in PATH. Windows build from gyan.dev recommended | [ffmpeg.org](https://ffmpeg.org/download.html) |
| **TwitchDownloaderCLI** | Downloads Twitch chat replays as JSON files (no API key needed) | Place in `tools/`. Developed by lay295 | [GitHub](https://github.com/lay295/TwitchDownloader/releases) |
| **whisper-cli** (whisper.cpp) | AMD GPU transcription via Vulkan backend. Built from source with CMake and Vulkan enabled | v1.7.6 confirmed working on AMD. Place compiled `whisper-cli.exe` + DLLs in `tools/` | [GitHub](https://github.com/ggerganov/whisper.cpp) |

---

## 🤖 AI Model Files

These go in the `models/` folder and are ignored by Git.

| Model | Size | What it does | Link |
|-------|------|--------------|------|
| **ggml-base.en.bin** | ~148 MB | English-only base Whisper model. Good balance of speed/accuracy. Used by whisper.cpp (Vulkan). (This tool DOES support other languages! Just drop the .en) | [HuggingFace](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin) |
| **ggml-small.en.bin** | ~488 MB | English-only small Whisper model. More accurate, slightly slower. Used by whisper.cpp (This tool DOES support other languages! Just drop the .en) | [HuggingFace](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin) |

---

## 💻 System Dependencies

These must be installed on YOUR system before the tool will work.

| Tool | What it provides | Link |
|------|------------------|------|
| **Python 3.12** | The runtime itself. Make sure "Add Python to PATH" is checked | [python.org](https://www.python.org/downloads/) |
| **Vulkan Runtime** | Allows whisper.cpp to talk to AMD GPUs. Install the Runtime (not the full SDK) | [vulkan.lunarg.com](https://vulkan.lunarg.com/) |
| **Visual Studio Build Tools 2022** | (Visual Studio 17 2022) C++ compiler (MSVC) needed to compile whisper.cpp from source. Select "Desktop development with C++" workload (I used used Visual Studio 18 2026 cause I'm a fangirl for NEW stuff) | [visualstudio.com](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) |
| **CMake** | Build system used to compile whisper.cpp. Download the Windows x64 installer | [cmake.org](https://cmake.org/download/) |
| **VLC Media Player** | *(optional, future)* Required by `python-vlc` for the GUI video player. The 64-bit installer provides `libvlc.dll` | [videolan.org](https://www.videolan.org/vlc/) |

---

## 🐞 Debugging / Dev Tools

These are optional but saved us many times during development.

| Tool | When to use it | Link |
|------|----------------|------|
| **Dependencies (by lucasg)** | Shows which DLLs a `.exe` needs and which are missing. Essential when `whisper-cli.exe` fails to launch | [GitHub](https://github.com/lucasg/Dependencies/releases) |
| **Ninja** | Alternative build system – faster than MSVC for compiling whisper.cpp. Install with `pip install ninja` | [PyPI](https://pypi.org/project/ninja/) |

---

## 📂 GitHub Repositories

| Repository | Role in the project | Link |
|------------|---------------------|------|
| **ggml-org/whisper.cpp** | Port of OpenAI Whisper in C/C++ with Vulkan GPU support | [GitHub](https://github.com/ggerganov/whisper.cpp) |
| **SYSTRAN/faster-whisper** | Python wrapper around CTranslate2 for CPU/NVIDIA transcription | [GitHub](https://github.com/guillaumekln/faster-whisper) |
| **lay295/TwitchDownloader** | CLI + GUI tool for downloading Twitch VODs and chat | [GitHub](https://github.com/lay295/TwitchDownloader) |
| **justinshenk/fer** | Facial expression recognition library used in visual scoring | [GitHub](https://github.com/justinshenk/fer) |
| **rdbende/Sun-Valley-ttk-theme** | Windows 11 theme for tkinter – makes the GUI look modern | [GitHub](https://github.com/rdbende/Sun-Valley-ttk-theme) |
| **openai/whisper** | The original Whisper model by OpenAI (used as reference, not directly in the code) | [GitHub](https://github.com/openai/whisper) |
| **OpenNMT/CTranslate2** | Fast inference engine for Transformer models – used by faster-whisper | [GitHub](https://github.com/OpenNMT/CTranslate2) |

---

**Last updated:** 4/28/2026  
**Maintainer:** [JamesBaughnd](https://github.com/jamesbaughnd)