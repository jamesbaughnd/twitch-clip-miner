

# 🐣 The Absolute Noob’s Guide to Getting Twitch Clip Miner Working

This guide will walk you through some of the most common problems and exactly how to fix them. If you still get stuck after trying all of these steps, open an issue on GitHub and paste the full terminal output — I'LL MOVE THE HEAVENS TO HELP YOU, CITIZEN! 🤝

---

## 1. I installed Python but the `python` command doesn’t work
- On Windows, make sure you ticked **“Add Python to PATH”** during installation. 
  - (To find it/check it manually: find the FULL path to your Python.exe (C:\Users\Daddys\HUGEPYTHON\python3.12\python.exe). Then open the search and type: "edit system environment variables", open it, then click "environment variables". Under "system variables" there's one called Path, double-click it to open another window, and the full path to your python.exe and save it.)
- Restart your terminal (and maybe your computer) after installing.
- Try `python3` or `py` if `python` still isn’t found.

---

## 2. "No module named *fer*" or weird `pkg_resources` errors
- I fought with fer for a few hours to get it to work correctly, so remember to use the pinned dependencies exactly as they are:
```bash
pip install -r requirements.txt
```
- This installs `fer==22.0.1` and `setuptools==68.2.2`, which are known to work together (lowky fckn fr).
- If you still see `ModuleNotFoundError: No module named 'pkg_resources'`, run:
```bash
pip install --upgrade setuptools==68.2.2
```

---

## 3. FFmpeg isn’t found
- Download FFmpeg from [ffmpeg.org](https://ffmpeg.org) and add its `bin` folder to your PATH.
- On Windows, the quickest fix is to put `ffmpeg.exe` directly into the project folder (next to `main.py`).
- Restart your terminal after adding FFmpeg to PATH.

---

## 4. Chat downloading fails
- Did you download it? Forgot? Fair. Download **TwitchDownloaderCLI** from [its releases page](https://github.com/lay295/TwitchDownloader/releases).
- Place `TwitchDownloaderCLI.exe` inside the `tools/` folder (create the folder if it doesn’t exist).
- No installation needed — the tool will find it automatically.

---

## 5. My AMD GPU isn’t being used for transcription
- You need a ***Vulkan‑enabled*** **`whisper-cli.exe`** and a GGML model inside `models/`.
- Install the **Vulkan Runtime** from [vulkan.lunarg.com](https://vulkan.lunarg.com) (choose the **Runtime** installer).  
- Reboot your PC, then run a quick manual test:
```bash
tools\whisper-cli.exe -m models\ggml-base.en.bin -f data\audio_chunks\some.wav -oj -of test --split-on-word --max-len 0 --language en
```
- Look for the line `ggml_vulkan: Found 1 Vulkan devices` — if you see it, your GPU is detected and ready.
- Don’t forget to set the backend in `config.yaml`:
```yaml
transcription:
  backend: whisper-cpp
```

---

## 6. Visual scoring (face emotions) is disabled, not working, or crashes
- Make sure `visual_enabled: true` is set under `detector:` in `config.yaml`.
- If the console warns you that `fer` is not installed:
```bash
pip install fer==22.0.1
```
- If you see a buttload of TensorFlow or LiteRT warnings, don't worry, they are **harmless**. You can silence them by following the tips in the main README.

---

## 7. The tool seems stuck during transcription
- The very first run downloads the Whisper model (a few hundred MB). Let it finish — it only happens once.
- If you’re using whisper.cpp, **delete any old empty transcript cache files** in `data/transcripts/` (look for `*_words_cpp.json`). Run with `--force` once to rebuild them.

---

## 8. Where are my F*&^$N' clips?
They’re inside `data/clips/<vod_id>/`. Each clip comes with a companion `_insights.json` file that includes editing tips, keywords, and category suggestions.

---

## 9. I want vertical (TikTok / Shorts) output
In `config.yaml` set:
```yaml
output:
  vertical: true
```
Or simply add `--vertical` when running the tool.

---

## 10. Nothing is working and I’m about to throw my PC (and maybe myself) out the window
I'm 1000% with you. Have you eaten yet today? Drank water? Took a dump? Take a breather, then try again:
1. Open the project folder in VS Code (or any editor) and make sure you’re inside the correct Python environment.
2. Run the tool with `--force` to ignore all caches:
```bash
python main.py --force "your-vod-url"
```
3. If it still fails—and if you're this far down *it probably will*—**open an issue on GitHub**. Paste the entire terminal output and anything else you've tried (explitives accepted) and tell me what's going on. I’ll help you fix it or I too will throw myself (and my PC) out a window.

---

##### Made with ❤️ by JamesBaughnd, a fellow streamer who also hates editing 6-hour gdmn vods.