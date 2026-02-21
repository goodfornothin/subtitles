# Subtitle Maker — Instagram-Style Subtitles

Generates **2–3 word** Instagram-style subtitles from audio or video files,
with bold uppercase text, drop shadow, and positioning at ¾ down the screen.

---

## Quick Start

```bash
# 1. Install dependencies (one-time)
pip install -r requirements.txt
brew install ffmpeg          # if not already installed

# 2. Create a project folder and put your files in it
mkdir my_song
cp ~/Downloads/song.mp4 my_song/
cp ~/Downloads/lyrics.txt my_song/   # optional

# 3. Run
python subtitle_maker.py my_song
```

## Folder Layout

Place your files inside a subfolder of this project:

```
subtitles/
├── subtitle_maker.py
├── requirements.txt
├── my_song/                 ← your project folder
│   ├── video.mp4            ← audio OR video file (required)
│   ├── lyrics.txt           ← lyrics / spoken text (optional)
│   │
│   ├── output.srt           ← generated SRT subtitle file
│   ├── output.ass           ← generated styled ASS subtitle file
│   └── video_subtitled.mp4  ← generated video with burned-in subtitles
```

### Input

| File | Required | Notes |
|------|----------|-------|
| **Audio / Video** | ✅ | `.mp4 .mkv .mov .avi .webm .mp3 .wav .flac .aac .ogg .m4a` |
| **Lyrics text** | ❌ | `.txt .lyrics .text .lrc` — if omitted, Whisper auto-transcribes |

### Output

| File | When |
|------|------|
| `output.srt` | Always |
| `output.ass` | Always (styled for Instagram look) |
| `<name>_subtitled.mp4` | Only when input is a video |

## Options

```
python subtitle_maker.py <folder> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `base` | Whisper model: `tiny`, `base`, `small`, `medium`, `large` |
| `--words` | `3` | Words per subtitle chunk (2 or 3 recommended) |
| `--font-ratio` | `0.055` | Font size as fraction of video height |
| `--no-uppercase` | off | Keep original letter casing |

### Examples

```bash
# Auto-transcribe, 3 words per subtitle
python subtitle_maker.py my_video

# Use provided lyrics, 2 words at a time, larger Whisper model
python subtitle_maker.py my_song --model medium --words 2

# Bigger font
python subtitle_maker.py my_reel --font-ratio 0.07
```

## How It Works

1. **Whisper** transcribes the audio and produces **word-level timestamps**.
2. If you provide a `lyrics.txt`, the lyrics words are mapped onto Whisper's
   timeline (proportional alignment).
3. Words are grouped into **2–3 word chunks** with proper timing.
4. An **SRT** and a styled **ASS** file are written.
5. For video input, **ffmpeg** burns the ASS subtitles directly into the video
   with Impact font, white text, drop shadow, and ¾-down positioning.

## Style

The subtitles are designed to mimic Instagram Reel / TikTok captions:

- **Impact** bold font
- **WHITE UPPERCASE** text
- Black **drop shadow** + outline
- Positioned at roughly **75 % from the top** of the frame
- **2–3 words** shown at a time for readability

## Requirements

- Python 3.10+
- `openai-whisper` (+ PyTorch)
- `ffmpeg` with **libass** support (standard Homebrew/apt build)
