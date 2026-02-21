#!/usr/bin/env python3
"""
Subtitle Maker — Instagram-style subtitle generator

Place your media files in a subfolder, then run:
    python subtitle_maker.py <folder_name>

Folder structure:
    subtitles/<folder>/
        ├── video.mp4  (or audio.mp3, .wav, etc.)
        └── lyrics.txt (optional — lyrics or spoken text)

Output:
    - <folder>/output.srt   (always created)
    - <folder>/output.ass   (always created, styled)
    - <folder>/<name>_subtitled.mp4  (only when input is video)
"""

import os
import sys
import subprocess
import argparse
import json
import re
import math
import shutil
import tempfile
from pathlib import Path
from datetime import timedelta

# ──────────────────────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────────────────────
WORDS_PER_CHUNK = 3
WHISPER_MODEL = "base"
SILENCE_GAP_THRESHOLD = 0.4      # seconds — gap between words that triggers a chunk break
PRE_DISPLAY_OFFSET = 0.05        # seconds — how early subtitle appears before the word is sung

# ASS subtitle styling (Instagram look)
FONT_NAME = "Impact"
FONT_SIZE_RATIO = 0.055          # font size as fraction of video height
SUBTITLE_POSITION_RATIO = 0.18   # MarginV as fraction of height (≈75 % from top)
TEXT_COLOUR = "&H00FFFFFF"       # white  (ASS AABBGGRR)
OUTLINE_COLOUR = "&H40000000"    # semi-transparent black outline
SHADOW_COLOUR = "&HA0000000"     # drop-shadow colour
OUTLINE_WIDTH = 4
SHADOW_DEPTH = 3

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.opus'}
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v'}
TEXT_EXTENSIONS  = {'.txt', '.lrc', '.text', '.lyrics'}

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def find_media_file(folder: Path):
    """Return the first audio / video file found in *folder*."""
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in (AUDIO_EXTENSIONS | VIDEO_EXTENSIONS):
            return f
    return None


def find_text_file(folder: Path):
    """Return the first lyrics / text file found in *folder*."""
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in TEXT_EXTENSIONS:
            return f
    return None


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def get_video_info(path: Path) -> dict:
    """Get video width, height, duration using ffmpeg -i (no ffprobe needed)."""
    # Try ffprobe first (if available)
    if shutil.which("ffprobe"):
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(path),
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(out.stdout)
            for s in data.get("streams", []):
                if s.get("codec_type") == "video":
                    return {
                        "width":  int(s["width"]),
                        "height": int(s["height"]),
                        "duration": float(s.get("duration", 0)),
                    }
        except Exception:
            pass

    # Fallback: parse ffmpeg -i stderr output
    cmd = ["ffmpeg", "-i", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True)
        # Parse "Stream ... Video: ... 1920x1080"
        m = re.search(r"(\d{2,5})x(\d{2,5})", out.stderr)
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            # Parse duration "Duration: HH:MM:SS.FF"
            dm = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", out.stderr)
            dur = 0.0
            if dm:
                dur = int(dm.group(1)) * 3600 + int(dm.group(2)) * 60 + float(dm.group(3))
            return {"width": w, "height": h, "duration": dur}
    except Exception:
        pass

    return {"width": 1080, "height": 1920, "duration": 0}   # default (portrait)


# ──────────────────────────────────────────────────────────────
# Whisper transcription
# ──────────────────────────────────────────────────────────────

def transcribe(media_path: Path, model_size: str = "base"):
    """
    Transcribe with OpenAI Whisper and return a list of
    {'word': str, 'start': float, 'end': float} dicts.
    """
    import whisper

    print(f"    Loading Whisper model '{model_size}' …")
    model = whisper.load_model(model_size)

    print(f"    Transcribing (this may take a while) …")
    result = model.transcribe(
        str(media_path),
        word_timestamps=True,
        verbose=False,
    )

    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append({
                "word":  w["word"].strip(),
                "start": w["start"],
                "end":   w["end"],
            })

    full_text = result.get("text", "")
    return words, full_text


# ──────────────────────────────────────────────────────────────
# Lyrics alignment
# ──────────────────────────────────────────────────────────────

def clean_lyrics(text: str) -> list[str]:
    """Split lyrics text into a flat list of words, stripping blanks."""
    # Remove common section markers like [Verse 1], [Chorus], etc.
    text = re.sub(r"\[.*?\]", "", text)
    words = text.split()
    return [w for w in words if w.strip()]


def parse_lyrics_lines(text: str) -> list[list[str]]:
    """
    Parse lyrics into a list of lines, each line being a list of words.
    Blank lines and section markers are removed.  Preserves line structure
    so we can insert silence breaks between lines.
    """
    text = re.sub(r"\[.*?\]", "", text)
    lines = []
    for raw_line in text.splitlines():
        words = raw_line.split()
        words = [w for w in words if w.strip()]
        if words:
            lines.append(words)
    return lines


def align_lyrics(lyrics_words: list, whisper_words: list,
                 lyrics_lines=None) -> list:
    """
    Map provided lyrics onto the timeline produced by Whisper.

    Strategy: align LINE-by-LINE.  Each lyrics line is mapped to a
    proportional slice of Whisper words.  Within that slice, each
    lyrics word gets exactly one Whisper word's start/end timestamp.
    This preserves natural silences between lines because the gaps
    live between slices.

    If *lyrics_lines* is provided, a linebreak marker is injected
    after each line so that `chunk_words` never merges words across
    lines.
    """
    if not whisper_words:
        dur = 60.0
        step = dur / max(len(lyrics_words), 1)
        return [
            {"word": w, "start": i * step, "end": (i + 1) * step}
            for i, w in enumerate(lyrics_words)
        ]

    n_lyr = len(lyrics_words)
    n_whi = len(whisper_words)

    # ── If we have lyrics_lines, do line-by-line alignment ────
    if lyrics_lines:
        result = []
        total_lyr_words = sum(len(line) for line in lyrics_lines)

        # Distribute Whisper words across lyrics lines proportionally
        wi = 0  # running Whisper index
        for line_num, line_words in enumerate(lyrics_lines):
            n_line = len(line_words)
            if n_line == 0:
                continue

            # How many Whisper words this line gets (proportional)
            remaining_lyrics = total_lyr_words - sum(
                len(lyrics_lines[k]) for k in range(line_num)
            )
            remaining_whi = n_whi - wi
            if remaining_lyrics > 0:
                n_slice = max(n_line, round(n_line * remaining_whi / remaining_lyrics))
            else:
                n_slice = n_line
            n_slice = min(n_slice, remaining_whi)
            n_slice = max(n_slice, n_line)       # at least one Whisper word per lyrics word
            n_slice = min(n_slice, remaining_whi) # can't exceed what's left

            whisper_slice = whisper_words[wi : wi + n_slice]

            # Map each lyrics word in this line to a Whisper word in the slice
            for j, lw in enumerate(line_words):
                idx = min(int(j * len(whisper_slice) / n_line), len(whisper_slice) - 1)
                result.append({
                    "word":  lw,
                    "start": whisper_slice[idx]["start"],
                    "end":   whisper_slice[idx]["end"],
                })

            wi += n_slice

            # Insert linebreak marker after each line (except the last)
            if line_num < len(lyrics_lines) - 1 and result:
                last = result[-1]
                result.append({
                    "word": "",
                    "start": last["end"],
                    "end":   last["end"],
                    "_linebreak": True,
                })

        return result

    # ── Fallback: flat proportional mapping (no lyrics_lines) ─
    aligned = []
    for i, lw in enumerate(lyrics_words):
        idx = min(int(i * n_whi / n_lyr), n_whi - 1)
        aligned.append({
            "word":  lw,
            "start": whisper_words[idx]["start"],
            "end":   whisper_words[idx]["end"],
        })
    return aligned


# ──────────────────────────────────────────────────────────────
# Chunking (2-3 words per subtitle)
# ──────────────────────────────────────────────────────────────

def chunk_words(words: list[dict], size: int = 3,
                gap_threshold: float = SILENCE_GAP_THRESHOLD,
                pre_display: float = PRE_DISPLAY_OFFSET) -> list[dict]:
    """
    Group timestamped words into chunks of up to *size* words.

    Silence-aware: if there is a gap >= *gap_threshold* seconds between
    two consecutive words, the current chunk is closed even if it has
    fewer than *size* words.  This keeps the screen blank during silence.

    *pre_display* controls how many seconds before the first word the
    subtitle appears (set to 0 for exact sync).
    """
    chunks = []
    i = 0
    n = len(words)

    while i < n:
        # Skip linebreak markers
        if words[i].get("_linebreak"):
            i += 1
            continue

        group = [words[i]]
        j = i + 1

        while j < n and len(group) < size:
            # A linebreak marker forces the chunk to close
            if words[j].get("_linebreak"):
                break
            # Check for silence gap between end of last word and start of next
            gap = words[j]["start"] - words[j - 1]["end"]
            if gap >= gap_threshold:
                break                       # silence detected — close chunk
            group.append(words[j])
            j += 1

        text  = " ".join(w["word"] for w in group)
        start = max(0, group[0]["start"] - pre_display)
        end   = group[-1]["end"]

        # enforce a minimum display time
        if end - start < 0.15:
            end = start + 0.15

        chunks.append({"text": text, "start": start, "end": end})
        i = j

    # remove overlaps (keep each chunk's end <= next chunk's start)
    for k in range(1, len(chunks)):
        if chunks[k]["start"] < chunks[k - 1]["end"]:
            chunks[k - 1]["end"] = chunks[k]["start"]
        if chunks[k - 1]["end"] <= chunks[k - 1]["start"]:
            chunks[k - 1]["end"] = chunks[k - 1]["start"] + 0.05

    return chunks


# ──────────────────────────────────────────────────────────────
# SRT generation
# ──────────────────────────────────────────────────────────────

def _ts_srt(seconds: float) -> str:
    """HH:MM:SS,mmm"""
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(chunks: list[dict], path: Path):
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(str(i))
        lines.append(f"{_ts_srt(c['start'])} --> {_ts_srt(c['end'])}")
        lines.append(c["text"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# ASS generation (styled)
# ──────────────────────────────────────────────────────────────

def _ts_ass(seconds: float) -> str:
    """H:MM:SS.CC"""
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def write_ass(chunks: list[dict], path: Path,
              width: int = 1080, height: int = 1920,
              font_size_ratio: float = FONT_SIZE_RATIO):
    """Write a fully styled ASS subtitle file."""
    font_size = max(24, int(height * font_size_ratio))
    margin_v  = max(30, int(height * SUBTITLE_POSITION_RATIO))

    header = (
        "[Script Info]\n"
        "Title: Instagram Style Subtitles\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{FONT_NAME},{font_size},"
        f"{TEXT_COLOUR},&H000000FF,{OUTLINE_COLOUR},{SHADOW_COLOUR},"
        f"-1,0,0,0,100,100,2,0,1,{OUTLINE_WIDTH},{SHADOW_DEPTH},"
        f"2,10,10,{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events = []
    for c in chunks:
        start = _ts_ass(c["start"])
        end   = _ts_ass(c["end"])
        text  = c["text"].upper()       # UPPERCASE for Instagram look
        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# Burn subtitles into video (ffmpeg)
# ──────────────────────────────────────────────────────────────

def burn_subtitles(video_path: Path, ass_path: Path, output_path: Path) -> bool:
    """Overlay ASS subtitles onto *video_path* → *output_path*."""

    # Copy ASS to a temp file with a safe name (avoids path-escaping issues)
    tmp_dir = tempfile.mkdtemp()
    tmp_ass = Path(tmp_dir) / "subs.ass"
    shutil.copy2(ass_path, tmp_ass)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"ass={tmp_ass}",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-c:a", "copy",
        str(output_path),
    ]

    print(f"    Running ffmpeg …")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # clean up
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if result.returncode != 0:
        print(f"    ✗ ffmpeg failed:\n{result.stderr[-1500:]}")
        return False

    return True


# ──────────────────────────────────────────────────────────────
# Dependency check
# ──────────────────────────────────────────────────────────────

def check_deps() -> list[str]:
    errors = []
    if shutil.which("ffmpeg") is None:
        errors.append("ffmpeg not found.  Install:  brew install ffmpeg")
    try:
        import whisper  # noqa: F401
    except ImportError:
        errors.append("openai-whisper not installed.  Run:  pip install openai-whisper")
    return errors


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Instagram-style subtitle generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python subtitle_maker.py my_song\n"
            "  python subtitle_maker.py my_video --model medium\n"
            "  python subtitle_maker.py my_project --words 2\n"
        ),
    )
    parser.add_argument("folder",
                        help="Sub-folder (inside this project) that contains your files")
    parser.add_argument("--model", default=WHISPER_MODEL,
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--words", type=int, default=WORDS_PER_CHUNK,
                        help="Words per subtitle chunk (default: 3)")
    parser.add_argument("--font-ratio", type=float, default=FONT_SIZE_RATIO,
                        help="Font-size as a fraction of video height (default: 0.055)")
    parser.add_argument("--gap", type=float, default=SILENCE_GAP_THRESHOLD,
                        help="Silence gap in seconds that triggers a chunk break (default: 0.4)")
    parser.add_argument("--pre-display", type=float, default=PRE_DISPLAY_OFFSET,
                        help="Seconds subtitle appears before the word is sung (default: 0.05)")
    parser.add_argument("--uppercase", action="store_true", default=True,
                        help="UPPERCASE subtitle text (default: on)")
    parser.add_argument("--no-uppercase", dest="uppercase", action="store_false",
                        help="Keep original casing")

    args = parser.parse_args()

    base_dir    = Path(__file__).resolve().parent
    folder_path = base_dir / args.folder

    if not folder_path.is_dir():
        print(f"✗ Folder not found: {folder_path}")
        print("  Create it and place your media (+ optional lyrics.txt) inside.")
        sys.exit(1)

    bar = "=" * 60
    print(bar)
    print("  SUBTITLE MAKER  ·  Instagram Style")
    print(bar)
    print(f"  Folder : {folder_path.name}/")

    # ── dependency check ──────────────────────────────────────
    errors = check_deps()
    if errors:
        print("\n  Missing dependencies:")
        for e in errors:
            print(f"    ✗ {e}")
        sys.exit(1)

    # ── locate files ──────────────────────────────────────────
    media_file = find_media_file(folder_path)
    if media_file is None:
        all_ext = sorted(AUDIO_EXTENSIONS | VIDEO_EXTENSIONS)
        print(f"\n  ✗ No media file found in {folder_path.name}/")
        print(f"    Supported: {', '.join(all_ext)}")
        sys.exit(1)

    video_mode = is_video(media_file)
    print(f"  Media  : {media_file.name}  ({'video' if video_mode else 'audio'})")

    text_file   = find_text_file(folder_path)
    lyrics_text = None
    if text_file:
        lyrics_text = text_file.read_text(encoding="utf-8").strip()
        print(f"  Lyrics : {text_file.name}  ({len(clean_lyrics(lyrics_text))} words)")
    else:
        print("  Lyrics : (none — will auto-transcribe)")

    total_steps = 3 if video_mode else 2

    # ── 1. Transcribe ────────────────────────────────────────
    print(f"\n  [{1}/{total_steps}] Transcribing with Whisper …")
    whisper_words, full_text = transcribe(media_file, args.model)
    print(f"    ✓ Whisper found {len(whisper_words)} words")

    if not whisper_words:
        print("    ✗ No speech detected — cannot continue.")
        sys.exit(1)

    # ── 2. Build word list ────────────────────────────────────
    if lyrics_text:
        lw = clean_lyrics(lyrics_text)
        lyrics_lines = parse_lyrics_lines(lyrics_text)
        print(f"\n  [{2}/{total_steps}] Aligning {len(lw)} lyrics words with audio …")
        words = align_lyrics(lw, whisper_words, lyrics_lines=lyrics_lines)
    else:
        print(f"\n  [{2}/{total_steps}] Preparing subtitles …")
        words = whisper_words

    # ── chunk & write ─────────────────────────────────────────
    chunks = chunk_words(words, size=args.words,
                         gap_threshold=args.gap,
                         pre_display=args.pre_display)
    if not args.uppercase:
        pass  # keep casing as-is; ASS writer will uppercase when flag is on
    print(f"    ✓ {len(chunks)} subtitle chunks  ({args.words} words each)")

    srt_path = folder_path / "output.srt"
    ass_path = folder_path / "output.ass"

    write_srt(chunks, srt_path)
    print(f"    ✓ SRT saved  → {srt_path.name}")

    # for ASS we need resolution
    if video_mode:
        vinfo = get_video_info(media_file)
        w, h  = vinfo["width"], vinfo["height"]
    else:
        w, h = 1080, 1920   # sensible default for portrait

    # Patch ASS writer to respect --no-uppercase flag
    _orig_chunks = chunks
    if not args.uppercase:
        for c in _orig_chunks:
            c["_keep_case"] = True

    write_ass(chunks, ass_path, width=w, height=h, font_size_ratio=args.font_ratio)
    print(f"    ✓ ASS saved  → {ass_path.name}")

    # ── 3. Burn into video ────────────────────────────────────
    if video_mode:
        print(f"\n  [{3}/{total_steps}] Burning subtitles into video …")
        out_name   = f"{media_file.stem}_subtitled{media_file.suffix}"
        video_out  = folder_path / out_name

        ok = burn_subtitles(media_file, ass_path, video_out)
        if ok:
            print(f"    ✓ Video saved → {out_name}")
        else:
            print("    ✗ Video encoding failed (SRT/ASS files are still usable)")
    else:
        print("\n  Audio-only input — SRT & ASS generated (no video output).")

    # ── done ──────────────────────────────────────────────────
    print(f"\n{bar}")
    print("  DONE ✓")
    print(f"    SRT : {srt_path}")
    print(f"    ASS : {ass_path}")
    if video_mode:
        print(f"    VID : {folder_path / f'{media_file.stem}_subtitled{media_file.suffix}'}")
    print(bar)


if __name__ == "__main__":
    main()
