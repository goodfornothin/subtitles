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
from difflib import SequenceMatcher
from pathlib import Path
from datetime import timedelta

# ──────────────────────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────────────────────
WORDS_PER_CHUNK = 3
WHISPER_MODEL = "base"
SILENCE_GAP_THRESHOLD = 0.4      # seconds — gap between words that triggers a chunk break
PRE_DISPLAY_OFFSET = 0.15        # seconds — how early subtitle appears before the word is sung

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
# Lyrics alignment  (DTW with fuzzy matching)
# ──────────────────────────────────────────────────────────────

def clean_lyrics(text: str) -> list:
    """Split lyrics text into a flat list of words, stripping blanks."""
    text = re.sub(r"\[.*?\]", "", text)
    words = text.split()
    return [w for w in words if w.strip()]


def parse_lyrics_lines(text: str) -> list:
    """
    Parse lyrics into a list of lines, each line being a list of words.
    Blank lines and section markers are removed.
    """
    text = re.sub(r"\[.*?\]", "", text)
    lines = []
    for raw_line in text.splitlines():
        words = raw_line.split()
        words = [w for w in words if w.strip()]
        if words:
            lines.append(words)
    return lines


def _flat_lyrics(lines):
    """Flatten lyrics lines into a single word list with line indices."""
    result = []
    for line_idx, line in enumerate(lines):
        for word_idx, word in enumerate(line):
            result.append({
                "word": word,
                "line_idx": line_idx,
                "word_idx": word_idx,
                "is_line_end": word_idx == len(line) - 1,
            })
    return result


def _word_similarity(a, b):
    """Compute similarity between two words (0.0 to 1.0)."""
    a_clean = re.sub(r"[^a-z]", "", a.lower())
    b_clean = re.sub(r"[^a-z]", "", b.lower())
    if not a_clean or not b_clean:
        return 0.0
    if a_clean == b_clean:
        return 1.0
    ratio = SequenceMatcher(None, a_clean, b_clean).ratio()
    if a_clean.startswith(b_clean) or b_clean.startswith(a_clean):
        ratio = max(ratio, 0.7)
    return ratio


def _merged_word_similarity(lyrics_words, whisper_word):
    """Check if a Whisper word is a merge of multiple lyrics words."""
    concat = "".join(re.sub(r"[^a-z]", "", w.lower()) for w in lyrics_words)
    w_clean = re.sub(r"[^a-z]", "", whisper_word.lower())
    if not concat or not w_clean:
        return 0.0
    return SequenceMatcher(None, concat, w_clean).ratio()


def _split_word_similarity(lyrics_word, whisper_words):
    """Check if multiple Whisper words correspond to a single lyrics word."""
    concat = "".join(re.sub(r"[^a-z]", "", w.lower()) for w in whisper_words)
    l_clean = re.sub(r"[^a-z]", "", lyrics_word.lower())
    if not concat or not l_clean:
        return 0.0
    return SequenceMatcher(None, l_clean, concat).ratio()


def _preprocess_whisper(whisper_words):
    """Merge hyphenated splits (e.g. 'star' + '-shaped' → 'star-shaped')."""
    merged = []
    for w in whisper_words:
        if w["word"].startswith("-") and merged:
            prev = merged[-1]
            prev["word"] = prev["word"] + w["word"]
            prev["end"] = w["end"]
        else:
            merged.append(dict(w))
    return merged


def align_lyrics(lyrics_words, whisper_words, lyrics_lines=None):
    """
    Map provided lyrics onto the timeline produced by Whisper using
    Dynamic Time Warping with fuzzy text matching.

    Handles word splits, merges, and mishearings automatically.
    """
    # Pre-process: merge hyphenated Whisper splits
    whisper_words = _preprocess_whisper(whisper_words)

    if not whisper_words:
        dur = 60.0
        step = dur / max(len(lyrics_words), 1)
        return [
            {"word": w, "start": i * step, "end": (i + 1) * step,
             "line_idx": 0, "is_line_end": i == len(lyrics_words) - 1}
            for i, w in enumerate(lyrics_words)
        ]

    # Build flat lyrics with line info
    if lyrics_lines:
        flat = _flat_lyrics(lyrics_lines)
    else:
        flat = [{"word": w, "line_idx": 0, "word_idx": i,
                 "is_line_end": i == len(lyrics_words) - 1}
                for i, w in enumerate(lyrics_words)]

    n = len(flat)
    m = len(whisper_words)

    # DP table: dp[i][j] = (cost, backpointer)
    INF = float("inf")
    dp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = (0.0, None)
    for j in range(1, m + 1):
        dp[0][j] = (j * 0.1, (0, 1))
    for i in range(1, n + 1):
        dp[i][0] = (i * 2.0, (1, 0))

    for i in range(1, n + 1):
        lw = flat[i - 1]["word"]
        for j in range(1, m + 1):
            ww = whisper_words[j - 1]["word"]
            candidates = []

            # Match
            sim = _word_similarity(lw, ww)
            match_cost = (1.0 - sim) * 2.0
            if dp[i - 1][j - 1] is not None:
                candidates.append((dp[i - 1][j - 1][0] + match_cost, (1, 1)))
            # Skip lyrics word
            if dp[i - 1][j] is not None:
                candidates.append((dp[i - 1][j][0] + 1.5, (1, 0)))
            # Skip Whisper word
            if dp[i][j - 1] is not None:
                candidates.append((dp[i][j - 1][0] + 0.3, (0, 1)))
            # Merge: 2 lyrics words → 1 Whisper word
            if i >= 2 and dp[i - 2][j - 1] is not None:
                merge_sim = _merged_word_similarity(
                    [flat[i - 2]["word"], lw], ww)
                if merge_sim > 0.4:
                    candidates.append(
                        (dp[i - 2][j - 1][0] + (1.0 - merge_sim) * 1.5, (2, 1)))
            # Split: 1 lyrics word → 2 Whisper words
            if j >= 2 and dp[i - 1][j - 2] is not None:
                split_sim = _split_word_similarity(
                    lw, [whisper_words[j - 2]["word"], ww])
                if split_sim > 0.4:
                    candidates.append(
                        (dp[i - 1][j - 2][0] + (1.0 - split_sim) * 1.5, (1, 2)))

            if candidates:
                dp[i][j] = min(candidates, key=lambda x: x[0])

    # Traceback
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0 and j > 0:
            j -= 1
            continue
        if j == 0 and i > 0:
            alignment.append((i - 1, None, None))
            i -= 1
            continue
        cell = dp[i][j]
        if cell is None:
            i -= 1
            j -= 1
            continue
        _, bp = cell
        if bp == (1, 1):
            alignment.append((i - 1, j - 1, j - 1))
            i -= 1
            j -= 1
        elif bp == (1, 0):
            alignment.append((i - 1, None, None))
            i -= 1
        elif bp == (0, 1):
            j -= 1
        elif bp == (2, 1):
            alignment.append((i - 1, j - 1, j - 1))
            alignment.append((i - 2, j - 1, j - 1))
            i -= 2
            j -= 1
        elif bp == (1, 2):
            alignment.append((i - 1, j - 2, j - 1))
            i -= 1
            j -= 2
        else:
            i -= 1
            j -= 1
    alignment.reverse()

    # Build result
    result = []
    for (li, wj_start, wj_end) in alignment:
        entry = flat[li]
        if wj_start is not None:
            start = whisper_words[wj_start]["start"]
            end = whisper_words[wj_end]["end"]
        else:
            start = None
            end = None
        result.append({
            "word": entry["word"],
            "start": start,
            "end": end,
            "line_idx": entry["line_idx"],
            "is_line_end": entry["is_line_end"],
        })

    _interpolate_missing(result)
    _spread_shared_timestamps(result)
    _fix_cross_line_timestamps(result)
    return result


def _interpolate_missing(aligned):
    """Fill in None timestamps by interpolating from neighbors."""
    n = len(aligned)
    for i in range(n):
        if aligned[i]["start"] is None:
            prev_end = 0.0
            for k in range(i - 1, -1, -1):
                if aligned[k]["end"] is not None:
                    prev_end = aligned[k]["end"]
                    break
            next_start = prev_end + 1.0
            for k in range(i + 1, n):
                if aligned[k]["start"] is not None:
                    next_start = aligned[k]["start"]
                    break
            count = 0
            for k in range(i, n):
                if aligned[k]["start"] is None:
                    count += 1
                else:
                    break
            step = (next_start - prev_end) / max(count, 1)
            for k in range(count):
                aligned[i + k]["start"] = prev_end + k * step
                aligned[i + k]["end"] = prev_end + (k + 1) * step


def _spread_shared_timestamps(aligned):
    """Split shared timestamps evenly among consecutive words."""
    n = len(aligned)
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(aligned[j]["start"] - aligned[i]["start"]) < 0.01:
            j += 1
        run_len = j - i
        if run_len > 1:
            start = aligned[i]["start"]
            end = aligned[j - 1]["end"]
            step = (end - start) / run_len
            for k in range(run_len):
                aligned[i + k]["start"] = start + k * step
                aligned[i + k]["end"] = start + (k + 1) * step
        i = j


def _fix_cross_line_timestamps(aligned):
    """Fix words at line starts that inherit the previous line's timestamp."""
    n = len(aligned)
    for i in range(1, n):
        if aligned[i]["line_idx"] != aligned[i - 1]["line_idx"]:
            if aligned[i]["start"] <= aligned[i - 1]["end"] + 0.05:
                new_start = aligned[i - 1]["end"] + 0.05
                if new_start < aligned[i]["end"]:
                    aligned[i]["start"] = new_start


# ──────────────────────────────────────────────────────────────
# Chunking (2-3 words per subtitle, line-aware)
# ──────────────────────────────────────────────────────────────

def _balanced_chunk_sizes(n, max_size=3):
    """
    Divide n words into chunks of up to max_size, avoiding 1-word orphans.
    E.g. 7 → [3,2,2], 4 → [2,2], 8 → [3,3,2].
    """
    if n <= max_size:
        return [n]
    full, rem = divmod(n, max_size)
    if rem == 0:
        return [max_size] * full
    if rem >= 2:
        return [max_size] * full + [rem]
    return [max_size] * (full - 1) + [max_size - 1, 2]


def chunk_words(words, size=3, gap_threshold=SILENCE_GAP_THRESHOLD,
                pre_display=PRE_DISPLAY_OFFSET):
    """
    Group timestamped words into chunks of up to *size* words.
    Line-aware: distributes words within each line to avoid orphans.
    Silence-aware: gaps >= *gap_threshold* force chunk breaks.
    """
    # Group words by line
    lines = []
    current_line = []
    current_line_idx = words[0].get("line_idx", 0) if words else -1

    for w in words:
        # Skip old-style linebreak markers
        if w.get("_linebreak"):
            continue
        li = w.get("line_idx", 0)
        if li != current_line_idx:
            if current_line:
                lines.append(current_line)
            current_line = [w]
            current_line_idx = li
        else:
            current_line.append(w)
    if current_line:
        lines.append(current_line)

    chunks = []
    for line_words in lines:
        # Split by silence gaps within the line (but tolerate more for
        # the first word to avoid single-word orphans at line start)
        segments = []
        seg = [line_words[0]]
        for k in range(1, len(line_words)):
            gap = line_words[k]["start"] - line_words[k - 1]["end"]
            effective_gap = gap_threshold if len(seg) > 1 else gap_threshold * 3
            if gap >= effective_gap:
                segments.append(seg)
                seg = [line_words[k]]
            else:
                seg.append(line_words[k])
        segments.append(seg)

        for seg_words in segments:
            chunk_sizes = _balanced_chunk_sizes(len(seg_words), size)
            idx = 0
            for cs in chunk_sizes:
                group = seg_words[idx : idx + cs]
                text = " ".join(w["word"] for w in group)
                raw_start = group[0]["start"]
                if raw_start < 0.3:
                    start = raw_start
                else:
                    start = max(0, raw_start - pre_display)
                end = group[-1]["end"]
                if end - start < 0.2:
                    end = start + 0.3
                chunks.append({"text": text, "start": start, "end": end})
                idx += cs

    # Remove overlaps
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
