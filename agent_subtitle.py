#!/usr/bin/env python3
"""
Agent Subtitle — Intelligent subtitle generator with fuzzy DTW alignment.

Uses Dynamic Time Warping with text similarity to properly align lyrics
to Whisper transcription, handling word splits, merges, and mishearings.
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from difflib import SequenceMatcher

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
WORDS_PER_CHUNK = 3
SILENCE_GAP_THRESHOLD = 0.4
PRE_DISPLAY_OFFSET = 0.15

FONT_NAME = "Impact"
FONT_SIZE_RATIO = 0.055
SUBTITLE_POSITION_RATIO = 0.18
TEXT_COLOUR = "&H00FFFFFF"
OUTLINE_COLOUR = "&H40000000"
SHADOW_COLOUR = "&HA0000000"
OUTLINE_WIDTH = 4
SHADOW_DEPTH = 3


# ──────────────────────────────────────────────────────────────
# Step 1 — Load & preprocess Whisper words
# ──────────────────────────────────────────────────────────────

def load_whisper_words(json_path):
    """Load and preprocess Whisper words: merge hyphenated splits."""
    with open(json_path) as f:
        raw = json.load(f)

    # Merge words starting with "-" into the preceding word
    merged = []
    for w in raw:
        if w["word"].startswith("-") and merged:
            prev = merged[-1]
            prev["word"] = prev["word"] + w["word"]
            prev["end"] = w["end"]
        else:
            merged.append(dict(w))  # copy

    return merged


# ──────────────────────────────────────────────────────────────
# Step 2 — Parse lyrics
# ──────────────────────────────────────────────────────────────

def parse_lyrics(text):
    """Parse lyrics into lines of words, stripping section markers."""
    text = re.sub(r"\[.*?\]", "", text)
    lines = []
    for raw in text.splitlines():
        words = raw.split()
        words = [w for w in words if w.strip()]
        if words:
            lines.append(words)
    return lines


def flat_lyrics(lines):
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


# ──────────────────────────────────────────────────────────────
# Step 3 — DTW alignment with fuzzy matching
# ──────────────────────────────────────────────────────────────

def word_similarity(a, b):
    """
    Compute similarity between two words (0.0 to 1.0).
    Handles common Whisper mishearings.
    """
    a_clean = re.sub(r"[^a-z]", "", a.lower())
    b_clean = re.sub(r"[^a-z]", "", b.lower())

    if not a_clean or not b_clean:
        return 0.0

    # Exact match
    if a_clean == b_clean:
        return 1.0

    # Use SequenceMatcher for fuzzy matching
    ratio = SequenceMatcher(None, a_clean, b_clean).ratio()

    # Boost if one is a prefix/suffix of the other
    if a_clean.startswith(b_clean) or b_clean.startswith(a_clean):
        ratio = max(ratio, 0.7)

    return ratio


def merged_word_similarity(lyrics_words, whisper_word):
    """
    Check if a single Whisper word is actually a merge of multiple
    lyrics words (e.g., "brimney" ≈ "brave" + "new").
    Returns similarity score against concatenation.
    """
    concat = "".join(re.sub(r"[^a-z]", "", w.lower()) for w in lyrics_words)
    w_clean = re.sub(r"[^a-z]", "", whisper_word.lower())
    if not concat or not w_clean:
        return 0.0
    return SequenceMatcher(None, concat, w_clean).ratio()


def split_word_similarity(lyrics_word, whisper_words):
    """
    Check if multiple Whisper words correspond to a single lyrics word
    (e.g., "earth's" + "bounce" ≈ "earthbound").
    """
    concat = "".join(re.sub(r"[^a-z]", "", w.lower()) for w in whisper_words)
    l_clean = re.sub(r"[^a-z]", "", lyrics_word.lower())
    if not concat or not l_clean:
        return 0.0
    return SequenceMatcher(None, l_clean, concat).ratio()


def dtw_align(lyrics_flat, whisper_words):
    """
    Align lyrics words to Whisper words using Dynamic Time Warping.

    Returns a list of aligned entries:
      {"word": str, "start": float, "end": float,
       "line_idx": int, "is_line_end": bool}
    """
    n = len(lyrics_flat)
    m = len(whisper_words)

    # Cost matrix: cost[i][j] = best cost to align lyrics[0..i-1] with whisper[0..j-1]
    INF = float("inf")

    # We use a DP table where:
    #   dp[i][j] = (cost, backpointer)
    # Backpointer types:
    #   (1,1) = match lyrics[i-1] to whisper[j-1]
    #   (1,0) = skip lyrics[i-1] (no Whisper match — interpolate later)
    #   (0,1) = skip whisper[j-1] (extra Whisper word)
    #   (2,1) = merge: lyrics[i-2..i-1] matched to whisper[j-1]
    #   (1,2) = split: lyrics[i-1] matched to whisper[j-2..j-1]

    dp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = (0.0, None)

    # Initialize: skipping Whisper words at start costs little
    for j in range(1, m + 1):
        dp[0][j] = (j * 0.1, (0, 1))

    # Skipping lyrics words costs more (they MUST appear)
    for i in range(1, n + 1):
        dp[i][0] = (i * 2.0, (1, 0))

    for i in range(1, n + 1):
        lw = lyrics_flat[i - 1]["word"]
        for j in range(1, m + 1):
            ww = whisper_words[j - 1]["word"]
            candidates = []

            # Option 1: Match lyrics[i-1] to whisper[j-1]
            sim = word_similarity(lw, ww)
            match_cost = (1.0 - sim) * 2.0  # low cost for good match
            if dp[i - 1][j - 1] is not None:
                candidates.append((dp[i - 1][j - 1][0] + match_cost, (1, 1)))

            # Option 2: Skip lyrics word (costly — lyrics must appear)
            if dp[i - 1][j] is not None:
                candidates.append((dp[i - 1][j][0] + 1.5, (1, 0)))

            # Option 3: Skip Whisper word (cheap — extra detection)
            if dp[i][j - 1] is not None:
                candidates.append((dp[i][j - 1][0] + 0.3, (0, 1)))

            # Option 4: Merge — 2 lyrics words → 1 Whisper word
            # (e.g., "brave new" → "brimney")
            if i >= 2 and dp[i - 2][j - 1] is not None:
                prev_lw = lyrics_flat[i - 2]["word"]
                merge_sim = merged_word_similarity([prev_lw, lw], ww)
                if merge_sim > 0.4:
                    merge_cost = (1.0 - merge_sim) * 1.5
                    candidates.append((dp[i - 2][j - 1][0] + merge_cost, (2, 1)))

            # Option 5: Split — 1 lyrics word → 2 Whisper words
            # (e.g., "earthbound" → "earth's" + "bounce")
            if j >= 2 and dp[i - 1][j - 2] is not None:
                prev_ww = whisper_words[j - 2]["word"]
                split_sim = split_word_similarity(lw, [prev_ww, ww])
                if split_sim > 0.4:
                    split_cost = (1.0 - split_sim) * 1.5
                    candidates.append((dp[i - 1][j - 2][0] + split_cost, (1, 2)))

            if candidates:
                best = min(candidates, key=lambda x: x[0])
                dp[i][j] = best

    # Traceback
    alignment = []  # list of (lyrics_idx or None, whisper_idx_start, whisper_idx_end)
    i, j = n, m

    while i > 0 or j > 0:
        if i == 0 and j > 0:
            j -= 1  # skip remaining Whisper words
            continue
        if j == 0 and i > 0:
            alignment.append((i - 1, None, None))
            i -= 1
            continue

        cell = dp[i][j]
        if cell is None:
            # Shouldn't happen, but handle gracefully
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
            j -= 1  # skip Whisper word
        elif bp == (2, 1):
            # Merge: 2 lyrics words → 1 Whisper word
            alignment.append((i - 1, j - 1, j - 1))
            alignment.append((i - 2, j - 1, j - 1))
            i -= 2
            j -= 1
        elif bp == (1, 2):
            # Split: 1 lyrics word → 2 Whisper words
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
        lentry = lyrics_flat[li]
        if wj_start is not None:
            start = whisper_words[wj_start]["start"]
            end = whisper_words[wj_end]["end"]
        else:
            start = None
            end = None
        result.append({
            "word": lentry["word"],
            "start": start,
            "end": end,
            "line_idx": lentry["line_idx"],
            "is_line_end": lentry["is_line_end"],
        })

    # Interpolate missing timestamps
    _interpolate_missing(result)

    # Spread words that share timestamps
    _spread_shared_timestamps(result)

    # Fix cross-line timestamp inheritance
    _fix_cross_line_timestamps(result)

    return result


def _interpolate_missing(aligned):
    """Fill in None timestamps by interpolating from neighbors."""
    n = len(aligned)
    for i in range(n):
        if aligned[i]["start"] is None:
            # Find nearest previous and next with timestamps
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
            # Count how many consecutive Nones
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
    """
    When consecutive words share the same start/end timestamps
    (due to merge alignment), split the time range evenly among them.
    """
    n = len(aligned)
    i = 0
    while i < n:
        # Find a run of words with the same start time
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
    """
    Fix cases where a word at the start of a new line inherits the
    timestamp of the last word on the previous line.  This happens when
    DTW skips a Whisper word and falls back to the previous timestamp.

    Fix: if word[i] is on a different line than word[i-1] AND their
    timestamps overlap or are identical, push word[i]'s start forward
    while keeping its original end time (to preserve natural duration).
    """
    n = len(aligned)
    for i in range(1, n):
        if aligned[i]["line_idx"] != aligned[i - 1]["line_idx"]:
            # Different line — check for timestamp overlap
            if aligned[i]["start"] <= aligned[i - 1]["end"] + 0.05:
                # Push start just past previous word's end
                new_start = aligned[i - 1]["end"] + 0.05
                original_end = aligned[i]["end"]
                # Only fix if it doesn't create a degenerate range
                if new_start < original_end:
                    aligned[i]["start"] = new_start
                    # Keep original end — don't shrink duration


# ──────────────────────────────────────────────────────────────
# Step 4 — Chunk into 2-3 word subtitle groups
# ──────────────────────────────────────────────────────────────

def chunk_words(aligned, size=3, gap_threshold=0.4, pre_display=0.15):
    """
    Group aligned words into subtitle chunks.
    Respects line boundaries and silence gaps.

    Line-aware: distributes words within each line to avoid
    single-word orphans at line ends.  E.g., a 7-word line
    becomes 3+2+2 instead of 3+3+1.
    """
    # First, group words by line
    lines = []
    current_line = []
    current_line_idx = aligned[0]["line_idx"] if aligned else -1

    for w in aligned:
        if w["line_idx"] != current_line_idx:
            if current_line:
                lines.append(current_line)
            current_line = [w]
            current_line_idx = w["line_idx"]
        else:
            current_line.append(w)
    if current_line:
        lines.append(current_line)

    chunks = []

    for line_words in lines:
        # Further split by silence gaps within the line
        # But use a higher threshold for the first word to avoid
        # single-word orphans at line starts
        segments = []
        seg = [line_words[0]]
        for k in range(1, len(line_words)):
            gap = line_words[k]["start"] - line_words[k - 1]["end"]
            # Use a higher gap threshold for the first word in a segment
            # to avoid single-word orphan at the start
            effective_gap = gap_threshold if len(seg) > 1 else gap_threshold * 3
            if gap >= effective_gap:
                segments.append(seg)
                seg = [line_words[k]]
            else:
                seg.append(line_words[k])
        segments.append(seg)

        # Chunk each segment with balanced sizes
        for seg_words in segments:
            n = len(seg_words)
            # Determine chunk sizes to avoid 1-word orphans
            chunk_sizes = _balanced_chunk_sizes(n, size)

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

                # Minimum display time
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


def _balanced_chunk_sizes(n, max_size=3):
    """
    Divide n words into chunks of up to max_size, avoiding single-word
    orphans at the end.

    Examples (max_size=3):
      1 → [1]         5 → [3, 2]
      2 → [2]         6 → [3, 3]
      3 → [3]         7 → [3, 2, 2]
      4 → [2, 2]      8 → [3, 3, 2]
      9 → [3, 3, 3]  10 → [3, 3, 2, 2]
    """
    if n <= max_size:
        return [n]

    # Start with full chunks
    full, rem = divmod(n, max_size)
    if rem == 0:
        return [max_size] * full
    if rem >= 2:
        return [max_size] * full + [rem]
    # rem == 1: redistribute to avoid single-word orphan
    # Take one from the last full chunk: (max_size-1) + 2
    return [max_size] * (full - 1) + [max_size - 1, 2]


# ──────────────────────────────────────────────────────────────
# Step 5 — Write SRT
# ──────────────────────────────────────────────────────────────

def ts_srt(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(chunks, path):
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(str(i))
        lines.append(f"{ts_srt(c['start'])} --> {ts_srt(c['end'])}")
        lines.append(c["text"])
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# Step 6 — Write ASS
# ──────────────────────────────────────────────────────────────

def ts_ass(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def write_ass(chunks, path, width=1080, height=1920):
    font_size = max(24, int(height * FONT_SIZE_RATIO))
    margin_v = max(30, int(height * SUBTITLE_POSITION_RATIO))

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
        start = ts_ass(c["start"])
        end = ts_ass(c["end"])
        text = c["text"].upper()
        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    Path(path).write_text(header + "\n".join(events) + "\n", encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# Step 7 — Burn subtitles into video
# ──────────────────────────────────────────────────────────────

def burn_subtitles(video_path, ass_path, output_path):
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

    print("  Burning subtitles into video …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if result.returncode != 0:
        print(f"  ✗ ffmpeg failed:\n{result.stderr[-1500:]}")
        return False
    return True


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("starship 2")
    base = Path(__file__).resolve().parent
    folder = base / folder

    print("=" * 60)
    print("  AGENT SUBTITLE — Intelligent Fuzzy Alignment")
    print("=" * 60)

    # Find files
    whisper_json = folder / "whisper_words.json"
    lyrics_file = folder / "lyrics.txt"

    # Find video
    video_file = None
    for f in folder.iterdir():
        if f.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov", ".webm"):
            if "_subtitled" not in f.name:
                video_file = f
                break

    if not whisper_json.exists():
        print(f"  ✗ {whisper_json} not found — run Whisper first")
        sys.exit(1)
    if not lyrics_file.exists():
        print(f"  ✗ {lyrics_file} not found")
        sys.exit(1)

    # Load
    print(f"\n  [1/5] Loading Whisper words …")
    whisper_words = load_whisper_words(whisper_json)
    print(f"    ✓ {len(whisper_words)} words (after merging splits)")

    print(f"\n  [2/5] Parsing lyrics …")
    lyrics_text = lyrics_file.read_text(encoding="utf-8")
    lyrics_lines = parse_lyrics(lyrics_text)
    lyrics = flat_lyrics(lyrics_lines)
    print(f"    ✓ {len(lyrics)} words across {len(lyrics_lines)} lines")

    # Align
    print(f"\n  [3/5] Running DTW alignment …")
    aligned = dtw_align(lyrics, whisper_words)
    print(f"    ✓ Aligned {len(aligned)} lyrics words")

    # Show alignment summary
    print(f"\n    --- Alignment preview (first 30 words) ---")
    for i, a in enumerate(aligned[:30]):
        print(f"    {i:3d}  {a['start']:7.3f} - {a['end']:7.3f}  L{a['line_idx']:2d}  {a['word']}")
    print(f"    --- End of preview ---")

    # Chunk
    print(f"\n  [4/5] Creating subtitle chunks …")
    chunks = chunk_words(aligned, size=WORDS_PER_CHUNK,
                         gap_threshold=SILENCE_GAP_THRESHOLD,
                         pre_display=PRE_DISPLAY_OFFSET)
    print(f"    ✓ {len(chunks)} chunks")

    # Write files
    srt_path = folder / "output.srt"
    ass_path = folder / "output.ass"

    write_srt(chunks, srt_path)
    print(f"    ✓ SRT → {srt_path.name}")

    write_ass(chunks, ass_path, width=1080, height=1920)
    print(f"    ✓ ASS → {ass_path.name}")

    # Burn
    if video_file:
        print(f"\n  [5/5] Burning subtitles into video …")
        out_name = f"{video_file.stem}_subtitled{video_file.suffix}"
        output = folder / out_name
        ok = burn_subtitles(video_file, ass_path, output)
        if ok:
            print(f"    ✓ Video → {out_name}")
        else:
            print("    ✗ Video encoding failed")
    else:
        print("\n  [5/5] No video file found — skipping burn")

    print(f"\n{'=' * 60}")
    print("  DONE ✓")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
