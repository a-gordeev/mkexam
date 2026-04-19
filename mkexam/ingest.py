import re
import tempfile
from pathlib import Path


def ingest_youtube(url: str) -> str:
    """Extract captions/transcript from a YouTube video using yt-dlp."""
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")

    with tempfile.TemporaryDirectory() as tmpdir:
        opts = {
            "skip_download": True,
            "writeautomaticsub": True,
            "writesubtitles": True,
            "subtitleslangs": ["en", "en-US", "en-GB"],
            "subtitlesformat": "vtt",
            "outtmpl": str(Path(tmpdir) / "video"),
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)

        vtt_files = list(Path(tmpdir).glob("*.vtt"))
        if vtt_files:
            return _parse_vtt(vtt_files[0].read_text(encoding="utf-8"))

    raise RuntimeError(
        "No captions found for this video. "
        "Try enabling auto-generated captions or paste the transcript manually."
    )


def _parse_vtt(vtt: str) -> str:
    """Parse VTT subtitles into a transcript with embedded time markers every ~30 seconds."""
    result: list[str] = []
    current_secs: int | None = None
    last_marker_secs: int = -60
    seen: set[str] = set()

    for line in vtt.splitlines():
        line = line.strip()
        if not line or line.startswith("WEBVTT") or re.match(r"^\d+$", line):
            continue
        # Timestamp line: "00:01:23.456 --> ..."
        ts = re.match(r'(\d+):(\d+):(\d+)', line)
        if "-->" in line and ts:
            h, m, s = int(ts.group(1)), int(ts.group(2)), int(ts.group(3))
            current_secs = h * 3600 + m * 60 + s
            continue
        text = re.sub(r"<[^>]+>", "", line).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        # Embed a time marker at ~30-second intervals
        if current_secs is not None and current_secs - last_marker_secs >= 30:
            total_m = current_secs // 60
            s_val = current_secs % 60
            result.append(f"[{total_m}:{s_val:02d}]")
            last_marker_secs = current_secs
        result.append(text)

    return " ".join(result)


def ingest_pdf(file_path: Path) -> list[dict]:
    """Extract text from a PDF as a list of sections {heading, text}.

    Detects up to 3 heading levels by font size; builds full hierarchical paths
    like "Chapter 3 / Section 3.2 (p. 45)" for each section.
    Falls back to a single flat section if no font-size info is available.
    """
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber is not installed. Run: pip install pdfplumber")

    from collections import Counter

    all_lines: list[dict] = []   # {"text": str, "size": float, "page": int}
    size_counts: Counter = Counter()

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            chars = page.chars
            if not chars:
                continue

            # Group chars into visual lines (3-point y buckets)
            line_map: dict[int, list] = {}
            for c in chars:
                key = round(float(c.get("top", 0)) / 3) * 3
                line_map.setdefault(key, []).append(c)

            for key in sorted(line_map):
                line_chars = sorted(line_map[key], key=lambda c: float(c.get("x0", 0)))

                # Reconstruct text with inter-word spacing
                parts: list[str] = []
                prev_x1: float | None = None
                for c in line_chars:
                    txt = c.get("text", "")
                    if not txt or not txt.strip():
                        continue
                    x0 = float(c.get("x0", 0))
                    if prev_x1 is not None and x0 - prev_x1 > 2:
                        parts.append(" ")
                    parts.append(txt)
                    prev_x1 = float(c.get("x1", x0))

                line_text = "".join(parts).strip()
                if not line_text:
                    continue

                sizes = [
                    float(c["size"])
                    for c in line_chars
                    if c.get("size") and float(c.get("size", 0)) > 0
                ]
                avg_size = sum(sizes) / len(sizes) if sizes else 0.0
                if avg_size > 0:
                    size_counts[round(avg_size, 1)] += 1

                all_lines.append({"text": line_text, "size": avg_size, "page": page_num})

    if not all_lines:
        raise RuntimeError("Could not extract text from the PDF.")

    body_size = float(size_counts.most_common(1)[0][0]) if size_counts else 12.0
    heading_threshold = body_size * 1.15

    # Map each distinct heading-size to a hierarchy level (1 = largest = chapter)
    heading_sizes = sorted(
        {round(l["size"], 1) for l in all_lines if l["size"] >= heading_threshold},
        reverse=True,
    )
    size_to_level: dict[float, int] = {sz: i + 1 for i, sz in enumerate(heading_sizes)}

    # Skip table of contents — detected by heading name or dot-leader content
    _TOC_HEADINGS = {
        "table of contents", "contents", "toc", "index", "list of figures",
        "list of tables", "list of abbreviations", "abbreviations", "acronyms",
        "acknowledgements", "acknowledgments", "preface", "foreword",
        "about the author", "about the authors", "author", "authors",
        "copyright", "legal notice", "disclaimer", "license", "licence",
        "colophon", "bibliography", "references", "further reading",
        "table des matières", "inhaltsverzeichnis",
    }

    def _is_toc(section: dict) -> bool:
        # Check the leaf component of the full path (strip page suffix)
        heading_full = section["heading"].strip()
        leaf = heading_full.rsplit(" / ", 1)[-1]
        leaf = re.sub(r'\s*\(p\.\s*\d+\)\s*$', '', leaf).strip().lower()
        if leaf in _TOC_HEADINGS or heading_full.lower() in _TOC_HEADINGS:
            return True
        # Content test: majority of lines look like "Title ......... 12"
        lines = [l for l in section["text"].splitlines() if l.strip()]
        if not lines:
            return False
        toc_lines = sum(
            1 for l in lines
            if re.search(r'\.{3,}\s*\d+\s*$', l) or re.search(r'\s{4,}\d+\s*$', l)
        )
        return toc_lines / len(lines) > 0.4

    sections: list[dict] = []
    heading_stack: dict[int, str] = {}   # level → heading text
    section_start_page: int = 1
    current_lines: list[str] = []

    def _flush() -> None:
        body = "\n".join(current_lines).strip()
        if not body:
            return
        path_parts = [heading_stack[l] for l in sorted(heading_stack.keys())]
        label = " / ".join(path_parts)
        if label:
            label += f" (p. {section_start_page})"
        sections.append({"heading": label, "text": body})

    for line in all_lines:
        level = size_to_level.get(round(line["size"], 1), 0)
        is_heading = level > 0 and 0 < len(line["text"]) <= 200
        if is_heading:
            _flush()
            current_lines = []
            # Drop levels >= current (sub-headings of the same or lower rank)
            for l in [k for k in list(heading_stack) if k >= level]:
                del heading_stack[l]
            heading_stack[level] = line["text"]
            section_start_page = line["page"]
        else:
            current_lines.append(line["text"])

    _flush()

    # Drop sections too short to yield a meaningful question
    sections = [s for s in sections if len(s["text"]) > 100]
    sections = [s for s in sections if not _is_toc(s)]

    if not sections:
        raise RuntimeError("Could not extract text from the PDF.")

    return sections


def ingest_mp4(file_path: Path, model: str = "small", progress_cb=None) -> str:
    """Transcribe a video file to a timestamped transcript string using Whisper."""
    try:
        import imageio_ffmpeg
    except ImportError:
        raise RuntimeError("imageio-ffmpeg not installed. Run: pip install imageio-ffmpeg")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")

    import subprocess

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "audio.wav"

        if progress_cb:
            progress_cb("Extracting audio…")

        result = subprocess.run(
            [ffmpeg_exe, "-i", str(file_path),
             "-ar", "16000", "-ac", "1", "-f", "wav", str(audio_path),
             "-y", "-loglevel", "error"],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg audio extraction failed: {result.stderr.decode(errors='replace')}"
            )

        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        is_cached = (cache_dir / f"models--Systran--faster-whisper-{model}").exists()

        if progress_cb:
            if not is_cached:
                progress_cb(
                    f"Downloading Whisper '{model}' model (first use only, may take a few minutes)…"
                )
            else:
                progress_cb("Loading Whisper model…")

        whisper_model = WhisperModel(model, device="cpu", compute_type="int8")

        if progress_cb:
            progress_cb("Transcribing audio…")

        segments, _ = whisper_model.transcribe(str(audio_path), beam_size=5)
        return _whisper_segments_to_transcript(segments)


def _whisper_segments_to_transcript(segments) -> str:
    """Convert faster-whisper segments to a transcript string with [M:SS] markers every ~30s."""
    result: list[str] = []
    last_marker_secs: float = -60.0

    for seg in segments:
        start: float = seg.start
        text = seg.text.strip()
        if not text:
            continue
        if start - last_marker_secs >= 30:
            total_m = int(start) // 60
            s_val = int(start) % 60
            result.append(f"[{total_m}:{s_val:02d}]")
            last_marker_secs = start
        result.append(text)

    return " ".join(result)


def ingest_url(url: str) -> str:
    """Extract main article text from a web URL."""
    try:
        import trafilatura
    except ImportError:
        raise RuntimeError("trafilatura is not installed. Run: pip install trafilatura")

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise RuntimeError(f"Could not fetch URL: {url}")
    text = trafilatura.extract(downloaded)
    if not text:
        raise RuntimeError("Could not extract readable text from that URL.")
    return text
