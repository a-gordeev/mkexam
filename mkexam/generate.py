import json
import os
import re
import threading
import time
import uuid
from datetime import date

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
BATCH_SIZE = 5

# Thread-local storage: app sets stream_cb to receive per-token progress updates
_tl = threading.local()

SYSTEM_PROMPT = (
    "You are an expert educator creating examination questions from source material. "
    "CRITICAL: never invent, infer, or hallucinate any fact, term, number, or claim. "
    "Every answer option and every source excerpt must be word-for-word from the provided content. "
    "Always work source-first: locate a verbatim passage, then derive answers from it. "
    "Each question must address exactly one clear fact literally stated in the content. "
    "Never combine facts from unrelated sections."
)

_OPTION_FORMAT = """\
For each question follow this exact sequence:

1. LOCATE — find a verbatim passage of 1–2 sentences in the content that states one \
clear, unambiguous, testable fact.
2. QUOTE — copy that passage word-for-word; it becomes the "sources" entry.
3. DERIVE — write exactly 2 correct answer options that are literally true according \
to the quoted passage only. Do not infer, extrapolate, or combine with other passages.
4. DISTRACT — write 3 distractor options with the same sentence skeleton, differing by \
only 1–2 words (a key term, number, or qualifier). Distractors must be plausible but \
contradicted by or absent from the content.
5. VERIFY — re-read each correct option against the quoted passage. If the passage does \
not directly support an option word-for-word, that option is wrong — replace it.

Rules:
- All 5 options must share the same sentence skeleton, differing by only 1–2 words.
- The 2 correct answers must differ from each other by at least 1 meaningful word. \
Reordering words or values does NOT count as a difference.
- A question must NOT combine facts from different passages.

Example of good options (differ by 1-3 words):
  1) The mitochondria produce ATP through oxidative phosphorylation
  2) The mitochondria produce NADH through oxidative phosphorylation
  3) The mitochondria produce ATP through substrate-level phosphorylation
  4) The chloroplasts produce ATP through oxidative phosphorylation
  5) The mitochondria consume ATP through oxidative phosphorylation

Each JSON object must have:
- "question": question text
- "options": ["1) ...", "2) ...", "3) ...", "4) ...", "5) ..."]
- "answer": array of exactly 2 one-based indices e.g. [1, 3]
- "comments": {{"1": "...", "2": "...", "3": "...", "4": "...", "5": "..."}} — \
one sentence per option: for correct options state what makes it right; for wrong options \
state what the content actually says that contradicts it
- "explanation": one sentence on what distinguishes the correct answers from the distractors
- "sources": {{"1": "[Section] <verbatim excerpt ≤ 2 sentences proving or disproving option 1>", \
"2": "[Section] <verbatim excerpt>", "3": "...", "4": "...", "5": "..."}} — for EVERY option, \
the exact passage (word-for-word from the content) that confirms the correct options or \
contradicts the wrong ones; prefix each with the section/chapter heading

Return ONLY a JSON array. No markdown fences, no extra text."""

KEY_POINTS_PROMPT = """\
Analyze the content below. List every distinct key point, concept, or fact worth testing in an exam.
A "point" is one coherent idea (may span multiple sentences) that yields exactly one good question.
Do not list trivial, redundant, or overlapping points.

Return a JSON array of brief strings (max 20 words each). Return ONLY the array, no extra text.

Content:
{content}"""

FOR_POINTS_PROMPT = """\
Generate exactly {n} examination questions, one per key point listed below:
{points}

Use the content for factual accuracy. Do not duplicate existing questions.

{existing_hint}

{fmt}

Content:
{content}"""

FIXED_PROMPT = """\
Generate exactly {n} examination questions from the content below.
Cover distinct concepts. Do not duplicate existing questions.

{existing_hint}

{fmt}

Content:
{content}"""

CONTRADICTION_PROMPT = """\
Analyze the following content from multiple sources. Identify factual contradictions — \
cases where two or more sources make incompatible claims about the same topic.

Return a JSON object with key "contradictions": an array of objects, each with:
- "topic": a brief label (max 10 words)
- "versions": array of {{"source": <1-indexed int>, "statement": "<the conflicting claim>"}}

If no contradictions, return {{"contradictions": []}}.
Return ONLY the JSON object. No markdown, no extra text.

Sources:
{content}"""

SEGMENT_PROMPT = """\
Split the following video transcript into logical sections covering the substantive content.
The transcript contains time markers like [5:23] — use the nearest preceding marker as the \
section start time.

Rules:
- Omit entirely: introductions, outros, sponsor segments, credits, jokes, audience Q&A not \
relevant to the main subject, and any content unrelated to the topic being taught.
- Each kept section must cover a coherent, self-contained concept worth examination questions.
- Aim for 3-10 sections.

Return a JSON array of objects, each with:
- "heading": a concise section title followed by its start time, e.g. "Topic Name [5:23]"
- "text": the verbatim transcript text belonging to that section

Return ONLY the JSON array. No markdown, no extra text.

Transcript:
{text}"""


MAX_CONTENT_CHARS = 80_000


_TRANSIENT_HINTS = (
    "server disconnected",
    "connection reset",
    "connection error",
    "remote protocol error",
    "read timeout",
    "stream ended",
    "eof occurred",
    "broken pipe",
    "response ended prematurely",
    "incomplete read",
    "chunked encoding error",
)


def _call_gemini(prompt: str, use_json_schema: bool = False) -> tuple[str, dict]:
    """Call Gemini API with retry on rate-limit and transient errors."""
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config: dict = {"system_instruction": SYSTEM_PROMPT}
    if use_json_schema:
        config["response_mime_type"] = "application/json"
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=os.environ.get("GEMINI_MODEL", GEMINI_MODEL),
                contents=prompt,
                config=config,
            )
            text = resp.text.strip()
            usage: dict = {}
            if resp.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(resp.usage_metadata, "prompt_token_count", 0) or 0,
                    "output_tokens": getattr(resp.usage_metadata, "candidates_token_count", 0) or 0,
                    "thinking_tokens": getattr(resp.usage_metadata, "thoughts_token_count", 0) or 0,
                }
            return text, usage
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if any(c in msg for c in ("429", "503")) and attempt < 3:
                time.sleep(30 * (2 ** attempt))
            elif any(h in msg for h in _TRANSIENT_HINTS) and attempt < 3:
                time.sleep(5 * (2 ** attempt))
            else:
                raise
    raise last_exc


def _call_openai(prompt: str, use_json_schema: bool = False) -> tuple[str, dict]:
    """Call an OpenAI-compatible endpoint (streaming) with retry on transient errors."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    base_url = os.environ.get("OPENAI_BASE_URL", "").rstrip("/") + "/"
    api_key  = os.environ.get("OPENAI_API_KEY", "no-key")
    model    = os.environ.get("OPENAI_MODEL", "")
    if not base_url.strip("/ "):
        raise RuntimeError("OPENAI_BASE_URL is not configured")
    if not model:
        raise RuntimeError("OPENAI_MODEL is not configured")

    client = OpenAI(base_url=base_url, api_key=api_key)
    system = (
        SYSTEM_PROMPT
        + "\nIMPORTANT: your entire response must be valid JSON only."
        " Output a JSON array [...] with no surrounding text, no markdown,"
        " no explanation before or after."
    )
    create_kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if use_json_schema:
        create_kwargs["response_format"] = {"type": "json_object"}

    cb = getattr(_tl, "stream_cb", None)
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(4):
        try:
            if cb is not None:
                cb(0)
            parts: list[str] = []
            stream_count = 0
            prompt_tokens = 0
            output_tokens = 0
            with client.chat.completions.create(**create_kwargs) as stream:
                for chunk in stream:
                    if chunk.choices:
                        delta = chunk.choices[0].delta.content or ""
                        if delta:
                            parts.append(delta)
                            stream_count += 1
                            if cb is not None:
                                cb(stream_count)
                    if getattr(chunk, "usage", None):
                        prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                        output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0
            text = "".join(parts).strip()
            if not text:
                raise ValueError("empty response from OpenAI-compatible endpoint")
            return text, {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens, "thinking_tokens": 0}
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if (any(h in msg for h in _TRANSIENT_HINTS) or "empty response" in msg) and attempt < 3:
                time.sleep(5 * (2 ** attempt))
            else:
                raise
    raise last_exc


def _call(prompt: str, use_json_schema: bool = False) -> tuple[str, dict]:
    backend = os.environ.get("LLM_BACKEND", "gemini").lower()
    if backend == "openai":
        return _call_openai(prompt, use_json_schema=use_json_schema)
    return _call_gemini(prompt, use_json_schema=use_json_schema)


def _parse_json(raw: str):
    """Parse JSON from an LLM response, tolerating markdown fences and extra text."""
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw)
    raw = raw.strip()
    if not raw:
        raise ValueError("Model returned an empty response")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        # If valid JSON followed by trailing garbage, truncate at the error position
        if "extra data" in str(exc).lower() and exc.pos:
            try:
                return json.loads(raw[:exc.pos])
            except json.JSONDecodeError:
                pass
        # Try to extract the outermost JSON array or object
        for opener, closer in [('[', ']'), ('{', '}')]:
            start = raw.find(opener)
            end = raw.rfind(closer)
            if 0 <= start < end:
                try:
                    return json.loads(raw[start:end + 1])
                except json.JSONDecodeError:
                    pass
        raise


def _call_json(prompt: str, usage_acc: list | None = None, retries: int = 4):
    """Call LLM and parse the JSON response. Retries up to `retries` times on any parse failure."""
    last_exc: Exception = RuntimeError("no attempts made")
    stats = getattr(_tl, "stats", None)
    for attempt in range(1 + retries):
        if attempt > 0 and stats is not None:
            stats["retries"] = stats.get("retries", 0) + 1
        raw, usage = _call(prompt)
        try:
            result = _parse_json(raw)
            if usage_acc is not None:
                usage_acc.append(usage)
            return result
        except (json.JSONDecodeError, ValueError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
    raise last_exc


def _stamp_one(q: dict, today: str) -> None:
    """Validate and normalise a single question dict in-place.

    Raises ValueError for unrecoverable problems; fixes what can be fixed silently.
    """
    q.setdefault("id", str(uuid.uuid4()))
    q["type"] = "multi2"

    # ── Options ─────────────────────────────────────────────────────────────
    raw_opts = q.get("options", [])
    if not isinstance(raw_opts, list) or len(raw_opts) != 5:
        raise ValueError(
            f"Expected 5 options, got {len(raw_opts) if isinstance(raw_opts, list) else type(raw_opts).__name__}"
        )
    opts = [re.sub(r'^\d+\)\s*', '', str(o)).strip() for o in raw_opts]
    if any(not o for o in opts):
        raise ValueError("One or more options are empty")
    q["options"] = opts

    # ── Question text ────────────────────────────────────────────────────────
    if not str(q.get("question", "")).strip():
        raise ValueError("Empty question text")

    # ── Answer indices (1-based from LLM, stored 1-based, bounds-checked) ──────
    ans_raw = q.get("answer", [])
    if isinstance(ans_raw, str):
        ans_raw = [a.strip() for a in ans_raw.split(",")]
    indices = []
    for a in ans_raw:
        a = str(a).strip().rstrip(")")
        if a.lstrip("-").isdigit():
            idx = int(a)
            if 1 <= idx <= len(opts):
                indices.append(idx)
    q["answer"] = sorted(set(indices))

    if len(q["answer"]) != 2:
        raise ValueError(
            f"Question must have exactly 2 correct answers, got {len(q['answer'])}"
        )

    # ── Comments → list[str] of exactly len(opts) ───────────────────────────
    raw_c = q.get("comments", {})
    if isinstance(raw_c, dict):
        def _cidx(k: str) -> int:
            k = k.strip().rstrip(")")
            if k.lstrip("-").isdigit():
                i = int(k) - 1              # 1-based key → 0-based list index
                return i if 0 <= i < len(opts) else -1
            return -1
        c_map = {_cidx(k): v for k, v in raw_c.items() if _cidx(k) >= 0}
        q["comments"] = [c_map.get(i, "") for i in range(len(opts))]
    elif isinstance(raw_c, list):
        padded = list(raw_c) + [""] * len(opts)
        q["comments"] = [str(x) for x in padded[:len(opts)]]
    else:
        q["comments"] = [""] * len(opts)

    # ── Sources → dict keyed by 1-based string index ─────────────────────────
    raw_s = q.get("sources", {})
    if isinstance(raw_s, dict):
        def _sidx(k: str) -> str | None:
            k = k.strip().rstrip(")")
            if k.lstrip("-").isdigit():
                idx = int(k)                   # 1-based from LLM, stored as-is
                return str(idx) if 1 <= idx <= len(opts) else None
            return None
        q["sources"] = {_sidx(k): v for k, v in raw_s.items() if _sidx(k) is not None}
    elif isinstance(raw_s, list):
        # Convert list to dict using the two correct-answer positions
        q["sources"] = {
            str(q["answer"][i]): str(raw_s[i])
            for i in range(min(len(raw_s), len(q["answer"])))
        }
    else:
        q["sources"] = {}

    # ── Duplicate / near-duplicate option check ──────────────────────────────
    def _norm(text: str) -> frozenset:
        return frozenset(text.lower().split())
    norm_opts = [_norm(o) for o in opts]
    if len(set(norm_opts)) < len(norm_opts):
        raise ValueError("Duplicate options detected")
    correct_texts = [opts[i - 1] for i in q["answer"]]
    if len(set(_norm(t) for t in correct_texts)) < len(correct_texts):
        raise ValueError("Correct answers are identical or word-reorder duplicates")

    q.setdefault("ease_factor", 2.5)
    q.setdefault("interval", 1)
    q.setdefault("repetitions", 0)
    q.setdefault("next_review", today)


def _stamp(questions) -> list[dict]:
    """Normalise a batch of raw LLM question dicts.

    Validates each question individually — a single bad question no longer
    discards the whole batch.  Raises only when the entire batch is empty.
    """
    # Unwrap {"questions": [...]} or any single-key wrapper dict
    if isinstance(questions, dict):
        for v in questions.values():
            if isinstance(v, list):
                questions = v
                break
    if not isinstance(questions, list):
        raise ValueError(f"Expected a JSON array of questions, got {type(questions).__name__}")
    candidates = [q for q in questions if isinstance(q, dict)]
    if not candidates:
        raise ValueError("Response contained no question objects")

    today = date.today().isoformat()
    stats = getattr(_tl, "stats", None)
    valid_qs: list[dict] = []

    for q in candidates:
        try:
            _stamp_one(q, today)
            valid_qs.append(q)
        except (ValueError, TypeError):
            if stats is not None:
                stats["invalid"] = stats.get("invalid", 0) + 1

    if not valid_qs:
        raise ValueError("No valid questions produced in this batch")
    return valid_qs


def _call_questions(prompt: str, usage_acc: list | None = None) -> list[dict]:
    """Call LLM, parse JSON, validate structure. Retries up to 4 times on any parse/structure error."""
    last_exc: Exception = RuntimeError("no attempts made")
    stats = getattr(_tl, "stats", None)
    for attempt in range(5):
        if attempt > 0 and stats is not None:
            stats["retries"] = stats.get("retries", 0) + 1
        raw, usage = _call(prompt, use_json_schema=True)
        try:
            result = _stamp(_parse_json(raw))
            if usage_acc is not None:
                usage_acc.append(usage)
            return result
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            last_exc = exc
            if attempt < 4:
                time.sleep(3 * (attempt + 1))
    raise last_exc


def _existing_hint(existing: list[dict]) -> str:
    if not existing:
        return ""
    topics = [q.get("question", "")[:120] for q in existing[:40]]
    return "Already covered — do NOT create similar questions:\n" + "\n".join(f"- {t}" for t in topics)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_key_points(content: str, usage_acc: list | None = None) -> list[str]:
    """Identify distinct testable points in content."""
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS]
    return _call_json(KEY_POINTS_PROMPT.format(content=content), usage_acc)


def generate_for_points(
    content: str,
    points: list[str],
    existing: list[dict] | None = None,
    usage_acc: list | None = None,
) -> list[dict]:
    """Generate one question per point (batch)."""
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS]
    pts = "\n".join(f"{i+1}. {p}" for i, p in enumerate(points))
    prompt = FOR_POINTS_PROMPT.format(
        n=len(points),
        points=pts,
        existing_hint=_existing_hint(existing or []),
        fmt=_OPTION_FORMAT,
        content=content,
    )
    return _call_questions(prompt, usage_acc)


def generate_batch(
    content: str,
    n: int,
    existing: list[dict] | None = None,
    usage_acc: list | None = None,
) -> list[dict]:
    """Generate n questions (fixed mode batch)."""
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS]
    prompt = FIXED_PROMPT.format(
        n=n,
        existing_hint=_existing_hint(existing or []),
        fmt=_OPTION_FORMAT,
        content=content,
    )
    return _call_questions(prompt, usage_acc)


VERIFY_PROMPT = """\
Review these existing exam questions against new source content being added to the deck.
For each question that is CONTRADICTED or MADE FACTUALLY OUTDATED by the new content, \
output a classification.
Questions not listed are assumed still valid — do NOT remove questions just because the \
new content doesn't mention the same topic.

Classify only as:
- "drop": the question is factually wrong or directly contradicted by the new content

Return a JSON array of objects: [{{"id": "...", "action": "drop", "reason": "..."}}]
If no questions are affected, return an empty array [].
Return ONLY the JSON array.

Existing questions:
{questions}

New content:
{content}
"""


def verify_cards(
    new_content: str,
    cards: list[dict],
    usage_acc: list | None = None,
) -> list[dict]:
    """Check existing cards against newly added content.
    Returns list of {id, action, reason} for cards that need to be dropped.
    """
    if not cards:
        return []
    q_summary = "\n".join(
        f'[{q["id"]}] {q["question"]}' for q in cards[:60]
    )
    prompt = VERIFY_PROMPT.format(
        questions=q_summary,
        content=new_content[:MAX_CONTENT_CHARS],
    )
    try:
        result = _call_json(prompt, usage_acc)
        if isinstance(result, list):
            return [r for r in result if isinstance(r, dict) and "id" in r]
        if isinstance(result, dict):
            inner = next((v for v in result.values() if isinstance(v, list)), [])
            return [r for r in inner if isinstance(r, dict) and "id" in r]
    except Exception:
        pass
    return []



def segment_transcript(text: str, usage_acc: list | None = None) -> list[dict]:
    """Split a flat video transcript into chapter-like sections using Gemini.
    Returns list[{"heading": str, "text": str}]. Falls back to a single section on failure.
    """
    if len(text) > MAX_CONTENT_CHARS:
        text = text[:MAX_CONTENT_CHARS]
    try:
        result = _call_json(SEGMENT_PROMPT.format(text=text), usage_acc)
        if isinstance(result, list):
            sections = [s for s in result if isinstance(s, dict) and "text" in s]
            if sections:
                return sections
    except Exception:
        pass
    return [{"heading": "", "text": text}]


def analyze_contradictions(parts: list[str], usage_acc: list | None = None) -> list[dict]:
    if len(parts) < 2:
        return []
    labeled = "\n\n---\n\n".join(f"[Source {i+1}]\n{p}" for i, p in enumerate(parts))
    if len(labeled) > MAX_CONTENT_CHARS:
        labeled = labeled[:MAX_CONTENT_CHARS]
    try:
        result = _call_json(CONTRADICTION_PROMPT.format(content=labeled), usage_acc)
        if isinstance(result, dict):
            return result.get("contradictions", [])
        return []
    except Exception:
        return []
