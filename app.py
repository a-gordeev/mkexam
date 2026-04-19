import json
import os
import threading
import time
import uuid
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect,
    url_for, jsonify, flash, session, after_this_request,
)

load_dotenv()

from mkexam.storage import DeckStorage
from mkexam.ingest import ingest_youtube, ingest_pdf, ingest_url, ingest_mp4
from mkexam.generate import (
    analyze_contradictions, verify_cards, list_key_points,
    generate_for_points, generate_batch, segment_transcript,
    BATCH_SIZE,
)
from mkexam.spaced import due_cards

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
TEMP_DIR = DATA_DIR / "temp"
JOBS_DIR = DATA_DIR / "jobs"
for _d in (UPLOAD_DIR, TEMP_DIR, JOBS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

storage = DeckStorage(DATA_DIR / "decks")

_KEY_FILE = DATA_DIR / ".secret_key"
if _KEY_FILE.exists():
    _secret = _KEY_FILE.read_bytes()
else:
    _secret = os.urandom(24)
    _KEY_FILE.write_bytes(_secret)

app = Flask(__name__)
app.secret_key = _secret


# ---------------------------------------------------------------------------
# Persistent job helpers
# ---------------------------------------------------------------------------

def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"

def _content_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}_content.json"

def _load_job(job_id: str) -> dict:
    p = _job_path(job_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

def _save_job(job_id: str, data: dict) -> None:
    _job_path(job_id).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

def _update_job(job_id: str, **kwargs) -> dict:
    job = _load_job(job_id)
    job.update(kwargs)
    _save_job(job_id, job)
    return job

def _aborted(job_id: str) -> bool:
    return _load_job(job_id).get("abort_requested", False)

def _check_stop(job_id: str) -> bool:
    """Check abort or pause request, update status, return True if thread should exit."""
    job = _load_job(job_id)
    if job.get("abort_requested"):
        _update_job(job_id, status="aborted")
        return True
    if job.get("pause_requested"):
        _update_job(job_id, status="paused")
        return True
    return False

def _stream_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}_stream"

def _cleanup_job(job_id: str) -> None:
    _job_path(job_id).unlink(missing_ok=True)
    _content_path(job_id).unlink(missing_ok=True)
    _stream_path(job_id).unlink(missing_ok=True)


def _estimate_cost(n: int | None, num_parts: int, gen_units: list) -> dict:
    """Rough billing estimate for a generation job.

    Gemini 2.5 Flash pricing (approximate, non-thinking output only):
      Input:  $0.15 / 1M tokens
      Output: $0.60 / 1M tokens
    """
    CHARS_PER_TOKEN = 4
    INPUT_PRICE  = 0.15 / 1_000_000
    OUTPUT_PRICE = 0.60 / 1_000_000
    total_len = sum(len(u["text"]) for u in gen_units) or 1

    input_tokens = 0
    output_tokens = 0
    est_questions = 0

    # Contradiction check cost (one call over all source parts)
    if num_parts > 1:
        input_tokens += total_len // CHARS_PER_TOKEN + 200
        output_tokens += 300

    for unit in gen_units:
        unit_tokens = max(1, len(unit["text"]) // CHARS_PER_TOKEN)
        if n is None:
            unit_q = max(1, unit_tokens // 400)         # ~1 key point per 400 tokens
        else:
            unit_q = max(1, round(n * len(unit["text"]) / total_len))
        est_questions += unit_q
        num_batches = max(1, (unit_q + BATCH_SIZE - 1) // BATCH_SIZE)

        if n is None:
            input_tokens  += unit_tokens + 300           # list_key_points call
            output_tokens += max(50, unit_q * 8)         # key-points list output
        input_tokens  += num_batches * (unit_tokens + 600)   # generation batches
        output_tokens += num_batches * 1500

    content_tokens = total_len // CHARS_PER_TOKEN
    backend = os.environ.get("LLM_BACKEND", "gemini").lower()
    cost_usd = (input_tokens * INPUT_PRICE + output_tokens * OUTPUT_PRICE) if backend == "gemini" else 0.0

    return {
        "content_tokens": content_tokens,
        "est_questions": est_questions,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 4),
        "free_backend": backend != "gemini",
    }


def _compute_actual_cost(usage_acc: list[dict]) -> dict:
    """Compute actual billing from accumulated usage metadata.

    Gemini 2.5 Flash pricing:
      Input:    $0.15 / 1M tokens
      Output:   $0.60 / 1M tokens
      Thinking: $3.50 / 1M tokens
    """
    prompt   = sum(u.get("prompt_tokens",   0) for u in usage_acc)
    output   = sum(u.get("output_tokens",   0) for u in usage_acc)
    thinking = sum(u.get("thinking_tokens", 0) for u in usage_acc)

    INPUT_PRICE    = 0.15 / 1_000_000
    OUTPUT_PRICE   = 0.60 / 1_000_000
    THINKING_PRICE = 3.50 / 1_000_000
    cost = prompt * INPUT_PRICE + output * OUTPUT_PRICE + thinking * THINKING_PRICE

    return {
        "actual_prompt_tokens":   prompt,
        "actual_output_tokens":   output,
        "actual_thinking_tokens": thinking,
        "actual_cost_usd":        round(cost, 5),
    }


# ---------------------------------------------------------------------------
# Background generation (resumable, abortable)
# ---------------------------------------------------------------------------

def _background_generate(job_id: str, source_specs: list) -> None:
    with app.app_context():
        try:
            _do_background_generate(job_id, source_specs)
        except Exception as exc:
            try:
                _update_job(job_id, status="error", error=f"Unexpected error: {exc}")
            except Exception:
                pass


def _do_background_generate(job_id: str, source_specs: list) -> None:
    job = _load_job(job_id)
    deck_name = job["deck_name"]
    n = job.get("n")                      # None = auto
    target_deck_id = job.get("target_deck_id")
    source_labels = job.get("source_labels", [])

    # Apply per-job LLM config (overrides env for this thread)
    llm_config = job.get("llm_config", {})
    if llm_config.get("backend"):
        os.environ["LLM_BACKEND"] = llm_config["backend"]
    if llm_config.get("gemini_api_key"):
        os.environ["GEMINI_API_KEY"] = llm_config["gemini_api_key"]
    if llm_config.get("gemini_model"):
        os.environ["GEMINI_MODEL"] = llm_config["gemini_model"]
    if llm_config.get("openai_base_url"):
        os.environ["OPENAI_BASE_URL"] = llm_config["openai_base_url"]
    if llm_config.get("openai_api_key"):
        os.environ["OPENAI_API_KEY"] = llm_config["openai_api_key"]
    if llm_config.get("openai_model"):
        os.environ["OPENAI_MODEL"] = llm_config["openai_model"]

    # Wire up streaming progress: write token count to a sidecar file (throttled)
    from mkexam import generate as _gen_mod
    _sp = _stream_path(job_id)
    _last_stream_t = [0.0]
    def _stream_cb(n: int) -> None:
        now = time.time()
        if n == 0 or now - _last_stream_t[0] >= 0.5:
            try:
                _sp.write_text(str(n))
            except Exception:
                pass
            _last_stream_t[0] = now
    _gen_mod._tl.stream_cb = _stream_cb
    _gen_mod._tl.stats = {"retries": 0, "invalid": 0}

    # ── Phase 1: extract ────────────────────────────────────────────────
    if job.get("phase") != "generating" and not _content_path(job_id).exists():
        _update_job(job_id, phase="extracting")
        parts = []          # one flat string per source (for contradiction check)
        pdf_units = []      # one unit per PDF section
        nonpdf_texts = []   # flat text from non-PDF sources

        # Use pre-extracted PDF units when the user already selected chapters
        pre_pdf_units = job.get("pre_extracted_pdf_units", [])
        if pre_pdf_units:
            pdf_units.extend(pre_pdf_units)
            parts.extend(u["text"] for u in pre_pdf_units)

        for spec in source_specs:
            if _check_stop(job_id): return
            try:
                src = spec["type"]
                if src == "youtube":
                    text = ingest_youtube(spec["url"])
                    parts.append(text)
                    sections = segment_transcript(text)
                    for sec in sections:
                        label = sec.get("heading", "")
                        body = (f"[{label}]\n{sec['text']}" if label else sec["text"])
                        pdf_units.append({"label": label, "text": body})
                elif src == "mp4":
                    p = Path(spec["mp4_path"])
                    def _mp4_cb(msg: str, _jid=job_id) -> None:
                        _update_job(_jid, ingest_msg=msg)
                    text = ingest_mp4(p, model=spec.get("whisper_model", "small"),
                                      progress_cb=_mp4_cb)
                    _update_job(job_id, ingest_msg="")
                    p.unlink(missing_ok=True)
                    parts.append(text)
                    sections = segment_transcript(text)
                    for sec in sections:
                        label = sec.get("heading", "")
                        body = (f"[{label}]\n{sec['text']}" if label else sec["text"])
                        pdf_units.append({"label": label, "text": body})
                elif src == "url":
                    text = ingest_url(spec["url"])
                    parts.append(text)
                    nonpdf_texts.append(text)
                elif src == "pdf":
                    p = Path(spec["pdf_path"])
                    sections = ingest_pdf(p)   # list[{heading, text}]
                    p.unlink(missing_ok=True)
                    flat = "\n\n".join(
                        (f"[{s['heading']}]\n{s['text']}" if s['heading'] else s['text'])
                        for s in sections
                    )
                    parts.append(flat)
                    for sec in sections:
                        label = sec["heading"]
                        body = (f"[{label}]\n{sec['text']}" if label else sec["text"])
                        pdf_units.append({"label": label, "text": body})
                elif src == "text":
                    text = spec["text"]
                    parts.append(text)
                    nonpdf_texts.append(text)
            except Exception as exc:
                _update_job(job_id, status="error", error=f"Source: {exc}"); return

        parts = [p.strip() for p in parts if p and p.strip()]
        if not parts:
            _update_job(job_id, status="error", error="No text could be extracted."); return

        # Build generation units: PDF sections first, then combined non-PDF
        gen_units = list(pdf_units)
        if nonpdf_texts:
            gen_units.append({"label": "", "text": "\n\n---\n\n".join(nonpdf_texts)})

        _content_path(job_id).write_text(
            json.dumps({"parts": parts, "gen_units": gen_units}, ensure_ascii=False),
            encoding="utf-8",
        )
        _update_job(job_id, phase="analyzing")

    if _check_stop(job_id): return

    # ── Phase 2: contradiction check ────────────────────────────────────
    job = _load_job(job_id)
    content_data = json.loads(_content_path(job_id).read_text(encoding="utf-8"))
    parts = content_data["parts"]
    content = content_data.get("merged") or "\n\n---\n\n".join(parts)

    if job.get("phase") == "analyzing":
        contradictions = analyze_contradictions(parts) if len(parts) > 1 else []
        if contradictions:
            temp_id = str(uuid.uuid4())
            (TEMP_DIR / f"{temp_id}.json").write_text(
                json.dumps({
                    "parts": parts, "deck_name": deck_name,
                    "n": n, "target_deck_id": target_deck_id,
                    "contradictions": contradictions,
                }, ensure_ascii=False), encoding="utf-8",
            )
            _update_job(job_id, status="review_needed", temp_id=temp_id); return
        _update_job(job_id, phase="listing" if n is None else "generating")

    if _check_stop(job_id): return

    # ── Confirmation: show cost estimate and wait for user approval ───────
    job = _load_job(job_id)
    if not job.get("confirmed"):
        gen_units_for_est = content_data.get("gen_units") or [{"text": content}]
        estimates = _estimate_cost(n, len(parts), gen_units_for_est)
        if estimates.get("free_backend"):
            _update_job(job_id, confirmed=True, **estimates)
        else:
            _update_job(job_id, status="pending_confirm", phase="confirming", **estimates)
            while True:
                time.sleep(2)
                if _check_stop(job_id): return
                if _load_job(job_id).get("confirmed"):
                    _update_job(job_id, status="running"); break

    if _check_stop(job_id): return

    # ── Phase 2b: verify existing cards against new content ──────────────
    if target_deck_id and not job.get("regenerate") and not job.get("verified"):
        existing_deck = storage.get_deck(target_deck_id)
        existing_cards = (existing_deck or {}).get("cards", [])
        if existing_cards:
            _update_job(job_id, phase="verifying")
            flagged = verify_cards(content, existing_cards, usage_acc)
            drop_ids = {c["id"] for c in flagged if c.get("action") == "drop"}
            if drop_ids:
                deck = storage.get_deck(target_deck_id)
                if deck:
                    deck["cards"] = [c for c in deck["cards"] if c["id"] not in drop_ids]
                    storage.save_deck(deck)
            _update_job(job_id, verified=True, cards_removed=len(drop_ids))

    if _check_stop(job_id): return

    # ── Phase 3: generate per unit ────────────────────────────────────────
    # Each PDF section is its own unit; non-PDF sources are combined into one.
    # For merged content (post-review), treat as a single flat unit.
    job = _load_job(job_id)
    if content_data.get("merged"):
        gen_units = [{"label": "", "text": content_data["merged"]}]
    else:
        gen_units = content_data.get("gen_units") or [{"label": "", "text": content}]

    # Seed existing-hint with already-saved deck cards (avoids duplicates)
    if target_deck_id and not job.get("regenerate"):
        _existing_deck = storage.get_deck(target_deck_id)
        _hint_cards = (_existing_deck or {}).get("cards", [])
    else:
        _hint_cards = []
    questions: list[dict] = job.get("questions", []) or list(_hint_cards)
    usage_acc: list[dict] = []
    unit_index: int = job.get("unit_index", 0)
    total_text_len = sum(len(u["text"]) for u in gen_units) or 1
    skipped_points: list[str] = []   # key points that failed after all retries
    skipped_count: int = 0           # total questions skipped

    while unit_index < len(gen_units):
        if _aborted(job_id):
            _update_job(job_id, status="aborted"); return

        unit = gen_units[unit_index]
        unit_text = unit["text"]
        unit_label = unit.get("label", "")
        job = _load_job(job_id)
        resuming = job.get("unit_index", 0) == unit_index

        if n is None:
            # ── Auto mode: key points per unit ──────────────────────────
            unit_kp = job.get("unit_key_points") if resuming else None
            unit_done = job.get("unit_done", 0) if resuming else 0

            if unit_kp is None:
                _update_job(job_id, phase="listing", unit_index=unit_index)
                try:
                    unit_kp = list_key_points(unit_text, usage_acc)
                except Exception:
                    # Key-point listing failed — fall back to fixed batch for this unit
                    unit_kp = None
                unit_done = 0
                if unit_kp is None:
                    # Fallback: generate a fixed batch instead of per-point questions
                    _update_job(job_id, phase="generating", unit_index=unit_index,
                                unit_key_points=[], unit_done=0)
                    try:
                        new_qs = generate_batch(unit_text, BATCH_SIZE, questions, usage_acc)
                    except Exception as exc:
                        if _is_fatal(exc):
                            _update_job(job_id, status="error", error=_gen_error(exc)); return
                        new_qs = []
                        skipped_count += BATCH_SIZE
                    for q in new_qs:
                        q["chapter"] = unit_label
                    questions.extend(new_qs)
                    _update_job(job_id, questions=questions, done=len(questions), unit_done=0)
                    unit_index += 1
                    _update_job(job_id, unit_index=unit_index, unit_key_points=None, unit_done=0)
                    continue
                old_total = _load_job(job_id).get("total") or 0
                _update_job(job_id, phase="generating", unit_index=unit_index,
                            unit_key_points=unit_kp, unit_done=0,
                            total=old_total + len(unit_kp))

            while unit_done < len(unit_kp):
                if _aborted(job_id):
                    _update_job(job_id, status="aborted"); return
                batch = unit_kp[unit_done: unit_done + BATCH_SIZE]
                try:
                    new_qs = generate_for_points(unit_text, batch, questions, usage_acc)
                except Exception as exc:
                    if _is_fatal(exc):
                        _update_job(job_id, status="error", error=_gen_error(exc)); return
                    skipped_points.extend(batch)
                    skipped_count += len(batch)
                    new_qs = []
                for q in new_qs:
                    q["chapter"] = unit_label
                questions.extend(new_qs)
                unit_done += len(batch)
                _update_job(job_id, questions=questions,
                            done=len(questions), unit_done=unit_done)

        else:
            # ── Fixed mode: proportional share per unit ──────────────────
            unit_n = max(1, round(n * len(unit_text) / total_text_len))
            done_in_unit = 0
            while done_in_unit < unit_n:
                if _aborted(job_id):
                    _update_job(job_id, status="aborted"); return
                batch_n = min(BATCH_SIZE, unit_n - done_in_unit)
                try:
                    new_qs = generate_batch(unit_text, batch_n, questions, usage_acc)
                except Exception as exc:
                    if _is_fatal(exc):
                        _update_job(job_id, status="error", error=_gen_error(exc)); return
                    skipped_count += batch_n
                    new_qs = []
                for q in new_qs:
                    q["chapter"] = unit_label
                questions.extend(new_qs)
                done_in_unit += len(new_qs)
                _update_job(job_id, questions=questions, done=len(questions))

        unit_index += 1
        _update_job(job_id, unit_index=unit_index, unit_key_points=None, unit_done=0)

    # ── Phase 4: save deck ───────────────────────────────────────────────
    regenerate = job.get("regenerate", False)
    # Strip the hint cards — only save questions generated in this job
    new_questions = questions[len(_hint_cards):]
    try:
        backend = os.environ.get("LLM_BACKEND", "gemini").lower()
        if backend == "openai":
            llm_model = os.environ.get("OPENAI_MODEL", "")
        else:
            llm_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        _stats = getattr(_gen_mod._tl, "stats", {})
        gen_stats = {
            "generated":      len(new_questions),
            "skipped":        skipped_count,
            "invalid":        _stats.get("invalid", 0),
            "skipped_points": skipped_points[:100],
            "dropped":        job.get("cards_removed", 0),
            "retries":        _stats.get("retries", 0),
        }
        deck_id = _finalise_deck(new_questions, deck_name, target_deck_id,
                                  gen_units, regenerate, llm_model=llm_model,
                                  gen_stats=gen_stats, source_labels=source_labels)
    except Exception as exc:
        _update_job(job_id, status="error", error=str(exc)); return

    billing = _compute_actual_cost(usage_acc)
    _update_job(job_id, status="done", deck_id=deck_id, gen_stats=gen_stats, **billing)
    _content_path(job_id).unlink(missing_ok=True)


def _gen_error(exc: Exception) -> str:
    msg = str(exc)
    if "429" in msg:
        return "Gemini rate limit reached. Wait a minute and try again."
    if "503" in msg:
        return "Gemini is temporarily overloaded. Will retry automatically."
    return f"Generation failed: {msg}"


def _is_fatal(exc: Exception) -> bool:
    """True if the error should stop the whole job; False if the batch can be skipped."""
    msg = str(exc).lower()
    # Structural validation failures (wrong answer count, duplicate options, etc.)
    # are skippable — the model couldn't produce valid output for this batch.
    skippable = (
        "exactly 2 correct answers" in msg
        or "duplicate options" in msg
        or "identical or word-reorder" in msg
        or "no question objects" in msg
        or isinstance(exc, (ValueError, TypeError))
    )
    return not skippable


def _finalise_deck(
    questions: list,
    deck_name: str,
    target_deck_id: str | None,
    gen_units: list | None = None,
    regenerate: bool = False,
    llm_model: str = "",
    gen_stats: dict | None = None,
    source_labels: list | None = None,
) -> str:
    if target_deck_id:
        deck = storage.get_deck(target_deck_id)
        if deck:
            if regenerate:
                deck["cards"] = questions
                if gen_units is not None:
                    deck["gen_units"] = gen_units
                if gen_stats is not None:
                    deck["gen_stats"] = gen_stats
            else:
                deck["cards"].extend(questions)
                if gen_units:
                    deck.setdefault("gen_units", [])
                    deck["gen_units"].extend(gen_units)
                if gen_stats is not None:
                    # Accumulate stats across sessions
                    prev = deck.get("gen_stats", {})
                    deck["gen_stats"] = {
                        "generated": prev.get("generated", 0) + gen_stats["generated"],
                        "skipped":   prev.get("skipped",   0) + gen_stats["skipped"],
                        "invalid":   prev.get("invalid",   0) + gen_stats["invalid"],
                        "dropped":   prev.get("dropped",   0) + gen_stats["dropped"],
                        "retries":   prev.get("retries",   0) + gen_stats["retries"],
                        "skipped_points": (prev.get("skipped_points", []) + gen_stats["skipped_points"])[:100],
                    }
            if llm_model:
                deck["llm_model"] = llm_model
            if source_labels:
                prev = deck.get("sources", [])
                deck["sources"] = prev + [s for s in source_labels if s not in prev]
            storage.save_deck(deck)
            return target_deck_id
    deck = {"name": deck_name, "cards": questions,
            "gen_units": gen_units or [], "llm_model": llm_model,
            "gen_stats": gen_stats or {}, "sources": source_labels or []}
    return storage.save_deck(deck)


def _resume_pending_jobs() -> None:
    """Called at startup — resume any jobs interrupted by a server restart."""
    for f in JOBS_DIR.glob("*.json"):
        if "_content" in f.stem:
            continue
        try:
            job = json.loads(f.read_text(encoding="utf-8"))
            if job.get("status") == "running" and _content_path(f.stem).exists():
                job_id = f.stem
                threading.Thread(
                    target=_background_generate,
                    args=(job_id, []),   # source_specs not needed; content already saved
                    daemon=True,
                ).start()
            elif job.get("status") == "running":
                # Content not saved → can't resume; mark failed
                job["status"] = "error"
                job["error"] = "Server restarted before extraction completed. Please resubmit."
                f.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    decks = storage.list_decks()
    active_jobs = _list_active_jobs()
    return render_template("index.html", decks=decks, active_jobs=active_jobs)


GEMINI_MODEL_CATALOG = [
    {
        "id": "gemini-2.5-flash",
        "label": "Gemini 2.5 Flash",
        "cost": "~$0.01 / deck",
        "speed": 5,
        "quality": 4,
        "note": "Recommended — fast, cheap, excellent JSON compliance.",
    },
    {
        "id": "gemini-2.5-pro",
        "label": "Gemini 2.5 Pro",
        "cost": "~$0.10 / deck",
        "speed": 3,
        "quality": 5,
        "note": "Highest quality distractors and explanations; 10× more expensive.",
    },
]

def _backend_ctx() -> dict:
    """Template context vars for the LLM backend selector."""
    return {
        "env_backend":          os.environ.get("LLM_BACKEND", "gemini").lower(),
        "env_gemini_key":       bool(os.environ.get("GEMINI_API_KEY")),
        "env_gemini_model":     os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        "gemini_model_catalog": GEMINI_MODEL_CATALOG,
        "env_openai_base_url":  os.environ.get("OPENAI_BASE_URL", ""),
        "env_openai_api_key":   bool(os.environ.get("OPENAI_API_KEY")),
        "env_openai_model":     os.environ.get("OPENAI_MODEL", ""),
    }


def _list_active_jobs() -> list[dict]:
    """Return jobs that are running, paused, or pending confirmation."""
    active = []
    for f in JOBS_DIR.glob("*.json"):
        if "_content" in f.stem:
            continue
        try:
            job = json.loads(f.read_text(encoding="utf-8"))
            if job.get("status") in ("running", "paused", "pending_confirm"):
                active.append({
                    "job_id": f.stem,
                    "deck_name": job.get("deck_name", "Untitled"),
                    "status": job.get("status"),
                    "done": job.get("done", 0),
                    "total": job.get("total"),
                    "phase": job.get("phase", ""),
                })
        except Exception:
            pass
    return active


@app.route("/generate", methods=["GET", "POST"])
def generate():
    if request.method == "GET":
        return render_template("generate.html", **_backend_ctx())
    return _handle_generate_post(request, target_deck_id=None)


@app.route("/deck/<deck_id>/add", methods=["GET", "POST"])
def deck_add_sources(deck_id):
    deck = storage.get_deck(deck_id)
    if not deck:
        flash("Deck not found.", "danger")
        return redirect(url_for("index"))
    if request.method == "GET":
        return render_template("generate.html", target_deck=deck, **_backend_ctx())
    return _handle_generate_post(request, target_deck_id=deck_id)


@app.route("/generate/waiting/<job_id>")
def generate_waiting(job_id):
    return render_template("waiting.html", job_id=job_id)


@app.route("/generate/select-chapters")
def select_chapters():
    token = request.args.get("token", "")
    tmp = TEMP_DIR / f"chapsel_{token}.json"
    if not tmp.exists():
        flash("Session expired. Please start again.", "danger")
        return redirect(url_for("generate"))
    data = json.loads(tmp.read_text(encoding="utf-8"))
    return render_template("select_chapters.html",
                           token=token,
                           sections=data["pdf_sections"],
                           deck_name=data["deck_name"])


@app.route("/generate/select-chapters/confirm", methods=["POST"])
def select_chapters_confirm():
    token = request.form.get("token", "")
    tmp = TEMP_DIR / f"chapsel_{token}.json"
    if not tmp.exists():
        flash("Session expired. Please start again.", "danger")
        return redirect(url_for("generate"))
    data = json.loads(tmp.read_text(encoding="utf-8"))

    selected = set(int(i) for i in request.form.getlist("sections"))
    if not selected:
        flash("Select at least one chapter.", "warning")
        return redirect(url_for("select_chapters", token=token))

    tmp.unlink(missing_ok=True)

    filtered = [s for i, s in enumerate(data["pdf_sections"]) if i in selected]
    pre_units = [
        {"label": s["label"],
         "text": (f"[{s['label']}]\n{s['text']}" if s["label"] else s["text"])}
        for s in filtered
    ]
    return _launch_job(
        data["deck_name"], data["n"], data["target_deck_id"],
        data["llm_config"], data["non_pdf_specs"],
        pre_extracted_pdf_units=pre_units,
        source_labels=data.get("source_labels", []),
    )


@app.route("/api/job/<job_id>")
def job_status(job_id):
    job = _load_job(job_id)
    if not job:
        return jsonify(status="error", error="Job not found"), 404
    if job.get("status") == "review_needed":
        session["pending_temp_id"] = job.get("temp_id")
    # Don't send full questions list to the poller — only metadata
    sp = _stream_path(job_id)
    try:
        stream_tokens = int(sp.read_text())
    except Exception:
        stream_tokens = 0
    return jsonify(stream_tokens=stream_tokens,
                   **{k: v for k, v in job.items() if k not in ("questions", "key_points")})


@app.route("/api/job/<job_id>/abort", methods=["POST"])
def job_abort(job_id):
    job = _load_job(job_id)
    if not job:
        return jsonify(error="Job not found"), 404
    _update_job(job_id, abort_requested=True)
    return jsonify(ok=True)


@app.route("/api/job/<job_id>/pause", methods=["POST"])
def job_pause(job_id):
    job = _load_job(job_id)
    if not job:
        return jsonify(error="Job not found"), 404
    _update_job(job_id, pause_requested=True)
    return jsonify(ok=True)


@app.route("/api/job/<job_id>/continue", methods=["POST"])
def job_continue(job_id):
    job = _load_job(job_id)
    if not job:
        return jsonify(error="Job not found"), 404
    _update_job(job_id, status="running", pause_requested=False)
    threading.Thread(
        target=_background_generate,
        args=(job_id, []),
        daemon=True,
    ).start()
    return jsonify(ok=True)


@app.route("/api/job/<job_id>/confirm", methods=["POST"])
def job_confirm(job_id):
    job = _load_job(job_id)
    if not job:
        return jsonify(error="Job not found"), 404
    _update_job(job_id, confirmed=True)
    return jsonify(ok=True)


@app.route("/generate/review", methods=["GET", "POST"])
def generate_review():
    temp_id = session.get("pending_temp_id")
    if not temp_id:
        return redirect(url_for("generate"))

    temp_file = TEMP_DIR / f"{temp_id}.json"
    if not temp_file.exists():
        flash("Session expired. Please resubmit your sources.", "warning")
        return redirect(url_for("generate"))

    pending = json.loads(temp_file.read_text(encoding="utf-8"))

    if request.method == "GET":
        target_deck = None
        if pending.get("target_deck_id"):
            target_deck = storage.get_deck(pending["target_deck_id"])
        return render_template(
            "review.html",
            contradictions=pending["contradictions"],
            target_deck=target_deck,
            deck_name=pending.get("deck_name", ""),
        )

    # POST: user submitted resolutions
    parts = pending["parts"]
    resolutions = []
    for i, c in enumerate(pending["contradictions"]):
        choice = request.form.get(f"resolution_{i}", "both")
        if choice == "both":
            resolutions.append(f"[Note on '{c['topic']}': both versions may be valid]")
        else:
            try:
                src_idx = int(choice) - 1
                chosen = c["versions"][src_idx]["statement"]
                resolutions.append(f"[Authoritative on '{c['topic']}': {chosen}]")
            except (ValueError, IndexError):
                pass

    content = "\n\n---\n\n".join(parts)
    if resolutions:
        content += "\n\n[Resolution notes]\n" + "\n".join(resolutions)

    temp_file.unlink(missing_ok=True)
    session.pop("pending_temp_id", None)

    n = pending.get("n")
    job_id = str(uuid.uuid4())
    _save_job(job_id, {
        "status": "running",
        "phase": "listing" if n is None else "generating",
        "deck_name": pending["deck_name"],
        "n": n,
        "target_deck_id": pending.get("target_deck_id"),
        "questions": [],
        "done": 0,
        "total": n,
    })
    _content_path(job_id).write_text(
        json.dumps({"parts": parts, "merged": content}, ensure_ascii=False),
        encoding="utf-8",
    )
    threading.Thread(
        target=_background_generate,
        args=(job_id, []),
        daemon=True,
    ).start()
    return redirect(url_for("generate_waiting", job_id=job_id))


@app.route("/deck/<deck_id>")
def deck_view(deck_id):
    deck = storage.get_deck(deck_id)
    if not deck:
        flash("Deck not found.", "danger")
        return redirect(url_for("index"))
    due = len(due_cards(deck["cards"]))
    return render_template("deck.html", deck=deck, due_count=due)


@app.route("/deck/<deck_id>/regenerate", methods=["POST"])
def deck_regenerate(deck_id):
    deck = storage.get_deck(deck_id)
    if not deck:
        flash("Deck not found.", "danger")
        return redirect(url_for("index"))
    gen_units = deck.get("gen_units")
    if not gen_units:
        flash("No stored sources — add sources first.", "warning")
        return redirect(url_for("deck_view", deck_id=deck_id))
    job_id = str(uuid.uuid4())
    _save_job(job_id, {
        "status": "running",
        "phase": "listing",
        "deck_name": deck["name"],
        "n": None,
        "target_deck_id": deck_id,
        "regenerate": True,
        "done": 0,
        "total": None,
    })
    _content_path(job_id).write_text(
        json.dumps({"parts": [u["text"] for u in gen_units], "gen_units": gen_units},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    threading.Thread(
        target=_background_generate, args=(job_id, []), daemon=True,
    ).start()
    return redirect(url_for("generate_waiting", job_id=job_id))


@app.route("/deck/<deck_id>/delete", methods=["POST"])
def deck_delete(deck_id):
    storage.delete_deck(deck_id)
    flash("Deck deleted.", "info")
    return redirect(url_for("index"))


@app.route("/deck/<deck_id>/rename", methods=["POST"])
def deck_rename(deck_id):
    new_name = (request.json or {}).get("name", "").strip()
    if not new_name:
        return jsonify(error="Name cannot be empty"), 400
    deck = storage.get_deck(deck_id)
    if not deck:
        return jsonify(error="Deck not found"), 404
    deck["name"] = new_name
    storage.save_deck(deck)
    return jsonify(ok=True, name=new_name)


@app.route("/deck/<deck_id>/export")
def deck_export(deck_id):
    import io
    from flask import send_file
    deck = storage.get_deck(deck_id)
    if not deck:
        flash("Deck not found.", "danger")
        return redirect(url_for("index"))
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in deck["name"])
    buf = io.BytesIO(json.dumps(deck, indent=2, ensure_ascii=False).encode())
    return send_file(buf, as_attachment=True,
                     download_name=f"{safe_name}.json",
                     mimetype="application/json")


@app.route("/deck/<deck_id>/export/anki")
def deck_export_anki(deck_id):
    import hashlib
    import tempfile
    import os as _os
    from flask import send_file
    import genanki

    deck = storage.get_deck(deck_id)
    if not deck:
        flash("Deck not found.", "danger")
        return redirect(url_for("index"))

    def _id(s):
        return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

    model = genanki.Model(
        _id(deck_id + ":model"),
        "mkexam Multi-Select",
        fields=[
            {"name": "Question"},
            {"name": "Options"},
            {"name": "OptionsRevealed"},
            {"name": "Explanation"},
        ],
        templates=[{
            "name": "Card",
            "qfmt": (
                "<div class='question'>{{Question}}</div>"
                "<div class='hint'>Select 2 correct answers</div>"
                "<hr>{{Options}}"
            ),
            "afmt": (
                "<div class='question'>{{Question}}</div>"
                "<hr>{{OptionsRevealed}}"
                "<hr><div class='explanation'>{{Explanation}}</div>"
            ),
        }],
        css="""
        .card { font-family: Arial, sans-serif; font-size: 15px; }
        .question { font-weight: bold; font-size: 17px; margin-bottom: 8px; }
        .hint { color: #888; font-size: 13px; margin-bottom: 8px; }
        ol { padding-left: 1.4em; margin: 0; }
        li { margin: 6px 0; }
        .correct { color: #198754; font-weight: bold; }
        .wrong { color: #dc3545; }
        .explanation { color: #555; font-style: italic; font-size: 14px; }
        """,
    )

    anki_deck = genanki.Deck(_id(deck_id + ":deck"), deck["name"])

    for card in deck["cards"]:
        correct_set = set(int(a) for a in card.get("answer", []))
        comments = card.get("comments", [])   # list, 0-based
        sources  = card.get("sources", {})    # dict, 1-based string keys

        # Front: plain numbered list
        opts_html = "<ol>" + "".join(
            f"<li>{opt}</li>"
            for opt in card.get("options", [])
        ) + "</ol>"

        # Back: coloured + comment per option (i is 0-based, displayed as i+1)
        def _revealed_li(i, opt):
            comment = comments[i] if i < len(comments) else ""
            one = i + 1   # 1-based
            cls = "correct" if one in correct_set else ""
            marker = " ✓" if one in correct_set else ""
            comment_html = f"<br><small>{comment}</small>" if comment else ""
            return f"<li class='{cls}'>{opt}{marker}{comment_html}</li>"

        opts_revealed = "<ol>" + "".join(
            _revealed_li(i, opt)
            for i, opt in enumerate(card.get("options", []))
        ) + "</ol>"

        note = genanki.Note(
            model=model,
            fields=[
                card["question"],
                opts_html,
                opts_revealed,
                card.get("explanation", ""),
            ],
            guid=card["id"],
        )
        anki_deck.add_note(note)

    tmp = tempfile.NamedTemporaryFile(suffix=".apkg", delete=False)
    tmp.close()
    genanki.Package(anki_deck).write_to_file(tmp.name)

    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in deck["name"])

    @after_this_request
    def _cleanup(response):
        try:
            _os.unlink(tmp.name)
        except Exception:
            pass
        return response

    return send_file(tmp.name, as_attachment=True,
                     download_name=f"{safe_name}.apkg",
                     mimetype="application/octet-stream")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _handle_generate_post(req, target_deck_id: str | None):
    """Shared POST handler for /generate and /deck/<id>/add."""
    deck_name = req.form.get("deck_name", "").strip()
    target_deck = storage.get_deck(target_deck_id) if target_deck_id else None
    effective_name = deck_name or (target_deck["name"] if target_deck else "")

    if not effective_name:
        flash("Please enter a deck name.", "danger")
        return render_template("generate.html")

    # Resolve backend config: form overrides env
    backend = (req.form.get("llm_backend") or os.environ.get("LLM_BACKEND") or "gemini").lower()

    if backend == "openai":
        openai_base_url = req.form.get("openai_base_url", "").strip() or os.environ.get("OPENAI_BASE_URL", "")
        openai_api_key  = req.form.get("openai_api_key",  "").strip() or os.environ.get("OPENAI_API_KEY",  "")
        openai_model    = req.form.get("openai_model",    "").strip() or os.environ.get("OPENAI_MODEL",    "")
        if not openai_base_url:
            flash("OpenAI-compatible base URL is required.", "danger")
            return render_template("generate.html", target_deck=target_deck, **_backend_ctx())
        if not openai_model:
            flash("Model name is required for OpenAI-compatible endpoint.", "danger")
            return render_template("generate.html", target_deck=target_deck, **_backend_ctx())
        llm_config = {"backend": "openai", "openai_base_url": openai_base_url,
                      "openai_api_key": openai_api_key, "openai_model": openai_model}
    else:
        gemini_key   = req.form.get("gemini_api_key", "").strip() or os.environ.get("GEMINI_API_KEY", "")
        gemini_model = req.form.get("gemini_model",   "").strip() or os.environ.get("GEMINI_MODEL",   "gemini-2.5-flash")
        if not gemini_key:
            flash("Gemini API key is required. Enter it in the form or set GEMINI_API_KEY in .env.", "danger")
            return render_template("generate.html", target_deck=target_deck, **_backend_ctx())
        llm_config = {"backend": "gemini", "gemini_api_key": gemini_key, "gemini_model": gemini_model}

    # Collect source specs — save uploaded files to disk immediately (request ends after return)
    source_specs = []
    idx = 0
    while True:
        src_type = req.form.get(f"src_type_{idx}")
        if src_type is None:
            break
        spec = {"type": src_type}
        if src_type == "youtube":
            spec["url"] = req.form.get(f"youtube_url_{idx}", "").strip()
            if not spec["url"]:
                flash(f"Source {idx+1}: enter a YouTube URL.", "danger")
                return render_template("generate.html", target_deck=target_deck)
            spec["label"] = spec["url"]
        elif src_type == "url":
            spec["url"] = req.form.get(f"web_url_{idx}", "").strip()
            if not spec["url"]:
                flash(f"Source {idx+1}: enter a URL.", "danger")
                return render_template("generate.html", target_deck=target_deck)
            spec["label"] = spec["url"]
        elif src_type == "mp4":
            f = req.files.get(f"mp4_file_{idx}")
            if not f or not f.filename:
                flash(f"Source {idx+1}: upload a video file.", "danger")
                return render_template("generate.html", target_deck=target_deck)
            tmp = UPLOAD_DIR / f"{uuid.uuid4()}.mp4"
            f.save(tmp)
            spec["mp4_path"] = str(tmp)
            spec["whisper_model"] = req.form.get(f"whisper_model_{idx}", "small")
            spec["label"] = f.filename
        elif src_type == "pdf":
            f = req.files.get(f"pdf_file_{idx}")
            if not f or not f.filename:
                flash(f"Source {idx+1}: upload a PDF file.", "danger")
                return render_template("generate.html", target_deck=target_deck)
            tmp = UPLOAD_DIR / f"{uuid.uuid4()}.pdf"
            f.save(tmp)
            spec["pdf_path"] = str(tmp)
            spec["label"] = f.filename
        elif src_type == "text":
            spec["text"] = req.form.get(f"paste_text_{idx}", "").strip()
            if not spec["text"]:
                flash(f"Source {idx+1}: paste some text.", "danger")
                return render_template("generate.html", target_deck=target_deck)
            spec["label"] = "(pasted text)"
        source_specs.append(spec)
        idx += 1

    if not source_specs:
        flash("Add at least one source.", "danger")
        return render_template("generate.html", target_deck=target_deck)

    auto = req.form.get("auto_count") == "1"
    n = None if auto else min(max(int(req.form.get("num_questions", 2)), 2), 64)

    # If any PDF sources, extract chapters and let the user choose
    pdf_specs = [s for s in source_specs if s["type"] == "pdf"]
    if pdf_specs:
        pdf_sections = []
        for spec in source_specs:
            if spec["type"] != "pdf":
                continue
            try:
                sections = ingest_pdf(Path(spec["pdf_path"]))
                for sec in sections:
                    pdf_sections.append({"label": sec["heading"], "text": sec["text"]})
            except Exception as exc:
                flash(f"Could not read PDF: {exc}", "danger")
                return render_template("generate.html", target_deck=target_deck, **_backend_ctx())

        if pdf_sections:
            token = str(uuid.uuid4())
            non_pdf_specs = [s for s in source_specs if s["type"] != "pdf"]
            source_labels = [s.get("label", s.get("url", "")) for s in source_specs]
            (TEMP_DIR / f"chapsel_{token}.json").write_text(
                json.dumps({
                    "non_pdf_specs": non_pdf_specs,
                    "pdf_sections": pdf_sections,
                    "deck_name": effective_name,
                    "n": n,
                    "target_deck_id": target_deck_id,
                    "llm_config": llm_config,
                    "source_labels": source_labels,
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            return redirect(url_for("select_chapters", token=token))

    # No PDFs — launch background job directly
    source_labels = [s.get("label", s.get("url", "")) for s in source_specs]
    return _launch_job(effective_name, n, target_deck_id, llm_config, source_specs,
                       source_labels=source_labels)


def _launch_job(deck_name, n, target_deck_id, llm_config, source_specs,
                pre_extracted_pdf_units=None, source_labels=None):
    job_id = str(uuid.uuid4())
    _save_job(job_id, {
        "status": "running",
        "phase": "extracting",
        "deck_name": deck_name,
        "n": n,
        "target_deck_id": target_deck_id,
        "llm_config": llm_config,
        "pre_extracted_pdf_units": pre_extracted_pdf_units or [],
        "source_labels": source_labels or [],
        "done": 0,
        "total": None,
    })
    threading.Thread(
        target=_background_generate,
        args=(job_id, source_specs),
        daemon=True,
    ).start()
    return redirect(url_for("generate_waiting", job_id=job_id))



_resume_pending_jobs()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    print(f"Starting mkexam at http://localhost:{args.port}")
    app.run(debug=True, port=args.port)
