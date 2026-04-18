import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect,
    url_for, jsonify, flash, session, after_this_request,
)

load_dotenv()

from mkexam.storage import DeckStorage
from mkexam.spaced import sm2_update, due_cards

BASE_DIR = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"

storage = DeckStorage(DATA_DIR / "decks")

_KEY_FILE = DATA_DIR / ".secret_key"
_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
if _KEY_FILE.exists():
    _secret = _KEY_FILE.read_bytes()
else:
    _secret = os.urandom(24)
    _KEY_FILE.write_bytes(_secret)

app = Flask(__name__)
app.secret_key = _secret

_quiz_sessions: dict[str, dict] = {}


@app.context_processor
def _inject_quiz_only():
    return {"quiz_only": True}


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    decks = storage.list_decks()
    return render_template("index.html", decks=decks, active_jobs=[], quiz_only=True)


@app.route("/import", methods=["GET", "POST"])
def import_deck():
    if request.method == "GET":
        return render_template("import.html")
    f = request.files.get("deck_file")
    if not f or not f.filename:
        flash("Select a .json deck file.", "danger")
        return render_template("import.html")
    try:
        deck = json.loads(f.read().decode("utf-8"))
        if "cards" not in deck or "name" not in deck:
            raise ValueError("Not a valid mkexam deck file.")
        deck_id = storage.save_deck(deck)
        flash(f"Imported \"{deck['name']}\" — {len(deck['cards'])} cards.", "success")
        return redirect(url_for("deck_view", deck_id=deck_id))
    except Exception as exc:
        flash(f"Import failed: {exc}", "danger")
        return render_template("import.html")


@app.route("/deck/<deck_id>")
def deck_view(deck_id):
    deck = storage.get_deck(deck_id)
    if not deck:
        flash("Deck not found.", "danger")
        return redirect(url_for("index"))
    due = len(due_cards(deck["cards"]))
    return render_template("deck.html", deck=deck, due_count=due, quiz_only=True)


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


@app.route("/deck/<deck_id>/quiz")
def quiz_single(deck_id):
    return redirect(url_for("quiz", decks=deck_id))


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
    import hashlib, tempfile, os as _os
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
            {"name": "Question"}, {"name": "Options"},
            {"name": "OptionsRevealed"}, {"name": "Explanation"},
        ],
        templates=[{
            "name": "Card",
            "qfmt": "<div class='question'>{{Question}}</div><div class='hint'>Select 2 correct answers</div><hr>{{Options}}",
            "afmt": "<div class='question'>{{Question}}</div><hr>{{OptionsRevealed}}<hr><div class='explanation'>{{Explanation}}</div>",
        }],
        css=".card{font-family:Arial,sans-serif;font-size:15px}.question{font-weight:bold;font-size:17px;margin-bottom:8px}.hint{color:#888;font-size:13px;margin-bottom:8px}ol{padding-left:1.4em;margin:0}li{margin:6px 0}.correct{color:#198754;font-weight:bold}.explanation{color:#555;font-style:italic;font-size:14px}",
    )

    anki_deck = genanki.Deck(_id(deck_id + ":deck"), deck["name"])

    for card in deck["cards"]:
        correct_set = set(int(a) for a in card.get("answer", []))
        comments = card.get("comments", [])

        opts_html = "<ol>" + "".join(f"<li>{o}</li>" for o in card.get("options", [])) + "</ol>"

        def _li(i, opt):
            comment = comments[i] if i < len(comments) else ""
            one = i + 1
            cls = "correct" if one in correct_set else ""
            marker = " ✓" if one in correct_set else ""
            cmt = f"<br><small>{comment}</small>" if comment else ""
            return f"<li class='{cls}'>{opt}{marker}{cmt}</li>"

        opts_rev = "<ol>" + "".join(_li(i, o) for i, o in enumerate(card.get("options", []))) + "</ol>"
        anki_deck.add_note(genanki.Note(
            model=model,
            fields=[card["question"], opts_html, opts_rev, card.get("explanation", "")],
            guid=card["id"],
        ))

    tmp = tempfile.NamedTemporaryFile(suffix=".apkg", delete=False)
    tmp.close()
    genanki.Package(anki_deck).write_to_file(tmp.name)
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in deck["name"])

    @after_this_request
    def _cleanup(response):
        try: _os.unlink(tmp.name)
        except Exception: pass
        return response

    return send_file(tmp.name, as_attachment=True,
                     download_name=f"{safe_name}.apkg",
                     mimetype="application/octet-stream")


@app.route("/quiz")
def quiz():
    deck_ids = request.args.get("decks", "").split(",")
    deck_ids = [d.strip() for d in deck_ids if d.strip()]
    if not deck_ids:
        flash("No decks selected.", "danger")
        return redirect(url_for("index"))
    decks = [d for d in (storage.get_deck(i) for i in deck_ids) if d]
    if not decks:
        flash("None of the selected decks were found.", "danger")
        return redirect(url_for("index"))
    total_due = sum(len(due_cards(d["cards"])) for d in decks)
    return render_template("quiz.html", decks=decks, due_count=total_due)


# ---------------------------------------------------------------------------
# Quiz API
# ---------------------------------------------------------------------------

@app.route("/api/quiz/start", methods=["POST"])
def quiz_start():
    body = request.json
    deck_ids = body.get("decks", [])
    mode = body.get("mode", "sequential")

    if not deck_ids:
        return jsonify(error="No decks provided"), 400

    all_cards = []
    for deck_id in deck_ids:
        deck = storage.get_deck(deck_id)
        if not deck:
            continue
        cards = due_cards(deck["cards"]) if mode == "spaced" else deck["cards"]
        for card in cards:
            if not card.get("hidden"):
                all_cards.append({**card, "_deck_id": deck_id, "_deck_name": deck["name"]})

    if not all_cards:
        if mode != "spaced":
            return jsonify(error="No cards found in the selected decks."), 400
        for deck_id in deck_ids:
            deck = storage.get_deck(deck_id)
            if not deck:
                continue
            for card in sorted(deck["cards"], key=lambda c: c.get("next_review", "")):
                if not card.get("hidden"):
                    all_cards.append({**card, "_deck_id": deck_id, "_deck_name": deck["name"]})
        if not all_cards:
            return jsonify(error="No cards found."), 400

    card_by_id = {c["id"]: c for c in all_cards}
    ordered_refs = [{"deck_id": c["_deck_id"], "card_id": c["id"]} for c in all_cards]

    session_id = str(uuid.uuid4())
    _quiz_sessions[session_id] = {
        "mode": mode, "refs": ordered_refs,
        "index": 0, "correct": 0, "total": len(ordered_refs),
    }

    first = card_by_id[ordered_refs[0]["card_id"]]
    return jsonify(session_id=session_id, total=len(ordered_refs),
                   card=_safe_card(first, deck_name=first["_deck_name"]))


@app.route("/api/quiz/answer", methods=["POST"])
def quiz_answer():
    data = request.json
    session_id = data.get("session_id")
    card_id = data.get("card_id")
    user_answer = (data.get("answer") or "").strip()

    qs = _quiz_sessions.get(session_id)
    if not qs:
        return jsonify(error="Invalid or expired session"), 400

    ref = next((r for r in qs["refs"] if r["card_id"] == card_id), None)
    if not ref:
        return jsonify(error="Card not in session"), 400

    deck = storage.get_deck(ref["deck_id"])
    if not deck:
        return jsonify(error="Deck not found"), 404

    card = next((c for c in deck["cards"] if c["id"] == card_id), None)
    if not card:
        return jsonify(error="Card not found"), 404

    correct, display_answer = _evaluate_answer(card, user_answer)
    updated = sm2_update(card, 4 if correct else 1)
    storage.update_card(ref["deck_id"], card_id,
                        {k: updated[k] for k in ("ease_factor", "interval", "repetitions", "next_review")})

    if correct:
        qs["correct"] += 1
    qs["index"] += 1

    next_card = None
    if qs["index"] < len(qs["refs"]):
        next_ref = qs["refs"][qs["index"]]
        next_deck = storage.get_deck(next_ref["deck_id"])
        if next_deck:
            nc = next((c for c in next_deck["cards"] if c["id"] == next_ref["card_id"]), None)
            if nc:
                next_card = _safe_card(nc, deck_name=next_deck["name"])

    return jsonify(
        correct=correct, correct_answer=display_answer,
        explanation=card.get("explanation", ""),
        sources=card.get("sources", {}),
        next_review=updated["next_review"], interval=updated["interval"],
        score=qs["correct"], answered=qs["index"], total=qs["total"],
        next_card=next_card,
    )


@app.route("/api/quiz/skip", methods=["POST"])
def quiz_skip():
    data = request.json
    session_id = data.get("session_id")
    qs = _quiz_sessions.get(session_id)
    if not qs:
        return jsonify(error="Invalid or expired session"), 400
    qs["index"] += 1
    next_card = _load_next_card(qs)
    return jsonify(next_card=next_card, answered=qs["index"],
                   score=qs["correct"], total=qs["total"])


@app.route("/api/card/hide", methods=["POST"])
def card_hide():
    data = request.json
    session_id = data.get("session_id")
    card_id = data.get("card_id")
    advance = data.get("advance", False)
    qs = _quiz_sessions.get(session_id)
    if not qs:
        return jsonify(error="Invalid session"), 400
    ref = next((r for r in qs["refs"] if r["card_id"] == card_id), None)
    if ref:
        storage.update_card(ref["deck_id"], card_id, {"hidden": True})
    if advance:
        qs["index"] += 1
    next_card = _load_next_card(qs)
    return jsonify(ok=True, next_card=next_card, answered=qs["index"],
                   score=qs["correct"], total=qs["total"])


def _load_next_card(qs: dict) -> dict | None:
    if qs["index"] < len(qs["refs"]):
        ref = qs["refs"][qs["index"]]
        deck = storage.get_deck(ref["deck_id"])
        if deck:
            card = next((c for c in deck["cards"] if c["id"] == ref["card_id"]), None)
            if card:
                return _safe_card(card, deck_name=deck["name"])
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evaluate_answer(card: dict, user_answer: str) -> tuple[bool, str]:
    correct_set = set(int(a) for a in card.get("answer", []))
    selected = {int(a) for a in user_answer.split(",") if a.strip().isdigit()}
    is_correct = selected == correct_set
    display = " and ".join(str(i) for i in sorted(correct_set))
    return is_correct, display


def _safe_card(card: dict, deck_name: str = "") -> dict:
    return {
        "id": card["id"],
        "type": card["type"],
        "question": card["question"],
        "options": card.get("options", []),
        "comments": card.get("comments", []),
        "deck_name": deck_name or card.get("_deck_name", ""),
        "chapter": card.get("chapter", ""),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    print(f"Starting mkexam quiz at http://localhost:{args.port}")
    app.run(debug=True, port=args.port)
