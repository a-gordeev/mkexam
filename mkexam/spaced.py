from datetime import date, timedelta


def sm2_update(card: dict, quality: int) -> dict:
    """
    Update SM-2 spaced repetition fields on a card.
    quality: 0-5 (0-2 = fail, 3-5 = pass)
    """
    ef = card.get("ease_factor", 2.5)
    interval = card.get("interval", 1)
    reps = card.get("repetitions", 0)

    if quality < 3:
        reps = 0
        interval = 1
    else:
        if reps == 0:
            interval = 1
        elif reps == 1:
            interval = 6
        else:
            interval = round(interval * ef)
        reps += 1
        ef = ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        ef = max(1.3, ef)

    next_review = (date.today() + timedelta(days=interval)).isoformat()

    return {
        **card,
        "ease_factor": round(ef, 2),
        "interval": interval,
        "repetitions": reps,
        "next_review": next_review,
    }


def due_cards(cards: list) -> list:
    """Return cards due for review today or earlier (never-reviewed cards are always due)."""
    today = date.today().isoformat()
    return [c for c in cards if c.get("next_review", today) <= today]
