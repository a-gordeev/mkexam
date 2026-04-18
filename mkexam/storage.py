import json
import uuid
from datetime import date
from pathlib import Path


class DeckStorage:
    def __init__(self, decks_dir: Path):
        self.dir = decks_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def list_decks(self) -> list:
        decks = []
        for f in sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                decks.append({
                    "id": data["id"],
                    "name": data["name"],
                    "created": data.get("created", ""),
                    "card_count": len(data.get("cards", [])),
                    "sources": data.get("sources", []),
                })
            except Exception:
                pass
        return decks

    def get_deck(self, deck_id: str) -> dict | None:
        path = self.dir / f"{deck_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_deck(self, deck: dict) -> str:
        if "id" not in deck:
            deck["id"] = str(uuid.uuid4())
        if "created" not in deck:
            deck["created"] = date.today().isoformat()
        path = self.dir / f"{deck['id']}.json"
        path.write_text(json.dumps(deck, indent=2, ensure_ascii=False), encoding="utf-8")
        return deck["id"]

    def update_card(self, deck_id: str, card_id: str, updates: dict) -> bool:
        deck = self.get_deck(deck_id)
        if not deck:
            return False
        for i, card in enumerate(deck["cards"]):
            if card["id"] == card_id:
                deck["cards"][i].update(updates)
                self.save_deck(deck)
                return True
        return False

    def delete_deck(self, deck_id: str) -> bool:
        path = self.dir / f"{deck_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False
