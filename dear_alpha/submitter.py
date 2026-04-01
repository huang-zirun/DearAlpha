"""
Alpha persistence and submission management.

Stores passing alphas to a JSON results file and handles
rate-limited, once-per-day submission to WorldQuant Brain.
"""

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class ResultStore:
    """
    Append-only JSON-lines store for passing alphas.
    Each record is one line of JSON for easy streaming reads.
    """

    def __init__(self, path: str = "results/passing_alphas.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, record: dict):
        """Append one record to the store."""
        record.setdefault("saved_at", datetime.utcnow().isoformat())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info("Saved alpha %s → %s", record.get("alpha_id", "?"), self.path)

    def load_all(self) -> list[dict]:
        """Return all saved records."""
        if not self.path.exists():
            return []
        records = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def load_unsubmitted(self) -> list[dict]:
        return [r for r in self.load_all() if not r.get("submitted")]

    def mark_submitted(self, alpha_id: str):
        """Rewrite the file marking the given alpha as submitted."""
        records = self.load_all()
        for r in records:
            if r.get("alpha_id") == alpha_id:
                r["submitted"] = True
                r["submitted_at"] = datetime.utcnow().isoformat()
        # Rewrite
        with self.path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


class Submitter:
    """
    Submits passing alphas to WorldQuant Brain.

    Enforces a daily submission limit and checks PROD_CORRELATION
    before submitting.
    """

    def __init__(
        self,
        brain,                          # BrainClient
        store: ResultStore,
        max_per_day: int = 10,
        max_prod_corr: float = 0.70,
    ):
        self.brain = brain
        self.store = store
        self.max_per_day = max_per_day
        self.max_prod_corr = max_prod_corr
        self._submitted_today: list[str] = []
        self._today = date.today()

    def _reset_if_new_day(self):
        today = date.today()
        if today != self._today:
            self._today = today
            self._submitted_today = []

    def _slots_remaining(self) -> int:
        self._reset_if_new_day()
        return max(0, self.max_per_day - len(self._submitted_today))

    def run(self, dry_run: bool = False) -> list[str]:
        """
        Submit up to max_per_day alphas that pass the prod-correlation check.
        Returns list of submitted alpha IDs.
        """
        slots = self._slots_remaining()
        if slots == 0:
            log.info("Daily submission quota reached (%d). Skipping.", self.max_per_day)
            return []

        candidates = self.store.load_unsubmitted()
        log.info("%d unsubmitted alphas, %d submission slots today", len(candidates), slots)

        submitted = []
        for record in candidates:
            if len(submitted) >= slots:
                break

            alpha_id = record.get("alpha_id")
            if not alpha_id:
                continue

            # Check prod correlation
            pc = self.brain.prod_correlation(alpha_id)
            if pc is None:
                log.info("Alpha %s failed submission checks, skipping", alpha_id)
                continue
            if pc > self.max_prod_corr:
                log.info(
                    "Alpha %s PROD_CORRELATION %.3f > %.3f, skipping",
                    alpha_id, pc, self.max_prod_corr,
                )
                continue

            if not dry_run:
                ok = self.brain.submit_alpha(alpha_id)
                if ok:
                    self.store.mark_submitted(alpha_id)
                    self._submitted_today.append(alpha_id)
                    submitted.append(alpha_id)
            else:
                log.info("[dry_run] Would submit alpha %s (pc=%.3f)", alpha_id, pc)
                submitted.append(alpha_id)

        log.info("Submitted %d alphas this run", len(submitted))
        return submitted
