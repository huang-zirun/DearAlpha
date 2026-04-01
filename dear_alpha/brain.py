"""
WorldQuant Brain API client.
Handles auth, simulation, alpha retrieval, and submission checks.
"""

import json
import logging
import re
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)

WQ_BASE = "https://api.worldquantbrain.com"


class BrainClient:
    """Thin wrapper around the WorldQuant Brain REST API."""

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.session = requests.Session()
        self._login()

    # ------------------------------------------------------------------ auth

    def _login(self):
        log.info("Authenticating with WorldQuant Brain …")
        self.session.auth = (self.username, self.password)
        r = self.session.post(f"{WQ_BASE}/authentication")
        if r.status_code != 201:
            raise RuntimeError(f"Auth failed ({r.status_code}): {r.text}")
        self.session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )
        self.session.auth = None  # session cookie now handles auth
        log.info("Authenticated OK")

    def _ensure_session(self):
        """Re-login if the session has expired (401)."""
        self._login()

    # -------------------------------------------------------------- simulate

    def simulate(
        self,
        expression: str,
        region: str = "USA",
        universe: str = "TOP3000",
        neutralization: str = "SUBINDUSTRY",
        decay: int = 0,
        delay: int = 1,
        truncation: float = 0.08,
    ) -> Optional[dict]:
        """
        Submit one simulation and block until it finishes.
        Returns the full result dict, or None on failure.
        """
        payload = {
            "type": "REGULAR",
            "settings": {
                "instrumentType": "EQUITY",
                "region": region,
                "universe": universe,
                "delay": delay,
                "decay": decay,
                "neutralization": neutralization,
                "truncation": truncation,
                "pasteurization": "ON",
                "unitHandling": "VERIFY",
                "nanHandling": "OFF",
                "language": "FASTEXPR",
                "visualization": False,
            },
            "regular": expression,
        }

        for attempt in range(3):
            r = self.session.post(f"{WQ_BASE}/simulations", json=payload)
            if r.status_code == 401:
                self._ensure_session()
                continue
            if r.status_code != 201:
                log.warning("Simulation submit failed: %s", r.text)
                return None
            break
        else:
            return None

        progress_url = r.headers.get("Location")
        if not progress_url:
            log.error("No Location header returned")
            return None

        return self._poll(progress_url)

    def _poll(self, url: str) -> Optional[dict]:
        while True:
            r = self.session.get(url)
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                time.sleep(float(retry_after))
                continue
            data = r.json()
            status = data.get("status")
            if status == "COMPLETE":
                return data
            if status in ("FAILED", "ERROR"):
                log.warning("Simulation %s: %s", status, data.get("message", ""))
                return None
            # Still running but no Retry-After – shouldn't happen, but guard it
            time.sleep(2)

    # --------------------------------------------------------- data fields

    def get_datafields(
        self,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        dataset_id: str = "",
        search: str = "",
    ) -> list[dict]:
        """Return all data fields available for the given settings."""
        base = (
            f"{WQ_BASE}/data-fields?"
            f"instrumentType=EQUITY&region={region}&delay={delay}"
            f"&universe={universe}&dataset.id={dataset_id}&limit=50"
        )
        if search:
            base += f"&search={search}"

        # First call to get total count
        r = self.session.get(base + "&offset=0")
        data = r.json()
        count = data.get("count", 50) if not search else 100

        fields = list(data.get("results", []))
        for offset in range(50, count, 50):
            r = self.session.get(base + f"&offset={offset}")
            fields.extend(r.json().get("results", []))

        return fields

    # ----------------------------------------------------------- alpha list

    def get_user_alphas(
        self,
        limit: int = 100,
        offset: int = 0,
        min_sharpe: float = 1.25,
        min_fitness: float = 1.0,
        region: str = "USA",
    ) -> list[dict]:
        """Fetch the current user's unsubmitted, passing alphas."""
        url = (
            f"{WQ_BASE}/users/self/alphas"
            f"?limit={limit}&offset={offset}"
            f"&status=UNSUBMITTED%1FIS_FAIL"
            f"&is.fitness%3E{min_fitness}&is.sharpe%3E{min_sharpe}"
            f"&settings.region={region}&order=-is.sharpe"
            f"&hidden=false&type!=SUPER"
        )
        r = self.session.get(url)
        if r.status_code != 200:
            log.warning("get_user_alphas: %s", r.text)
            return []
        return r.json().get("results", [])

    # -------------------------------------------------- submission check

    def check_submission(self, alpha_id: str) -> Optional[dict]:
        """
        Poll the /check endpoint until ready.
        Returns the full JSON (which includes is.checks), or None.
        """
        url = f"{WQ_BASE}/alphas/{alpha_id}/check"
        while True:
            r = self.session.get(url)
            if "Retry-After" in r.headers:
                time.sleep(float(r.headers["Retry-After"]))
                continue
            if r.status_code != 200:
                log.warning("check_submission failed: %s", r.text)
                return None
            return r.json()

    def prod_correlation(self, alpha_id: str) -> Optional[float]:
        """Return PROD_CORRELATION value, or None if check fails/is in FAIL."""
        data = self.check_submission(alpha_id)
        if not data:
            return None
        try:
            checks = data["is"]["checks"]
            if any(c["result"] == "FAIL" for c in checks):
                return None
            for c in checks:
                if c["name"] == "PROD_CORRELATION":
                    return float(c["value"])
        except (KeyError, TypeError, ValueError):
            return None
        return None

    # ----------------------------------------------------------- submission

    def submit_alpha(self, alpha_id: str) -> bool:
        """Submit an alpha for review."""
        r = self.session.post(f"{WQ_BASE}/alphas/{alpha_id}/submit")
        if r.status_code in (200, 201):
            log.info("Submitted alpha %s", alpha_id)
            return True
        log.warning("Submit failed for %s: %s", alpha_id, r.text)
        return False

    # ------------------------------------------------------- helpers

    @staticmethod
    def extract_metrics(result: dict) -> dict:
        """Pull key IS metrics from a completed simulation result."""
        is_ = result.get("is", {})
        return {
            "alpha_id": result.get("id"),
            "sharpe": is_.get("sharpe"),
            "fitness": is_.get("fitness"),
            "turnover": is_.get("turnover"),
            "margin": is_.get("margin"),
            "long_count": is_.get("longCount"),
            "short_count": is_.get("shortCount"),
            "returns": is_.get("returns"),
        }
