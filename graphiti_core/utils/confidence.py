"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

Per-edge confidence: a (rating, uncertainty) pair with corroboration/contest
dynamics and read-time recency decay.

This is a deliberately simplified, Glicko-*shaped* model (not Glicko-2). The
point is to get the DYNAMICS right so the rest of the pipeline can gate on
trust:

- corroboration (an independent episode re-asserting a fact) raises rating and
  narrows uncertainty;
- contestation (a contradicting fact) lowers rating and widens uncertainty;
- elapsed time without reinforcement widens uncertainty (applied at read time
  only, never persisted).

The constants below are PLACEHOLDERS pending calibration against real ingest
data -- do not treat them as tuned. They are intentionally shared as module
constants so a future calibration changes one place.

Ported in spirit from the memory-engine prototype's ``confidence.py``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from graphiti_core.utils.datetime_utils import ensure_utc, utc_now

# --- Dynamics constants (UNCALIBRATED placeholders) ---------------------------
CORROBORATION_WEIGHT = 0.3
CONTESTATION_WEIGHT = 0.3
UNCERTAINTY_DECAY_PER_DAY = 0.01  # widening rate absent any new evidence
MIN_UNCERTAINTY = 0.05
MAX_UNCERTAINTY = 1.0
INITIAL_UNCERTAINTY = 0.5
# Uncertainty at/above which a fact is "contested" -- wide enough to warrant
# hedging or retrieval escalation rather than confident proactive injection.
CONTESTED_UNCERTAINTY_THRESHOLD = 0.6


class Confidence(BaseModel):
    """A fact's trust state.

    rating: how likely the fact is currently true (~0-1).
    uncertainty: how little recent corroboration has been seen; widens over
        time without reinforcement, narrows with corroboration.
    last_touched_at: when rating/uncertainty were last updated by evidence --
        the anchor for read-time decay.
    corroboration_count: how many independent episodes have asserted this fact.
    """

    rating: float = 0.75
    uncertainty: float = INITIAL_UNCERTAINTY
    last_touched_at: datetime | None = None
    corroboration_count: int = 1


def initial_confidence(extraction_certainty: float, now: datetime | None = None) -> Confidence:
    """Confidence for a brand-new fact, before any corroboration/contestation.

    ``extraction_certainty`` seeds the rating (1.0 = plainly stated, lower =
    hedged in the source text).
    """
    return Confidence(
        rating=extraction_certainty,
        uncertainty=INITIAL_UNCERTAINTY,
        last_touched_at=now or utc_now(),
        corroboration_count=1,
    )


def with_time_decay(confidence: Confidence, now: datetime | None = None) -> Confidence:
    """Widen uncertainty by elapsed time since last reinforcement.

    READ-TIME ONLY: returns a copy and never mutates stored state. Naive
    ``last_touched_at`` (e.g. from a FalkorDB round-trip) is treated as UTC.
    """
    now = now or utc_now()
    anchor = ensure_utc(confidence.last_touched_at)
    if anchor is None:
        return confidence
    elapsed_days = max(0.0, (now - anchor).total_seconds() / 86400.0)
    widened = confidence.uncertainty + elapsed_days * UNCERTAINTY_DECAY_PER_DAY
    return confidence.model_copy(update={'uncertainty': min(MAX_UNCERTAINTY, widened)})


def corroborate(confidence: Confidence, now: datetime | None = None) -> Confidence:
    """An independent new episode restates the same fact: raise rating, narrow
    uncertainty, bump the count. Decays first so age is accounted for."""
    now = now or utc_now()
    decayed = with_time_decay(confidence, now)
    new_rating = decayed.rating + (1.0 - decayed.rating) * CORROBORATION_WEIGHT
    new_uncertainty = max(MIN_UNCERTAINTY, decayed.uncertainty * (1.0 - CORROBORATION_WEIGHT))
    return Confidence(
        rating=new_rating,
        uncertainty=new_uncertainty,
        last_touched_at=now,
        corroboration_count=confidence.corroboration_count + 1,
    )


def contest(confidence: Confidence, now: datetime | None = None) -> Confidence:
    """A new fact conflicts with this one: lower rating, widen uncertainty. The
    corroboration count is unchanged (contestation is not corroboration)."""
    now = now or utc_now()
    decayed = with_time_decay(confidence, now)
    new_rating = decayed.rating * (1.0 - CONTESTATION_WEIGHT)
    new_uncertainty = min(MAX_UNCERTAINTY, decayed.uncertainty + CONTESTATION_WEIGHT)
    return Confidence(
        rating=new_rating,
        uncertainty=new_uncertainty,
        last_touched_at=now,
        corroboration_count=confidence.corroboration_count,
    )


def is_contested(
    confidence: Confidence, uncertainty_threshold: float = CONTESTED_UNCERTAINTY_THRESHOLD
) -> bool:
    """Whether this fact's uncertainty is wide enough to warrant hedging."""
    return confidence.uncertainty >= uncertainty_threshold
