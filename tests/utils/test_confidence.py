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
"""

from datetime import datetime, timedelta, timezone

from graphiti_core.utils.confidence import (
    MAX_UNCERTAINTY,
    MIN_UNCERTAINTY,
    Confidence,
    contest,
    corroborate,
    initial_confidence,
    is_contested,
    with_time_decay,
)

NOW = datetime(2026, 7, 11, 12, 0, 0, tzinfo=timezone.utc)


def test_initial_confidence_seeds_from_extraction_certainty():
    c = initial_confidence(0.9, now=NOW)
    assert c.rating == 0.9
    assert c.corroboration_count == 1
    assert c.last_touched_at == NOW


def test_with_time_decay_widens_uncertainty_and_does_not_mutate():
    c = Confidence(rating=0.7, uncertainty=0.3, last_touched_at=NOW - timedelta(days=10))
    decayed = with_time_decay(c, NOW)
    assert decayed.uncertainty == 0.3 + 10 * 0.01  # 10 days * 0.01/day
    assert c.uncertainty == 0.3  # original untouched (read-time only)


def test_with_time_decay_clamps_to_max():
    c = Confidence(rating=0.5, uncertainty=0.95, last_touched_at=NOW - timedelta(days=100))
    assert with_time_decay(c, NOW).uncertainty == MAX_UNCERTAINTY


def test_with_time_decay_none_anchor_is_noop():
    c = Confidence(rating=0.7, uncertainty=0.3, last_touched_at=None)
    assert with_time_decay(c, NOW).uncertainty == 0.3


def test_with_time_decay_treats_naive_anchor_as_utc():
    naive = (NOW - timedelta(days=5)).replace(tzinfo=None)
    c = Confidence(rating=0.7, uncertainty=0.3, last_touched_at=naive)
    assert with_time_decay(c, NOW).uncertainty == 0.3 + 5 * 0.01


def test_corroborate_raises_rating_narrows_uncertainty_bumps_count():
    c = Confidence(rating=0.5, uncertainty=0.4, last_touched_at=NOW, corroboration_count=1)
    out = corroborate(c, NOW)
    assert out.rating == 0.5 + (1 - 0.5) * 0.3  # 0.65
    assert out.uncertainty == 0.4 * 0.7  # 0.28
    assert out.corroboration_count == 2
    assert out.last_touched_at == NOW


def test_corroborate_floors_uncertainty_at_min():
    c = Confidence(rating=0.9, uncertainty=0.06, last_touched_at=NOW)
    assert corroborate(c, NOW).uncertainty == MIN_UNCERTAINTY


def test_contest_lowers_rating_widens_uncertainty_count_unchanged():
    c = Confidence(rating=0.8, uncertainty=0.3, last_touched_at=NOW, corroboration_count=3)
    out = contest(c, NOW)
    assert out.rating == 0.8 * 0.7  # 0.56
    assert out.uncertainty == 0.3 + 0.3  # 0.6
    assert out.corroboration_count == 3  # unchanged


def test_contest_clamps_uncertainty_to_max():
    c = Confidence(rating=0.5, uncertainty=0.9, last_touched_at=NOW)
    assert contest(c, NOW).uncertainty == MAX_UNCERTAINTY


def test_corroborate_and_contest_apply_decay_first():
    # last touched 20 days ago: decay adds 0.2 to uncertainty before the update.
    c = Confidence(rating=0.5, uncertainty=0.2, last_touched_at=NOW - timedelta(days=20))
    # corroborate: decayed uncertainty 0.4, then *0.7 -> 0.28
    assert abs(corroborate(c, NOW).uncertainty - 0.28) < 1e-9
    # contest: decayed uncertainty 0.4, then +0.3 -> 0.7
    assert abs(contest(c, NOW).uncertainty - 0.7) < 1e-9


def test_is_contested_threshold():
    assert is_contested(Confidence(uncertainty=0.6)) is True
    assert is_contested(Confidence(uncertainty=0.59)) is False
    assert is_contested(Confidence(uncertainty=0.4), uncertainty_threshold=0.3) is True
