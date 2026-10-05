from __future__ import annotations

import re

import numpy as np
import pandas as pd

# How far form is allowed to move a horse away from its market price.
# 0.25 means a horse one standard deviation better than its rivals on form
# gets its market probability multiplied by ~1.28. Keep this modest until
# BET_RECS_LOG results show the features actually predict winners.
FORM_STRENGTH = 0.25

# Relative importance of each feature inside the form score
WEIGHTS = {"rating_z": 0.45, "form_z": 0.35, "recency_z": 0.10, "cd": 0.10}

_POS_POINTS = {"1": 1.0, "2": 0.7, "3": 0.5, "4": 0.3}


def form_string_score(form: str) -> float:
    """
    Score recent finishing positions, most recent run weighted highest.
    '1-23' -> runs 1, 2, 3 with 3 the most recent. Falls/PU/0 score zero.
    Returns NaN when there is no usable form.
    """
    s = str(form or "").strip().upper()
    if not s or s == "NAN":
        return np.nan
    # Keep only the most recent season (after the last '/' or '-')
    s = re.split(r"[/\-]", s)[-1] or s.replace("/", "").replace("-", "")
    runs = [c for c in s if c.isdigit() or c.isalpha()][-6:]
    if not runs:
        return np.nan
    total, wsum = 0.0, 0.0
    for age, c in enumerate(reversed(runs)):  # age 0 = latest run
        w = 0.75 ** age
        total += w * _POS_POINTS.get(c, 0.0)
        wsum += w
    return total / wsum if wsum else np.nan


def _zscore_in_race(df: pd.DataFrame, col: str) -> pd.Series:
    """Standardise a feature within each race; missing values -> 0 (neutral)."""
    g = df.groupby("race_id")[col]
    mean = g.transform("mean")
    std = g.transform("std")
    n = g.transform("count")
    z = (df[col] - mean) / std.replace(0, np.nan)
    z = z.where(n >= 2)  # need at least two horses with data to compare
    return z.fillna(0.0).clip(-2.5, 2.5)


def build_runner_scores(runners: pd.DataFrame) -> pd.DataFrame:
    df = runners.copy()

    df["best_price_dec"] = pd.to_numeric(df["best_price_dec"], errors="coerce")
    df = df.dropna(subset=["best_price_dec"]).copy()
    if df.empty:
        return df

    # Market probability, overround removed within race
    df["market_prob_raw"] = 1.0 / df["best_price_dec"]
    df["market_prob"] = df.groupby("race_id")["market_prob_raw"].transform(
        lambda s: s / max(1e-9, s.sum())
    )
    df["runner_count"] = df.groupby("race_id")["runner"].transform("count")

    # --- Features (each NaN when not available) ---
    df["rating"] = pd.to_numeric(df.get("rating", np.nan), errors="coerce")
    df.loc[df["rating"] <= 0, "rating"] = np.nan

    df["form"] = df.get("form", "").astype(str).replace({"nan": ""})
    df["form_score"] = df["form"].apply(form_string_score)

    df["days_since"] = pd.to_numeric(df.get("days_since", np.nan), errors="coerce")
    # Freshness: best around 14-35 days, worse when very long absent
    df["recency"] = np.where(
        df["days_since"].notna(),
        1.0 - (df["days_since"].clip(7, 180) - 21).abs() / 160.0,
        np.nan,
    )

    cd_raw = df.get("course_distance", "")
    cd_str = cd_raw.astype(str) if not isinstance(cd_raw, str) else pd.Series("", index=df.index)
    df["cd"] = cd_str.str.contains(r"CD|C&D|C\s*D", case=False, na=False).astype(float)
    df.loc[cd_str.str.contains(r"\b[CD]\b", case=False, na=False) & (df["cd"] == 0), "cd"] = 0.5

    # --- Standardise within race so outsiders don't get a free lift ---
    df["rating_z"] = _zscore_in_race(df, "rating")
    df["form_z"] = _zscore_in_race(df, "form_score")
    df["recency_z"] = _zscore_in_race(df, "recency")
    df["cd_z"] = df["cd"] - df.groupby("race_id")["cd"].transform("mean")

    df["form_composite"] = (
        WEIGHTS["rating_z"] * df["rating_z"]
        + WEIGHTS["form_z"] * df["form_z"]
        + WEIGHTS["recency_z"] * df["recency_z"]
        + WEIGHTS["cd"] * df["cd_z"]
    )

    # A race "has form" if at least two runners have rating or form data
    has_feat = df["rating"].notna() | df["form_score"].notna()
    df["has_form"] = (has_feat.groupby(df["race_id"]).transform("sum") >= 2).astype(int)

    # --- Model probability: market price nudged by form, renormalised ---
    adj = np.exp(FORM_STRENGTH * df["form_composite"].where(df["has_form"] == 1, 0.0))
    df["score_blended"] = df["market_prob"] * adj
    df["model_prob"] = df.groupby("race_id")["score_blended"].transform(
        lambda s: s / max(1e-9, s.sum())
    )

    df["value_edge"] = df["model_prob"] - df["market_prob"]
    df["fair_odds"] = (1.0 / df["model_prob"]).round(2)
    df["confidence"] = (df["model_prob"] * 100).round(1)

    return df.sort_values(
        ["date", "course", "off_time", "value_edge"],
        ascending=[True, True, True, False],
    )


def build_value_bets(scored: pd.DataFrame, min_edge: float) -> pd.DataFrame:
    df = scored.copy()
    if df.empty:
        return df

    df["value_edge"] = pd.to_numeric(df["value_edge"], errors="coerce")
    df = df.dropna(subset=["value_edge"]).copy()
    df = df[(df["value_edge"] >= float(min_edge)) & (df.get("has_form", 1) == 1)].copy()
    if df.empty:
        return df

    df["suggested_stake_units"] = 1
    return df.sort_values("value_edge", ascending=False).head(50)
