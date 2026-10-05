from __future__ import annotations

import pandas as pd
import numpy as np


def build_runner_scores(runners: pd.DataFrame) -> pd.DataFrame:
    df = runners.copy()

    df["best_price_dec"] = pd.to_numeric(df["best_price_dec"], errors="coerce")
    df = df.dropna(subset=["best_price_dec"]).copy()
    if df.empty:
        return df

    # Market probability (raw then remove overround within race)
    df["market_prob_raw"] = 1.0 / df["best_price_dec"]
    df["market_prob"] = df.groupby("race_id")["market_prob_raw"].transform(
        lambda s: s / max(1e-9, s.sum())
    )

    # Rating feature (only if present)
    df["rating"] = pd.to_numeric(df.get("rating", np.nan), errors="coerce")
    has_rating = df["rating"].notna() & (df["rating"] > 0)

    df["rating_norm"] = 0.0
    if has_rating.any():
        df.loc[has_rating, "rating_norm"] = (
            df[has_rating]
            .groupby("race_id")["rating"]
            .transform(lambda s: (s - s.min()) / max(1e-9, (s.max() - s.min())))
        )

    # Recency
    df["days_since"] = (
        pd.to_numeric(df.get("days_since", 60), errors="coerce").fillna(60).clip(0, 120)
    )
    df["recency"] = 1.0 - (df["days_since"].clip(0, 60) / 60.0)

    # Course & distance indicator
    cd_raw = df.get("course_distance", "")
    df["cd"] = (
        cd_raw.astype(str).str.contains("cd", case=False, na=False).astype(int)
        if not isinstance(cd_raw, str)
        else 0
    )

    # Field size (smaller fields → slightly higher confidence on top picks)
    df["runner_count"] = df.groupby("race_id")["runner"].transform("count")
    df["field_factor"] = (1.0 / df["runner_count"].clip(lower=4)).clip(0.05, 0.25)

    # Composite score
    # When we have real ratings we use them; otherwise lean more on market + field size
    df["score_raw"] = (
        0.45 * df["rating_norm"]
        + 0.20 * df["recency"]
        + 0.15 * df["cd"]
        + 0.20 * df["field_factor"]
    ).clip(0, 1)

    # Soft blend with market so we never produce pure noise when features are empty
    df["score_blended"] = (0.35 * df["score_raw"] + 0.65 * df["market_prob"]).clip(1e-6, 1.0)

    # Model probability per race
    df["model_prob"] = df.groupby("race_id")["score_blended"].transform(
        lambda s: s / max(1e-9, s.sum())
    )

    # Only claim an edge where real form features exist in the race.
    # Without them the blend just shrinks prices toward 1/N, which makes
    # every outsider look like "value" - that is maths, not an edge.
    has_form = (df["rating_norm"] > 0) | (df["recency"] > 0) | (df["cd"] > 0)
    race_has_form = has_form.groupby(df["race_id"]).transform("any")
    df.loc[~race_has_form, "model_prob"] = df.loc[~race_has_form, "market_prob"]
    df["has_form"] = race_has_form.astype(int)

    # Value edge vs normalised market
    df["value_edge"] = df["model_prob"] - df["market_prob"]

    df["confidence"] = (df["score_blended"] * 100).round(1)

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
    df = df[df["value_edge"] >= float(min_edge)].copy()
    if df.empty:
        return df

    df["suggested_stake_units"] = 1
    return df.sort_values("value_edge", ascending=False).head(50)
