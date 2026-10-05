from __future__ import annotations

import os
import re
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from src.sources.irishracing import IrishRacingClient
from src.scoring import build_runner_scores, build_value_bets
from src.sheets import SheetsWriter

TZ = ZoneInfo("Europe/Dublin")
NIGHT_BEFORE_HOUR = 22  # 22:00 Dublin time

MIN_MOVE_PCT = 0.02
MAX_RUNNERS_FOR_SIGNAL = 18


def env(name: str, default: str | None = None) -> str:
    v = os.getenv(name)
    if v is None or v == "":
        if default is None:
            raise RuntimeError(f"Missing required env var: {name}")
        return default
    return v


def _normalize_off_time(s: str) -> str:
    """
    Racecard times are 12-hour without am/pm ("1.55" = 13:55).
    Sheets may also have turned "4.00" into "4" or "1.50" into "1.5".
    Returns 24-hour "HH:MM".
    """
    t = str(s).strip().replace(":", ".")
    if not t:
        return t
    m = re.match(r"^(\d{1,2})(?:\.(\d{1,2}))?$", t)
    if not m:
        return t
    hh = int(m.group(1))
    mm_s = m.group(2) or "00"
    if len(mm_s) == 1:
        mm_s += "0"
    mm = int(mm_s)
    if 1 <= hh <= 9:
        hh += 12
    return f"{hh:02d}:{mm:02d}"


def _prep_snapshots_df(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required = {"snapshot_time", "race_id", "runner", "best_price_dec"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], errors="coerce", utc=True)
    df["best_price_dec"] = pd.to_numeric(df["best_price_dec"], errors="coerce")
    df = df.dropna(subset=["snapshot_time", "best_price_dec", "race_id", "runner"]).copy()

    df["race_id"] = df["race_id"].astype(str).str.strip()
    df["runner"] = df["runner"].astype(str).str.strip()
    df["snapshot_local"] = df["snapshot_time"].dt.tz_convert(TZ)

    if "off_time" in df.columns:
        df["off_time_norm"] = df["off_time"].astype(str).apply(_normalize_off_time)
    else:
        df["off_time_norm"] = ""

    df["race_dt"] = pd.NaT
    if "date" in df.columns and "off_time_norm" in df.columns:
        race_dt = pd.to_datetime(
            df["date"].astype(str) + " " + df["off_time_norm"].astype(str), errors="coerce"
        )
        race_dt = race_dt.dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT")
        df["race_dt"] = race_dt

    return df


def compute_movers_last_two(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.dropna(subset=["snapshot_local", "best_price_dec", "race_id", "runner"]).copy()
    d = d.sort_values(["race_id", "runner", "snapshot_local"])
    key = ["race_id", "runner"]

    last_two = d.groupby(key).tail(2).copy()
    counts = last_two.groupby(key).size().reset_index(name="n")
    valid = counts[counts["n"] >= 2][key]
    if valid.empty:
        return pd.DataFrame()

    last_two = last_two.merge(valid, on=key, how="inner")
    last_two["rn"] = last_two.groupby(key).cumcount()

    prev = last_two[last_two["rn"] == 0].copy()
    curr = last_two[last_two["rn"] == 1].copy()

    curr_out = curr[key].copy()
    curr_out["price_now"] = curr["best_price_dec"].values
    curr_out["time_now"] = curr["snapshot_local"].values
    for c in ["date", "course", "off_time", "race_name"]:
        curr_out[c] = curr[c].values if c in curr.columns else ""

    prev_out = prev[key].copy()
    prev_out["price_prev"] = prev["best_price_dec"].values
    prev_out["time_prev"] = prev["snapshot_local"].values

    merged = prev_out.merge(curr_out, on=key, how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged["pct_change"] = (merged["price_now"] - merged["price_prev"]) / merged["price_prev"]
    merged["direction"] = merged["pct_change"].apply(lambda x: "SHORTENING" if x < 0 else "DRIFTING")
    merged = merged[merged["pct_change"].abs() >= MIN_MOVE_PCT].copy()

    keep = [
        "date", "course", "off_time", "race_name", "race_id", "runner",
        "price_prev", "price_now", "pct_change", "direction", "time_prev", "time_now",
    ]
    return merged[keep].sort_values("pct_change", ascending=True)


def compute_movers_night_before(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "race_dt" not in df.columns:
        return pd.DataFrame()

    d = df.dropna(subset=["race_dt"]).copy()
    if d.empty:
        return pd.DataFrame()

    d["night_start"] = (d["race_dt"].dt.normalize() - pd.Timedelta(days=1)) + pd.Timedelta(
        hours=NIGHT_BEFORE_HOUR
    )
    d["cutoff_30m"] = d["race_dt"] - pd.Timedelta(minutes=30)

    key = ["race_id", "runner"]

    night = d[d["snapshot_local"] >= d["night_start"]].sort_values("snapshot_local")
    night_first = night.groupby(key, as_index=False).first()

    pre = d[d["snapshot_local"] <= d["cutoff_30m"]].sort_values("snapshot_local")
    pre_last = pre.groupby(key, as_index=False).last()

    merged = night_first.merge(
        pre_last[key + ["best_price_dec", "snapshot_local"]],
        on=key,
        how="inner",
        suffixes=("_start", "_30m"),
    )
    if merged.empty:
        return pd.DataFrame()

    merged = merged.rename(
        columns={
            "best_price_dec_start": "start_price_night_before",
            "best_price_dec_30m": "price_30min_before",
            "snapshot_local_start": "time_start",
            "snapshot_local_30m": "time_30min_before",
        }
    )

    merged["pct_change"] = (
        merged["price_30min_before"] - merged["start_price_night_before"]
    ) / merged["start_price_night_before"]
    merged["direction"] = merged["pct_change"].apply(
        lambda x: "SHORTENING" if x < 0 else "DRIFTING"
    )
    merged = merged[merged["pct_change"].abs() >= MIN_MOVE_PCT].copy()

    for c in ["date", "course", "off_time", "race_name"]:
        if c not in merged.columns:
            merged[c] = ""

    keep = [
        "date", "course", "off_time", "race_name", "race_id", "runner",
        "start_price_night_before", "price_30min_before", "pct_change", "direction",
        "time_start", "time_30min_before",
    ]
    return merged[keep].sort_values("pct_change", ascending=True)


def compute_persistent_shorteners(snap_df: pd.DataFrame) -> pd.DataFrame:
    if snap_df is None or snap_df.empty:
        return pd.DataFrame()

    d = snap_df.dropna(subset=["race_id", "runner", "snapshot_local", "best_price_dec"]).copy()
    d = d.sort_values(["race_id", "runner", "snapshot_local"])
    key = ["race_id", "runner"]

    last4 = d.groupby(key).tail(4).copy()
    last4["prev_price"] = last4.groupby(key)["best_price_dec"].shift(1)
    last4["down"] = (last4["best_price_dec"] < last4["prev_price"]).astype(int)

    return (
        last4.groupby(key, as_index=False)["down"]
        .sum()
        .rename(columns={"down": "shorten_steps_last4"})
    )


def build_signals(
    scored: pd.DataFrame,
    movers_2h: pd.DataFrame,
    movers_night: pd.DataFrame,
    persistence: pd.DataFrame,
) -> pd.DataFrame:
    df = scored.copy()
    df["runner_count"] = pd.to_numeric(df.get("runner_count", 0), errors="coerce").fillna(0)
    df = df[df["runner_count"] <= MAX_RUNNERS_FOR_SIGNAL].copy()
    df["value_edge"] = pd.to_numeric(df.get("value_edge", 0), errors="coerce").fillna(0.0)

    m2 = (
        movers_2h[["race_id", "runner", "pct_change"]].rename(columns={"pct_change": "mover_2h_pct"})
        if (movers_2h is not None and not movers_2h.empty)
        else pd.DataFrame(columns=["race_id", "runner", "mover_2h_pct"])
    )
    mn = (
        movers_night[["race_id", "runner", "pct_change"]].rename(
            columns={"pct_change": "mover_night_pct"}
        )
        if (movers_night is not None and not movers_night.empty)
        else pd.DataFrame(columns=["race_id", "runner", "mover_night_pct"])
    )
    ps = (
        persistence
        if (persistence is not None and not persistence.empty)
        else pd.DataFrame(columns=["race_id", "runner", "shorten_steps_last4"])
    )

    for t in (m2, mn, ps):
        if not t.empty:
            t["race_id"] = t["race_id"].astype(str).str.strip()
            t["runner"] = t["runner"].astype(str).str.strip()

    df["race_id"] = df["race_id"].astype(str).str.strip()
    df["runner"] = df["runner"].astype(str).str.strip()

    out = (
        df.merge(m2, on=["race_id", "runner"], how="left")
        .merge(mn, on=["race_id", "runner"], how="left")
        .merge(ps, on=["race_id", "runner"], how="left")
    )

    out["mover_2h_pct"] = pd.to_numeric(out.get("mover_2h_pct", 0), errors="coerce").fillna(0.0)
    out["mover_night_pct"] = pd.to_numeric(out.get("mover_night_pct", 0), errors="coerce").fillna(0.0)
    out["shorten_steps_last4"] = (
        pd.to_numeric(out.get("shorten_steps_last4", 0), errors="coerce").fillna(0).astype(int)
    )

    out["shorten_2h_score"] = (-out["mover_2h_pct"]).clip(lower=0)
    out["shorten_night_score"] = (-out["mover_night_pct"]).clip(lower=0)

    out["signal_score"] = (
        1.0 * out["value_edge"].clip(lower=0)
        + 0.8 * out["shorten_2h_score"]
        + 1.2 * out["shorten_night_score"]
        + 0.15 * out["shorten_steps_last4"]
    )

    return out.sort_values(
        ["date", "course", "off_time", "signal_score"], ascending=[True, True, True, False]
    )


def build_bets_to_place(signals: pd.DataFrame, min_edge: float = 0.0) -> pd.DataFrame:
    if signals is None or signals.empty:
        return pd.DataFrame()

    df = signals.copy()
    df["signal_score"] = pd.to_numeric(df.get("signal_score", 0), errors="coerce").fillna(0.0)
    df["value_edge"] = pd.to_numeric(df.get("value_edge", 0), errors="coerce").fillna(0.0)
    df["mover_2h_pct"] = pd.to_numeric(df.get("mover_2h_pct", 0), errors="coerce").fillna(0.0)
    df["mover_night_pct"] = pd.to_numeric(df.get("mover_night_pct", 0), errors="coerce").fillna(0.0)

    # Keep only horses that have at least one clear reason
    edge_floor = max(float(min_edge), 1e-9)
    has_reason = (
        (df["value_edge"] >= edge_floor)
        | (df["mover_2h_pct"] <= -MIN_MOVE_PCT)
        | (df["mover_night_pct"] <= -MIN_MOVE_PCT)
    )
    df = df[has_reason].copy()
    if df.empty:
        return df

    # Prefer horses that have BOTH value and some shortening
    df["has_shortening"] = (
        (df["mover_2h_pct"] <= -MIN_MOVE_PCT) | (df["mover_night_pct"] <= -MIN_MOVE_PCT)
    ).astype(int)
    df["has_value"] = (df["value_edge"] >= edge_floor).astype(int)
    df["priority"] = df["has_value"] + df["has_shortening"]  # 0, 1 or 2

    # Rank inside each race by priority first, then signal_score
    df["_sort_key"] = df["priority"] * 1000 + df["signal_score"]
    df["rank_in_race"] = df.groupby("race_id")["_sort_key"].rank(ascending=False, method="first")
    df = df.drop(columns=["_sort_key"])
    df = df[df["rank_in_race"] <= 2].copy()

    df["suggested_stake_units"] = 1
    df["bet_key"] = (
        df["date"].astype(str)
        + "|"
        + df["course"].astype(str)
        + "|"
        + df["off_time"].astype(str)
        + "|"
        + df["runner"].astype(str)
    )

    return df.sort_values(
        ["date", "course", "off_time", "priority", "signal_score"],
        ascending=[True, True, True, False, False],
    )


def update_bet_recs_log(writer: SheetsWriter, bets_to_place: pd.DataFrame) -> None:
    if bets_to_place is None or bets_to_place.empty or "bet_key" not in bets_to_place.columns:
        return

    existing = writer.read_df("BET_RECS_LOG")
    existing_keys = (
        set(existing["bet_key"].astype(str).tolist())
        if (not existing.empty and "bet_key" in existing.columns)
        else set()
    )

    new_rows = bets_to_place[~bets_to_place["bet_key"].astype(str).isin(existing_keys)].copy()
    if new_rows.empty:
        return

    out = pd.DataFrame(index=new_rows.index)
    out["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["bet_key"] = new_rows["bet_key"]
    for c in [
        "date", "course", "off_time", "race_name", "runner", "best_price_dec",
        "signal_score", "value_edge", "mover_2h_pct", "mover_night_pct",
        "shorten_steps_last4", "suggested_stake_units", "fair_odds", "has_form",
    ]:
        out[c] = new_rows.get(c, "")
    out["result"] = ""
    out["pnl_units"] = ""
    out["notes"] = ""

    cols = [
        "timestamp_utc", "bet_key", "date", "course", "off_time", "race_name", "runner",
        "best_price_dec", "signal_score", "value_edge", "mover_2h_pct", "mover_night_pct",
        "shorten_steps_last4", "suggested_stake_units", "fair_odds", "has_form",
        "result", "pnl_units", "notes",
    ]
    writer.append_df("BET_RECS_LOG", out[cols])


def build_dashboard(
    races: pd.DataFrame,
    runners: pd.DataFrame,
    movers2h: pd.DataFrame,
    moversnight: pd.DataFrame,
    bets: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "metric": "last_run_local",
            "value": datetime.now(timezone.utc).astimezone(TZ).isoformat(timespec="seconds"),
        },
        {"metric": "races_target_day", "value": int(len(races)) if races is not None else 0},
        {"metric": "runners_target_day", "value": int(len(runners)) if runners is not None else 0},
        {"metric": "movers_2h_rows", "value": int(len(movers2h)) if movers2h is not None else 0},
        {
            "metric": "movers_night_rows",
            "value": int(len(moversnight)) if moversnight is not None else 0,
        },
        {"metric": "bets_to_place", "value": int(len(bets)) if bets is not None else 0},
    ]
    return pd.DataFrame(rows)


def main() -> int:
    sheet_name = env("SHEET_NAME")
    region = env("REGION", "all").lower()
    min_edge = float(env("MIN_VALUE_EDGE", "0.00"))

    client = IrishRacingClient(region=region)

    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(TZ)

    today = now_local.date()
    tomorrow = today + timedelta(days=1)

    # After 22:00 Dublin → target TOMORROW, else TODAY
    after_cutoff = now_local.hour >= NIGHT_BEFORE_HOUR
    target_label = "TOMORROW" if after_cutoff else "TODAY"
    target_date = tomorrow if after_cutoff else today

    races_df, runners_df = client.fetch_for_date(target_date)
    if races_df.empty or runners_df.empty:
        print(f"No races/runners found for target_date={target_date} ({target_label}).")
        return 0

    runners_df = client.enrich_with_best_prices(runners_df)

    scored = build_runner_scores(runners_df)
    value_bets = build_value_bets(scored, min_edge=min_edge)

    writer = SheetsWriter(sheet_name=sheet_name, credentials_path="credentials.json")

    # Debug / status
    writer.write_df(
        "TARGET_DAY",
        pd.DataFrame(
            [
                {
                    "run_utc": now_utc.isoformat(timespec="seconds"),
                    "run_local_dublin": now_local.isoformat(timespec="seconds"),
                    "dublin_hour": int(now_local.hour),
                    "after_22_rule": bool(after_cutoff),
                    "target_label": target_label,
                    "target_date": target_date.isoformat(),
                    "night_before_hour_local": NIGHT_BEFORE_HOUR,
                    "region": region,
                }
            ]
        ),
    )

    if getattr(client, "debug_lines", None):
        writer.write_df("DEBUG_CARD", pd.DataFrame({"line": client.debug_lines}))

    writer.write_df("RACES_TARGET", races_df)
    writer.write_df("RUNNERS_TARGET", scored)
    writer.write_df("VALUE_BETS_TARGET", value_bets)

    # Snapshots
    snapshots_new = scored[
        ["date", "course", "off_time", "race_name", "race_id", "runner", "best_price_dec"]
    ].copy()
    snapshots_new.insert(0, "snapshot_time", now_utc.isoformat(timespec="seconds"))
    snapshots_new.insert(1, "target_label", target_label)

    writer.write_df("MARKET_SNAPSHOTS_TARGET", snapshots_new)
    writer.append_df("MARKET_SNAPSHOTS_LOG", snapshots_new)

    # Compute movers from the LOG (history), not the overwritten TARGET
    snap_log = _prep_snapshots_df(writer.read_df("MARKET_SNAPSHOTS_LOG"))

    # Keep only rows that belong to the current target date
    if not snap_log.empty and "date" in snap_log.columns:
        snap_log = snap_log[snap_log["date"].astype(str) == target_date.isoformat()].copy()

    movers_2h = compute_movers_last_two(snap_log)
    movers_night = compute_movers_night_before(snap_log)
    persistence = compute_persistent_shorteners(snap_log)

    writer.write_df("MARKET_MOVERS_2H", movers_2h)
    writer.write_df("MARKET_MOVERS", movers_night)

    signals = build_signals(scored, movers_2h, movers_night, persistence)
    writer.write_df("SIGNALS", signals)

    bets_to_place = build_bets_to_place(signals, min_edge=min_edge)
    writer.write_df("BETS_TO_PLACE", bets_to_place)

    update_bet_recs_log(writer, bets_to_place)

    dashboard = build_dashboard(races_df, scored, movers_2h, movers_night, bets_to_place)
    writer.write_df("DASHBOARD", dashboard)

    writer.append_df(
        "RUN_LOG",
        pd.DataFrame(
            [
                {
                    "run_utc": now_utc.isoformat(timespec="seconds"),
                    "run_local_dublin": now_local.isoformat(timespec="seconds"),
                    "region": region,
                    "target_label": target_label,
                    "target_date": target_date.isoformat(),
                    "races": int(len(races_df)),
                    "runners": int(len(scored)),
                    "movers_2h": int(len(movers_2h)),
                    "movers_night": int(len(movers_night)),
                    "bets": int(len(bets_to_place)),
                }
            ]
        ),
    )

    print(
        f"Update complete. target={target_label} {target_date} | "
        f"races={len(races_df)} runners={len(scored)} "
        f"movers_2h={len(movers_2h)} bets={len(bets_to_place)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
