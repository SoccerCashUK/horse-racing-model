from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Tuple, Union, Set

import pandas as pd
import requests
from bs4 import BeautifulSoup

UA = "Mozilla/5.0 (compatible; HorseRacingSheetsBot/1.1)"
BASE = "https://www.irishracing.com"

# Known Irish courses (normalised: lowercase, hyphens)
IRE_COURSES: Set[str] = {
    "curragh", "leopardstown", "fairyhouse", "punchestown", "navan", "cork",
    "galway", "killarney", "listowel", "tipperary", "dundalk", "gowran-park",
    "gowran", "naas", "roscommon", "sligo", "down-royal", "downroyal",
    "downpatrick", "clonmel", "thurles", "limerick", "ballinrobe", "tramore",
    "wexford", "kilbeggan", "bellewstown", "laytown", "punchestown",
    "fairyhouse", "navan", "cork", "tipperary",
}

# Pure overseas meetings we usually want to drop even on "all"
OVERSEAS_COURSES: Set[str] = {
    "tokyo", "sha-tin", "kochi", "kanazawa", "ohi", "hanshin", "kyoto",
    "nakayama", "chongqing", "seoul", "busan", "singapore", "kembla-grange",
}


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _frac_to_decimal(frac: str) -> float | None:
    if not frac or "/" not in frac:
        return None
    try:
        a, b = frac.strip().split("/")
        a = float(a)
        b = float(b)
        if b == 0:
            return None
        return 1.0 + (a / b)
    except Exception:
        return None


def _suffix(day: int) -> str:
    if 10 <= day % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def _date_label(d: date) -> str:
    """IrishRacing date label format: Wed-4th-Feb-2026"""
    day = d.day
    return f"{d.strftime('%a')}-{day}{_suffix(day)}-{d.strftime('%b')}-{d.strftime('%Y')}"


def _parse_date_input(d: Union[str, date]) -> date:
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d), "%Y-%m-%d").date()


def _norm_course(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")


_FORM_RE = re.compile(r"^(?=.*\d)[0-9PFURBCOSVDL/\-]{2,15}$", re.I)
_WEIGHT_RE = re.compile(r"^(\d{1,2})\s*[-\u2013]\s*(\d{1,2})$")
_DAYS_RE = re.compile(r"(\d{1,4})\s*(?:days?|d)\b", re.I)
_BADGE_RE = re.compile(r"^(?:(?:C|D|CD|BF|C&D)\s*)+$")


def _parse_runner_context(before: list[str], after: list[str]) -> dict:
    """
    Pull form / age / weight / trainer / jockey / OR / days-since / C&D
    from the lines around a runner's name. Anything not found stays blank,
    so a layout change degrades to "no form" instead of crashing.
    """
    out = {
        "form": "", "rating": "", "days_since": "", "course_distance": "",
        "trainer": "", "jockey": "", "weight": "", "age": "", "sex": "", "draw": "",
    }

    for ln in reversed(before):
        t = ln.replace(" ", "")
        if (_FORM_RE.match(t) and not t.isdigit()) or (t.isdigit() and len(t) >= 3):
            out["form"] = t
            break

    text_fields = []
    weight_seen = False
    for ln in after:
        t = ln.strip()
        if not t:
            continue
        if _BADGE_RE.match(t):
            out["course_distance"] = (out["course_distance"] + " " + t).strip()
            continue
        m_days = _DAYS_RE.search(t)
        if m_days and not out["days_since"] and len(t) <= 12:
            out["days_since"] = m_days.group(1)
            continue
        m_w = _WEIGHT_RE.match(t)
        if m_w and not out["weight"]:
            out["weight"] = t
            weight_seen = True
            continue
        if t.isdigit():
            n = int(t)
            if not out["age"] and 2 <= n <= 15 and not weight_seen:
                out["age"] = t
            elif weight_seen and not out["rating"] and 20 <= n <= 190:
                out["rating"] = t
            continue
        m_or = re.match(r"^OR\s*(\d{2,3})$", t, re.I)
        if m_or:
            out["rating"] = m_or.group(1)
            continue
        if re.search(r"[A-Za-z]{2,}", t) and len(t) <= 40 and len(text_fields) < 2:
            text_fields.append(t)

    if text_fields:
        out["trainer"] = text_fields[0]
    if len(text_fields) > 1:
        out["jockey"] = text_fields[1]
    return out


@dataclass(frozen=True)
class Race:
    date: str
    date_label: str
    course: str
    off_time: str
    race_name: str
    distance: str
    going: str
    class_band: str
    race_url: str
    race_key: str


class IrishRacingClient:
    def __init__(self, region: str = "all", timeout: int = 25) -> None:
        self.region = (region or "all").lower().strip()
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA})
        self.debug_lines: list[str] = []

    def fetch_today(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.fetch_for_date(datetime.now().date())

    def fetch_tomorrow(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.fetch_for_date(datetime.now().date() + timedelta(days=1))

    def fetch_for_date(self, d: Union[str, date]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        target_date = _parse_date_input(d)
        date_label = _date_label(target_date)
        day_url = f"{BASE}/racecards/{date_label}"

        try:
            html = self.session.get(day_url, timeout=self.timeout).text
        except Exception as e:
            print(f"Failed to fetch day page {day_url}: {e}")
            return pd.DataFrame(), pd.DataFrame()

        soup = BeautifulSoup(html, "lxml")

        meeting_links = []
        for a in soup.select("a[href^='/racecards/']"):
            href = a.get("href", "")
            if re.fullmatch(rf"/racecards/{re.escape(date_label)}/[^/]+", href or ""):
                meeting_links.append(href)

        meeting_links = list(dict.fromkeys(meeting_links))

        races_all = []
        runners_all = []

        for href in meeting_links:
            course = href.split("/")[-1]
            course_norm = _norm_course(course)

            # Region filter
            if self.region == "ire":
                if course_norm not in IRE_COURSES:
                    continue
            elif self.region == "gb":
                if course_norm in IRE_COURSES or course_norm in OVERSEAS_COURSES:
                    continue
            else:  # all → drop pure overseas
                if course_norm in OVERSEAS_COURSES:
                    continue

            course_url = BASE + href
            try:
                races_df, runners_df = self._parse_course_all_races(course_url, date_label, course)
            except Exception as e:
                print(f"Failed parsing {course_url}: {e}")
                continue

            if not races_df.empty:
                races_all.append(races_df)
            if not runners_df.empty:
                runners_all.append(runners_df)

        races = pd.concat(races_all, ignore_index=True) if races_all else pd.DataFrame()
        runners = pd.concat(runners_all, ignore_index=True) if runners_all else pd.DataFrame()
        return races, runners

    def enrich_with_best_prices(self, runners_df: pd.DataFrame) -> pd.DataFrame:
        df = runners_df.copy()
        df["best_price_dec"] = pd.to_numeric(df["best_price_dec"], errors="coerce")
        df = df.dropna(subset=["best_price_dec"]).copy()
        return df

    def _parse_course_all_races(self, url: str, date_label: str, course: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        html = self.session.get(url, timeout=self.timeout).text
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text("\n", strip=True)
        if not self.debug_lines:
            self.debug_lines = [f"URL: {url}"] + text.split("\n")[:800]

        going = ""
        m_going = re.search(r"Going\s*-\s*(.+?)\.", text)
        if m_going:
            going = _clean(m_going.group(1))

        date_iso = self._label_to_date(date_label)

        blocks = re.split(r"\n(?=\d{1,2}\.\d{2}\n)", text)

        races_rows = []
        runner_rows = []

        for block in blocks:
            m_time = re.match(r"(\d{1,2}\.\d{2})\n", block)
            if not m_time:
                continue
            off_time = m_time.group(1)

            lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
            if len(lines) < 3:
                continue

            race_name = ""
            for ln in lines[1:10]:
                if ln.lower() in ("no", "form", "horse age weight", "trainer", "jockey", "or", "horse"):
                    continue
                if re.match(r"^\d+\^\{", ln):
                    continue
                if re.search(r"Probable SP", ln, re.I):
                    break
                if re.search(r"\b(of €|Race Conditions|Weights|Penalties)\b", ln, re.I):
                    continue
                race_name = ln
                break

            dist = ""
            m_dist = re.search(r"(\d+m\.\s*\d+f\.\s*\d+yds\.|\d+m\.\s*\d+f\.|\d+f\.)", block)
            if m_dist:
                dist = _clean(m_dist.group(1)).replace(" .", ".")

            class_band = ""
            m_class = re.search(r"\(Class\s*(\d+)\)", block)
            if m_class:
                class_band = f"Class {m_class.group(1)}"

            m_psp = re.search(r"Probable SP\s*-\s*(.+)", block)
            if not m_psp:
                continue
            psp = m_psp.group(1)

            race_key = f"{course}_{off_time}_{re.sub(r'[^A-Za-z0-9]+', '_', race_name)[:40]}"

            races_rows.append({
                "date": date_iso,
                "date_label": date_label,
                "course": course,
                "off_time": off_time,
                "race_name": race_name,
                "distance": dist,
                "going": going,
                "class_band": class_band,
                "race_url": url,
                "race_id": race_key,
            })

            horse_to_odds = {}
            for part in psp.split(","):
                part = part.strip().rstrip(".")
                m = re.match(r"(\d+/\d+)\s+(.+)$", part)
                if not m:
                    continue
                frac = m.group(1).strip()
                name = _clean(m.group(2))
                name = re.sub(r"\.\s*$", "", name)
                horse_to_odds[name.lower()] = frac

            # Index of each runner's name line in this race block
            name_idx = []
            for i, ln in enumerate(lines):
                if ln.lower().startswith("probable sp"):
                    break
                if ln.lower() in horse_to_odds:
                    name_idx.append(i)

            seen = set()
            for k, i in enumerate(name_idx):
                ln = lines[i]
                key = ln.lower()
                if key in seen:
                    continue
                seen.add(key)
                frac = horse_to_odds[key]
                dec = _frac_to_decimal(frac)
                if not dec:
                    continue

                prev_i = name_idx[k - 1] if k > 0 else max(0, i - 6)
                next_i = name_idx[k + 1] if k + 1 < len(name_idx) else min(len(lines), i + 12)
                before = lines[max(prev_i + 1, i - 6):i]
                after = lines[i + 1:min(next_i, i + 14)]
                info = _parse_runner_context(before, after)

                runner_rows.append({
                    "date": date_iso,
                    "date_label": date_label,
                    "course": course,
                    "off_time": off_time,
                    "race_name": race_name,
                    "distance": dist,
                    "going": going,
                    "class_band": class_band,
                    "race_id": race_key,
                    "runner": ln,
                    "best_price_frac": frac,
                    "best_price_dec": float(dec),
                    **info,
                })

        races_df = pd.DataFrame(races_rows)
        runners_df = (
            pd.DataFrame(runner_rows).drop_duplicates(subset=["race_id", "runner"])
            if runner_rows
            else pd.DataFrame()
        )
        return races_df, runners_df

    def _label_to_date(self, label: str) -> str:
        try:
            parts = label.split("-")
            day = re.sub(r"\D", "", parts[1])
            mon = parts[2]
            year = parts[3]
            dt = datetime.strptime(f"{day} {mon} {year}", "%d %b %Y")
            return dt.date().isoformat()
        except Exception:
            return datetime.utcnow().date().isoformat()
