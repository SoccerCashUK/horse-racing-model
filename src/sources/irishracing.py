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


_AGE_WT_RE = re.compile(r"^(\d{1,2})\s+(\d{1,2}-\d{1,2})$")
_BADGES = {"c", "d", "cd", "bf"}
_GEAR = {"b", "v", "p", "cp", "t", "ts", "h", "e/s", "es", "e", "s", "tb", "vs", "hb"}


def _is_form(t: str) -> bool:
    return bool(re.fullmatch(r"[0-9PFURBCOSVDL/\-]+", t, re.I)) and len(t) <= 15


def _parse_runner_context(before: list[str], after: list[str]) -> dict:
    """
    irishracing.com "all races" card, one runner, as text lines:

      before name: [prev runner's trailing lines...] No, Draw(flat), ShortForm, FullForm
      after name:  [badge, count]*  "3 9-9"  Trainer  Jockey  [OR]  [next No, Draw...]

    Badges: c / d / cd / bf with a count; gear: cp, ts, v, b, h, t ... with a count.
    Fields that can't be found are left blank.
    """
    out = {
        "form": "", "rating": "", "days_since": "", "course_distance": "",
        "gear": "", "trainer": "", "jockey": "", "weight": "", "age": "", "sex": "", "draw": "",
    }

    # --- before the name: form (short form repeated as suffix of full form) ---
    rest = list(before)
    if len(rest) >= 2 and _is_form(rest[-1]) and _is_form(rest[-2]) and \
            rest[-1].replace("-", "").endswith(rest[-2].replace("-", "")) and \
            len(rest) >= 3 and rest[-3].isdigit():
        # only form if a saddle number/draw is still left in front of it;
        # otherwise the pair was "No, Draw" for an unraced horse (e.g. "4", "4")
        out["form"] = rest[-1]
        rest = rest[:-2]
    # draw: the line after the saddle number (flat races only)
    if len(rest) >= 2 and rest[-1].isdigit() and rest[-2].isdigit():
        # rest[-2] = saddle no, rest[-1] = draw; but in jumps rest[-2] may be prev OR
        no, dr = int(rest[-2]), int(rest[-1])
        if no <= 40 and dr <= 40:
            out["draw"] = rest[-1]

    # --- after the name ---
    j = 0
    badges, gear = [], []
    while j < len(after) and not _AGE_WT_RE.match(after[j]):
        tok = after[j].strip().lower()
        cnt = after[j + 1] if j + 1 < len(after) and after[j + 1].strip().isdigit() else ""
        if tok in _BADGES:
            badges.append(tok + cnt)
            j += 2 if cnt else 1
            continue
        if tok.rstrip("+") in _GEAR:
            gear.append(tok + cnt)
            j += 2 if cnt else 1
            continue
        m_days = re.fullmatch(r"\(?(\d{1,4})\)?", tok)
        if m_days and tok.startswith("("):
            out["days_since"] = m_days.group(1)
        j += 1

    if j < len(after):
        m = _AGE_WT_RE.match(after[j])
        out["age"], out["weight"] = m.group(1), m.group(2)
        tail = after[j + 1:]
        if len(tail) >= 1:
            out["trainer"] = tail[0]
        if len(tail) >= 2:
            out["jockey"] = tail[1]
        if len(tail) >= 3 and tail[2].isdigit():
            v = int(tail[2])
            # OR is 20+; a small number here is the next runner's saddle number
            if v >= 20:
                out["rating"] = tail[2]

    out["course_distance"] = " ".join(badges)
    out["gear"] = " ".join(gear)
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
            last_frac = None
            for part in psp.split(","):
                part = part.strip().rstrip(".")
                if not part:
                    continue
                m = re.match(r"(\d+/\d+|evens|evs)\s+(.+)$", part, re.I)
                if m:
                    frac = m.group(1).strip()
                    if frac.lower() in ("evens", "evs"):
                        frac = "1/1"
                    last_frac = frac
                    name = _clean(m.group(2))
                elif last_frac:
                    # "14/1 Dark Supremacy, Zabeel Express" -> both 14/1
                    frac = last_frac
                    name = _clean(part)
                else:
                    continue
                name = re.sub(r"\.\s*$", "", name)
                horse_to_odds[name.lower()] = frac

            # Index of each runner's name line in this race block
            name_idx = []
            for i, ln in enumerate(lines):
                if ln.lower().startswith("probable sp"):
                    break
                if ln.lower() in horse_to_odds:
                    name_idx.append(i)

            # Probable SP line index (end of runner section)
            psp_i = next((j for j, l in enumerate(lines) if l.lower().startswith("probable sp")), len(lines))
            hdr_i = next((j for j, l in enumerate(lines) if l.upper() == "OR"), -1)

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

                start = name_idx[k - 1] + 1 if k > 0 else hdr_i + 1
                stop = name_idx[k + 1] if k + 1 < len(name_idx) else psp_i
                before = lines[start:i]
                after = lines[i + 1:stop]
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
