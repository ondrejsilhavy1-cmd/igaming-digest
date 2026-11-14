"""
The Cashout — iGaming weekly rhythm
Telegram Channel Bot

Mon 13:00 overview · Tue 11:00 news · Wed 11:30 poll · Thu 11:00 onchain update
Fri 15:30 afternoon drink · event countdowns — all Europe/Berlin

Stack: Python, APScheduler, feedparser, Groq (llama-3.3-70b-versatile), telebot, requests, matplotlib
Data: Tanzanite, Circus, FairGambling
Deploy: Railway via GitHub
"""

import os
import io
import re
import html
import json
import time
import hashlib
import logging
import threading
import requests
import feedparser
import telebot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from groq import Groq
from telebot.apihelper import ApiTelegramException
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timezone, timedelta, date
from collections import defaultdict
from zoneinfo import ZoneInfo

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("cashout-bot")

# ── Environment variables ─────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
CHANNEL_ID     = os.environ["CHANNEL_ID"]
GROQ_API_KEY   = os.environ["GROQ_API_KEY"]
RSSHUB_URL     = os.environ.get("RSSHUB_URL", "")


def env_flag(name: str, default: bool = True) -> bool:
    """Optional toggle — on unless set to 0 / false / off / no."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in ("0", "false", "off", "no")


# Optional data sources / features — all default ON
ENABLE_CIRCUS           = env_flag("ENABLE_CIRCUS")
ENABLE_FAIRGAMBLING     = env_flag("ENABLE_FAIRGAMBLING")
ENABLE_EVENT_COUNTDOWNS = env_flag("ENABLE_EVENT_COUNTDOWNS")

# ── Schedule timezone ─────────────────────────────────────────────────────────
SCHEDULE_TZ = "Europe/Berlin"
try:
    LOCAL_TZ = ZoneInfo(SCHEDULE_TZ)
except Exception as e:   # no tz database on the image — "tzdata" in requirements.txt covers this
    log.warning(f"Timezone {SCHEDULE_TZ} unavailable ({e}) — dates fall back to UTC")
    LOCAL_TZ = timezone.utc

# ── State directory ───────────────────────────────────────────────────────────
def resolve_state_dir() -> str:
    """STATE_DIR env var, else /app if it exists and is writable, else ./.state"""
    path = os.environ.get("STATE_DIR", "").strip()
    if not path:
        path = "/app" if os.path.isdir("/app") and os.access("/app", os.W_OK) else "./.state"
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        log.warning(f"Could not create state dir {path}: {e}")
    return path


STATE_DIR           = resolve_state_dir()
STATE_FILE          = os.path.join(STATE_DIR, "state.json")
SEEN_HEADLINES_FILE = os.path.join(STATE_DIR, "seen_headlines.json")

# ── Clients ───────────────────────────────────────────────────────────────────
GROQ_MODEL  = "llama-3.3-70b-versatile"
bot         = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="HTML")
groq_client = Groq(api_key=GROQ_API_KEY)

# ── Extra data sources (optional, fail-soft) ──────────────────────────────────
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/126.0.0.0 Safari/537.36",
    "Accept":     "application/json",
}
CIRCUS_ANALYTICS_API        = "https://www.circus.fyi/api/streamer-analytics?days=7"
CIRCUS_UPDATES_API          = "https://www.circus.fyi/api/updates"
CIRCUS_MIN_PREV_HOURS       = 1000   # ignore movers with a tiny previous-week base
FAIRGAMBLING_BUNDLE_API     = "https://api.fairgambling.com/api/analytics/home-bundle"
FAIRGAMBLING_COMPLAINTS_API = "https://api.fairgambling.com/api/complaints/stats/global"

# ── Casino name display map ───────────────────────────────────────────────────
# Maps raw API slugs → clean display names
# Add more as you discover them in the data
CASINO_NAMES = {
    # Stake — two separate brands
    "stakeus":         "Stake.us",
    "stakecom":        "Stake.com",
    "stakecasino":     "Stake.com",
    "stake":           "Stake.com",

    # Major crypto casinos
    "roobetcom":       "Roobet",
    "roobet":          "Roobet",
    "rollbitcom":      "Rollbit",
    "rollbit":         "Rollbit",
    "bcgame":          "BC.Game",
    "bc_game":         "BC.Game",
    "betfury":         "BetFury",
    "yologroup":       "Yolo Group",
    "acebet":          "AceBet",
    "_500casino":      "500 Casino",
    "500casino":       "500 Casino",
    "winr":            "WINR",
    "crashino":        "Crashino",
    "duelbitscom":     "Duelbits",
    "duelbits":        "Duelbits",
    "thunderpick":     "Thunderpick",
    "hypedrop":        "HypeDrop",
    "csgoempire":      "CSGOEmpire",
    "rakebackgg":      "Rakeback.gg",
    "gamdomcom":       "Gamdom",
    "gamdom":          "Gamdom",
    "fortunejack":     "FortuneJack",
    "metaspins":       "Metaspins",
    "bitstarz":        "BitStarz",
    "cloudbet":        "Cloudbet",
    "1xbit":           "1xBit",
    "sportsbetio":     "Sportsbet.io",
    "sportsbet":       "Sportsbet.io",
    "primedice":       "Primedice",
    "winstrike":       "Winstrike",
    "bspin":           "Bspin",
    "trustdice":       "TrustDice",
    "nitrobetting":    "Nitrobetting",
    "shufflecom":      "Shuffle",
    "shuffle":         "Shuffle",
    "luckio":          "Luck.io",
    "luck":            "Luck.io",
    "solpot":          "Solpot",
    "solpump":         "Solpump",
    "rainbet":         "Rainbet",
    "duel":            "Duel",
    "winz":            "Winz.io",
    "winzio":          "Winz.io",
    "vave":            "Vave",
    "mystake":         "MyStake",
    "jackbit":         "Jackbit",
    "mirax":           "Mirax",
    "bets":            "Bets.io",
    "betsio":          "Bets.io",
    "coinpoker":       "CoinPoker",
    "pokerrrr":        "Pokerrrr",
}

# Populated at startup from Tanzanite meta.entity_display — auto-updates when
# Tanzanite renames slugs. Falls back to CASINO_NAMES, then slug cleanup.
CASINO_LABELS: dict[str, str] = {}


def clean_name(raw: str) -> str:
    """Return display name for a casino slug, checking API labels first."""
    key = raw.lower()
    return (
        CASINO_LABELS.get(key)
        or CASINO_NAMES.get(key)
        or raw.replace("_", " ").strip().title()
    ).strip()

# ── Chain emoji map ───────────────────────────────────────────────────────────
CHAIN_EMOJI = {
    "ARB":      "🔵",   # Arbitrum
    "ETH":      "⬡",    # Ethereum
    "BNB":      "🟡",   # BNB Chain
    "MATIC":    "🟣",   # Polygon
    "SOL":      "🟢",   # Solana
    "TRX":      "🔴",   # Tron
    "AVAX":     "🔺",   # Avalanche
    "BASE":     "🔷",   # Base
    "OP":       "🔴",   # Optimism
}

def chain_tag(chains: list[str]) -> str:
    """Return emoji string for a list of chains."""
    tags = [CHAIN_EMOJI.get(c.upper(), f"[{c}]") for c in chains]
    return " ".join(dict.fromkeys(tags))  # dedup, preserve order

# ── Volume threshold ──────────────────────────────────────────────────────────
MIN_VOLUME_USD = 50_000   # Minimum daily deposit volume to qualify for rankings

# ── Big events ────────────────────────────────────────────────────────────────
# Countdown posts go to the main channel: weekly from 35 days before the start,
# plus one final call 3 days before. venue is optional (None).
BIG_EVENTS = [
    {"name": "ICE Barcelona",      "city": "Barcelona", "venue": "Fira Gran Via",
     "start": date(2027, 1, 18),   "end": date(2027, 1, 20)},
    {"name": "SiGMA World Rome",   "city": "Rome",      "venue": None,
     "start": date(2027, 5, 24),   "end": date(2027, 5, 27)},
    {"name": "iGB L!VE Amsterdam", "city": "Amsterdam", "venue": "RAI",
     "start": date(2027, 7, 14),   "end": date(2027, 7, 15)},
    {"name": "SBC Summit Lisbon",  "city": "Lisbon",    "venue": "FIL",
     "start": date(2027, 9, 21),   "end": date(2027, 9, 23)},
    {"name": "SiGMA Europe Malta", "city": "Malta",     "venue": None,
     "start": date(2027, 11, 3),   "end": date(2027, 11, 5)},
]

# ── RSS feeds ─────────────────────────────────────────────────────────────────
RSS_FEEDS = [
    "https://sbcnews.co.uk/feed",
    "https://igamingbusiness.com/feed",
    "https://igamingexpert.com/feed",
    "https://www.yogonet.com/international/rss.xml",
    "https://gamblingnews.com/feed",
    "https://gamblinginsider.com/feed",
    "https://cryptogamblingnews.com/feed",
]

TWITTER_ACCOUNTS = ["SBCnews", "iGamingBusiness", "GamblingInsider", "tanzanite_xyz"]
if RSSHUB_URL:
    for account in TWITTER_ACCOUNTS:
        RSS_FEEDS.append(f"{RSSHUB_URL.rstrip('/')}/twitter/user/{account}")

# ── Tanzanite API ─────────────────────────────────────────────────────────────
TANZANITE_API          = "https://www.tanzanite.xyz/api/public/casino-analytics/deposit-volume"
TANZANITE_TRENDING_API = "https://www.tanzanite.xyz/api/public/casino-analytics/trending"
# deposit-volume  → Thursday update (interval day-on-day pct) + Monday chart
# trending        → Monday overview (pre-calculated week-on-week pct)
# deposit-distribution → weekly player profile + meta.entity_display labels


def fetch_tanzanite() -> dict | None:
    """
    Fetches the single deposit-volume endpoint which returns:
      data["trending"]                                         — pre-ranked list with usd_change_pct
      data["deposit_volume"]["timeframes"]["monthly"]["sites"] — per-site chain/token breakdown
    """
    try:
        resp = requests.get(TANZANITE_API, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        n_trending = len(data.get("trending", []))
        n_sites    = len(data.get("deposit_volume", {}).get("timeframes", {}).get("monthly", {}).get("sites", {}))
        log.info(f"Tanzanite fetched OK — {n_trending} trending, {n_sites} sites")
        return data
    except Exception as e:
        log.error(f"Tanzanite API error: {e}")
        return None


def fetch_tanzanite_trending() -> list:
    """
    Fetches the dedicated trending endpoint — returns week-on-week pct changes.
    Used exclusively by the Monday weekly recap.
    Returns the trending list directly, or [] on error.
    """
    try:
        resp = requests.get(TANZANITE_TRENDING_API, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        entries = data.get("trending", [])
        log.info(f"Tanzanite trending fetched OK — {len(entries)} entries")
        return entries
    except Exception as e:
        log.error(f"Tanzanite trending API error: {e}")
        return []


def fetch_casino_labels() -> None:
    """
    Fetches meta.entity_display from deposit-distribution and populates
    CASINO_LABELS with slug -> label mappings. Called once at startup.
    Failures are non-fatal — falls back to CASINO_NAMES.
    """
    global CASINO_LABELS
    try:
        resp = requests.get(TANZANITE_DISTRIBUTION_API, timeout=20)
        resp.raise_for_status()
        body = resp.json()
        # meta used to be top-level; it now sits under deposit_distribution_buckets
        meta = body.get("meta") or body.get("deposit_distribution_buckets", {}).get("meta", {})
        entity_display = meta.get("entity_display", {})
        CASINO_LABELS = {slug: info["label"] for slug, info in entity_display.items() if "label" in info}
        log.info(f"Casino labels loaded — {len(CASINO_LABELS)} entries from API")
    except Exception as e:
        log.warning(f"Could not load casino labels from API: {e} — using hardcoded CASINO_NAMES")


def parse_casino_movements(data: dict) -> list:
    """
    Computes day-on-day pct change from intervals[0] vs intervals[1] in deposit_volume.
    This is reliable regardless of whether trending[] is present in the response.

    Returns list of dicts:
      {name, raw_name, current, previous, pct, chains}
    sorted by pct DESC, filtered by MIN_VOLUME_USD.
    """
    casino_data = defaultdict(lambda: {"current": 0.0, "previous": 0.0, "chains": set()})

    try:
        sites = data["deposit_volume"]["timeframes"]["monthly"]["sites"]
    except (KeyError, TypeError):
        log.error("Tanzanite: could not find deposit_volume.timeframes.monthly.sites")
        return []

    for site_name, site_info in sites.items():
        for chain_name, chain_info in site_info.get("chains", {}).items():
            for token_name, token_info in chain_info.get("tokens", {}).items():
                intervals = token_info.get("intervals", [])
                if len(intervals) >= 1:
                    usd0 = float(intervals[0].get("usd", 0) or 0)
                    if usd0 < 50_000_000:  # skip corrupted values
                        casino_data[site_name]["current"] += usd0
                        casino_data[site_name]["chains"].add(chain_name)
                if len(intervals) >= 2:
                    usd1 = float(intervals[1].get("usd", 0) or 0)
                    if usd1 < 50_000_000:
                        casino_data[site_name]["previous"] += usd1

    movements = []
    for raw_name, vols in casino_data.items():
        curr = vols["current"]
        prev = vols["previous"]

        if curr < MIN_VOLUME_USD:
            continue

        if prev > 0:
            pct = ((curr - prev) / prev) * 100
        elif curr > 0:
            pct = 100.0
        else:
            continue

        movements.append({
            "name":     clean_name(raw_name),
            "raw_name": raw_name,
            "current":  curr,
            "previous": prev,
            "pct":      pct,
            "chains":   list(vols["chains"]),
        })

    movements.sort(key=lambda x: x["pct"], reverse=True)
    return movements


def parse_weekly_volumes(data: dict) -> dict[str, float]:
    """
    Returns dict of {display_name: 7_day_total_usd} for top casinos.
    Sums all intervals within the last 7 days.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    casino_vols = defaultdict(float)

    try:
        sites = data["deposit_volume"]["timeframes"]["monthly"]["sites"]
    except (KeyError, TypeError):
        return {}

    for casino_name, casino_info in sites.items():
        for chain_name, chain_info in casino_info.get("chains", {}).items():
            for token_name, token_info in chain_info.get("tokens", {}).items():
                for interval in token_info.get("intervals", []):
                    try:
                        day = datetime.strptime(interval["day"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        if day >= cutoff:
                            usd = float(interval.get("usd", 0) or 0)
                            # Skip corrupted values
                            if usd < 50_000_000:
                                casino_vols[casino_name] += usd
                    except (ValueError, KeyError):
                        continue

    # Clean names and filter low volume
    cleaned = {}
    for raw, vol in casino_vols.items():
        if vol >= MIN_VOLUME_USD:
            cleaned[clean_name(raw)] = cleaned.get(clean_name(raw), 0) + vol

    return cleaned


TANZANITE_DISTRIBUTION_API = "https://www.tanzanite.xyz/api/public/casino-analytics/deposit-distribution"


def fetch_tanzanite_distribution() -> dict | None:
    """
    Fetches the deposit-distribution endpoint.
    Returns data containing deposit_distribution_buckets with:
      - buckets: list of bucket definitions (key, label, min_usd, max_usd)
      - rows: flat list of {timeframe, site, chain, bucket_key,
              deposit_count, wallet_count, deposit_total_usd, wallet_total_usd}
    Timeframe is always "weekly" — used for Monday recap only.
    """
    try:
        resp = requests.get(TANZANITE_DISTRIBUTION_API, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        n_rows = len(data.get("deposit_distribution_buckets", {}).get("rows", []))
        log.info(f"Tanzanite distribution fetched OK — {n_rows} rows")
        return data
    except Exception as e:
        log.error(f"Tanzanite distribution API error: {e}")
        return None


def parse_player_profile(dist_data: dict) -> dict | None:
    """
    Processes deposit_distribution_buckets rows to find:
      - whale_casino:   highest % of volume from $100K+ buckets
      - retail_casino:  highest % of volume from <$1K buckets
      - balanced_casino: most evenly spread across all buckets (lowest max bucket %)

    Returns dict with keys: whale, whale_pct, retail, retail_pct, balanced, balanced_pct
    or None if data is unavailable.
    """
    try:
        rows = dist_data["deposit_distribution_buckets"]["rows"]
    except (KeyError, TypeError):
        log.error("Distribution: could not find deposit_distribution_buckets.rows")
        return None

    if not rows:
        return None

    # Aggregate deposit_total_usd per site per bucket_key
    site_buckets: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        site      = row.get("site", "")
        bucket    = row.get("bucket_key", "")
        usd       = float(row.get("deposit_total_usd", 0) or 0)
        if site and bucket:
            site_buckets[site][bucket] += usd

    WHALE_BUCKETS  = {"100000_1000000", "gte_1000000"}
    RETAIL_BUCKETS = {"lt_100", "100_1000"}

    whale_casino    = None;  whale_pct    = 0.0
    retail_casino   = None;  retail_pct   = 0.0
    balanced_casino = None;  balanced_pct = 100.0  # lower = more balanced

    for site, buckets in site_buckets.items():
        total = sum(buckets.values())
        if total < MIN_VOLUME_USD:
            continue

        whale_share  = sum(buckets.get(b, 0) for b in WHALE_BUCKETS)  / total * 100
        retail_share = sum(buckets.get(b, 0) for b in RETAIL_BUCKETS) / total * 100
        max_bucket   = max(buckets.values()) / total * 100  # concentration score

        display = clean_name(site) or site

        if whale_share > whale_pct:
            whale_pct    = whale_share
            whale_casino = display

        if retail_share > retail_pct:
            retail_pct    = retail_share
            retail_casino = display

        if max_bucket < balanced_pct:
            balanced_pct    = max_bucket
            balanced_casino = display

    if not whale_casino:
        return None

    return {
        "whale":        whale_casino,
        "whale_pct":    whale_pct,
        "retail":       retail_casino,
        "retail_pct":   retail_pct,
        "balanced":     balanced_casino,
        "balanced_pct": balanced_pct,
    }


def format_volume(vol: float) -> str:
    if vol >= 1_000_000:
        return f"${vol/1_000_000:.2f}M"
    if vol >= 1_000:
        return f"${vol/1_000:.1f}K"
    return f"${vol:.0f}"


# ── Chart generator ───────────────────────────────────────────────────────────

def generate_weekly_chart(weekly_vols: dict[str, float]) -> io.BytesIO | None:
    """Generate a polished dark bar chart of top 10 casinos by 7-day volume."""
    if not weekly_vols:
        return None

    sorted_items = sorted(weekly_vols.items(), key=lambda x: x[1], reverse=True)[:10]
    names  = [item[0] for item in reversed(sorted_items)]
    values = [item[1] for item in reversed(sorted_items)]
    n = len(names)

    BG        = "#18181f"
    ACCENT    = "#00e56b"
    GOLD      = "#f0c040"
    TEXT      = "#ffffff"
    SUBTEXT   = "#55556a"
    ROW_BG    = "#1e1e28"
    DIM_GREEN = "#00703a"

    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    max_val   = max(values)
    bar_h     = 0.44
    row_h     = 1.0
    bar_start = max_val * 0.01
    positions = [i * row_h for i in range(n)]

    for i, (name, val, y) in enumerate(zip(names, values, positions)):
        is_top = (val == max_val)
        rank   = n - i

        # Row card
        row = mpatches.FancyBboxPatch(
            (-max_val * 0.32, y - row_h * 0.46), max_val * 1.58, row_h * 0.88,
            boxstyle="round,pad=0.004", facecolor=ROW_BG, edgecolor="none", zorder=1)
        ax.add_patch(row)

        # Rank badge
        badge_w = max_val * 0.052
        badge_h = row_h * 0.52
        badge_x = -max_val * 0.305
        badge = mpatches.FancyBboxPatch(
            (badge_x, y - badge_h / 2), badge_w, badge_h,
            boxstyle="round,pad=0.003",
            facecolor=ACCENT if is_top else "#2a2a3a", edgecolor="none", zorder=3)
        ax.add_patch(badge)
        ax.text(badge_x + badge_w / 2, y, str(rank),
                ha="center", va="center",
                color="#000" if is_top else SUBTEXT,
                fontsize=9, fontweight="bold", zorder=4)

        # Bar
        bar = mpatches.FancyBboxPatch(
            (bar_start, y - bar_h / 2), val, bar_h,
            boxstyle="round,pad=0.004",
            facecolor=ACCENT if is_top else DIM_GREEN,
            edgecolor="none", zorder=2,
            alpha=0.75 + 0.25 * (val / max_val))
        ax.add_patch(bar)

        # Casino name
        ax.text(-max_val * 0.235, y, name,
                va="center", ha="left",
                color=TEXT if is_top else "#bbbccc",
                fontsize=11, fontweight="bold" if is_top else "normal")

        # Volume label
        ax.text(val + max_val * 0.016, y, format_volume(val),
                va="center", ha="left",
                color=GOLD if is_top else "#888899",
                fontsize=9.5,
                fontweight="bold" if is_top else "normal",
                fontfamily="monospace")

    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(-max_val * 0.33, max_val * 1.28)
    ax.set_ylim(positions[0] - row_h * 0.6, positions[-1] + row_h * 0.6)

    week_end   = datetime.now(timezone.utc).strftime("%d %b %Y")
    week_start = (datetime.now(timezone.utc) - timedelta(days=6)).strftime("%d %b")

    fig.text(0.04, 0.975, "Top Onchain Casinos",
             color=ACCENT, fontsize=18, fontweight="bold", va="top", ha="left")
    fig.text(0.04, 0.918, f"7-Day Deposit Volume  ·  {week_start} – {week_end}",
             color=SUBTEXT, fontsize=10, va="top", ha="left")
    fig.text(0.96, 0.975, "The Cashout",
             color=TEXT, fontsize=14, fontweight="bold", va="top", ha="right")
    fig.text(0.96, 0.922, "TG  @The_Cashout",
             color=ACCENT, fontsize=9.5, va="top", ha="right")
    fig.text(0.96, 0.018, "Data: Tanzanite Terminal",
             color=SUBTEXT, fontsize=8, va="bottom", ha="right")

    plt.subplots_adjust(left=0.02, right=0.97, top=0.89, bottom=0.03)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Small helpers: escaping, lengths, formatting ─────────────────────────────

TG_TEXT_LIMIT    = 4096   # Telegram text message limit
TG_CAPTION_LIMIT = 1024   # Telegram photo caption limit

FOOTER_COMMENTS = "💬 Drop a take in the comments"


def esc(value) -> str:
    """HTML-escape dynamic text for Telegram HTML mode."""
    return html.escape(str(value), quote=False)


def esc_url(url: str) -> str:
    """Escape a URL for use inside href="…"."""
    return html.escape(str(url), quote=True)


def tg_len(text: str) -> int:
    """Length in UTF-16 code units — how Telegram counts (emoji count as 2)."""
    return len(text.encode("utf-16-le")) // 2


def truncate_message(text: str, footer: str = "", limit: int = TG_TEXT_LIMIT) -> str:
    """
    Append footer and make sure the result fits in `limit`.
    Drops whole lines from the end of `text` — every line in our messages has
    balanced HTML tags, so cutting on line boundaries never breaks the markup.
    """
    tail = f"\n{footer}" if footer else ""
    if tg_len(text + tail) <= limit:
        return text + tail
    lines = text.split("\n")
    tail  = "\n…" + tail
    while lines and tg_len("\n".join(lines) + tail) > limit:
        lines.pop()
    log.warning("Message exceeded Telegram limit — truncated.")
    return "\n".join(lines).rstrip() + tail


def shorten(text: str, max_chars: int) -> str:
    """Collapse whitespace and cut at a word boundary."""
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:.-") + "…"


def clean_ai(text, max_chars: int) -> str:
    """Normalise Groq output to a single escaped paragraph."""
    if not isinstance(text, str):
        return ""
    return esc(shorten(text.strip().strip('"').strip(), max_chars))


def num(value) -> float | None:
    """Safe float conversion — None for null / junk / NaN."""
    try:
        f = float(value)
        return f if f == f and abs(f) != float("inf") else None
    except (TypeError, ValueError):
        return None


def as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def as_list(value) -> list:
    return value if isinstance(value, list) else []


def format_count(value: float) -> str:
    """1234567 → 1.23M, 45600 → 45.6K"""
    if value >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:.0f}"


def format_pct(value: float) -> str:
    return f"{value:+.0f}%" if abs(value) >= 100 else f"{value:+.1f}%"


def berlin_today() -> date:
    return datetime.now(LOCAL_TZ).date()


def news_item_lines(h: dict, note: str = "") -> list[str]:
    """Two balanced lines for one story: title, then optional note + link."""
    link = f'<a href="{esc_url(h["link"])}">Read more →</a>'
    return [
        f"• {esc(shorten(h['title'], 180))}",
        f"  <i>{note}</i> {link}" if note else f"  {link}",
    ]


# ── Telegram send wrapper (rate limits) ───────────────────────────────────────

MIN_SEND_GAP = 1.1   # seconds between sends to the same chat
_send_lock   = threading.Lock()
_last_send: dict[str, float] = {}


def tg_send(method: str, chat_id, *args, **kwargs):
    """
    Call bot.<method>(chat_id, …) keeping >= MIN_SEND_GAP between sends to the
    same chat. On a 429 it waits retry_after and retries once.
    Use for every send_message / send_photo / send_poll.
    """
    key = str(chat_id)
    with _send_lock:
        for attempt in (1, 2):
            last = _last_send.get(key)
            if last is not None:
                wait = MIN_SEND_GAP - (time.monotonic() - last)
                if wait > 0:
                    time.sleep(wait)
            try:
                return getattr(bot, method)(chat_id, *args, **kwargs)
            except ApiTelegramException as e:
                if e.error_code != 429 or attempt == 2:
                    raise
                params      = as_dict(as_dict(getattr(e, "result_json", None)).get("parameters"))
                retry_after = num(params.get("retry_after")) or 5
                log.warning(f"Telegram 429 on {method} — retrying in {retry_after:.0f}s")
                time.sleep(retry_after + 0.5)
                photo = kwargs.get("photo")
                if hasattr(photo, "seek"):
                    photo.seek(0)   # rewind the chart buffer for the retry
            finally:
                _last_send[key] = time.monotonic()


def send_text(text: str):
    return tg_send("send_message", CHANNEL_ID, text,
                   parse_mode="HTML", disable_web_page_preview=True)


def send_channel_poll(question: str, options: list[str]):
    return tg_send("send_poll", CHANNEL_ID, question, options,
                   is_anonymous=True, type="regular", allows_multiple_answers=False)


# ── State (posted links, event posts) ─────────────────────────────────────────

POSTED_LINKS_TTL = 14   # days to remember a posted link
EVENT_POSTS_TTL  = 60   # days to remember an event countdown post


def load_state() -> dict:
    """
    state.json:
      posted_links: {link: {"date": iso_date, "title": str, "slot": str}}
      event_posts:  {event_key: iso_date of the last countdown post}
    """
    state = {}
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                state = as_dict(json.load(f))
    except Exception as e:
        log.warning(f"Could not load state: {e}")

    posted = {}
    for link, info in as_dict(state.get("posted_links")).items():
        if isinstance(info, str):          # plain {link: iso_date} form
            info = {"date": info}
        if isinstance(info, dict) and isinstance(info.get("date"), str):
            posted[link] = {
                "date":  info["date"],
                "title": str(info.get("title") or ""),
                "slot":  str(info.get("slot") or ""),
            }
    events = {k: v for k, v in as_dict(state.get("event_posts")).items() if isinstance(v, str)}
    return {"posted_links": posted, "event_posts": events}


def save_state(state: dict):
    """Save state atomically, pruning expired entries."""
    today      = berlin_today()
    link_cut   = (today - timedelta(days=POSTED_LINKS_TTL)).isoformat()
    event_cut  = (today - timedelta(days=EVENT_POSTS_TTL)).isoformat()
    pruned = {
        "posted_links": {k: v for k, v in state.get("posted_links", {}).items() if v["date"] >= link_cut},
        "event_posts":  {k: v for k, v in state.get("event_posts", {}).items() if v >= event_cut},
    }
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(pruned, f, indent=1)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        log.warning(f"Could not save state: {e}")


def record_posted(items: list[dict], slot: str):
    """
    Remember stories that were actually posted: links go to state.json, title
    hashes to the seen-headlines file. Called by senders after a successful send.
    A link that is already recorded keeps its original date and slot.
    """
    if not items:
        return
    today = berlin_today().isoformat()
    state = load_state()
    seen  = load_seen_headlines()
    for h in items:
        state["posted_links"].setdefault(h["link"], {"date": today, "title": h["title"], "slot": slot})
        seen.setdefault(headline_hash(h["title"]), today)
    save_state(state)
    save_seen_headlines(seen)


def posted_last_week(today: date | None = None, state: dict | None = None) -> list[dict]:
    """Stories posted in the 7 days before `today` (Monday's own picks excluded)."""
    today = today or berlin_today()
    state = state or load_state()
    start = (today - timedelta(days=7)).isoformat()
    end   = today.isoformat()
    items = [
        {"title": info["title"], "link": link, "date": info["date"]}
        for link, info in state["posted_links"].items()
        if start <= info["date"] < end and info["slot"] != "monday" and info["title"]
    ]
    items.sort(key=lambda x: x["date"])
    return items


# ── News ──────────────────────────────────────────────────────────────────────

SEEN_HEADLINES_TTL = 7  # days to remember a headline


def load_seen_headlines() -> dict:
    """Load persisted headline hashes with their date of first appearance."""
    try:
        if os.path.exists(SEEN_HEADLINES_FILE):
            with open(SEEN_HEADLINES_FILE, "r") as f:
                return as_dict(json.load(f))
    except Exception as e:
        log.warning(f"Could not load seen headlines: {e}")
    return {}


def save_seen_headlines(seen: dict):
    """Save headline hashes, pruning entries older than TTL."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=SEEN_HEADLINES_TTL)).strftime("%Y-%m-%d")
    pruned = {k: v for k, v in seen.items() if isinstance(v, str) and v >= cutoff}
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(SEEN_HEADLINES_FILE, "w") as f:
            json.dump(pruned, f)
    except Exception as e:
        log.warning(f"Could not save seen headlines: {e}")


def headline_hash(title: str) -> str:
    """Short hash of a normalised title for cross-day dedup."""
    normalised = " ".join(title.lower().split())
    return hashlib.md5(normalised.encode()).hexdigest()[:12]


def titles_are_similar(a: str, b: str) -> bool:
    """Return True if two titles are likely covering the same story."""
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    stops = {"a","an","the","in","on","at","to","for","of","and","or","but",
             "with","as","by","from","is","are","was","were","be","been"}
    a_words -= stops
    b_words -= stops
    if not a_words or not b_words:
        return False
    overlap = len(a_words & b_words) / min(len(a_words), len(b_words))
    return overlap > 0.6


def strip_html(text: str) -> str:
    """Plain text from an RSS summary."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text or "")).split())


def fetch_news_headlines(max_per_feed: int = 3, total_max: int = 10, skip_posted: bool = True) -> list[dict]:
    """
    Returns [{title, link, summary}] from RSS_FEEDS, near-duplicates removed.
    With skip_posted=True, anything already posted (seen title hash, posted link,
    or a title similar to a posted one) is left out — that is what keeps Thursday
    from repeating Tuesday.
    Read-only: nothing is marked as seen here. Senders call record_posted()
    after a successful send.
    """
    seen_hashes   = load_seen_headlines() if skip_posted else {}
    posted        = load_state()["posted_links"] if skip_posted else {}
    posted_titles = [p["title"] for p in posted.values() if p["title"]]
    headlines     = []
    seen_titles   = []

    for url in RSS_FEEDS:
        try:
            feed  = feedparser.parse(url)
            count = 0
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                link  = entry.get("link",  "").strip()
                if not title or not link:
                    continue
                if headline_hash(title) in seen_hashes or link in posted:
                    continue
                if any(titles_are_similar(title, t) for t in posted_titles):
                    continue
                # Skip near-duplicates within this batch
                if any(titles_are_similar(title, seen) for seen in seen_titles):
                    continue
                seen_titles.append(title)
                headlines.append({
                    "title":   title,
                    "link":    link,
                    "summary": shorten(strip_html(entry.get("summary", "")), 400),
                })
                count += 1
                if count >= max_per_feed:
                    break
        except Exception as e:
            log.warning(f"Feed error ({url}): {e}")

    return headlines[:total_max]


# ── Groq ──────────────────────────────────────────────────────────────────────

def groq_chat(prompt: str, max_tokens: int, temperature: float = 0.5, as_json: bool = False) -> str | None:
    """One Groq completion. Returns the text, or None on any error."""
    try:
        kwargs = {"response_format": {"type": "json_object"}} if as_json else {}
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return (response.choices[0].message.content or "").strip() or None
    except Exception as e:
        log.error(f"Groq error: {e}")
        return None


def groq_json(prompt: str, max_tokens: int, temperature: float = 0.5) -> dict | None:
    """Groq completion parsed as a JSON object, or None."""
    raw = groq_chat(prompt, max_tokens, temperature, as_json=True)
    if not raw:
        return None
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        parsed = json.loads(raw[start:end])
        return parsed if isinstance(parsed, dict) else None
    except ValueError as e:
        log.warning(f"Groq returned invalid JSON: {e}")
        return None


def numbered_headlines(headlines: list[dict], with_summary: bool = False) -> str:
    rows = []
    for i, h in enumerate(headlines):
        row = f"{i+1}. {h['title']}"
        if with_summary and h.get("summary"):
            row += f"\n   Summary: {h['summary']}"
        rows.append(row)
    return "\n".join(rows)


def pick_index(value, size: int) -> int | None:
    """1-based number from Groq → valid 0-based index, or None."""
    try:
        idx = int(value) - 1
    except (TypeError, ValueError):
        return None
    return idx if 0 <= idx < size else None


def ai_top_weekly_stories(headlines: list[dict], n: int = 3) -> list[dict]:
    """
    Ask Groq to pick the n most significant, topic-diverse stories.
    Always returns exactly n stories when at least n are available.
    """
    if not headlines:
        return []
    indices = []
    if len(headlines) > n:
        raw = groq_chat(
            "You are a senior iGaming industry analyst. "
            f"From the list of headlines below, pick the {n} most significant stories "
            "for a B2B iGaming audience — think regulation, major deals, market moves, or platform shifts. "
            "IMPORTANT: Do not pick two stories covering the same topic or event — diversity of topics matters. "
            f"Reply ONLY with {n} numbers on separate lines (e.g. 3) — no explanation, no text.\n\n"
            f"{numbered_headlines(headlines)}",
            max_tokens=30, temperature=0.3,
        ) or ""
        for token in re.findall(r"\d+", raw):
            idx = pick_index(token, len(headlines))
            if idx is not None and idx not in indices:
                indices.append(idx)
    # Pad with the first unpicked headlines so the count is always n
    for idx in range(len(headlines)):
        if len(indices) >= n:
            break
        if idx not in indices:
            indices.append(idx)
    return [headlines[i] for i in indices[:n]]


def ai_news_takes(headlines: list[dict], n: int = 5) -> tuple[list[dict], str]:
    """
    Ask Groq to pick n stories, add a one-line "why it matters" to each and an
    overall analyst take. Returns ([{title, link, summary, take}], overall).
    Falls back to the first n headlines without takes.
    """
    if not headlines:
        return [], ""
    result = groq_json(
        "You are a sharp iGaming industry analyst writing for a B2B Telegram channel. "
        f"From the headlines below pick the {min(n, len(headlines))} most significant for operators, "
        "suppliers and affiliates — regulation, deals, market moves, platform shifts. "
        "No two picks on the same topic. For each pick write one punchy line (max 20 words) on why it matters. "
        "Then write an overall analyst take of 2 sentences on what today's news says about the market. "
        "Use only what the headlines and summaries say — do not invent facts or numbers. No disclaimers.\n"
        'Reply with strict JSON only: {"stories": [{"n": <headline number>, "take": "<line>"}], "overall": "<take>"}\n\n'
        f"{numbered_headlines(headlines, with_summary=True)}",
        max_tokens=700, temperature=0.5,
    ) or {}

    picked, used = [], set()
    for story in as_list(result.get("stories")):
        idx = pick_index(as_dict(story).get("n"), len(headlines))
        if idx is None or idx in used:
            continue
        used.add(idx)
        picked.append({**headlines[idx], "take": clean_ai(as_dict(story).get("take"), 160)})
    for idx, h in enumerate(headlines):   # pad / fallback
        if len(picked) >= n:
            break
        if idx not in used:
            picked.append({**h, "take": ""})
    return picked[:n], clean_ai(result.get("overall"), 400)


def ai_deep_take(headlines: list[dict]) -> tuple[dict | None, str]:
    """
    Ask Groq to choose ONE story and write an 80–120 word take on it.
    Returns (headline, escaped_take). Take is "" if Groq fails.
    """
    if not headlines:
        return None, ""
    result = groq_json(
        "You are a senior iGaming industry analyst writing for a B2B Telegram channel. "
        "Choose the ONE story below with the biggest implications for operators, suppliers or affiliates, "
        "and write a deeper take of 80-120 words: what happened, why it matters, and what to watch next. "
        "One paragraph, present tense, no fluff, no disclaimers. "
        "Use only what the headline and summary say — do not invent facts, quotes or numbers.\n"
        'Reply with strict JSON only: {"n": <headline number>, "take": "<80-120 words>"}\n\n'
        f"{numbered_headlines(headlines, with_summary=True)}",
        max_tokens=400, temperature=0.6,
    ) or {}
    idx  = pick_index(result.get("n"), len(headlines))
    take = clean_ai(result.get("take"), 900)
    if idx is None or not take:
        return headlines[0], ""
    return headlines[idx], take


# ── Polls ─────────────────────────────────────────────────────────────────────

POLL_QUESTION_MAX = 300
POLL_OPTION_MAX   = 100
POLL_FOLLOWUP     = "💬 Vote above, then tell us why in the comments."

# Evergreen iGaming B2B topics — used when Groq fails or returns something invalid
POLL_FALLBACKS = [
    {"question": "What's the biggest growth lever for operators right now?",
     "options": ["New regulated markets", "Retention & CRM", "Crypto / onchain payments", "Content & streamers", "M&A"]},
    {"question": "Where does your player acquisition budget work hardest?",
     "options": ["Affiliates", "Streamers & creators", "Paid social / search", "Sponsorships", "Organic & SEO"]},
    {"question": "Which region will grow fastest over the next 12 months?",
     "options": ["LatAm", "North America", "Europe", "Africa", "Asia"]},
    {"question": "What's the biggest headache in payments today?",
     "options": ["Approval rates", "Chargebacks & fraud", "Processing fees", "Payout speed", "Crypto on/off-ramps"]},
    {"question": "Tighter regulation: net positive or net negative for the industry?",
     "options": ["Positive — it builds trust", "Negative — it feeds the black market", "Depends on the regulator", "Too early to tell"]},
    {"question": "Which vertical has the most upside right now?",
     "options": ["Sportsbook", "Slots & casino", "Live casino", "Crash & instant games", "Prediction markets"]},
    {"question": "Where will AI move the needle first in iGaming?",
     "options": ["Personalisation & CRM", "Fraud & risk", "Responsible gambling", "Customer support", "Trading & odds"]},
    {"question": "How do you see crypto casinos in three years?",
     "options": ["Mainstream and licensed", "Big but still offshore", "Squeezed by regulators", "Merged into traditional brands"]},
    {"question": "Which conference gives you the best ROI?",
     "options": ["ICE Barcelona", "SiGMA", "SBC Summit", "iGB L!VE", "I skip conferences"]},
    {"question": "What matters most when you pick a platform or game supplier?",
     "options": ["Price & rev share", "Speed of integration", "Content quality", "Licences & compliance", "Support"]},
]


def validate_poll(question, options) -> tuple[str, list[str]] | None:
    """
    Telegram poll rules: question 1–300 chars, 2–10 options of 1–100 chars each.
    Options are de-duplicated (case-insensitive). Returns None if invalid.
    """
    if not isinstance(question, str) or not isinstance(options, list):
        return None
    question = " ".join(question.split())
    if not question or tg_len(question) > POLL_QUESTION_MAX:
        return None
    if "<" in question or ">" in question:   # the bot's default HTML parse mode may apply to the question
        return None
    clean, seen = [], set()
    for opt in options:
        if not isinstance(opt, str):
            return None
        opt = " ".join(opt.split())
        if not opt or tg_len(opt) > POLL_OPTION_MAX:
            return None
        if opt.lower() not in seen:
            seen.add(opt.lower())
            clean.append(opt)
    if not 2 <= len(clean) <= 10:
        return None
    return question, clean


def fallback_poll(today: date | None = None) -> tuple[str, list[str]]:
    """Evergreen poll, rotated by ISO week so it doesn't repeat week to week."""
    week = (today or berlin_today()).isocalendar()[1]
    poll = POLL_FALLBACKS[week % len(POLL_FALLBACKS)]
    return poll["question"], list(poll["options"])


def ai_poll(headlines: list[dict]) -> tuple[str, list[str]] | None:
    """Ask Groq for a poll on this week's news. Returns None if it fails validation."""
    if not headlines:
        return None
    result = groq_json(
        "You run a B2B Telegram channel for iGaming professionals (operators, suppliers, affiliates, payments). "
        "Based on this week's headlines, write ONE poll that industry people would want to vote on — "
        "an opinion question about a trend or decision, not a quiz or a fact check. "
        "Rules: question max 200 characters; 3 to 5 answer options, each max 60 characters, "
        "all different, covering the realistic range of views. Neutral wording, no emojis.\n"
        'Reply with strict JSON only: {"question": "<question>", "options": ["<option>", "<option>", "<option>"]}\n\n'
        f"{numbered_headlines(headlines)}",
        max_tokens=250, temperature=0.7,
    )
    if not result:
        return None
    poll = validate_poll(result.get("question"), result.get("options"))
    if not poll:
        log.warning("Groq poll failed validation — using a fallback poll.")
    return poll


def build_wednesday_poll(today: date | None = None) -> tuple[str, list[str], str]:
    """Returns (question, options, follow-up text). Does not send."""
    today = today or berlin_today()
    week_start = (today - timedelta(days=today.weekday())).isoformat()
    # This week's headlines: what we posted since Monday, plus what's in the feeds now
    headlines = [
        {"title": info["title"]}
        for info in load_state()["posted_links"].values()
        if info["date"] >= week_start and info["title"]
    ]
    for h in fetch_news_headlines(max_per_feed=3, total_max=15, skip_posted=False):
        if not any(titles_are_similar(h["title"], x["title"]) for x in headlines):
            headlines.append(h)

    question, options = ai_poll(headlines[:20]) or fallback_poll(today)
    return question, options, POLL_FOLLOWUP


# ── Circus (creator market) ───────────────────────────────────────────────────

def fetch_json(url: str, source: str) -> dict | None:
    """One GET, fail-soft: returns the parsed JSON object or None."""
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError("unexpected response shape")
        log.info(f"{source} fetched OK")
        return data
    except Exception as e:
        log.warning(f"{source} unavailable: {e}")
        return None


def fetch_circus() -> tuple[dict | None, dict | None]:
    """Returns (streamer analytics, weekly digest). Either may be None."""
    if not ENABLE_CIRCUS:
        return None, None
    return (
        fetch_json(CIRCUS_ANALYTICS_API, "Circus streamer-analytics"),
        fetch_json(CIRCUS_UPDATES_API, "Circus updates"),
    )


def build_circus_block(analytics: dict | None, updates: dict | None) -> str:
    """Monday "Creator market" block — up to 5 lines, "" if there is no data."""
    try:
        return _circus_block(as_dict(analytics), as_dict(as_dict(updates).get("digest")))
    except Exception as e:
        log.warning(f"Circus block skipped: {e}")
        return ""


def _circus_block(analytics: dict, digest: dict) -> str:
    lines   = []
    summary = as_dict(analytics.get("summary"))
    total   = num(summary.get("totalViewerHours"))

    # 1. Total viewer-hours + week-on-week change
    if total and total > 0:
        line = f"• {format_count(total)} viewer-hours"
        streamers = num(summary.get("totalStreamers"))
        if streamers:
            line += f" across {streamers:.0f} streamers"
        prev = num(as_dict(analytics.get("previousPeriod")).get("totalViewerHours"))
        if prev and prev > 0:
            line += f" (<b>{format_pct((total - prev) / prev * 100)}</b> WoW)"
        lines.append(line)

    # 2. Top 3 casinos by viewer-hours, with share of total
    casinos = []
    for c in as_list(analytics.get("casinoBreakdown")):
        c     = as_dict(c)
        hours = num(c.get("viewerHours"))
        if c.get("casino") and hours and hours > 0:
            casinos.append((str(c["casino"]), hours))
    casinos.sort(key=lambda x: x[1], reverse=True)
    if casinos:
        parts = []
        for name, hours in casinos[:3]:
            share = hours / total * 100 if total and total > 0 else None
            parts.append(f"{esc(name)} {share:.0f}%" if share is not None and share <= 100
                         else f"{esc(name)} {format_count(hours)}h")
        lines.append("• Top casinos by watch time: " + " · ".join(parts))

    # 3. #1 streamer
    top = as_dict((as_list(analytics.get("topStreamers")) or [None])[0])
    most_watched = as_dict(as_dict(digest.get("streamers")).get("mostWatched"))
    if top.get("displayName") or top.get("streamer"):
        line = f"• #1 streamer: <b>{esc(top.get('displayName') or top.get('streamer'))}</b>"
        if top.get("platform"):
            line += f" ({esc(str(top['platform']).title())})"
        top_casinos = [str(c) for c in as_list(top.get("casinos")) if c]
        if top_casinos:
            line += f" — {esc(top_casinos[0])}"
        lines.append(line)
    elif most_watched.get("displayName"):
        line = f"• Most watched: <b>{esc(most_watched['displayName'])}</b>"
        if most_watched.get("currentCasino"):
            line += f" — {esc(most_watched['currentCasino'])}"
        lines.append(line)

    # 4. Market-share mover (deposits)
    casino_digest = as_dict(digest.get("casino"))
    mover = as_dict(casino_digest.get("marketShareMover"))
    delta = num(mover.get("shareDelta"))
    if mover.get("name") and delta is not None:
        lines.append(f"• Deposit share mover: {esc(mover['name'])} <b>{delta:+.1f} pts</b>")

    # 5. Hottest chain
    chain     = as_dict(casino_digest.get("hottestChain"))
    chain_pct = num(chain.get("pct"))
    if chain.get("chain") and chain_pct is not None:
        lines.append(f"• Hottest chain: {esc(str(chain['chain']).upper())} <b>{format_pct(chain_pct)}</b>")

    # Filler when something above is missing: biggest casino mover by watch time.
    # changePct is absurd on tiny bases, so require a real previous week.
    if len(lines) < 5:
        movers = []
        for m in as_list(as_dict(analytics.get("biggestMovers")).get("casinos")):
            m      = as_dict(m)
            prev   = num(m.get("prevViewerHours"))
            change = num(m.get("changePct"))
            if m.get("casino") and change is not None and prev is not None and prev >= CIRCUS_MIN_PREV_HOURS:
                movers.append((abs(change), str(m["casino"]), change))
        if movers:
            _, name, change = max(movers)
            lines.append(f"• Biggest watch-time mover: {esc(name)} <b>{format_pct(change)}</b>")

    if not lines:
        return ""
    return "\n".join(["🎥 <b>Creator market — last 7 days</b>"] + lines[:5] + ["<i>Data: circus.fyi</i>"])


# ── FairGambling (market & trust) ─────────────────────────────────────────────

def fetch_fairgambling() -> tuple[dict | None, dict | None]:
    """Returns (home bundle, global complaint stats). Either may be None."""
    if not ENABLE_FAIRGAMBLING:
        return None, None
    return (
        fetch_json(FAIRGAMBLING_BUNDLE_API, "FairGambling home-bundle"),
        fetch_json(FAIRGAMBLING_COMPLAINTS_API, "FairGambling complaints"),
    )


def build_fairgambling_block(bundle: dict | None, complaints: dict | None) -> str:
    """Thursday "Market & trust" block — up to 4 lines, "" if there is no data."""
    try:
        return _fairgambling_block(as_dict(as_dict(bundle).get("data")), as_dict(as_dict(complaints).get("data")))
    except Exception as e:
        log.warning(f"FairGambling block skipped: {e}")
        return ""


def _fairgambling_block(data: dict, stats: dict) -> str:
    lines = []

    # Casinos with a usable 7-day market share, biggest first
    ranked = []
    for row in as_list(data.get("breakdown7d")):
        row   = as_dict(row)
        share = num(row.get("marketShare"))
        if row.get("casinoName") and share is not None:
            ranked.append((share, str(row["casinoName"]), num(row.get("depositVolumeChange"))))
    ranked.sort(key=lambda x: x[0], reverse=True)

    # 1. Top 3 by 7-day market share, with deposit volume change
    if ranked:
        parts = []
        for share, name, change in ranked[:3]:
            part = f"{esc(name)} {share:.1f}%"
            if change is not None:
                part += f" (vol {format_pct(change)})"
            parts.append(part)
        lines.append("• 7d deposit share: " + " · ".join(parts))

    # 2. Biggest 7-day gainer among the top 10
    gainers = [(change, name) for _, name, change in ranked[:10] if change is not None and change > 0]
    if gainers:
        change, name = max(gainers)
        lines.append(f"• Biggest gainer in the top 10: {esc(name)} <b>{format_pct(change)}</b> deposits")

    # 3. Highest trust scores (trustScore is a string like "9.20", often null)
    trusted = []
    for c in as_list(data.get("casinos")):
        c     = as_dict(c)
        score = num(c.get("trustScore"))
        if c.get("name") and score is not None:
            trusted.append((score, str(c["name"])))
    trusted.sort(key=lambda x: x[0], reverse=True)
    if trusted:
        lines.append("• Highest trust scores: " + " · ".join(f"{esc(name)} {score:.1f}" for score, name in trusted[:3]))

    # 4. Complaints
    disputes = num(stats.get("totalDisputes"))
    if disputes is not None:
        line = f"• Complaints: {disputes:,.0f} disputes"
        resolved = num(stats.get("resolved"))
        if resolved is not None:
            line += f", {resolved:,.0f} resolved"
        hours = num(stats.get("avgResolutionHours"))
        if hours is not None and hours > 0:
            line += f", avg resolution {hours / 24:.1f} days"
        lines.append(line)

    if not lines:
        return ""
    return "\n".join(["🛡 <b>Market &amp; trust</b>"] + lines + ["<i>Data: fairgambling.com</i>"])


# ── Message builders (no sending, no state writes) ────────────────────────────

def build_monday_overview(
    data: dict | None,
    dist_data: dict | None = None,
    trending: list | None = None,
    circus_analytics: dict | None = None,
    circus_updates: dict | None = None,
) -> tuple[str, io.BytesIO | None, list[dict]]:
    """
    Monday overview: weekly chart + recap, top 3 stories of last week, optional
    Circus creator block. Returns (text, chart, stories shown).
    """
    today      = berlin_today()
    week_end   = today.strftime("%d %b %Y")
    week_start = (today - timedelta(days=6)).strftime("%d %b")

    lines = [
        f"☕ <b>Happy Monday, The Cashout community!</b>\n"
        f"Here's your weekly onchain recap.\n"
        f"<i>{week_start} – {week_end}</i>\n"
    ]

    chart_buf = None

    if data:
        weekly_vols = parse_weekly_volumes(data)
        chart_buf   = generate_weekly_chart(weekly_vols)

        if weekly_vols:
            sorted_vols = sorted(weekly_vols.items(), key=lambda x: x[1], reverse=True)[:10]
            medals = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]
            lines.append("📊 <b>Top 10 Onchain Casinos — 7-Day Volume</b>")
            for i, (name, vol) in enumerate(sorted_vols):
                lines.append(f"{medals[i]} {esc(name)} — {format_volume(vol)}")
        else:
            lines.append("<i>Volume data unavailable this week.</i>")
    else:
        lines.append("<i>Could not reach Tanzanite Terminal API.</i>")

    # Week-on-week movers from dedicated trending endpoint
    movers = []
    for e in as_list(trending):
        e   = as_dict(e)
        pct = num(e.get("usd_change_pct"))
        if e.get("label") and pct is not None:
            movers.append((str(e["label"]), pct))
    gainers = [m for m in movers if m[1] > 0][:3]
    losers  = [m for m in reversed(movers) if m[1] < 0][:3]
    if gainers or losers:
        lines.append("\n📈 <b>Week-on-Week Movers</b>")
    if gainers:
        lines.append("Winners:")
        for label, pct in gainers:
            lines.append(f"  ↑ {esc(label)} +{pct:.1f}%")
    if losers:
        lines.append("Losers:")
        for label, pct in losers:
            lines.append(f"  ↓ {esc(label)} {pct:.1f}%")

    # Player profile — retail vs whale breakdown from distribution endpoint
    if dist_data:
        profile = parse_player_profile(dist_data)
        if profile:
            lines.append("\n🎯 <b>Player Profile This Week</b>")
            lines.append(
                f"🐋 <b>Whale casino:</b> {esc(profile['whale'])} "
                f"— {profile['whale_pct']:.0f}% of volume from $100K+ deposits"
            )
            if profile["retail"]:
                lines.append(
                    f"🛒 <b>Most retail:</b> {esc(profile['retail'])} "
                    f"— {profile['retail_pct']:.0f}% of volume from &lt;$1K deposits"
                )
            if profile["balanced"]:
                lines.append(
                    f"📊 <b>Most balanced:</b> {esc(profile['balanced'])} "
                    f"— most evenly spread across deposit sizes"
                )

    # Top 3 stories of last week — prefer what we actually posted last week,
    # top up from the current feeds when state has fewer than 3
    candidates = posted_last_week(today)
    if len(candidates) < 3:
        for h in fetch_news_headlines(max_per_feed=5, total_max=25, skip_posted=False):
            if not any(h["link"] == c["link"] or titles_are_similar(h["title"], c["title"]) for c in candidates):
                candidates.append(h)
    top3 = ai_top_weekly_stories(candidates, n=3)
    if top3:
        lines.append("\n📰 <b>Top 3 Stories Last Week</b>")
        for h in top3:
            lines.extend(news_item_lines(h))

    circus_block = build_circus_block(circus_analytics, circus_updates)
    if circus_block:
        lines.append("\n" + circus_block)

    footer = f"\n<i>Data: tanzanite.xyz | {FOOTER_COMMENTS}</i>"
    return truncate_message("\n".join(lines), footer), chart_buf, top3


def build_tuesday_news() -> tuple[str, list[dict]]:
    """Tuesday: fresh news with analyst takes. Returns (text, stories shown); ("", []) if no news."""
    headlines = fetch_news_headlines(max_per_feed=3, total_max=14)
    stories, overall = ai_news_takes(headlines, n=5)
    if not stories:
        return "", []

    lines = [
        f"📰 <b>The Cashout — Tuesday News</b>\n"
        f"<i>{berlin_today().strftime('%d %B %Y')} · what the industry is talking about</i>\n"
    ]
    for h in stories:
        lines.extend(news_item_lines(h, note=h["take"]))
    if overall:
        lines.append(f"\n💡 <b>Analyst take:</b> <i>{overall}</i>")

    return truncate_message("\n".join(lines), f"\n<i>{FOOTER_COMMENTS}</i>"), stories


def build_onchain_block(data: dict | None, top_n: int = 3) -> list[str]:
    """Compact winner / loser / top gainers block from Tanzanite deposit-volume."""
    if not data:
        return ["📊 <i>Could not reach Tanzanite Terminal API today.</i>"]
    movements = parse_casino_movements(data)
    if not movements:
        return ["📊 <i>Onchain casino data unavailable today.</i>"]

    w = movements[0]
    lines = [
        f"🏆 <b>Winner:</b> {esc(w['name'])} — {format_volume(w['current'])} "
        f"(<b>{w['pct']:+.1f}%</b> vs prev. period)"
    ]
    if len(movements) > 1:
        l = movements[-1]
        lines.append(
            f"💀 <b>Loser:</b> {esc(l['name'])} — {format_volume(l['current'])} "
            f"(<b>{l['pct']:+.1f}%</b>)"
        )
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    lines.append(f"📈 <b>Top {min(top_n, len(movements))} gainers</b>")
    for i, m in enumerate(movements[:min(top_n, len(medals))]):
        lines.append(f"{medals[i]} {esc(m['name'])} — {format_volume(m['current'])} (<b>{m['pct']:+.1f}%</b>)")
    lines.append("<i>Data: tanzanite.xyz</i>")
    return lines


def build_thursday_update(
    data: dict | None,
    fg_bundle: dict | None = None,
    fg_complaints: dict | None = None,
) -> tuple[str, list[dict]]:
    """
    Thursday: compact onchain update, one deeper take on a story that has not
    been posted this week, optional FairGambling block. Returns (text, stories shown).
    """
    lines = [
        f"⛓️ <b>The Cashout — Onchain Update</b>\n"
        f"<i>{berlin_today().strftime('%d %B %Y')} · deposit volume, last period vs previous</i>\n"
    ]
    lines.extend(build_onchain_block(data))

    # fetch_news_headlines leaves out everything already posted (Tuesday included)
    story, take = ai_deep_take(fetch_news_headlines(max_per_feed=3, total_max=12))
    if story:
        lines.append("\n🔍 <b>The deeper take</b>")
        lines.append(f"<b>{esc(shorten(story['title'], 180))}</b>")
        if take:
            lines.append(take)
        lines.append(f'<a href="{esc_url(story["link"])}">Read more →</a>')

    fg_block = build_fairgambling_block(fg_bundle, fg_complaints)
    if fg_block:
        lines.append("\n" + fg_block)

    text = truncate_message("\n".join(lines), f"\n<i>{FOOTER_COMMENTS}</i>")
    return text, [story] if story else []


# ── Afternoon drink (Friday networking) ───────────────────────────────────────

DRINK_OPTIONS = ["Operator", "Supplier", "Affiliate", "Payments", "Media", "Just here for the news"]
DRINK_TEMPLATE = "name · company · looking for"

# Rotated by ISO week number
AFTERNOON_DRINK_VARIANTS = [
    {"question": "What are you working on this week?",
     "intro": "The week's nearly done. Pull up a chair and say hi — someone here is looking for exactly what you do."},
    {"question": "Which side of the industry are you on?",
     "intro": "Friday round is on us. Tell the room who you are — deals in this industry start with a hello."},
    {"question": "Who's at the bar this Friday?",
     "intro": "Laptops half closed, glasses half full. Good moment to meet someone new before the weekend."},
    {"question": "What's your corner of iGaming?",
     "intro": "iGaming is a small world — make it smaller. One line is enough to start a conversation."},
    {"question": "Where do you sit in the value chain?",
     "intro": "End-of-week networking, no badge needed. New here or a regular, drop a line and see who replies."},
]


def build_afternoon_drink(today: date | None = None) -> tuple[str, list[str], str]:
    """Friday networking post. Returns (poll question, poll options, text). Does not send."""
    week    = (today or berlin_today()).isocalendar()[1]
    variant = AFTERNOON_DRINK_VARIANTS[week % len(AFTERNOON_DRINK_VARIANTS)]
    text = (
        f"<b>Afternoon drink 🍻</b>\n\n"
        f"{esc(variant['intro'])}\n\n"
        f"Tap the poll below, then reply in the comments with one line:\n"
        f"<code>{DRINK_TEMPLATE}</code>"
    )
    return variant["question"], list(DRINK_OPTIONS), text


# ── Event countdowns (main channel) ───────────────────────────────────────────

COUNTDOWN_WINDOW_DAYS = 35   # weekly "who's going?" posts start this many days out
FINAL_CALL_DAYS       = 3    # one last post this many days before start (moved off weekends)


def final_call_date(event: dict) -> date:
    """
    Day of the final-call post: 3 days before start. If that falls on a weekend,
    use 2 days before when that's a weekday, otherwise the Friday before.
    """
    day = event["start"] - timedelta(days=FINAL_CALL_DAYS)
    if day.weekday() < 5:
        return day
    two_before = event["start"] - timedelta(days=2)
    if two_before.weekday() < 5:
        return two_before
    return day - timedelta(days=day.weekday() - 4)


def event_key(event: dict) -> str:
    return f"{event['name']}|{event['start'].isoformat()}"


def format_event_dates(start: date, end: date) -> str:
    """18–20 Jan 2027 / 30 Jun – 2 Jul 2027"""
    if (start.year, start.month) == (end.year, end.month):
        return f"{start.day}–{end.day} {end.strftime('%b %Y')}"
    return f"{start.day} {start.strftime('%b')} – {end.day} {end.strftime('%b %Y')}"


def same_iso_week(a: date, b: date) -> bool:
    return a.isocalendar()[:2] == b.isocalendar()[:2]


def due_event_posts(today: date, mode: str, last_posted: dict[str, str] | None = None) -> list[dict]:
    """
    Pure: which countdown posts are due on `today`?
      mode="weekly" — the weekly "who's going?" post, from 35 days before start
      mode="final"  — the final call, only on final_call_date(event) (2–4 days before, a weekday)
    `last_posted` maps event_key → ISO date of that event's last countdown post.
    Rules: never once the event has started; at most one post per event per ISO
    week (the weekly post gives way to the final call when both fall in one week).
    Returns [{"event", "days", "kind"}].
    """
    last_posted = last_posted or {}
    due = []
    for event in BIG_EVENTS:
        days = (event["start"] - today).days
        if days <= 0:
            continue
        last = last_posted.get(event_key(event))
        try:
            if last and same_iso_week(date.fromisoformat(last), today):
                continue
        except ValueError:
            pass

        final_day = final_call_date(event)
        if mode == "final":
            if today == final_day:
                due.append({"event": event, "days": days, "kind": "final"})
        elif mode == "weekly":
            if today < final_day and days <= COUNTDOWN_WINDOW_DAYS and not same_iso_week(final_day, today):
                due.append({"event": event, "days": days, "kind": "weekly"})
    return due


def build_event_countdown_message(event: dict, days: int, kind: str = "weekly") -> str:
    """Countdown text for one event. Does not send."""
    if days % 7 == 0:
        weeks = days // 7
        left  = f"{weeks} week{'s' if weeks != 1 else ''}"
    else:
        left = f"{days} day{'s' if days != 1 else ''}"

    place = f"{event['venue']}, {event['city']}" if event.get("venue") else event["city"]
    where = f"📅 {format_event_dates(event['start'], event['end'])} · 📍 {esc(place)}"
    name  = esc(event["name"])

    if kind == "final":
        return (
            f"🚨 <b>Final call — {name} starts in {left}</b>\n\n"
            f"{where}\n\n"
            f"Last chance to line up meetings before the floor opens. "
            f"Reply in the comments if you're going or want to meet — "
            f"<code>{DRINK_TEMPLATE}</code>."
        )
    return (
        f"⏳ <b>{left} to {name} — who's going?</b>\n\n"
        f"{where}\n\n"
        f"Reply in the comments if you're going or want to meet — "
        f"company, role, and what you'd like to talk about. "
        f"The more people connect before the doors open, the better."
    )


def build_event_countdowns(today: date | None = None, mode: str = "weekly",
                           last_posted: dict[str, str] | None = None) -> list[tuple[dict, str]]:
    """Returns [(event, text)] for every countdown due today. Empty if none."""
    return [
        (item["event"], build_event_countdown_message(item["event"], item["days"], item["kind"]))
        for item in due_event_posts(today or berlin_today(), mode, last_posted)
    ]


# ── Senders ───────────────────────────────────────────────────────────────────

CHART_CAPTION = "📊 <b>Weekly Onchain Deposit Rankings</b>\n<i>Data: tanzanite.xyz</i>"


def send_monday_overview():
    log.info("Building Monday overview…")
    if not CASINO_LABELS:
        fetch_casino_labels()   # startup load can fail (Tanzanite 5xx) — retry once per post
    try:
        data      = fetch_tanzanite()
        dist_data = fetch_tanzanite_distribution()
        trending  = fetch_tanzanite_trending()   # week-on-week
        circus_analytics, circus_updates = fetch_circus()
        message, chart, stories = build_monday_overview(
            data, dist_data, trending, circus_analytics, circus_updates)

        if chart:
            # Chart as photo with a short caption (1024-char limit), full text as follow-up
            caption = CHART_CAPTION if tg_len(CHART_CAPTION) <= TG_CAPTION_LIMIT else ""
            tg_send("send_photo", CHANNEL_ID, photo=chart, caption=caption, parse_mode="HTML")
        send_text(message)
        record_posted(stories, "monday")
        log.info("Monday overview posted successfully.")
    except Exception as e:
        log.error(f"Failed to post Monday overview: {e}")


def send_tuesday_news():
    log.info("Building Tuesday news…")
    try:
        message, stories = build_tuesday_news()
        if not message:
            log.warning("No fresh headlines — Tuesday news skipped.")
            return
        send_text(message)
        record_posted(stories, "tuesday")
        log.info("Tuesday news posted successfully.")
    except Exception as e:
        log.error(f"Failed to post Tuesday news: {e}")


def send_wednesday_poll():
    log.info("Building Wednesday poll…")
    try:
        question, options, followup = build_wednesday_poll()
        send_channel_poll(question, options)
        send_text(followup)
        log.info("Wednesday poll posted successfully.")
    except Exception as e:
        log.error(f"Failed to post Wednesday poll: {e}")


def send_thursday_update():
    log.info("Building Thursday update…")
    if not CASINO_LABELS:
        fetch_casino_labels()   # startup load can fail (Tanzanite 5xx) — retry once per post
    try:
        data = fetch_tanzanite()
        fg_bundle, fg_complaints = fetch_fairgambling()
        message, stories = build_thursday_update(data, fg_bundle, fg_complaints)
        send_text(message)
        record_posted(stories, "thursday")
        log.info("Thursday update posted successfully.")
    except Exception as e:
        log.error(f"Failed to post Thursday update: {e}")


def send_afternoon_drink():
    log.info("Posting Afternoon drink…")
    try:
        question, options, text = build_afternoon_drink()
        send_text(text)
        send_channel_poll(question, options)
        log.info("Afternoon drink posted successfully.")
    except Exception as e:
        log.error(f"Failed to post Afternoon drink: {e}")


def send_event_countdowns(mode: str = "weekly"):
    """Post due event countdowns to the main channel. Does nothing if none are due."""
    if not ENABLE_EVENT_COUNTDOWNS:
        return
    try:
        state = load_state()
        today = berlin_today()
        for event, text in build_event_countdowns(today, mode, state["event_posts"]):
            try:
                send_text(text)
                state["event_posts"][event_key(event)] = today.isoformat()
                save_state(state)
                log.info(f"Event countdown ({mode}) posted for {event['name']}")
            except Exception as e:
                log.error(f"Failed to post countdown for {event['name']}: {e}")
    except Exception as e:
        log.error(f"Event countdown check failed: {e}")


def send_event_final_calls():
    """Daily check — only posts on the exact final-call day of an event."""
    send_event_countdowns(mode="final")


# ── Scheduler ─────────────────────────────────────────────────────────────────

def main():
    log.info("Starting The Cashout bot…")
    log.info(f"State dir: {STATE_DIR} | Circus: {ENABLE_CIRCUS} | FairGambling: {ENABLE_FAIRGAMBLING} "
             f"| Event countdowns: {ENABLE_EVENT_COUNTDOWNS}")

    # Load casino display labels from Tanzanite API (non-fatal if unavailable)
    fetch_casino_labels()

    # ── Test lines — uncomment ONE to test on deploy, then re-comment ──────────
    # send_monday_overview()            # ← Monday overview test
    # send_tuesday_news()               # ← Tuesday news test
    # send_wednesday_poll()             # ← Wednesday poll test
    # send_thursday_update()            # ← Thursday update test
    # send_afternoon_drink()            # ← Friday afternoon drink test
    # send_event_countdowns()           # ← weekly event countdown test (posts only if one is due)
    # send_event_final_calls()          # ← event final-call test (posts only on the final-call day)

    # Europe/Berlin — DST shifts automatically
    scheduler = BlockingScheduler(
        timezone=SCHEDULE_TZ,
        job_defaults={"coalesce": True, "misfire_grace_time": 600},
    )

    jobs = [
        (send_monday_overview, "mon", 13,  0, "Mon 13:00 Monday overview"),
        (send_tuesday_news,    "tue", 11,  0, "Tue 11:00 news"),
        (send_wednesday_poll,  "wed", 11, 30, "Wed 11:30 poll"),
        (send_thursday_update, "thu", 11,  0, "Thu 11:00 onchain update"),
        (send_afternoon_drink, "fri", 15, 30, "Fri 15:30 afternoon drink"),
    ]
    if ENABLE_EVENT_COUNTDOWNS:
        jobs += [
            (send_event_countdowns,  "wed", 15, 0, "Wed 15:00 event countdowns"),
            (send_event_final_calls, "*",   10, 0, "daily 10:00 event final-call check"),
        ]
    for func, day_of_week, hour, minute, _ in jobs:
        scheduler.add_job(func, "cron", day_of_week=day_of_week, hour=hour, minute=minute, id=func.__name__)

    log.info(f"Scheduler running ({SCHEDULE_TZ}) — " + " | ".join(label for *_, label in jobs))

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Bot stopped.")


if __name__ == "__main__":
    main()
