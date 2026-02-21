"""
The Cashout — iGaming Daily Brief
Telegram Channel Bot

Stack: Python, APScheduler, feedparser, Groq (llama-3.3-70b-versatile), telebot, requests, matplotlib
Deploy: Railway via GitHub
"""

import os
import io
import logging
import requests
import feedparser
import telebot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from groq import Groq
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timezone, timedelta
from collections import defaultdict

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

# ── Clients ───────────────────────────────────────────────────────────────────
bot         = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="HTML")
groq_client = Groq(api_key=GROQ_API_KEY)

# ── Casino name display map ───────────────────────────────────────────────────
# Maps raw API slugs → clean display names
# Add more as you discover them in the data
CASINO_NAMES = {
    "roobetcom":       "Roobet",
    "rollbitcom":      "Rollbit",
    "stakeus":         "Stake",
    "stakecasino":     "Stake",
    "bcgame":          "BC.Game",
    "bc_game":         "BC.Game",
    "betfury":         "BetFury",
    "yologroup":       "Yolo Group",
    "acebet":          "AceBet",
    "_500casino":      "500 Casino",
    "500casino":       "500 Casino",
    "winr":            "WINR",
    "crashino":        "Crashino",
    "duelbits":        "Duelbits",
    "thunderpick":     "Thunderpick",
    "hypedrop":        "HypeDrop",
    "csgoempire":      "CSGOEmpire",
    "rakebackgg":      "Rakeback.gg",
    "gamdomcom":       "Gamdom",
    "fortunejack":     "FortuneJack",
    "metaspins":       "Metaspins",
    "bitstarz":        "BitStarz",
    "cloudbet":        "Cloudbet",
    "1xbit":           "1xBit",
    "sportsbet":       "Sportsbet.io",
    "sportsbetio":     "Sportsbet.io",
    "primedice":       "Primedice",
    "winstrike":       "Winstrike",
    "bspin":           "Bspin",
    "trustdice":       "TrustDice",
    "nitrobetting":    "Nitrobetting",
}

def clean_name(raw: str) -> str:
    """Return display name for a casino slug."""
    return CASINO_NAMES.get(raw.lower(), raw.replace("_", " ").title())

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

# ── Event channels ────────────────────────────────────────────────────────────
# Add each event as a dict below. Set active=True when the channel is live,
# False to disable without deleting. Create the Telegram channel manually,
# add the bot as admin, then paste the channel ID here.
#
# EVENT_CHANNELS = [
#     {
#         "id":       "@thecashout_sbcrio",   # or numeric ID
#         "name":     "SBC Rio 2026",
#         "date":     "10–12 June 2026",
#         "location": "Rio de Janeiro, Brazil",
#         "hashtag":  "#SBCRio2026",
#         "active":   True,
#     },
#     {
#         "id":       "@thecashout_sigmaeurope",
#         "name":     "SiGMA Europe 2026",
#         "date":     "17–20 November 2026",
#         "location": "Malta",
#         "hashtag":  "#SiGMAEurope",
#         "active":   False,   # flip to True when channel is ready
#     },
# ]

EVENT_CHANNELS = []  # Populate above and replace this line when ready

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
TANZANITE_API = "https://terminal.tanzanite.xyz/api/public/overview"


def fetch_tanzanite() -> dict | None:
    try:
        resp = requests.get(TANZANITE_API, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        log.info(f"Tanzanite fetched OK — {data.get('total_count', '?')} records")
        return data
    except Exception as e:
        log.error(f"Tanzanite API error: {e}")
        return None


def parse_casino_movements(data: dict) -> list:
    """
    Returns list of dicts:
    {name, raw_name, current, previous, pct, chains}
    sorted by pct DESC, filtered by MIN_VOLUME_USD.
    """
    casino_data = defaultdict(lambda: {"current": 0.0, "previous": 0.0, "chains": set()})

    try:
        sites = data["timeframes"]["monthly"]["sites"]
    except (KeyError, TypeError):
        log.error("Tanzanite: could not find timeframes.monthly.sites")
        return []

    for casino_name, casino_info in sites.items():
        for chain_name, chain_info in casino_info.get("chains", {}).items():
            for token_name, token_info in chain_info.get("tokens", {}).items():
                intervals = token_info.get("intervals", [])
                if len(intervals) >= 1:
                    casino_data[casino_name]["current"] += float(intervals[0].get("usd", 0) or 0)
                    casino_data[casino_name]["chains"].add(chain_name)
                if len(intervals) >= 2:
                    casino_data[casino_name]["previous"] += float(intervals[1].get("usd", 0) or 0)

    movements = []
    for raw_name, vols in casino_data.items():
        curr = vols["current"]
        prev = vols["previous"]

        # Filter out low-volume casinos
        if curr < MIN_VOLUME_USD:
            continue

        if prev > 0:
            pct = ((curr - prev) / prev) * 100
        elif curr > 0:
            pct = 100.0
        else:
            pct = 0.0

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
        sites = data["timeframes"]["monthly"]["sites"]
    except (KeyError, TypeError):
        return {}

    for casino_name, casino_info in sites.items():
        for chain_name, chain_info in casino_info.get("chains", {}).items():
            for token_name, token_info in chain_info.get("tokens", {}).items():
                for interval in token_info.get("intervals", []):
                    try:
                        day = datetime.strptime(interval["day"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        if day >= cutoff:
                            casino_vols[casino_name] += float(interval.get("usd", 0) or 0)
                    except (ValueError, KeyError):
                        continue

    # Clean names and filter low volume
    cleaned = {}
    for raw, vol in casino_vols.items():
        if vol >= MIN_VOLUME_USD:
            cleaned[clean_name(raw)] = cleaned.get(clean_name(raw), 0) + vol

    return cleaned


def format_volume(vol: float) -> str:
    if vol >= 1_000_000:
        return f"${vol/1_000_000:.2f}M"
    if vol >= 1_000:
        return f"${vol/1_000:.1f}K"
    return f"${vol:.0f}"


# ── Chart generator ───────────────────────────────────────────────────────────

def generate_weekly_chart(weekly_vols: dict[str, float]) -> io.BytesIO | None:
    """Generate a dark-themed horizontal bar chart of top 10 casinos by 7-day volume."""
    if not weekly_vols:
        return None

    # Top 10 by volume
    sorted_items = sorted(weekly_vols.items(), key=lambda x: x[1], reverse=True)[:10]
    names  = [item[0] for item in reversed(sorted_items)]
    values = [item[1] for item in reversed(sorted_items)]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#05050a")
    ax.set_facecolor("#05050a")

    # Bars
    colors = ["#00f078" if v == max(values) else "#00b050" for v in values]
    bars = ax.barh(names, values, color=colors, height=0.6, edgecolor="none")

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
            format_volume(val),
            va="center", ha="left",
            color="#d4af37", fontsize=9, fontweight="bold"
        )

    # Styling
    ax.set_xlabel("7-Day Deposit Volume (USD)", color="#888", fontsize=9)
    ax.tick_params(colors="#ccc", labelsize=9)
    ax.xaxis.label.set_color("#888")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format_volume(x)))
    ax.tick_params(axis="x", colors="#555")
    ax.tick_params(axis="y", colors="#ccc")
    ax.grid(axis="x", color="#1a2a1a", linewidth=0.5)

    # Title
    week_end   = datetime.now(timezone.utc).strftime("%d %b")
    week_start = (datetime.now(timezone.utc) - timedelta(days=6)).strftime("%d %b")
    ax.set_title(
        f"📊 Top Onchain Casinos by Deposit Volume  |  {week_start} – {week_end}",
        color="#00f078", fontsize=11, fontweight="bold", pad=12
    )

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Groq narrative ────────────────────────────────────────────────────────────

def ai_onchain_narrative(movements: list) -> str:
    """Ask Groq to write a 2-3 sentence analyst take on today's onchain data."""
    if not movements:
        return ""

    top5    = movements[:5]
    loser   = movements[-1]
    summary = ", ".join(f"{m['name']} ({m['pct']:+.1f}%)" for m in top5)
    loser_s = f"{loser['name']} ({loser['pct']:+.1f}%)"

    prompt = (
        "You are a sharp crypto gambling industry analyst writing for a B2B Telegram channel. "
        "Based on today's onchain deposit data, write 2-3 punchy sentences interpreting what's happening. "
        "Be specific — reference the casinos and numbers. No fluff, no disclaimers. "
        "Write in present tense as if this is breaking intelligence.\n\n"
        f"Top gainers today: {summary}\n"
        f"Biggest loser: {loser_s}\n"
        f"Winner volume: {format_volume(movements[0]['current'])}\n"
        f"Loser volume: {format_volume(loser['current'])}"
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.6,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"Groq narrative error: {e}")
        return ""


# ── News ──────────────────────────────────────────────────────────────────────

def titles_are_similar(a: str, b: str) -> bool:
    """Return True if two titles are likely covering the same story."""
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    # Remove common stop words
    stops = {"a","an","the","in","on","at","to","for","of","and","or","but",
             "with","as","by","from","is","are","was","were","be","been"}
    a_words -= stops
    b_words -= stops
    if not a_words or not b_words:
        return False
    overlap = len(a_words & b_words) / min(len(a_words), len(b_words))
    return overlap > 0.6


def fetch_news_headlines(max_per_feed: int = 3, total_max: int = 12) -> list[dict]:
    headlines = []
    seen_titles = []

    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                link  = entry.get("link",  "").strip()
                if not title or not link:
                    continue
                # Smart dedup — catch near-duplicate stories
                if any(titles_are_similar(title, seen) for seen in seen_titles):
                    continue
                seen_titles.append(title)
                headlines.append({"title": title, "link": link})
                count += 1
                if count >= max_per_feed:
                    break
        except Exception as e:
            log.warning(f"Feed error ({url}): {e}")

    return headlines[:total_max]


# ── Morning greeting basket ───────────────────────────────────────────────────
DAILY_GREETINGS = [
    "Good morning — let's see who moved the chips overnight. ☕",
    "Rise and grind. Onchain never sleeps, but you should. Here's what happened. 📊",
    "Morning. Markets moved. Casinos shifted. Let's get into it. 🎰",
    "Another day, another leaderboard. GM. 👋",
    "Your daily dose of onchain reality. No fluff, just numbers. 📈",
    "Doors are open, data is fresh. Good morning. ☀️",
    "While you were sleeping, the chains were busy. Morning briefing below. 🌅",
    "GM. Here's who won, who lost, and what the industry is talking about today. 🗞️",
    "Coffee in hand? Good. You'll need it for these numbers. ☕",
    "Fresh data, fresh week. Let's go. 🚀",
    "The onchain scorecard is in. Good morning. 🏆",
    "Morning. Roobet or Rollbit today? The data has opinions. 👀",
    "Wakey wakey. The leaderboard has spoken. 📣",
    "No algos, no spin — just raw onchain deposits. GM. ⛓️",
    "The industry doesn't pause. Neither do we. Good morning. ⚡",
]

# ── Message builders ──────────────────────────────────────────────────────────

def build_daily_message(data: dict | None) -> str:
    today = datetime.now(timezone.utc).strftime("%A, %d %B %Y")

    # Rotate greeting by day-of-year — consistent per day, never repeats for 15 days
    day_index = datetime.now(timezone.utc).timetuple().tm_yday
    greeting  = DAILY_GREETINGS[day_index % len(DAILY_GREETINGS)]

    lines = [
        f"<b>🎰 The Cashout — {today}</b>",
        f"<i>{greeting}</i>\n",
    ]

    # ── Casino section ────────────────────────────────────────────────────────
    if data:
        movements = parse_casino_movements(data)

        if movements:
            # Winner
            w = movements[0]
            chains = chain_tag(w["chains"])
            lines.append(
                f"🏆 <b>Winner of the Day</b>\n"
                f"<b>{w['name']}</b> {chains} — {format_volume(w['current'])} "
                f"(<b>+{w['pct']:.1f}%</b> vs yesterday)\n"
            )

            # Loser
            l = movements[-1]
            chains_l = chain_tag(l["chains"])
            lines.append(
                f"💀 <b>Loser of the Day</b>\n"
                f"<b>{l['name']}</b> {chains_l} — {format_volume(l['current'])} "
                f"(<b>{l['pct']:+.1f}%</b> vs yesterday)\n"
            )

            # Top 5
            medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
            gainers_lines = ["📈 <b>Top 5 Onchain Gainers</b>"]
            for i, m in enumerate(movements[:5]):
                gainers_lines.append(
                    f"{medals[i]} {m['name']} — {format_volume(m['current'])} "
                    f"(<b>{m['pct']:+.1f}%</b>)"
                )
            lines.append("\n".join(gainers_lines) + "\n")

            # Groq narrative
            narrative = ai_onchain_narrative(movements)
            if narrative:
                lines.append(f"💡 <i>{narrative}</i>\n")
        else:
            lines.append("📊 <i>Onchain casino data unavailable today.</i>\n")
    else:
        lines.append("📊 <i>Could not reach Tanzanite Terminal API today.</i>\n")

    # ── News section ──────────────────────────────────────────────────────────
    headlines = fetch_news_headlines()
    news_block = ["📰 <b>Industry News</b>"]
    for h in headlines:
        news_block.append(
            f'• {h["title"]}\n'
            f'  <a href="{h["link"]}">Read more →</a>'
        )
    lines.append("\n".join(news_block))

    # ── Footer ────────────────────────────────────────────────────────────────
    lines.append(
        "\n<i>Onchain data: <a href=\"https://terminal.tanzanite.xyz\">Tanzanite Terminal</a> "
        "| 💬 Drop a take in the comments</i>"
    )

    return "\n".join(lines)


def ai_top3_weekly_stories(headlines: list[dict]) -> list[dict]:
    """Ask Groq to pick the 3 most significant stories from the week's headlines."""
    if not headlines:
        return []
    numbered = "\n".join(f"{i+1}. {h['title']}" for i, h in enumerate(headlines))
    prompt = (
        "You are a senior iGaming industry analyst. "
        "From the list of headlines below, pick the 3 most significant stories "
        "for a B2B iGaming audience — think regulation, major deals, market moves, or platform shifts. "
        "Reply ONLY with three numbers on separate lines (e.g. 3) — no explanation, no text.\n\n"
        f"{numbered}"
    )
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        indices = []
        for line in raw.split("\n"):
            try:
                idx = int(line.strip()) - 1
                if 0 <= idx < len(headlines):
                    indices.append(idx)
            except ValueError:
                continue
        return [headlines[i] for i in indices[:3]]
    except Exception as e:
        log.error(f"Groq top3 error: {e}")
        return headlines[:3]


def build_weekly_message(data: dict | None) -> tuple[str, io.BytesIO | None]:
    """Build the Monday weekly recap message + chart image."""
    week_end   = datetime.now(timezone.utc).strftime("%d %b %Y")
    week_start = (datetime.now(timezone.utc) - timedelta(days=6)).strftime("%d %b")

    lines = [
        f"☕ <b>Happy Monday, The Cashout community!</b>\n"
        f"Grab your coffee — here's your weekly onchain recap.\n"
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
                lines.append(f"{medals[i]} {name} — {format_volume(vol)}")
        else:
            lines.append("<i>Volume data unavailable this week.</i>")
    else:
        lines.append("<i>Could not reach Tanzanite Terminal API.</i>")

    # Top 3 stories of the week via Groq
    headlines = fetch_news_headlines(max_per_feed=5, total_max=20)
    top3 = ai_top3_weekly_stories(headlines)
    if top3:
        lines.append("\n📰 <b>Top 3 Stories This Week</b>")
        for h in top3:
            lines.append(
                f'• {h["title"]}\n'
                f'  <a href="{h["link"]}">Read more →</a>'
            )

    lines.append(
        "\n<i>Onchain data: <a href=\"https://terminal.tanzanite.xyz\">Tanzanite Terminal</a> "
        "| 💬 Drop a take in the comments</i>"
    )

    return "\n".join(lines), chart_buf


def send_event_welcome(event: dict):
    """
    Post a welcome/networking message to an event channel.
    Call this manually or hook it to a scheduled date.
    """
    message = (
        f"👋 <b>Welcome to The Cashout — {event['name']}</b>\n\n"
        f"📅 <b>Date:</b> {event['date']}\n"
        f"📍 <b>Location:</b> {event['location']}\n\n"
        f"This channel is your networking hub for {event['name']}. "
        f"Drop your name, company, and what you're looking to connect on — "
        f"let's make the most of the week.\n\n"
        f"The Cashout daily onchain data + news brief runs in the main channel. "
        f"This room is for people, meetings, and side events.\n\n"
        f"{event['hashtag']} | 💬 Introduce yourself below"
    )
    try:
        bot.send_message(event["id"], message, parse_mode="HTML")
        log.info(f"Event welcome posted to {event['name']} ({event['id']})")
    except Exception as e:
        log.error(f"Failed to post event welcome to {event['id']}: {e}")


def send_event_countdown(event: dict, days_remaining: int):
    """Post a countdown message to an event channel."""
    message = (
        f"⏳ <b>{days_remaining} days to {event['name']}</b>\n\n"
        f"📅 {event['date']} · 📍 {event['location']}\n\n"
        f"Who's going? Drop a reply — company, role, and what you want to "
        f"talk about. The more people connect before the floor opens, the better.\n\n"
        f"{event['hashtag']}"
    )
    try:
        bot.send_message(event["id"], message, parse_mode="HTML")
        log.info(f"Countdown posted to {event['name']} — {days_remaining} days")
    except Exception as e:
        log.error(f"Failed to post countdown to {event['id']}: {e}")


# ── Senders ───────────────────────────────────────────────────────────────────

def send_daily_brief():
    log.info("Building daily brief…")
    try:
        data    = fetch_tanzanite()
        message = build_daily_message(data)
        bot.send_message(
            CHANNEL_ID,
            message,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        log.info("Daily brief posted successfully.")
    except Exception as e:
        log.error(f"Failed to post daily brief: {e}")


def send_weekly_recap():
    log.info("Building weekly recap…")
    try:
        data          = fetch_tanzanite()
        message, chart = build_weekly_message(data)

        if chart:
            bot.send_photo(
                CHANNEL_ID,
                photo=chart,
                caption=message,
                parse_mode="HTML",
            )
        else:
            bot.send_message(
                CHANNEL_ID,
                message,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        log.info("Weekly recap posted successfully.")
    except Exception as e:
        log.error(f"Failed to post weekly recap: {e}")


# ── Scheduler ─────────────────────────────────────────────────────────────────

def main():
    log.info("Starting The Cashout bot…")

    # ── Test lines — uncomment ONE, deploy, then re-comment and redeploy ─────
    # send_weekly_recap()  # ← weekly recap test
    send_daily_brief()     # ← daily brief test — remove after confirming

    scheduler = BlockingScheduler(timezone="UTC")

    # Daily brief — Tue–Fri at 07:00 UTC (= 8am CET winter / 9am CEST summer)
    scheduler.add_job(send_daily_brief, "cron", day_of_week="tue-fri", hour=7, minute=0)

    # Weekly recap — Monday at 07:00 UTC
    scheduler.add_job(send_weekly_recap, "cron", day_of_week="mon", hour=7, minute=0)

    log.info("Scheduler running — Mon: weekly recap | Tue–Fri: daily brief at 07:00 UTC | weekends off.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Bot stopped.")


if __name__ == "__main__":
    main()
