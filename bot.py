"""
iGaming Daily Brief — Telegram Channel Bot
Stack: Python, APScheduler, feedparser, Groq (llama-3.3-70b-versatile), telebot, requests
Deploy: Railway via GitHub
"""

import os
import logging
import requests
import feedparser
import telebot
from groq import Groq
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timezone
from collections import defaultdict

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("igaming-bot")

# ── Environment variables ─────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
CHANNEL_ID     = os.environ["CHANNEL_ID"]        # e.g. "@YourChannel" or "-100xxxxx"
GROQ_API_KEY   = os.environ["GROQ_API_KEY"]
RSSHUB_URL     = os.environ.get("RSSHUB_URL", "")  # e.g. "https://rsshub.yourdomain.com"

# ── Clients ───────────────────────────────────────────────────────────────────
bot        = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="HTML")
groq_client = Groq(api_key=GROQ_API_KEY)

# ── RSS feed list ─────────────────────────────────────────────────────────────
RSS_FEEDS = [
    "https://sbcnews.co.uk/feed",
    "https://igamingbusiness.com/feed",
    "https://igamingexpert.com/feed",
    "https://www.yogonet.com/international/rss.xml",
    "https://gamblingnews.com/feed",
    "https://gamblinginsider.com/feed",
    "https://cryptogamblingnews.com/feed",
]

# Append RSSHub Twitter feeds if configured
TWITTER_ACCOUNTS = ["SBCnews", "iGamingBusiness", "GamblingInsider", "tanzanite_xyz"]
if RSSHUB_URL:
    for account in TWITTER_ACCOUNTS:
        RSS_FEEDS.append(f"{RSSHUB_URL.rstrip('/')}/twitter/user/{account}")

# ── Tanzanite API ─────────────────────────────────────────────────────────────
TANZANITE_API = "https://terminal.tanzanite.xyz/api/public/overview"

def fetch_tanzanite() -> dict | None:
    """Fetch Tanzanite Terminal overview data."""
    try:
        resp = requests.get(TANZANITE_API, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.error(f"Tanzanite API error: {e}")
        return None


def parse_casino_movements(data: dict) -> tuple[list, list]:
    """
    Parse the Tanzanite JSON and return:
      - sorted list of (casino_name, current_vol, pct_change) by pct_change DESC
      - same list sorted ASC (losers first)

    The API returns per-casino data. We aggregate all chains/tokens per casino,
    compare current period vs previous period deposit volume.
    Adjust the key names below to match the actual API response schema.
    """
    casino_data = defaultdict(lambda: {"current": 0.0, "previous": 0.0})

    # ── Adapt this block to the real API schema ───────────────────────────────
    # Expected structure (example — verify against real response):
    # data = {
    #   "casinos": [
    #     {
    #       "name": "Casino XYZ",
    #       "chains": [
    #         {
    #           "tokens": [
    #             {
    #               "deposits": {"current": 12345.67, "previous": 9876.54}
    #             }
    #           ]
    #         }
    #       ]
    #     }
    #   ]
    # }
    casinos = data.get("casinos") or data.get("data") or []
    for casino in casinos:
        name = casino.get("name") or casino.get("casino") or "Unknown"
        for chain in casino.get("chains", []):
            for token in chain.get("tokens", []):
                dep = token.get("deposits", {})
                casino_data[name]["current"]  += float(dep.get("current", 0) or 0)
                casino_data[name]["previous"] += float(dep.get("previous", 0) or 0)
    # ─────────────────────────────────────────────────────────────────────────

    movements = []
    for name, vols in casino_data.items():
        curr = vols["current"]
        prev = vols["previous"]
        if prev > 0:
            pct = ((curr - prev) / prev) * 100
        elif curr > 0:
            pct = 100.0
        else:
            pct = 0.0
        movements.append((name, curr, pct))

    movements.sort(key=lambda x: x[2], reverse=True)
    return movements  # first = biggest gainer, last = biggest loser


def format_volume(vol: float) -> str:
    """Human-readable USD volume."""
    if vol >= 1_000_000:
        return f"${vol/1_000_000:.2f}M"
    if vol >= 1_000:
        return f"${vol/1_000:.1f}K"
    return f"${vol:.0f}"


# ── RSS / News ────────────────────────────────────────────────────────────────

def fetch_news_headlines(max_per_feed: int = 3, total_max: int = 12) -> list[dict]:
    """Collect recent headlines across all RSS feeds."""
    headlines = []
    seen_titles = set()

    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                link  = entry.get("link", "").strip()
                if not title or not link:
                    continue
                # Basic dedup on normalised title
                key = title.lower()[:60]
                if key in seen_titles:
                    continue
                seen_titles.add(key)
                headlines.append({"title": title, "link": link})
                count += 1
                if count >= max_per_feed:
                    break
        except Exception as e:
            log.warning(f"Feed error ({url}): {e}")

    return headlines[:total_max]


def ai_summarise_headlines(headlines: list[dict]) -> str:
    """Ask Groq to produce a punchy 1-liner summary for each headline."""
    if not headlines:
        return "No headlines available today."

    bullet_list = "\n".join(
        f"{i+1}. {h['title']}" for i, h in enumerate(headlines)
    )
    prompt = (
        "You are a concise iGaming industry analyst. "
        "For each numbered headline below, write ONE punchy sentence (max 20 words) "
        "summarising the story for a B2B audience. "
        "Reply ONLY with the numbered list — no intro, no preamble.\n\n"
        f"{bullet_list}"
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"Groq error: {e}")
        # Fallback: return plain titles
        return "\n".join(f"{i+1}. {h['title']}" for i, h in enumerate(headlines))


# ── Message builder ───────────────────────────────────────────────────────────

def build_daily_message() -> str:
    """Assemble the full daily brief."""
    today = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
    lines = [f"<b>🎰 iGaming Daily Brief — {today}</b>\n"]

    # ── Casino section ────────────────────────────────────────────────────────
    data = fetch_tanzanite()

    if data:
        movements = parse_casino_movements(data)

        if movements:
            # Winner
            winner_name, winner_vol, winner_pct = movements[0]
            lines.append(
                f"🏆 <b>Winner of the Day</b>\n"
                f"<b>{winner_name}</b> — {format_volume(winner_vol)} deposits "
                f"(<b>+{winner_pct:.1f}%</b> vs yesterday)\n"
            )

            # Loser
            loser_name, loser_vol, loser_pct = movements[-1]
            lines.append(
                f"💀 <b>Loser of the Day</b>\n"
                f"<b>{loser_name}</b> — {format_volume(loser_vol)} deposits "
                f"(<b>{loser_pct:+.1f}%</b> vs yesterday)\n"
            )

            # Top 5 Gainers
            top5 = movements[:5]
            gainers_lines = ["📈 <b>Top 5 Onchain Gainers</b>"]
            medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
            for i, (name, vol, pct) in enumerate(top5):
                gainers_lines.append(
                    f"{medals[i]} {name} — {format_volume(vol)} (<b>{pct:+.1f}%</b>)"
                )
            lines.append("\n".join(gainers_lines) + "\n")
        else:
            lines.append("📊 <i>Onchain casino data unavailable today.</i>\n")
    else:
        lines.append("📊 <i>Could not reach Tanzanite Terminal API today.</i>\n")

    # ── News section ──────────────────────────────────────────────────────────
    headlines = fetch_news_headlines()
    summaries_raw = ai_summarise_headlines(headlines)
    summary_lines = summaries_raw.strip().split("\n")

    news_block = ["📰 <b>Industry News</b>"]
    for i, h in enumerate(headlines):
        # Try to match summary line by index
        summary_text = ""
        if i < len(summary_lines):
            # Strip leading "1. " etc.
            summary_text = summary_lines[i].lstrip("0123456789. ").strip()
        news_block.append(
            f'• <a href="{h["link"]}">{h["title"]}</a>\n'
            f'  <i>{summary_text}</i>'
        )

    lines.append("\n".join(news_block))

    # ── Footer ────────────────────────────────────────────────────────────────
    lines.append(
        "\n<i>Onchain data: <a href=\"https://terminal.tanzanite.xyz\">Tanzanite Terminal</a> "
        "| 💬 Drop a take in the comments</i>"
    )

    return "\n".join(lines)


# ── Telegram sender ───────────────────────────────────────────────────────────

def send_daily_brief():
    log.info("Building daily brief…")
    try:
        message = build_daily_message()
        bot.send_message(
            CHANNEL_ID,
            message,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        log.info("Daily brief posted successfully.")
    except Exception as e:
        log.error(f"Failed to post daily brief: {e}")


# ── Scheduler ─────────────────────────────────────────────────────────────────

def main():
    log.info("Starting iGaming Daily Brief bot…")

    send_daily_brief()  # ← must be indented with 4 spaces, inside main()

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(send_daily_brief, "cron", hour=8, minute=0)
    log.info("Scheduler running. Next post at 08:00 UTC daily.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Bot stopped.")


if __name__ == "__main__":
    main()
