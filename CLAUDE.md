# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Purpose

**The Cashout** is a scheduled Telegram channel bot for the iGaming / crypto-gambling industry. It posts to one Telegram channel:

- **Daily brief** (Tue–Fri, 08:00 UTC): onchain casino deposit winner, loser, and top 5 gainers from the Tanzanite public API; a short analyst take written by Groq (LLM); and deduplicated industry headlines from RSS feeds.
- **Weekly recap** (Mon, 08:00 UTC): a matplotlib bar chart of the top 10 casinos by 7-day deposit volume, week-on-week movers, a whale/retail player profile, and Groq's pick of the week's top 5 stories.
- **Introduction Friday** (Fri, 08:30 UTC): a rotating networking prompt for the community.
- **Event channels** (built but switched off): `send_event_welcome` / `send_event_countdown` for conference networking channels. `EVENT_CHANNELS` is empty.

The bot only posts. It has no command or message handlers and doesn't poll Telegram for updates.

## Stack

- Python 3.10+ (the code uses `X | None` type syntax). No Python version is pinned in the repo.
- `pyTelegramBotAPI` (`telebot`): sends messages with `parse_mode="HTML"`.
- `APScheduler`: a `BlockingScheduler` with cron jobs in UTC.
- `groq`: model `llama-3.3-70b-versatile`, hardcoded in two places.
- `feedparser`: RSS news, plus optional Twitter/X feeds through RSSHub.
- `requests`: Tanzanite public API (`https://www.tanzanite.xyz/api/public/casino-analytics/...`: `deposit-volume`, `trending`, `deposit-distribution`). No auth needed.
- `matplotlib` (Agg backend) and `numpy`: weekly chart, rendered to an in-memory PNG.

## Structure

```
bot.py            # The whole application (~1040 lines)
requirements.txt  # Unpinned minimum versions (>=)
README.md         # One-line description
```

`bot.py` layout, top to bottom:

| Section | What it holds |
|---|---|
| Env vars and clients | Reads config and creates the `bot` and `groq_client` objects at import time |
| `CASINO_NAMES` / `CASINO_LABELS` / `clean_name` | Slug → display-name map. `CASINO_LABELS` is filled at startup from Tanzanite `meta.entity_display`, with `CASINO_NAMES` as the fallback |
| `CHAIN_*` / `chain_tag` | Emoji tags for each chain |
| `MIN_VOLUME_USD` | $50k minimum deposit volume to appear in the rankings |
| `EVENT_CHANNELS` | Event channel config (currently `[]`, with a commented example) |
| `RSS_FEEDS`, `TWITTER_ACCOUNTS` | News sources |
| Tanzanite fetch/parse | `fetch_tanzanite`, `fetch_tanzanite_trending`, `fetch_tanzanite_distribution`, `fetch_casino_labels`, `parse_casino_movements`, `parse_weekly_volumes`, `parse_player_profile` |
| `generate_weekly_chart` | matplotlib chart, returned as `io.BytesIO` |
| Groq | `ai_onchain_narrative`, `ai_top5_weekly_stories` |
| News | `fetch_news_headlines`, with deduplication across days through `SEEN_HEADLINES_FILE` (`/app/seen_headlines.json`, 7-day TTL) and fuzzy title matching within a batch |
| Message builders | `build_daily_message`, `build_weekly_message`, event messages |
| Senders | `send_daily_brief`, `send_weekly_recap`, `send_intro_friday` |
| `main()` | Loads casino labels, registers the cron jobs, starts the scheduler |

## How to run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Export the required env vars (see below). Use a TEST channel. The bot posts for real.
python bot.py
```

- The process blocks on the scheduler and only posts at the scheduled times.
- To trigger a post right away, temporarily uncomment one of the test lines in `main()` (`send_weekly_recap()`, `send_daily_brief()`, `send_intro_friday()`). Re-comment it before committing, because every push to `main` deploys to production.
- Another option is to call a builder without sending anything, for example `python -c "import bot; print(bot.build_daily_message(bot.fetch_tanzanite()))"`. The required env vars must still be set because clients are created at import time, and this calls Groq.
- Seen-headline persistence writes to `/app/seen_headlines.json`. Locally, `/app` usually doesn't exist, so loading and saving fail with a logged warning and cross-day deduplication is off.

## How to test

There are **no automated tests**, no test framework, and no linter or CI configuration. Testing today is manual:

1. Run the bot locally against a test channel, using one of the test lines in `main()` or a direct call to a builder or sender.
2. Watch the logs (`cashout-bot` logger, INFO level) for Tanzanite, Groq, and feed errors. Every external call catches its own exceptions and logs them, so a failure produces a degraded message rather than a crash.

If you add tests, the pure functions are easy to cover with fixture JSON: `parse_casino_movements`, `parse_weekly_volumes`, `parse_player_profile`, `clean_name`, `chain_tag`, `format_volume`, `titles_are_similar`, `headline_hash`, and `build_*_message` with the network calls mocked. Importing `bot` requires `TELEGRAM_TOKEN`, `CHANNEL_ID`, and `GROQ_API_KEY` to be set (dummy values are fine).

## Deployment

- **Platform:** Railway, auto-deploying from GitHub. The `bot.py` header comment says *"Deploy: Railway via GitHub"*.
- **Default branch:** `main`.
- **Trigger:** GitHub Deployments show `railway-app[bot]` creating a deployment to environment **`gallant-love / production`** for each commit on `main`. This is Railway's GitHub integration auto-deploying. **Every merge or push to `main` goes to production.**
- **Repo contents:** a single-file Python Telegram bot (`bot.py`, `requirements.txt`, `README.md`).
- **No deploy config in the repo:** no GitHub Actions workflows and no `railway.json`, `railway.toml`, `Procfile`, or `Dockerfile`. Railway detects the build and start command itself (Python, `requirements.txt`, presumably running `python bot.py`). Any start-command override, env vars, or volumes live in the Railway dashboard, not in this repo.
- Secrets and config are set as Railway service variables.
- The `/app/seen_headlines.json` path assumes Railway's `/app` working directory. Unless a Railway volume is mounted there, the file is lost on every redeploy or restart. A volume can't be confirmed from the repo.

## Environment variables

Names only, read from `bot.py`. There is no `.env.example`.

| Name | Required | Purpose |
|---|---|---|
| `TELEGRAM_TOKEN` | Yes | Telegram bot token |
| `CHANNEL_ID` | Yes | Target channel (`@name` or numeric ID) |
| `GROQ_API_KEY` | Yes | Groq API key |
| `RSSHUB_URL` | No | Base URL of an RSSHub instance. If set, Twitter/X feeds for `TWITTER_ACCOUNTS` are added to `RSS_FEEDS` |

Never commit values for these variables.

## Known issues / TODOs

- **No tests, CI, or linting.** Every push to `main` deploys to production without any checks.
- **DST is handled by hand.** The jobs are fixed at 08:00 UTC. A code comment says the time should change to 07:00 UTC when CEST starts (late March) to keep 9am local time. Nothing automates this. Using `timezone="Europe/..."` on the scheduler would fix it.
- **Headline persistence is probably ephemeral** on Railway (see Deployment), and it doesn't work locally because of the hardcoded `/app` path.
- **The weekly "Top 5 Stories Last Week" isn't last week's stories.** `build_weekly_message` calls `fetch_news_headlines`, which *skips* headlines already posted in the daily briefs and takes the newest items from each feed. It doesn't filter by date. It also records up to 25 more hashes as seen.
- **HTML isn't escaped.** RSS titles, links, Groq output, and casino names go straight into HTML-mode messages. A `<` or `&` in a title can make Telegram reject the whole post with a parse error. Use `html.escape` on dynamic text.
- **Message length isn't checked.** The weekly recap is sent as a single message. If it goes over Telegram's 4096-character limit, the send fails. The chart caption is kept short on purpose because captions are limited to 1024 characters.
- **Failures are silent.** Every error is caught and logged, and there is no alerting or retry. A failed scheduled post is simply skipped.
- **Test lines in `main()`** must be commented out by hand before pushing. Forgetting causes an extra post on every deploy.
- **Event channels are unused.** `EVENT_CHANNELS = []`, and the event senders aren't scheduled. They are only meant to be called by hand.
- **Groq model name is duplicated** in `ai_onchain_narrative` and `ai_top5_weekly_stories`.
- **Dependencies are unpinned** (`>=` only), and no Python version is pinned. Builds aren't reproducible.
- `TANZANITE_DISTRIBUTION_API` is defined after `fetch_casino_labels`, the function that uses it. This works because the name is resolved when the function is called, but it is easy to break when reordering code.
- `CASINO_NAMES` needs manual upkeep ("Add more as you discover them"). The API labels loaded at startup are the main source.
