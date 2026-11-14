# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Purpose

**The Cashout** is a scheduled Telegram channel bot for the iGaming / crypto-gambling industry. It posts to one Telegram channel:

Weekly rhythm (all times Europe/Berlin; APScheduler handles summer time):

| When | Job | Content |
|---|---|---|
| Mon 13:00 | `send_monday_overview` | Chart photo + weekly Tanzanite recap (top 10, WoW movers, player profile), top 3 stories of last week, optional "🎥 Creator market" block (Circus) |
| Tue 11:00 | `send_tuesday_news` | Fresh news only: 5 stories with a Groq "why it matters" line + overall analyst take |
| Wed 11:30 | `send_wednesday_poll` | Anonymous regular poll (Groq writes it from the week's headlines; falls back to `POLL_FALLBACKS`) + a comment-invite text |
| Thu 11:00 | `send_thursday_update` | Compact Tanzanite winner/loser/top 3, one Groq "deeper take" on a story not posted earlier in the week, optional "🛡 Market & trust" block (FairGambling) |
| Fri 15:30 | `send_afternoon_drink` | "Afternoon drink 🍻" text with the `name · company · looking for` template, then a role poll. Copy rotates by ISO week (`AFTERNOON_DRINK_VARIANTS`) |
| Wed 15:00 | `send_event_countdowns` | Weekly "who's going?" countdown for `BIG_EVENTS`, from 35 days before start |
| Daily 10:00 | `send_event_final_calls` | Final call on `final_call_date(event)`: 3 days before start, moved to a weekday (2 or 4 days) if that's a weekend |

Event countdowns go to the main channel. At most one countdown per event per ISO week; the weekly post is skipped in the final-call week. `BIG_EVENTS` (2027): ICE Barcelona, SiGMA World Rome, iGB L!VE Amsterdam, SBC Summit Lisbon, SiGMA Europe Malta. Add next year's dates there.

Every job builds its post with a `build_*` function (no sending, no state writes) and sends through `tg_send`, which keeps ≥1.1s between sends to the same chat and retries once after a 429 using `retry_after`.

The bot only posts. It has no command or message handlers and doesn't poll Telegram for updates.

## Stack

- Python 3.10+ (the code uses `X | None` type syntax). No Python version is pinned in the repo.
- `pyTelegramBotAPI` (`telebot`): sends messages with `parse_mode="HTML"`.
- `APScheduler`: a `BlockingScheduler(timezone="Europe/Berlin")` (`tzdata` in requirements for slim images).
- `groq`: model `GROQ_MODEL = "llama-3.3-70b-versatile"`, called through `groq_chat` / `groq_json`. Every Groq failure has a non-AI fallback.
- `feedparser`: RSS news, plus optional Twitter/X feeds through RSSHub.
- `requests`: Tanzanite public API (`https://www.tanzanite.xyz/api/public/casino-analytics/...`: `deposit-volume`, `trending`, `deposit-distribution`). No auth needed.
- Circus (`https://www.circus.fyi/api/streamer-analytics?days=7`, `/api/updates`) and FairGambling (`https://api.fairgambling.com/api/analytics/home-bundle`, `/api/complaints/stats/global`): undocumented public frontend APIs, no auth. One GET each per post, 15s timeout, `fetch_json` returns `None` on any error and the block is omitted. Field names used are documented in `build_circus_block` / `build_fairgambling_block`. **Both sites' robots.txt disallow /api/ and their terms restrict reuse; get permission before relying on them.**
- `matplotlib` (Agg backend) and `numpy`: weekly chart, rendered to an in-memory PNG.

## Structure

```
bot.py            # The whole application (~1850 lines)
.gitignore        # .state/, state files, .venv, .env
requirements.txt  # Unpinned minimum versions (>=)
README.md         # One-line description
```

`bot.py` layout, top to bottom:

| Section | What it holds |
|---|---|
| Env vars, toggles, state dir | Required env vars, `ENABLE_*` flags (`env_flag`), `STATE_DIR` resolution, clients, API URLs |
| `CASINO_NAMES` / `CASINO_LABELS` / `clean_name` | Slug → display name. Labels load from Tanzanite at startup and are retried before Mon/Thu posts if that failed |
| `BIG_EVENTS`, `RSS_FEEDS`, `TWITTER_ACCOUNTS` | Config |
| Tanzanite fetch/parse + `generate_weekly_chart` | Unchanged from before the rework |
| Helpers | `esc` / `esc_url` (HTML escaping), `tg_len` (UTF-16 length), `truncate_message` (≤4096, cuts whole lines) |
| `tg_send`, `send_text`, `send_channel_poll` | Rate-limited Telegram sends |
| State | `state.json` in `STATE_DIR`: `posted_links` {link: {date, title, slot}} (14-day TTL) and `event_posts` {event_key: date}. `seen_headlines.json` sits next to it |
| News + Groq | `fetch_news_headlines`, `ai_top_weekly_stories`, `ai_news_takes`, `ai_deep_take`, `ai_poll` / `validate_poll` / `POLL_FALLBACKS` |
| Data blocks | `fetch_circus` / `build_circus_block`, `fetch_fairgambling` / `build_fairgambling_block`, `build_onchain_block` |
| Builders | `build_monday_overview`, `build_tuesday_news`, `build_wednesday_poll`, `build_thursday_update`, `build_afternoon_drink`, `due_event_posts` (pure), `build_event_countdowns` |
| Senders + `main()` | `send_*` jobs, commented test hooks, scheduler registration |

## How to run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Export the required env vars (see below). Use a TEST channel. The bot posts for real.
python bot.py
```

- The process blocks on the scheduler and only posts at the scheduled times.
- To trigger a post right away, temporarily uncomment one of the test lines in `main()` (one per job: `send_monday_overview()`, `send_tuesday_news()`, `send_wednesday_poll()`, `send_thursday_update()`, `send_afternoon_drink()`, `send_event_countdowns()`, `send_event_final_calls()`). Re-comment it before committing, because every push to `main` deploys to production.
- Safer: dry-run with sending stubbed. Set dummy env vars, `STATE_DIR=/tmp/x`, replace `bot.bot.send_message/send_photo/send_poll` with functions that print, replace `bot.groq_client` with a stub (or let it fail; every Groq call has a fallback), then call the `send_*` jobs. Builders can also be called directly, e.g. `bot.build_afternoon_drink()`, `bot.due_event_posts(date(2027,1,15), "final")`.
- State lives in `STATE_DIR` (env var; default `/app` if writable, else `./.state`, which is gitignored).

## How to test

There are **no automated tests**, no test framework, and no linter or CI configuration. Testing today is manual:

1. Run the bot locally against a test channel, using one of the test lines in `main()` or a direct call to a builder or sender.
2. Watch the logs (`cashout-bot` logger, INFO level) for Tanzanite, Groq, and feed errors. Every external call catches its own exceptions and logs them, so a failure produces a degraded message rather than a crash.

If you add tests, the pure functions are easy to cover with fixture JSON: `parse_*`, `clean_name`, `format_volume`, `titles_are_similar`, `validate_poll`, `due_event_posts` / `final_call_date`, `build_circus_block` / `build_fairgambling_block` (take raw JSON), `build_afternoon_drink`. Importing `bot` requires `TELEGRAM_TOKEN`, `CHANNEL_ID`, and `GROQ_API_KEY` to be set (dummy values are fine).

## Deployment

- **Platform:** Railway, auto-deploying from GitHub. The `bot.py` header comment says *"Deploy: Railway via GitHub"*.
- **Default branch:** `main`.
- **Trigger:** GitHub Deployments show `railway-app[bot]` creating a deployment to environment **`gallant-love / production`** for each commit on `main`. This is Railway's GitHub integration auto-deploying. **Every merge or push to `main` goes to production.**
- **Repo contents:** a single-file Python Telegram bot (`bot.py`, `requirements.txt`, `README.md`).
- **No deploy config in the repo:** no GitHub Actions workflows and no `railway.json`, `railway.toml`, `Procfile`, or `Dockerfile`. Railway detects the build and start command itself (Python, `requirements.txt`, presumably running `python bot.py`). Any start-command override, env vars, or volumes live in the Railway dashboard, not in this repo.
- Secrets and config are set as Railway service variables.
- State (`state.json`, `seen_headlines.json`) goes to `STATE_DIR`, default `/app`. Unless a Railway volume is mounted and `STATE_DIR` points at it, state is lost on every redeploy or restart; then Thursday can repeat a Tuesday story and an event countdown could repeat in the same week after a redeploy.

## Environment variables

Names only, read from `bot.py`. There is no `.env.example`.

| Name | Required | Purpose |
|---|---|---|
| `TELEGRAM_TOKEN` | Yes | Telegram bot token |
| `CHANNEL_ID` | Yes | Target channel (`@name` or numeric ID) |
| `GROQ_API_KEY` | Yes | Groq API key |
| `RSSHUB_URL` | No | Base URL of an RSSHub instance. If set, Twitter/X feeds for `TWITTER_ACCOUNTS` are added to `RSS_FEEDS` |
| `STATE_DIR` | No | Where `state.json` and `seen_headlines.json` live (default `/app` if writable, else `./.state`) |
| `ENABLE_CIRCUS` | No | Circus creator block on Monday. On unless `0`/`false`/`off`/`no` |
| `ENABLE_FAIRGAMBLING` | No | FairGambling market & trust block on Thursday. Same rule |
| `ENABLE_EVENT_COUNTDOWNS` | No | Registers the two event countdown jobs. Same rule |

Never commit values for these variables.

## Known issues / TODOs

- **No tests, CI, or linting.** Every push to `main` deploys to production without any checks.
- **State is probably ephemeral** on Railway unless a volume is mounted at `STATE_DIR` (see Deployment).
- **Circus / FairGambling are unofficial APIs**: robots.txt disallows /api/ on both and their terms restrict reuse. Toggle them off with `ENABLE_CIRCUS=0` / `ENABLE_FAIRGAMBLING=0` if asked; field names can change without notice (blocks then disappear silently).
- **Tanzanite is flaky** (502/521/522 seen). Posts degrade to a "could not reach" line; the Monday chart is skipped if the volume endpoint is down.
- **Failures are silent.** Every error is caught and logged, and there is no alerting. A failed scheduled post is simply skipped (APScheduler `misfire_grace_time` is 600s).
- **Polls in a channel are anonymous**, so nobody's vote is visible; the discussion happens in the linked group's comments. The bot never reads or posts in the discussion group.
- **Friday copy** rotates through 5 variants by ISO week. Hosting (a person replying in the comments) is not automated.
- **Test lines in `main()`** must be commented out by hand before pushing. Forgetting causes an extra post on every deploy.
- **`BIG_EVENTS` needs yearly upkeep.** Past events are ignored automatically.
- **Dependencies are unpinned** (`>=` only), and no Python version is pinned. Builds aren't reproducible.
- `TANZANITE_DISTRIBUTION_API` is defined after `fetch_casino_labels`, the function that uses it. Works, but is easy to break when reordering code.
- `CASINO_NAMES` needs manual upkeep. Without API labels, unknown slugs fall back to title-case (e.g. "Rainbetcom").
