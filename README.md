# Telegram Bot Kit

A generic, self-hosted Telegram AI assistant bot. Give it a bot token, an
LLM, and (optionally) a persona — it runs standalone with no per-company
features baked in.

## Features

- Conversational memory per chat (recent history + rolling summary + long-term
  vector recall).
- Automatic web search via tool-calling (the model decides when to search).
- Web image search, diagram generation (Graphviz/Mermaid via Kroki), document
  generation (txt/docx/pptx/pdf), image editing (crop/resize/rotate/text).
- Vision: understands photos sent directly in chat.
- Per-chat document upload & Q&A: upload a PDF/Word/Excel/PowerPoint/txt/image
  and ask questions about it.
- Network diagnostics tool: ping, DNS, whois, subnet math.
- Works with any OpenAI-compatible LLM endpoint — Ollama, OpenAI, Groq,
  OpenRouter, etc.

**Not included:** ticketing/task tracking, curated knowledge-base RAG, and
scheduled reminders. This kit is intentionally a lean, generic assistant —
fork it if you need those.

## Before you start

You need two things decided before running `install.sh`:

1. **A Telegram bot token** — get one from [@BotFather](https://t.me/BotFather).
2. **An LLM backend** — either your own OpenAI-compatible endpoint (OpenAI,
   Groq, OpenRouter, a remote Ollama server, etc.), or the bundled Ollama
   option if you don't have one yet. Read [LLM backend options](#llm-backend-options)
   below and pick one *before* you run the final `docker compose up` step —
   the bot won't work without a valid `LLM_BASE_URL`.

Everything else (SearXNG, Kroki, database) is bundled and needs no decision.

## Quick start

```
git clone <this-repo-url>
cd telegram-bot-kit
./install.sh
```

`install.sh` installs Docker (if it's not already on the machine — this is
meant to run on a fresh server, like an XAMPP-style installer), copies
`.env.example` to `.env`, then stops and tells you the one thing that truly
can't be automated: **get a bot token from [@BotFather](https://t.me/BotFather)
and put it in `.env`**. Re-run `./install.sh` (or `docker compose up -d --build`)
once that's done and it brings everything up — the bot, plus bundled SearXNG
(web search) and Kroki+Mermaid (diagram rendering), no separate installs needed.

Optionally edit `system_prompt.txt` to give the bot its own persona — it's
reloaded live, no restart needed.

Prefer to do it by hand instead of running the script? Same steps, manually:
copy `.env.example` to `.env`, fill in `TELEGRAM_TOKEN` and an LLM backend,
then `docker compose up -d --build`.

## LLM backend options

**Option A — bring your own endpoint (default).** Point `LLM_BASE_URL` at any
OpenAI-compatible `/v1` endpoint you already have running: a remote Ollama
server (`http://your-host:11434/v1`), OpenAI, Groq, OpenRouter, etc. Set
`LLM_API_KEY` and `LLM_MODEL` to match.

**Option B — bundle a local Ollama.** If you don't have an LLM endpoint yet,
use the optional overlay to run one alongside the bot:
```
docker compose -f docker-compose.yml -f docker-compose.ollama.yml up -d --build
docker compose exec ollama ollama pull llama3.1
docker compose exec ollama ollama pull nomic-embed-text
```
Then set in `.env`:
```
LLM_BASE_URL=http://ollama:11434/v1
EMBED_BASE_URL=http://ollama:11434/v1
```

Embeddings (used for long-term memory and document Q&A) are configured
separately from chat (`EMBED_BASE_URL`/`EMBED_API_KEY`/`EMBED_MODEL`), since
not every OpenAI-compatible provider offers an embeddings endpoint. They
default to the `LLM_*` values if left unset.

## Web search & diagrams

`docker compose up` already includes these — [SearXNG](https://docs.searxng.org/)
for web search (config at `searxng/settings.yml`, JSON output enabled since
the bot needs it) and [Kroki](https://kroki.io/) + a Mermaid companion for
diagram rendering. Nothing to install separately.

If you're running `bot.py` directly without Docker (see below), point
`SEARXNG_URL`/`KROKI_URL` in `.env` at instances you run yourself. If neither
is configured, the bot still works — those specific tools just fail
gracefully when invoked.

## Running without Docker

```
python -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python bot.py
```

An example systemd unit is at `telegram-bot-kit.service.example` — copy it, edit
the paths/user, and install it if you want the bot to run as a service.

## Maintenance

`memory_prune.py` deletes vector memories older than `MEMORY_RETENTION_DAYS`
(default 180). Run it periodically via cron:
```
0 3 * * * /path/to/telegram-bot-kit/venv/bin/python /path/to/telegram-bot-kit/memory_prune.py
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
