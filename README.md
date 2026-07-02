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

## Quick start

1. Copy `.env.example` to `.env` and fill in `TELEGRAM_TOKEN` (get one from
   [@BotFather](https://t.me/BotFather)).
2. Choose an LLM backend (see below) and set `LLM_BASE_URL` / `LLM_API_KEY` /
   `LLM_MODEL` accordingly.
3. Optionally edit `system_prompt.txt` to give the bot its own persona — it's
   reloaded live, no restart needed.
4. Run it:
   ```
   docker compose up -d --build
   ```

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

## Optional: web search & diagrams

- Web search needs a [SearXNG](https://docs.searxng.org/) instance — set
  `SEARXNG_URL`.
- Diagram generation needs a [Kroki](https://kroki.io/) instance — set
  `KROKI_URL`. You can self-host both with Docker.

If neither is configured, the bot still works — those specific tools will
just fail gracefully when invoked.

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
