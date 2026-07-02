#!/usr/bin/env bash
# One-command installer/bootstrapper for Telegram Bot Kit.
#
# Usage (from the cloned repo directory):
#   ./install.sh
#
# What it does:
#   1. Installs Docker Engine + Compose plugin if not already present
#      (Linux servers, via the official get.docker.com script).
#   2. Creates .env from .env.example if it doesn't exist yet.
#   3. If .env is fully configured, brings the whole stack up.
#      Otherwise it stops and tells you what to fill in, then re-run.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f docker-compose.yml ]; then
  echo "ERROR: run this from the telegram-bot-kit repo root (docker-compose.yml not found here)."
  exit 1
fi

echo "=== Telegram Bot Kit installer ==="
echo

# --- 1. Docker Engine ---
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not found. Installing via the official get.docker.com script"
  echo "(this may ask for your sudo password)..."
  curl -fsSL https://get.docker.com | sh
else
  echo "Docker found: $(docker --version)"
fi

# --- 2. Figure out how to invoke docker (may need sudo right after a fresh
#        install, until you re-login into the 'docker' group). ---
DOCKER="docker"
if ! docker info >/dev/null 2>&1; then
  if sudo docker info >/dev/null 2>&1; then
    DOCKER="sudo docker"
    echo "Using 'sudo docker' for this run. To use docker without sudo going"
    echo "forward: sudo usermod -aG docker \$USER, then log out and back in."
  else
    echo "ERROR: can't talk to the Docker daemon. Is it running?"
    echo "Try: sudo systemctl start docker"
    exit 1
  fi
fi

# --- 3. Compose plugin ---
if ! $DOCKER compose version >/dev/null 2>&1; then
  echo "ERROR: 'docker compose' (v2 plugin) isn't available even after installing Docker."
  echo "See: https://docs.docker.com/compose/install/"
  exit 1
fi
echo "Docker Compose found: $($DOCKER compose version)"
echo

# --- 4. .env setup ---
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example."
else
  echo ".env already exists — leaving it as-is."
fi

# --- 5. Check the one thing that truly can't be automated: your bot token. ---
if ! grep -qE '^TELEGRAM_TOKEN=\S+' .env; then
  cat <<'EOF'

Almost there — one manual step left:

  1. Get a bot token from @BotFather on Telegram.
  2. Edit .env and set TELEGRAM_TOKEN=<your token>.
  3. Check LLM_BASE_URL / LLM_API_KEY / LLM_MODEL in .env. If you don't have
     an LLM endpoint yet, use the bundled Ollama option instead:
       docker compose -f docker-compose.yml -f docker-compose.ollama.yml up -d --build
     then set LLM_BASE_URL=http://ollama:11434/v1 in .env and pull a model:
       docker compose exec ollama ollama pull llama3.1
       docker compose exec ollama ollama pull nomic-embed-text
  4. Re-run ./install.sh (or just: docker compose up -d --build)

EOF
  exit 0
fi

# --- 6. Bring it up. ---
echo "Starting the bot (this also builds the image and starts SearXNG/Kroki)..."
$DOCKER compose up -d --build
echo
echo "Done. Tail logs with: docker compose logs -f bot"
