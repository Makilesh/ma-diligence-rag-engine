#!/usr/bin/env bash
# One-shot setup of the public backend on a fresh Ubuntu 22.04/24.04 VM
# (e.g. Oracle Cloud Always Free ARM). Run from the repository root:
#
#   git clone https://github.com/Makilesh/redline-diligence.git && cd redline-diligence
#   bash scripts/server_setup.sh
#
# Safe to re-run: every step checks before it acts. It stops, with the exact
# fix, at the first thing that would otherwise fail silently later — an unset
# secret, or a DOMAIN that does not point at this machine (Caddy then cannot
# get a certificate and the site is unreachable with no obvious error).
#
# What it does not do: the cloud firewall. On Oracle, open TCP 80 and 443 in the
# VCN's Security List yourself (DEPLOYMENT.md step 2); this script opens them on
# the host.

set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.prod.yml)
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mSTOP:\033[0m %s\n' "$*" >&2; exit 1; }
env_value() { grep -E "^$1=" .env | tail -1 | cut -d= -f2- || true; }

# --- 1. Docker ---------------------------------------------------------------
say "Docker"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
fi
# The docker group applies from the next login; use sudo until then.
if ! docker info >/dev/null 2>&1; then
  COMPOSE=(sudo "${COMPOSE[@]}")
fi
"${COMPOSE[@]}" version >/dev/null || fail "docker compose is not available."

# --- 2. Host firewall ----------------------------------------------------------
say "Host firewall: allow 80 and 443"
# Oracle's Ubuntu image REJECTs everything but SSH in iptables. Insert ACCEPT
# rules ahead of that REJECT, once.
for port in 80 443; do
  if ! sudo iptables -C INPUT -p tcp --dport "$port" -m state --state NEW -j ACCEPT 2>/dev/null; then
    sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport "$port" -j ACCEPT
  fi
done
if command -v netfilter-persistent >/dev/null 2>&1; then
  sudo netfilter-persistent save >/dev/null
else
  echo "   (netfilter-persistent not installed — rules last until reboot;"
  echo "    'sudo apt-get install -y iptables-persistent' makes them permanent)"
fi

# --- 3. .env -----------------------------------------------------------------
say "Configuration (.env)"
if [ ! -f .env ]; then
  cp .env.deploy.example .env
  echo "   created .env from .env.deploy.example"
fi
gen_secret() { python3 -c "import secrets; print(secrets.token_urlsafe(32))"; }
for name in POSTGRES_PASSWORD ADMIN_API_KEY; do
  if [ -z "$(env_value "$name")" ]; then
    sed -i "s|^$name=.*|$name=$(gen_secret)|" .env
    echo "   generated $name (stored in .env)"
  fi
done
missing=()
for name in GEMINI_API_KEYS DOMAIN CORS_ORIGINS; do
  [ -n "$(env_value "$name")" ] || missing+=("$name")
done
[ ${#missing[@]} -eq 0 ] || fail "set ${missing[*]} in .env (nano .env), then re-run this script."

DOMAIN="$(env_value DOMAIN)"

# --- 4. DNS points here? -------------------------------------------------------
say "DNS: does $DOMAIN resolve to this machine?"
public_ip="$(curl -fsS -4 --max-time 10 https://checkip.amazonaws.com | tr -d '[:space:]' || true)"
resolved="$(getent ahostsv4 "$DOMAIN" | awk 'NR==1 {print $1}' || true)"
echo "   this machine: ${public_ip:-unknown}   $DOMAIN: ${resolved:-does not resolve}"
if [ -z "$public_ip" ] || [ "$resolved" != "$public_ip" ]; then
  fail "$DOMAIN must point at ${public_ip:-the public IP of this machine}. Update it at duckdns.org, wait a minute, re-run."
fi

# --- 5. Build and start -------------------------------------------------------
say "Building and starting (first build: 20–30 min on 2 ARM cores)"
"${COMPOSE[@]}" up -d --build

# --- 6. Wait until it answers over HTTPS ---------------------------------------
say "Waiting for https://$DOMAIN/health (model loading takes a few minutes)"
for _ in $(seq 1 120); do
  if curl -fsS --max-time 10 "https://$DOMAIN/health" >/dev/null 2>&1; then
    echo "   up."
    echo
    echo "Next:"
    echo "  bash scripts/seed_demo.sh                 # index the sample data room"
    echo "  Vercel: NEXT_PUBLIC_API_URL=https://$DOMAIN, then Redeploy"
    exit 0
  fi
  sleep 15
done
fail "no answer after 30 min. Look at: ${COMPOSE[*]} logs --tail 100 api caddy"
