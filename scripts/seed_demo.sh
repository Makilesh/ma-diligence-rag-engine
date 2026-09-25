#!/usr/bin/env bash
# Indexes the sample data room (data/sample_deal/*.txt) into the demo deal on a
# running backend. Run from the repository root on the server:
#
#   bash scripts/seed_demo.sh                    # https://$DOMAIN from .env
#   bash scripts/seed_demo.sh https://host:port  # any other base URL
#
# Writing to the demo deal is an admin operation, so the X-Admin-Key header is
# read from .env. Re-running is safe: document and point IDs are derived from
# content, so identical files replace themselves instead of duplicating.
#
# CURL_INSECURE=1 skips TLS verification — only for a local test stack whose
# certificate comes from Caddy's internal CA (DOMAIN=localhost).

set -euo pipefail
cd "$(dirname "$0")/.."

env_value() { grep -E "^$1=" .env | tail -1 | cut -d= -f2- || true; }

BASE_URL="${1:-https://$(env_value DOMAIN)}"
BASE_URL="${BASE_URL%/}"
ADMIN_API_KEY="${ADMIN_API_KEY:-$(env_value ADMIN_API_KEY)}"
DEAL_ID="aurora_vertex_2024"
[ -n "$ADMIN_API_KEY" ] || { echo "ADMIN_API_KEY is not set in .env" >&2; exit 1; }

curl_opts=(-sS --fail-with-body --max-time 600 -H "X-Admin-Key: $ADMIN_API_KEY")
[ "${CURL_INSECURE:-0}" = "1" ] && curl_opts+=(-k)

# The same category pins run_demo.py uses, so the demo corpus is categorised the
# same way everywhere.
category_for() {
  case "$1" in
    board_deck_strategic_review_mar2024.txt) echo board ;;
    regulatory_and_data_privacy_memo.txt)    echo regulatory ;;
    employment_and_retention_agreements.txt) echo legal ;;
    *) echo "" ;;
  esac
}

failed=0
for f in data/sample_deal/*.txt; do
  name="$(basename "$f")"
  form=(-F "deal_id=$DEAL_ID" -F "is_current_version=true" -F "file=@$f")
  cat="$(category_for "$name")"
  [ -n "$cat" ] && form+=(-F "document_category=$cat")
  if out="$(curl "${curl_opts[@]}" "${form[@]}" "$BASE_URL/api/v1/ingest")"; then
    echo "  ok   $name  $(printf '%s' "$out" | grep -o '"chunks_created":[0-9]*' || true)"
  else
    echo "  FAIL $name  $out" >&2
    failed=1
  fi
done

echo
curl "${curl_opts[@]}" "$BASE_URL/api/v1/deals"; echo
exit "$failed"
