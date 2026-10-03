#!/bin/bash
# Saheli Lab helper for the bots.   labctl next <group> [limit] [role] | answer <file.json> | status
# Reads LAB_TOKEN from $SAHELI_LAB_DIR/lab.env (written when the bridge is set up).
set -euo pipefail
LAB_DIR="${SAHELI_LAB_DIR:-/home/m4dm4x/OpenBot/Shared/saheli-lab}"
[ -f "$LAB_DIR/lab.env" ] && . "$LAB_DIR/lab.env"
URL="${LAB_URL:-http://127.0.0.1:8787}"
H="X-Lab-Token: ${LAB_TOKEN:?no LAB_TOKEN in $LAB_DIR/lab.env}"
PLAYER="${LAB_PLAYER:-$(whoami)}"
case "${1:-}" in
  next)   curl -sf -H "$H" "$URL/lab/next?group=${2:?group}&player=$(printf %s "$PLAYER" | jq -sRr @uri)&limit=${3:-10}${4:+&role=$4}" ;;
  answer) jq --arg p "$PLAYER" '{player: $p, answers: .}' "${2:?answers.json}" | curl -sf -H "$H" -H 'Content-Type: application/json' -d @- "$URL/lab/answer" ;;
  status) curl -sf -H "$H" "$URL/lab/status" ;;
  *) echo "usage: labctl next <north|south_east|english|judge> [limit] [sim|judge] | answer <answers.json> | status"; exit 2 ;;
esac
