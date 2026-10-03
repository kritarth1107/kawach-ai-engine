#!/bin/bash
# Saheli training runs, staged and budgeted. Nothing runs without a budget and a "yes".
#
#   eval/train.sh <stage> [families…]
#
# Stages (estimates at ~₹31 per family-day full judge, ~₹12 cheap judge; see journal/2026-10-04_*_training-plan.md):
#   free       unit tests + agent bench + guard replay                       ₹0
#   smoke      3 families × 3 days, cheap judge                              ~₹110
#   regress    10 families × 7 days, aggressive events, cheap judge          ~₹850
#   learning   10 families × 21 days, learning storylines (patterns,         ~₹2,500
#              outcomes, baselines, nightly dream), cheap judge
#   full       10 families × 30 days, aggressive, full Pro judge             ~₹9,500 (only before launch)
#   lessons    learn a playbook from the simulated runs so far (score, corpus,  ~₹50
#              draft, safety gate) — the "teach her how to talk" step
#   lab N D    the Saheli Lab bots play N bot-designed families for D days (families from   Saheli's replies only
#              /home/m4dm4x/OpenBot/Shared/saheli-lab/families); start eval/lab/start_bridge.sh first
#   compare V  regress twice on the same families: without a playbook and with  ~₹1,700
#              playbook V; the report says which replies scored better
#
# Required: SIM_PROJECT (a separate GCP project for training, never the production project) and ADC
# credentials for it. The run stops at SIM_MAX_INR whatever the stage.
set -euo pipefail
cd "$(dirname "$0")/.."
stage="${1:-}"; shift || true
case "$stage" in
  free)
    .venv/bin/python -m pytest -q
    PYTHONPATH=. .venv/bin/python eval/agent_bench.py
    .venv/bin/python eval/replay_guards.py
    exit 0 ;;
  smoke)   days=3;  fams=(sharma gupta das); budget=${SIM_MAX_INR:-200};  export SIM_AGGRESSIVE=0 JUDGE_MODE=cheap JUDGE_SAMPLE=0.5 SIM_EXTRACT_PER_DAY=2 ;;
  regress) days=7;  fams=(); budget=${SIM_MAX_INR:-1200}; export SIM_AGGRESSIVE=1 JUDGE_MODE=cheap JUDGE_SAMPLE=0.4 SIM_EXTRACT_PER_DAY=2 ;;
  learning) days=21; fams=(); budget=${SIM_MAX_INR:-3000}; export SIM_AGGRESSIVE=1 SIM_LEARNING=1 JUDGE_MODE=cheap JUDGE_SAMPLE=0.3 SIM_EXTRACT_PER_DAY=2 ;;
  full)    days=30; fams=(); budget=${SIM_MAX_INR:-10000}; export SIM_AGGRESSIVE=1 SIM_LEARNING=1 JUDGE_MODE=full JUDGE_SAMPLE=1.0 SIM_EXTRACT_PER_DAY=5 ;;
  lab)
    n="${1:-20}"; d="${2:-14}"; shift 2 || true
    days=$d; fams=(); budget=${SIM_MAX_INR:-$(( n * d * 6 ))}
    export SIM_LAB=1 SIM_EXTERNAL_ROLES="${SIM_EXTERNAL_ROLES:-sim,judge}" SIM_EXTERNAL_TIMEOUT="${SIM_EXTERNAL_TIMEOUT:-3600}" \
           SIM_RUN_ID="${SIM_RUN_ID:-lab-$(date +%Y%m%d-%H%M)}" SIM_AGGRESSIVE=0 SIM_LEARNING=1 JUDGE_MODE=full JUDGE_SAMPLE=1.0 SIM_EXTRACT_PER_DAY=2
    keys=$(ls /home/m4dm4x/OpenBot/Shared/saheli-lab/families/*.json 2>/dev/null | head -n "$n" | xargs -n1 basename | sed 's/\.json$//' | tr '\n' ' ')
    read -r -a fams <<< "$keys"
    [ -f /home/m4dm4x/OpenBot/Shared/saheli-lab/lab.env ] || { echo "Start the bridge first: eval/lab/start_bridge.sh"; exit 2; } ;;
  lessons)
    : "${SIM_PROJECT:?set SIM_PROJECT to a separate GCP project for training (not kavach-care)}"
    [ "$SIM_PROJECT" = "kavach-care" ] && { echo "Refusing: training must not run in the production project."; exit 2; }
    read -r -p "Learn a playbook from the simulator database (~₹50 in $SIM_PROJECT)? Type yes: " ok; [ "$ok" = "yes" ] || exit 1
    export PYTHONPATH=. DEBUG=false GCP_PROJECT_ID="$SIM_PROJECT" LEARN_MIN_EXAMPLES="${LEARN_MIN_EXAMPLES:-15}"
    export DATABASE_URL="${SIM_DATABASE_URL:-postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_sim}"
    export MODEL_ROUTES='{"learn": ["gemini:gemini-3.1-pro-preview", "gemini:gemini-3.8-flash"]}'
    exec .venv/bin/python eval/learn_cycle.py ;;
  compare)
    v="${1:?usage: eval/train.sh compare <playbook version>}"; shift
    echo "Two regress runs: LEARN_FORCE_VERSION=0 then $v (same families, same events)."
    LEARN_FORCE_VERSION=0 SIM_MAX_INR="${SIM_MAX_INR:-900}" "$0" regress "$@"
    LEARN_FORCE_VERSION="$v" SIM_MAX_INR="${SIM_MAX_INR:-900}" exec "$0" regress "$@" ;;
  *) echo "usage: eval/train.sh free|smoke|regress|learning|full|lessons|compare V [families…]"; exit 2 ;;
esac
[ $# -gt 0 ] && fams=("$@")
: "${SIM_PROJECT:?set SIM_PROJECT to a separate GCP project for training (not kavach-care)}"
if [ "$SIM_PROJECT" = "kavach-care" ]; then echo "Refusing: training must not run in the production project."; exit 2; fi
n=${#fams[@]}; [ "$n" -eq 0 ] && n=10
echo "Stage $stage: $n families × $days days, judge=$JUDGE_MODE sample=$JUDGE_SAMPLE, budget ₹$budget, project $SIM_PROJECT"
read -r -p "Spend up to ₹$budget on Vertex AI in $SIM_PROJECT? Type yes: " ok
[ "$ok" = "yes" ] || { echo "Not started."; exit 1; }
export PYTHONUNBUFFERED=1 DEBUG=false PYTHONPATH=. GCP_PROJECT_ID="$SIM_PROJECT" SIM_MAX_INR="$budget"
export DATABASE_URL="${SIM_DATABASE_URL:-postgresql+asyncpg://postgres:postgres@localhost:5433/kawach_sim}"
export BRAIN_EFFORT="${BRAIN_EFFORT:-medium}" SIM_VOLUME="${SIM_VOLUME:-1.6}"
export MODEL_ROUTES='{"brain": ["gemini:gemini-3.5-flash@asia-south1", "gemini:gemini-3.8-flash"], "extract": ["gemini:gemini-3.8-flash"], "worker": ["gemini:gemini-3.8-flash"], "classify": ["gemini:gemini-3.8-flash"], "sim": ["gemini:gemini-3.8-flash"], "judge_fast": ["gemini:gemini-3.8-flash"], "judge": ["gemini:gemini-3.1-pro-preview", "gemini:gemini-3.8-flash"]}'
exec .venv/bin/python -u eval/month_sim.py "${fams[@]}" --days "$days" --parallel 5
