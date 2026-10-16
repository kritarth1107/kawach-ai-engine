#!/bin/bash
# Register kavach-saheli-check from the image just built, run the pre-deploy replay, and leave
# that revision only when it passes. The night command is the default, so cron can start it later.
# A founder override (OVERRIDE=true) skips the run. Never prints secrets.
set -euo pipefail
: "${IMAGE:?}"
: "${AWS_REGION:?}"
if [ "${OVERRIDE:-false}" = "true" ]; then
  echo "founder override: the Saheli check was skipped"
  exit 0
fi
aws ecs describe-task-definition --task-definition kavach-engine --query taskDefinition \
  | jq --arg IMG "$IMAGE" --arg NAME "kawach-ai-engine" '
      def setenv($n; $v):
        .environment = ((.environment // [])
          | map(if .name == $n then .value = $v else . end)
          | if any(.[]; .name == $n) then . else . + [{name:$n, value:$v}] end);
      .family = "kavach-saheli-check"
      | .cpu = "1024"
      | .memory = "4096"
      | .containerDefinitions |= (
          map(if ((.image | split("/") | last | split(":")[0]) == $NAME) or .name == "engine" then
            .image = $IMG
            | .name = "engine"
            | .command = ["python", "-m", "app.jobs.saheli_night"]
            | .essential = true
            | .logConfiguration.options["awslogs-stream-prefix"] = "saheli-check"
            | setenv("GCP_PROJECT_ID"; "kavach-eval-1015")
            | setenv("GOOGLE_CLOUD_PROJECT"; "kavach-eval-1015")
            | setenv("GOOGLE_CLOUD_LOCATION"; "asia-south1")
            | setenv("REPLAY_DATABASE_URL"; "postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/kawach_replay")
            | setenv("REPLAY_MAX_INR"; "20")
            | setenv("PROBES_MAX_INR"; "60")
            | setenv("JUDGE_MAX_INR"; "40")
            | setenv("RECALL_MAX_INR"; "20")
            | setenv("NUDGE_EVAL_MAX_INR"; "20")
            | setenv("POINTS_MAX_INR"; "10")
            | setenv("NIGHT_CAP_INR"; "150")
            | setenv("WEB_SEARCH_PROVIDER"; "off")
            | setenv("MEMORY_EMBEDDINGS"; "off")
            | setenv("EMBED_DIM_MIGRATE"; "off")
            | setenv("ENGINE_URL"; "http://engine.kavach.local:8000")
          else . end)
          + [{name:"replay-db", image:"pgvector/pgvector:pg16", essential:false,
              environment:[{name:"POSTGRES_USER",value:"postgres"},{name:"POSTGRES_PASSWORD",value:"postgres"},{name:"POSTGRES_DB",value:"kawach_replay"}],
              portMappings:[{containerPort:5432,protocol:"tcp"}]}])
      | del(.taskDefinitionArn, .revision, .status, .requiresAttributes, .compatibilities, .registeredAt, .registeredBy)' \
  > /tmp/saheli-check.json
REV=$(aws ecs register-task-definition --cli-input-json file:///tmp/saheli-check.json --query taskDefinition.revision --output text)
echo "registered kavach-saheli-check:$REV"
NET=$(aws ecs describe-services --cluster kavach --services engine --query 'services[0].networkConfiguration.awsvpcConfiguration' --output json)
SUBNETS=$(echo "$NET" | jq -r '.subnets | join(",")')
SGS=$(echo "$NET" | jq -r '.securityGroups | join(",")')
PUB=$(echo "$NET" | jq -r '.assignPublicIp')
OV='{"containerOverrides":[{"name":"engine","command":["python","-m","app.jobs.deploy_check"]}]}'
set +e
TASK=$(aws ecs run-task --cluster kavach --launch-type FARGATE --task-definition "kavach-saheli-check:$REV" \
  --started-by deploy-check --overrides "$OV" \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SGS],assignPublicIp=$PUB}" \
  --query 'tasks[0].taskArn' --output text)
RUN_CODE=$?
set -e
if [ "$RUN_CODE" != 0 ] || [ -z "$TASK" ] || [ "$TASK" = "None" ]; then
  echo "the Saheli check task did not start"
  aws ecs deregister-task-definition --task-definition "kavach-saheli-check:$REV" >/dev/null
  exit 1
fi
echo "check task $TASK"
aws ecs wait tasks-stopped --cluster kavach --tasks "$TASK"
CODE=$(aws ecs describe-tasks --cluster kavach --tasks "$TASK" --query "tasks[0].containers[?name=='engine'].exitCode | [0]" --output text)
echo "check exit $CODE"
if [ "$CODE" = "0" ]; then
  echo "Saheli check passed; the night task stays on kavach-saheli-check:$REV"
  exit 0
fi
echo "Saheli check failed; the live image stays"
TID=${TASK##*/}
aws logs tail /kavach/engine --log-stream-names "saheli-check/engine/$TID" --since 1h --format short 2>/dev/null \
  | grep -E 'DEPLOY CHECK|BLOCKING|refusing|FAILED|Error|error' | tail -40 > /tmp/saheli-fail.txt || true
python3 - << 'PY'
import os
from pathlib import Path
import importlib.util
text = Path("/tmp/saheli-fail.txt").read_text() if Path("/tmp/saheli-fail.txt").exists() else ""
body = "The pre-deploy Saheli check failed. The live engine image was not changed.\n\n" + (text or "No failure lines in the log.")
spec = importlib.util.spec_from_file_location("post_eval_scores", "scripts/post_eval_scores.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
if os.environ.get("RESEND_API_KEY"):
    mod.email("Saheli check failed", body)
else:
    print("no RESEND_API_KEY; the failure is in the job log")
PY
aws ecs deregister-task-definition --task-definition "kavach-saheli-check:$REV" >/dev/null
exit 1
