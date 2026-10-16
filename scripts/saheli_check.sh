#!/bin/bash
# Register kavach-saheli-check from the image just built. Cron starts it after the dream.
# This does not run the task and does not gate the deploy.
set -euo pipefail
: "${IMAGE:?}"
: "${AWS_REGION:?}"
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
            | .logConfiguration.options["awslogs-stream-prefix"] = "saheli-night"
            | setenv("GCP_PROJECT_ID"; "kavach-eval-1015")
            | setenv("GOOGLE_CLOUD_PROJECT"; "kavach-eval-1015")
            | setenv("GOOGLE_CLOUD_LOCATION"; "asia-south1")
            | setenv("REPLAY_DATABASE_URL"; "postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/kawach_replay")
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
  > /tmp/saheli-night.json
REV=$(aws ecs register-task-definition --cli-input-json file:///tmp/saheli-night.json --query taskDefinition.revision --output text)
echo "registered kavach-saheli-check:$REV"
