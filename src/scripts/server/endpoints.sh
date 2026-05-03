# copy this file into ~/.bashrc or ~/.zshrc and fill out placeholder

server_ls() {
  TPU_SERVER_URL='http://100.79.104.73:8000'
  command -v jq >/dev/null 2>&1 || {
    echo "server_jobs: need jq installed for table output (brew install jq)" >&2
    return 1
  }
  local json
  json=$(curl -sS -f "${TPU_SERVER_URL%/}/jobs") || return

  if [ -t 1 ]; then
    local max_cmd_len=30
    echo "$json" | jq -r --arg max_len "$max_cmd_len" '
      ["NODE_ID", "TPU_STATUS", "JOB_STATUS", "ZONE", "TPU_TYPE", "RUNTIME", "CMD", "RETRIES"],
      (.[] | [
        .node_id,
        .tpu_status,
        .job_status,
        .zone,
        .tpu_type,
        .runtime,
        (.cmd | gsub("\t"; " ") | gsub("\n"; " ") | if (length > ($max_len|tonumber)) then .[0:($max_len|tonumber)] + "…" else . end),
        (.retries_left | tostring)
      ])
      | @tsv
    ' | column -t -s $'\t'
  else
    echo "$json" | jq -r '
      ["NODE_ID", "TPU_STATUS", "JOB_STATUS", "ZONE", "TPU_TYPE", "RUNTIME", "CMD", "RETRIES"],
      (.[] | [
        .node_id,
        .tpu_status,
        .job_status,
        .zone,
        .tpu_type,
        .runtime,
        (.cmd | gsub("\t"; " ") | gsub("\n"; " ")),
        (.retries_left | tostring)
      ])
      | @tsv
    ' | column -t -s $'\t'
  fi
}

server_delete() {
  TPU_SERVER_URL='http://100.79.104.73:8000'
  local node_id="$1"
  if [[ -z "$node_id" ]]; then
    echo "usage: server_delete <node_id>" >&2
    return 1
  fi
  local enc
  enc=$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$node_id")
  curl -sS -f -X DELETE "${TPU_SERVER_URL%/}/jobs/${enc}"
  echo
}

server_logs() {
  local node_id="$1"
  TPU_SERVER_URL="http://100.79.104.73:8000"
  if [[ -z "$node_id" ]]; then
    echo "usage: server_stream <node_id>" >&2
    return 1
  fi
  local enc
  enc=$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$node_id")
  curl -sS -N -f "${TPU_SERVER_URL%/}/logs/${enc}/stream"
}