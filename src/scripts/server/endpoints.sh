export TPU_SERVER_URL='url here'

server_jobs() {
  command -v jq >/dev/null 2>&1 || {
    echo "server_jobs: need jq installed for table output (brew install jq)" >&2
    return 1
  }
  local json
  json=$(curl -sS -f "${TPU_SERVER_URL%/}/jobs") || return
  echo "$json" | jq -r '
    ["NODE_ID", "ZONE", "TPU_TYPE", "RUNTIME", "CMD", "RETRIES", "TPU_STATUS", "JOB_STATUS"],
    (.[] | [
      .node_id,
      .zone,
      .tpu_type,
      .runtime,
      (.cmd | gsub("\t"; " ") | gsub("\n"; " ")),
      (.retries_left | tostring),
      .tpu_status,
      .job_status
    ])
    | @tsv
  ' | column -t -s $'\t'
}

server_delete() {
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