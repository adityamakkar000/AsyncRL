# copy this file into ~/.bashrc or ~/.zshrc and fill out placeholder

TPU_SERVER="100.97.202.27"
TPU_SERVER_URL="http://${TPU_SERVER}:8000"

server_ls() {
  command -v jq >/dev/null 2>&1 || {
    echo "server_ls: need jq installed for table output (brew install jq)" >&2
    return 1
  }
  local json max_msg=60
  json=$(curl -sS -f --max-time 10 "${TPU_SERVER_URL%/}/jobs") || return
  [ -t 1 ] || max_msg=100000
  echo "$json" | jq -r --argjson now "$(date +%s)" --argjson max "$max_msg" '
    def dur: ($now - .) | floor | if . >= 86400 then "\(./86400|floor)d\((.%86400)/3600|floor)h"
      elif . >= 3600 then "\(./3600|floor)h\((.%3600)/60|floor)m" elif . >= 60 then "\(./60|floor)m" else "\(.)s" end;
    ["NODE_ID", "STATE", "FOR", "TPU", "TYPE", "ZONE", "ATTEMPTS", "PREEMPT", "RETRIES", "BY", "LATEST"],
    (.[] | [
      .node_id,
      .state,
      (.state_since | dur),
      .tpu_status,
      .tpu_type,
      .zone,
      (.attempts | tostring),
      (.preemptions | tostring),
      (.retries_left | tostring),
      (.launched_by // ""),
      (.message | gsub("[\t\n]"; " ") | if length > $max then .[0:$max] + "…" else . end)
    ])
    | @tsv
  ' | column -t -s $'\t'
  echo "dashboard: ${TPU_SERVER_URL%/}/"
}

server_job() {
  local node_id="$1"
  if [[ -z "$node_id" ]]; then
    echo "usage: server_job <node_id>" >&2
    return 1
  fi
  local enc
  enc=$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$node_id")
  curl -sS -f --max-time 10 "${TPU_SERVER_URL%/}/jobs/${enc}" | jq -r '
    "\(.node_id)  \(.state)  tpu=\(.tpu_status)  \(.tpu_type) \(.zone)  attempts=\(.attempts) preempted=\(.preemptions) retries_left=\(.retries_left)",
    "cmd: \(.cmd)",
    "logs: \(.log_dir)",
    "",
    (.events[] | "\(.[0] | localtime | strftime("%m-%d %H:%M:%S"))  \(.[1] | .[0:15] + (" " * (15 - length)))  \(.[2])\(if (.[3] // 1) > 1 then " (x\(.[3]))" else "" end)")
  '
}

server_delete() {
  local force=false
  if [[ "$1" == "--force" ]]; then force=true; shift; fi
  local node_id="$1"
  if [[ -z "$node_id" ]]; then
    echo "usage: server_delete [--force] <node_id>" >&2
    return 1
  fi
  local enc
  enc=$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$node_id")
  curl -sS -f -X DELETE "${TPU_SERVER_URL%/}/jobs/${enc}?force=${force}" | jq -r '"\(.node_id): \(.action)"'
}

server_logs() {
  local node_id="$1"
  if [[ -z "$node_id" ]]; then
    echo "usage: server_logs <node_id>" >&2
    return 1
  fi
  local enc
  enc=$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$node_id")
  curl -sS -N -f "${TPU_SERVER_URL%/}/logs/${enc}/stream"
}

server_ranks() {
  local node_id="$1"
  if [[ -z "$node_id" ]]; then
    echo "usage: server_ranks <node_id>" >&2
    return 1
  fi
  local enc
  enc=$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$node_id")
  curl -sS -f "${TPU_SERVER_URL%/}/logs/${enc}/ranks"
  echo
}

server_rank_logs() {
  local node_id="$1" index="$2"
  if [[ -z "$node_id" || -z "$index" ]]; then
    echo "usage: server_rank_logs <node_id> <rank_index>" >&2
    return 1
  fi
  local enc
  enc=$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$node_id")
  curl -sS -N -f "${TPU_SERVER_URL%/}/logs/${enc}/ranks/${index}/stream"
}