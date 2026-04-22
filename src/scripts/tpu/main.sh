
# delete a queued TPU resource
tpuq_rm() {
  local ID="$1"
  local ZONE="${2:-$GCLOUD_TPU_ZONE}"

  if [[ -z "$ID" ]]; then
    echo "Usage: tpuq_rm <id> [zone]"
    return 1
  fi

  gcloud compute tpus queued-resources delete "$ID" \
    --project "$GCLOUD_TPU_PROJECT" \
    --zone "$ZONE" \
    --force
}


# list queued TPU resources
tpuq_ls() {
  local ZONE="${1:-$GCLOUD_TPU_ZONE}"

  gcloud compute tpus queued-resources list \
    --project "$GCLOUD_TPU_PROJECT" \
    --zone "$ZONE"
}

# create a queued TPU resource
tpuq_create() {
  local ID="$1"
  local TPU_TYPE="$2"
  local RUNTIME="$3"
  local SPOT="$4"
  local ZONE="${5:-$GCLOUD_TPU_ZONE}"

  if [[ -z "$ID" || -z "$TPU_TYPE" ]]; then
    echo "Usage: tpuq_create <id> <tpu-type> [runtime] [spot] [zone]"
    return 1
  fi

  if [[ -z "$RUNTIME" || "$RUNTIME" == "spot" ]]; then
    RUNTIME="$GCLOUD_TPU_RUNTIME"
    SPOT="$3"
    ZONE="${4:-$GCLOUD_TPU_ZONE}"
  fi

  local SPOT_FLAG=""
  if [[ "$SPOT" == "spot" ]]; then
    SPOT_FLAG="--spot"
  fi

  gcloud compute tpus queued-resources create "$ID" \
    --node-id "$ID" \
    --project "$GCLOUD_TPU_PROJECT" \
    --zone "$ZONE" \
    --accelerator-type "$TPU_TYPE" \
    --runtime-version "$RUNTIME" \
    $SPOT_FLAG
}

# describe a queued TPU resource
tpuq_describe() {
  local ID="$1"
  local ZONE="${2:-$GCLOUD_TPU_ZONE}"

  if [[ -z "$ID" ]]; then
    echo "Usage: tpuq_describe <id> [zone]"
    return 1
  fi

  gcloud compute tpus queued-resources describe "$ID" \
    --project "$GCLOUD_TPU_PROJECT" \
    --zone "$ZONE"
}

# list TPU VMs in a zone
tpu_ls () {
  if [ -z "$1" ]; then
    echo "usage: tpu_ls <zone>"
    return 1
  fi

  local zone="$1"

  gcloud compute tpus tpu-vm list \
    --zone="$zone" \
    --format="table(
      name,
      acceleratorType,
      state,
      networkEndpoints[].accessConfig.externalIp
    )"
}

tpu_ls_all() {
  for zone in us-central1-a us-central1-b us-central1-c us-central1-f us-central2-b us-central2-c us-central2-d us-east1-c us-east1-d us-east4-a us-east4-b us-east4-c us-east5-b us-east5-c us-south1-a us-west1-a us-west1-b us-west1-c us-west4-a us-west4-b europe-west4-a europe-west4-b europe-west4-c; do
    tpus=$(gcloud compute tpus tpu-vm list --zone="$zone" --project="$GCLOUD_TPU_PROJECT" --format="table(name,acceleratorType,state)" 2>/dev/null)
    if [[ -n "$tpus" && "$tpus" != "Listed 0 items." ]]; then
      echo "--- Zone: $zone ---"
      echo "$tpus"
    fi
  done
}