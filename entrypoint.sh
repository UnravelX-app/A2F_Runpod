#!/usr/bin/env bash
set -Eeuo pipefail

log() {
  printf '[a2f-runpod] %s\n' "$*" >&2
}

log "build=${A2F_WRAPPER_BUILD:-unknown} port=${PORT:-8080}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L >&2 || true
fi

wait_for_a2f_ready() {
  local timeout="${A2F_READY_TIMEOUT_SEC:-3600}"
  local deadline=$((SECONDS + timeout))
  local http_url="http://127.0.0.1:8000/v1/health/ready"
  local grpc_host="127.0.0.1"
  local grpc_port="52000"

  log "waiting for A2F: http=${http_url} grpc=${grpc_host}:${grpc_port} timeout=${timeout}s"
  while (( SECONDS < deadline )); do
    if python3 - "$http_url" "$grpc_host" "$grpc_port" <<'PYREADY' >/dev/null 2>&1
import socket, sys, urllib.request
url, host, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
with urllib.request.urlopen(url, timeout=2) as r:
    if r.status < 200 or r.status >= 300: raise SystemExit(1)
with socket.create_connection((host, port), timeout=2): pass
PYREADY
    then
      log "A2F ready"
      return 0
    fi
    sleep "${A2F_READY_POLL_SEC:-5}"
  done
  log "timed out waiting for A2F"
  exit 70
}

start_api() {
  wait_for_a2f_ready
  log "starting uvicorn on port ${PORT:-8080}"
  exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"
}

start_api &

if [[ -n "${SERVER_START_SCRIPT_PATH:-}" ]]; then
  log "starting A2F via SERVER_START_SCRIPT_PATH=${SERVER_START_SCRIPT_PATH}"
  exec /bin/bash -c "$SERVER_START_SCRIPT_PATH"
elif [[ "$#" -gt 0 ]]; then
  exec "$@"
else
  log "ERROR: no A2F startup command. Set SERVER_START_SCRIPT_PATH or pass CMD args."
  log "Find the NIM entrypoint with: docker inspect nvcr.io/nim/nvidia/audio2face-3d:1.3.16 --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}}'"
  exit 64
fi
