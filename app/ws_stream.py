from typing import Optional

async def fetch_nvidia_blendshape_names(function_id: str, apikey: Optional[str]) -> Optional[list]:
    """
    Query the NVIDIA model metadata via DescribeFunction to extract blend shape names.
    Returns a list of blend shape names if available, else None.
    """
    from nvidia_ace.controller import v1_pb2 as ctrl_pb2
    from nvidia_ace.services.a2f_controller import v1_pb2_grpc as a2f_ctrl_grpc
    import grpc
    import os
    NVCF_ENDPOINT = os.environ.get("NVCF_ENDPOINT", "grpc.nvcf.nvidia.com:443")
    metadata = []
    if not apikey:
        apikey = get_nvcf_api_key()
    if apikey:
        metadata.append(("authorization", f"Bearer {apikey}"))
    metadata.append(("function-id", function_id))
    try:
        channel = grpc.aio.secure_channel(NVCF_ENDPOINT, grpc.ssl_channel_credentials())
        stub = a2f_ctrl_grpc.A2FControllerServiceStub(channel)
        # Robust DescribeFunction support across schema variants
        try:
            ReqCls = getattr(ctrl_pb2, 'DescribeFunctionRequest', None)
            if ReqCls is None:
                # Fallback name in some NVIDIA wheels
                ReqCls = getattr(ctrl_pb2, 'FunctionDescriptionRequest', None)
            if ReqCls is None:
                raise AttributeError('No DescribeFunctionRequest in ctrl_pb2')
            req = ReqCls()
            resp = await stub.DescribeFunction(req, metadata=metadata)
        except Exception as e:
            logger.info(f"[DescribeFunction] DescribeFunction unavailable: {e}")
            return None
        # Try to find blend shape names in the response
        # Response may contain .model_metadata.blend_shape_names or similar
        # We'll try several likely locations
        blend_names = []
        meta = getattr(resp, "model_metadata", None)
        if meta:
            # Try .blend_shape_names
            names = getattr(meta, "blend_shape_names", None)
            if names and isinstance(names, (list, tuple)):
                blend_names = list(names)
            elif names:  # protobuf repeated field
                blend_names = list(names)
        # Fallback: check .blendShapeNames (camelCase)
        if not blend_names:
            names = getattr(meta, "blendShapeNames", None) if meta else None
            if names:
                blend_names = list(names)
        await channel.close()
        if blend_names and len(blend_names) > 0:
            return blend_names
    except Exception as e:
        logger.info(f"[DescribeFunction] Could not fetch blend shape names: {e}")
    return None
# app/ws_stream.py
import asyncio
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import boto3
from botocore.exceptions import ClientError
from typing import Any, AsyncIterator, Optional, Dict, List, Set
import contextlib
from grpc import aio as grpc_aio
import logging
import sys
# --- Force a visible logger for this module ---
from google.protobuf.json_format import MessageToDict
from google.protobuf import descriptor_pool as _desc_pool
logger = logging.getLogger("a2f_stream")
if not logger.handlers:
    _h = logging.StreamHandler(stream=sys.stdout)
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
VERBOSE_LOGS = False  # Hard-coded for diagnostics

# --- Helper: safe protobuf → dict for verbose logging ---
def _proto_to_dict(msg) -> dict:
    """
    Convert any protobuf message to a plain dict for logging.
    Uses field names as defined in .proto (snake_case) to aid debugging across schema variants.
    Tries the default descriptor pool so google.protobuf.Any can unpack if descriptors are registered.
    If unpacking fails due to unknown Any types (e.g., metadata), we attempt normal decode; on failure we strip only problematic metadata and retry — emotion messages will pass through when stubs are registered.
    Always returns a dict; never raises.
    """
    try:
        return MessageToDict(
            msg,
            preserving_proto_field_name=True,
            use_integers_for_enums=True,
            descriptor_pool=_desc_pool.Default(),
        )
    except Exception as e:
        s = str(e)
        # If failure is due to missing descriptor for Any in metadata, drop metadata and retry
        if "type.googleapis.com" in s or "message descriptor" in s:
            try:
                tmp = type(msg)()
                tmp.CopyFrom(msg)
                # Best-effort path clears
                try:
                    # Common schema: msg.animation_data.metadata (map<string, Any>)
                    tmp.animation_data.ClearField("metadata")
                except Exception:
                    pass
                try:
                    # Some schemas: msg.metadata at top-level
                    tmp.ClearField("metadata")
                except Exception:
                    pass
                return MessageToDict(
                    tmp,
                    preserving_proto_field_name=True,
                    use_integers_for_enums=True,
                    descriptor_pool=_desc_pool.Default(),
                )
            except Exception as e2:
                logger.warning(f"[proto2dict] stripped metadata due to Any descriptor error; retry failed: {e2}")
        # Fallback to repr to keep pipeline flowing
        try:
            return {"_protobuf_repr": repr(msg), "_error": s}
        except Exception:
            return {"_protobuf_fallback": "unprintable", "_error": s}

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import grpc

# For server-driven audio fetch via ffmpeg
import asyncio.subprocess as asp


# --- Robust NVIDIA ACE proto resolver ---
import importlib, pkgutil, sys, os, pathlib, fnmatch

def _import_first(paths):
    last = None
    for p in paths:
        try:
            return importlib.import_module(p)
        except Exception as e:
            last = e
    if last:
        raise last
    raise ImportError(f"None importable: {paths}")

def _discover_pb2_module(root_hint: str, want: str, contains: str | None = None):
    """
    Search under /app for generated *_pb2*.py that match a pattern,
    derive module path, and import. Prefer paths containing `contains`.
    """
    search_roots = ["/app", "/app/Audio2Face-3D-Samples/proto", "/app/Audio2Face-3D-Samples/proto/protobuf_files"]
    candidates = []
    for root in search_roots:
        for dp, _, files in os.walk(root):
            for f in files:
                if f.endswith("_pb2.py") or f.endswith("_pb2_grpc.py"):
                    full = os.path.join(dp, f)
                    rel = full.replace("/app/", "")
                    mod = rel[:-3].replace("/", ".")
                    if fnmatch.fnmatch(mod, root_hint):
                        if (contains is None) or (contains in mod):
                            candidates.append(mod)
    # Prefer shortest module (shallower), then alphabetical for stability
    candidates.sort(key=lambda m: (m.count("."), m))
    if not candidates:
        raise ImportError(f"No generated modules found for pattern {root_hint}")
    # Try to import in order
    last = None
    for m in candidates:
        try:
            return importlib.import_module(m)
        except Exception as e:
            last = e
    raise last if last else ImportError(f"Could not import any discovered module for {root_hint}")

# --- Resolve modules ---
# ctrl_pb2 (controller messages)
try:
    ctrl_pb2 = _import_first([
        "nvidia_ace.controller.v1_pb2",
        "nvidia_ace.controller.v1.controller_pb2",
        "nvidia_ace.controller.controller_pb2",
    ])
except Exception:
    ctrl_pb2 = _discover_pb2_module("nvidia_ace.controller*_*pb2", want="controller")

# a2f controller GRPC stub
try:
    a2f_ctrl_grpc = _import_first([
        "nvidia_ace.services.a2f_controller.v1_pb2_grpc",
        "nvidia_ace.services.a2f_controller.v1.a2f_controller_pb2_grpc",
        "nvidia_ace.a2f_controller.v1.a2f_controller_pb2_grpc",
        "nvidia_ace.a2f_controller.v1_pb2_grpc",
    ])
except Exception:
    a2f_ctrl_grpc = _discover_pb2_module("nvidia_ace.*a2f_controller*_*pb2_grpc", want="a2f_controller", contains="services")

# audio messages
try:
    audio_pb2 = _import_first([
        "nvidia_ace.audio.v1_pb2",
        "nvidia_ace.audio.v1.audio_pb2",
    ])
except Exception:
    audio_pb2 = _discover_pb2_module("nvidia_ace.audio*_*pb2", want="audio")

#
# a2f messages
try:
    a2f_pb2 = _import_first([
        "nvidia_ace.a2f.v1_pb2",
        "nvidia_ace.a2f.v1.a2f_pb2",
    ])
except Exception:
    a2f_pb2 = _discover_pb2_module("nvidia_ace.a2f*_*pb2", want="a2f")


# emotion aggregate (metadata Any) — optional but preferred
try:
    emotion_pb2 = _import_first([
        "nvidia_ace.emotion_aggregate.v1_pb2",
        "nvidia_ace.emotion_aggregate.v1.emotion_aggregate_pb2",
    ])
except Exception:
    try:
        emotion_pb2 = _discover_pb2_module("nvidia_ace.emotion_aggregate*_*pb2", want="emotion_aggregate")
    except Exception:
        emotion_pb2 = None
if emotion_pb2:
    try:
        logger.info(f"[stubs] resolved: emotion_pb2={getattr(emotion_pb2,'__name__',emotion_pb2)} (registered in default pool)")
    except Exception:
        logger.info("[stubs] resolved: emotion_pb2 (registered)")
else:
    logger.info("[stubs] emotion_pb2 not found; Any(emotion_aggregate) may be stripped at runtime.")

logger.info(f"[stubs] resolved: ctrl_pb2={getattr(ctrl_pb2,'__name__',ctrl_pb2)} "
            f"a2f_ctrl_grpc={getattr(a2f_ctrl_grpc,'__name__',a2f_ctrl_grpc)} "
            f"audio_pb2={getattr(audio_pb2,'__name__',audio_pb2)} "
            f"a2f_pb2={getattr(a2f_pb2,'__name__',a2f_pb2)}")

# --- DIAG: list actual fields and oneofs of AudioStream to detect correct schema ---
try:
    _desc = ctrl_pb2.AudioStream.DESCRIPTOR
    flds = []
    for f in _desc.fields:
        flds.append({
            "name": f.name,
            "number": f.number,
            "type": f.type,
            "oneof": f.containing_oneof.name if f.containing_oneof else None,
        })
    oneofs = [o.name for o in _desc.oneofs]
    logger.info(f"[DIAG][AudioStream] oneofs={oneofs}")
    logger.info(f"[DIAG][AudioStream] fields={flds}")
except Exception as _e_diag:
    logger.error(f"[DIAG][AudioStream] descriptor inspection failed: {_e_diag}")

# Sanity check: is EmotionAggregate descriptor available?
try:
    _ = _desc_pool.Default().FindMessageTypeByName("nvidia_ace.emotion_aggregate.v1.EmotionAggregate")
    logger.info("[stubs] EmotionAggregate descriptor present in default pool.")
except Exception:
    logger.info("[stubs] EmotionAggregate descriptor NOT present; will strip metadata if needed.")

# emotion with timecode (explicit per-frame emotions in audio) — optional but preferred
try:
    emotion_wtc_pb2 = _import_first([
        "nvidia_ace.emotion_with_timecode.v1_pb2",
        "nvidia_ace.emotion_with_timecode.v1.emotion_with_timecode_pb2",
    ])
except Exception:
    try:
        emotion_wtc_pb2 = _discover_pb2_module("nvidia_ace.emotion_with_timecode*_*pb2", want="emotion_with_timecode")
    except Exception:
        emotion_wtc_pb2 = None

if emotion_wtc_pb2:
    try:
        logger.info(f"[stubs] resolved: emotion_wtc_pb2={getattr(emotion_wtc_pb2,'__name__',emotion_wtc_pb2)} (registered in default pool)")
    except Exception:
        logger.info("[stubs] resolved: emotion_wtc_pb2 (registered)")
else:
    logger.info("[stubs] emotion_wtc_pb2 not found; audio.emotions may be absent or opaque.")

# Sanity check: is EmotionWithTimeCode descriptor available?
try:
    _ = _desc_pool.Default().FindMessageTypeByName("nvidia_ace.emotion_with_timecode.v1.EmotionWithTimeCode")
    logger.info("[stubs] EmotionWithTimeCode descriptor present in default pool.")
except Exception:
    logger.info("[stubs] EmotionWithTimeCode descriptor NOT present; audio.emotions may not unpack.")

router = APIRouter()

NVCF_ENDPOINT = os.environ.get("NVCF_ENDPOINT", "grpc.nvcf.nvidia.com:443")
A2F_BACKEND = os.environ.get("A2F_BACKEND", "nvcf").lower().strip()
A2F_GRPC_ADDR = os.environ.get("A2F_GRPC_ADDR", NVCF_ENDPOINT).strip()
VAST_ROUTE_URL = os.environ.get("VAST_ROUTE_URL", "https://run.vast.ai/route/").strip()
VAST_ENDPOINT = (
    os.environ.get("VAST_ENDPOINT")
    or os.environ.get("VAST_A2F_ENDPOINT")
    or os.environ.get("VAST_ROUTE_ENDPOINT")
    or ""
).strip()
VAST_ROUTE_COST = float(os.environ.get("VAST_ROUTE_COST", "1") or "1")
VAST_ROUTE_TIMEOUT_SEC = float(os.environ.get("VAST_ROUTE_TIMEOUT_SEC", "15") or "15")
VAST_ROUTE_CACHE_TTL_SEC = float(os.environ.get("VAST_ROUTE_CACHE_TTL_SEC", "45") or "45")
VAST_GRPC_PORT = (
    os.environ.get("VAST_GRPC_PORT")
    or os.environ.get("VAST_A2F_GRPC_PORT")
    or ""
).strip()
VAST_ROUTE_URL_IS_GRPC = os.environ.get("VAST_ROUTE_URL_IS_GRPC", "0").strip().lower() in ("1", "true", "yes", "on")
_VAST_ROUTE_CACHE = {"addr": "", "route_url": "", "expires_at": 0.0}

# Shared JSON secret loader, matching avatar-api's URX_AWS_SECRET_NAME pattern.
_AWS_SECRETS_CACHE: Optional[Dict[str, Any]] = None


def _load_secrets_from_aws() -> Optional[Dict[str, Any]]:
    global _AWS_SECRETS_CACHE
    if _AWS_SECRETS_CACHE is not None:
        return _AWS_SECRETS_CACHE

    secret_name = (os.getenv("URX_AWS_SECRET_NAME", "") or "").strip() or "UnravelAI/AI_KEYS"
    region_name = (
        (os.getenv("URX_AWS_REGION", "") or "").strip()
        or (os.getenv("AWS_REGION", "") or "").strip()
        or "ap-southeast-1"
    )
    if not secret_name or not region_name:
        return None

    try:
        session = boto3.session.Session()
        client = session.client(service_name="secretsmanager", region_name=region_name)
        resp = client.get_secret_value(SecretId=secret_name)
        secret_str = resp.get("SecretString")
        if not secret_str:
            return None
        loaded = json.loads(secret_str)
        if not isinstance(loaded, dict):
            return None
        _AWS_SECRETS_CACHE = loaded
        return _AWS_SECRETS_CACHE
    except Exception as exc:
        logger.warning("[Secrets] Failed to retrieve shared AWS secret %s: %s", secret_name, exc)
        return None


def _get_secret(key: str) -> Optional[str]:
    secrets = _load_secrets_from_aws()
    if secrets and key in secrets and secrets[key]:
        return str(secrets[key])
    value = os.getenv(key)
    return value if value else None


# Hosted NVCF function IDs your clients already send. In self-hosted mode these
# become routing keys to dedicated A2F NIM instances, because a single NIM
# container only runs one active character/profile.
_A2F_MARK_FUNCTION_IDS = {
    "8efc55f5-6f00-424e-afe9-26212cd2c630",
    "cf145b84-423b-4222-bfdd-15bb0142b0fd",
}
_A2F_CLAIRE_FUNCTION_IDS = {
    "0961a6da-fb9e-4f2e-8491-247e5fd7bf8d",
    "617f80a7-85e4-4bf0-9dd6-dcb61e886142",
}
_A2F_JAMES_FUNCTION_IDS = {
    "9327c39f-a361-4e02-bd72-e11b4c9b7b5e",
    "8082bdcb-9968-4dc5-8705-423ea98b8fc2",
}


def _is_self_hosted_a2f() -> bool:
    return A2F_BACKEND in ("self_hosted", "self-hosted", "local", "vast", "runpod")


def _vast_api_key() -> str:
    raw = (
        _get_secret("VAST_API_KEY")
        or _get_secret("VAST_ROUTE_API_KEY")
        or _get_secret("VAST_API_TOKEN")
        or _get_secret("VASTAI_API_KEY")
        or ""
    ).strip()
    if raw.startswith("{") and raw.endswith("}"):
        try:
            data = json.loads(raw)
            raw = str(data.get("VAST_API_KEY") or data.get("token") or data.get("api_key") or "").strip()
        except Exception:
            pass
    return raw


def _host_port_from_url(url: str, grpc_port: str) -> str:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    if not parsed.hostname:
        raise RuntimeError(f"Vast route returned an invalid url: {url!r}")
    host = parsed.hostname
    if grpc_port:
        return f"{host}:{grpc_port}"
    if VAST_ROUTE_URL_IS_GRPC and parsed.port:
        return f"{host}:{parsed.port}"
    raise RuntimeError(
        "Vast route returned an HTTP worker URL, but a2f-wrapper needs a raw A2F gRPC host:port. "
        "Set VAST_GRPC_PORT to the public port mapped to container 52000, or set "
        "VAST_ROUTE_URL_IS_GRPC=1 only if the route URL itself is a gRPC endpoint."
    )


def _request_vast_route() -> Dict[str, Any]:
    if not VAST_ENDPOINT:
        raise RuntimeError("A2F_BACKEND=vast requires VAST_ENDPOINT.")

    api_key = _vast_api_key()
    if not api_key:
        raise RuntimeError("VAST_ENDPOINT is set, but VAST_API_KEY/VAST_ROUTE_API_KEY is missing.")

    payload = json.dumps({"endpoint": VAST_ENDPOINT, "cost": VAST_ROUTE_COST}).encode("utf-8")
    req = urllib.request.Request(
        VAST_ROUTE_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=VAST_ROUTE_TIMEOUT_SEC) as resp:
            body = resp.read().decode("utf-8", "replace")
            data = json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace") if exc.fp else ""
        raise RuntimeError(f"Vast route failed with HTTP {exc.code}: {detail[:300]}") from exc
    except Exception as exc:
        raise RuntimeError(f"Vast route request failed: {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeError("Vast route returned a non-object response.")
    return data


async def _probe_grpc_port(host: str, port: int) -> bool:
    """Single non-blocking TCP probe. Returns True if the port accepts a connection."""
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=2.0)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    except (OSError, asyncio.TimeoutError):
        return False


def _fetch_vast_port_mappings(api_key: str, host_ip: str) -> Dict[int, str]:
    """Query Vast.ai instances API and return {container_port: external_port} for the matching instance."""
    result: Dict[int, str] = {}
    try:
        url = f"https://console.vast.ai/api/v0/instances/?api_key={api_key}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        instances = data.get("instances", [])
        logger.info("[VAST] instances API returned %d instances", len(instances))
        for inst in instances:
            pub_ip = str(inst.get("public_ipaddr") or "").strip()
            if pub_ip != host_ip:
                continue
            logger.info("[VAST] instance id=%s ip=%s ports=%s", inst.get("id"), pub_ip, inst.get("ports"))
            for port_key, mappings in (inst.get("ports") or {}).items():
                try:
                    container_port = int(port_key.split("/")[0])
                except ValueError:
                    continue
                for m in (mappings or []):
                    hp = str(m.get("HostPort") or "").strip()
                    if hp and container_port not in result:
                        result[container_port] = hp
            break
    except Exception as e:
        logger.warning("[VAST] Failed to fetch port mappings from instances API: %s", e)
    logger.info("[VAST] port mappings for %s: %s", host_ip, result)
    return result


def _resolve_vast_route_addr() -> str:
    explicit = (os.environ.get("VAST_A2F_GRPC_ADDR") or "").strip()
    if explicit:
        return explicit
    if not VAST_ENDPOINT:
        if A2F_GRPC_ADDR and A2F_GRPC_ADDR != NVCF_ENDPOINT:
            return A2F_GRPC_ADDR
        raise RuntimeError("A2F_BACKEND=vast requires VAST_ENDPOINT or A2F_GRPC_ADDR/VAST_A2F_GRPC_ADDR.")

    now = time.time()
    cached_addr = str(_VAST_ROUTE_CACHE.get("addr") or "")
    if cached_addr and now < float(_VAST_ROUTE_CACHE.get("expires_at") or 0):
        return cached_addr

    data = _request_vast_route()
    logger.info("[VAST] route API response keys: %s", list(data.keys()))
    logger.info("[VAST] route API response: %s", data)
    route_url = str(data.get("url") or "").strip()

    if not route_url:
        raise RuntimeError(
            "Vast.ai endpoint has no route assigned yet — worker is still starting up. "
            f"endpoint={VAST_ENDPOINT!r}"
        )

    if VAST_GRPC_PORT:
        addr = _host_port_from_url(route_url, VAST_GRPC_PORT)
    elif VAST_ROUTE_URL_IS_GRPC:
        addr = _host_port_from_url(route_url, "")
    else:
        # Auto-discover the external port for container:52000 via the Vast.ai instances API.
        # The route URL points to the HTTP API port (8000), not the gRPC port (52000).
        # Add scheme if missing so urlparse extracts hostname correctly.
        normalized = route_url if "://" in route_url else f"http://{route_url}"
        parsed = urllib.parse.urlparse(normalized)
        host_ip = parsed.hostname or ""
        api_key = _vast_api_key()
        logger.info("[VAST] auto-discover: route_url=%r host_ip=%r has_api_key=%s", route_url, host_ip, bool(api_key))
        port_map = _fetch_vast_port_mappings(api_key, host_ip) if (host_ip and api_key) else {}
        grpc_port = port_map.get(52000)
        if not grpc_port:
            raise RuntimeError(
                f"Could not auto-discover gRPC port for {host_ip!r} (route_url={route_url!r}). "
                f"has_api_key={bool(api_key)}. "
                "Set VAST_GRPC_PORT to the external port mapped to container port 52000, "
                "or ensure VAST_API_KEY is present in AWS Secrets Manager."
            )
        addr = f"{host_ip}:{grpc_port}"
        # Cache the HTTP health port (container:8000) so a2f_vast_ready can probe NIM readiness.
        http_port = port_map.get(8000)
        if http_port:
            _VAST_ROUTE_CACHE["http_health_addr"] = f"{host_ip}:{http_port}"

    _VAST_ROUTE_CACHE.update({
        "addr": addr,
        "route_url": route_url,
        "expires_at": now + max(1.0, VAST_ROUTE_CACHE_TTL_SEC),
    })
    logger.info("[VAST] endpoint=%s route_url=%s grpc_addr=%s", VAST_ENDPOINT, route_url, addr)
    return addr


@router.get("/a2f/vast/ready")
async def a2f_vast_ready():
    """Check whether Vast can route to a ready A2F worker without exposing Vast credentials."""
    if A2F_BACKEND != "vast":
        return {
            "ready": True,
            "backend": A2F_BACKEND,
            "skipped": True,
            "reason": "backend_not_vast",
        }

    try:
        # Always resolve (uses cache for addr discovery, but always re-checks NIM health).
        addr = await asyncio.to_thread(_resolve_vast_route_addr)
        host, grpc_port_str = addr.rsplit(":", 1)
        grpc_port = int(grpc_port_str)

        # Check NIM HTTP health first, then confirm gRPC port accepts connections.
        # Both must pass — HTTP can return 200 before gRPC is fully bound.
        http_health_addr = _VAST_ROUTE_CACHE.get("http_health_addr") or ""
        nim_ready = False
        nim_status = "unknown"

        if http_health_addr:
            try:
                health_url = f"http://{http_health_addr}/v1/health/ready"
                health_req = urllib.request.Request(health_url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(health_req, timeout=5) as r:
                    body = r.read().decode("utf-8", "replace")
                    nim_status = f"http_{r.status} body={body[:80]}"
                    if r.status == 200:
                        # Also verify gRPC port is actually accepting connections.
                        grpc_ok = await _probe_grpc_port(host, grpc_port)
                        nim_ready = grpc_ok
                        nim_status += f" grpc_probe={'ok' if grpc_ok else 'refused'}"
            except urllib.error.HTTPError as e:
                nim_status = f"http_{e.code}"
            except Exception as e:
                nim_status = f"error:{e}"
        else:
            nim_ready = await _probe_grpc_port(host, grpc_port)
            nim_status = f"tcp_probe={'ok' if nim_ready else 'refused'}"

        logger.info("[VAST] ready check: addr=%s http_health=%s nim_ready=%s", addr, http_health_addr, nim_ready)

        if nim_ready:
            return {
                "ready": True,
                "backend": "vast",
                "endpoint": VAST_ENDPOINT,
                "grpc_addr": addr,
            }

        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "backend": "vast",
                "endpoint": VAST_ENDPOINT,
                "status": f"NIM not ready ({nim_status}) — model still loading",
            },
        )
    except Exception as exc:
        logger.warning("[VAST] readiness check failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "backend": "vast",
                "endpoint": VAST_ENDPOINT,
                "error": str(exc)[:300],
            },
        )


def _resolve_self_hosted_a2f_addr(function_id: Optional[str]) -> str:
    if A2F_BACKEND == "vast":
        return _resolve_vast_route_addr()

    fid = (function_id or "").strip().lower()
    if fid in _A2F_CLAIRE_FUNCTION_IDS:
        return os.environ.get("A2F_GRPC_ADDR_CLAIRE", "").strip() or A2F_GRPC_ADDR
    if fid in _A2F_MARK_FUNCTION_IDS:
        return os.environ.get("A2F_GRPC_ADDR_MARK", "").strip() or A2F_GRPC_ADDR
    if fid in _A2F_JAMES_FUNCTION_IDS:
        return os.environ.get("A2F_GRPC_ADDR_JAMES", "").strip() or A2F_GRPC_ADDR
    return A2F_GRPC_ADDR


# --- Pub/Sub: conversation_id -> WebSocket subscribers ---
_SUBSCRIBERS: Dict[str, Set[WebSocket]] = {}
_SUB_LOCK = asyncio.Lock()
_WS_CLIENT_INFO: Dict[WebSocket, Dict[str, str]] = {}
_AUDIO_TEXT_BY_CONVO: Dict[str, str] = {}
_AUDIO_SENTENCE_BY_CONVO: Dict[str, str] = {}
_LAST_LOGGED_SENTENCE_BY_CONVO: Dict[str, str] = {}
_PUB_LOG_EVERY = int(os.getenv("A2F_PUB_LOG_EVERY", "1"))
_PUB_COUNTS: Dict[str, int] = {}
_RESOURCE_EXHAUSTED_TIMES: List[float] = []
_RESOURCE_EXHAUSTED_WINDOW_SEC = int(os.getenv("A2F_RESOURCE_EXHAUSTED_WINDOW_SEC", "120"))
_RESOURCE_EXHAUSTED_BASE_RETRY_MS = int(os.getenv("A2F_RESOURCE_EXHAUSTED_BASE_RETRY_MS", "1500"))
_RESOURCE_EXHAUSTED_MAX_RETRY_MS = int(os.getenv("A2F_RESOURCE_EXHAUSTED_MAX_RETRY_MS", "20000"))

# Emotion state per conversation: conversation_id -> 10-element float vector
# Order: [amazement, anger, cheekiness, disgust, fear, grief, joy, outofbreath, pain, sadness]
_EMOTION_STATE: Dict[str, List[float]] = {}
_A2F_EMOTION_LABELS = ["amazement", "anger", "cheekiness", "disgust", "fear", "grief", "joy", "outofbreath", "pain", "sadness"]


def set_emotion(conversation_id: str, emotion_vector: List[float]) -> None:
    """Store an emotion vector for the given conversation. Called from the REST endpoint."""
    _EMOTION_STATE[conversation_id] = list(emotion_vector)
    logger.info(f"[EMOTION] set convo={conversation_id!r} vector={emotion_vector}")


_BUILD_AWE_CALL_COUNTS: Dict[str, int] = {}

def _build_awe(audio_buffer: bytes, conversation_id: str):
    """Build an AudioWithEmotion message, injecting emotion state for the conversation if available.

    EmotionWithTimeCode.emotion is a map<string, float> — set directly, no EmotionAggregate needed.
    """
    count = _BUILD_AWE_CALL_COUNTS.get(conversation_id, 0) + 1
    _BUILD_AWE_CALL_COUNTS[conversation_id] = count
    if count == 1 or count % 100 == 0:
        logger.info(f"[EMOTION][DBG] _build_awe convo={conversation_id!r} call#{count} known_convos={list(_EMOTION_STATE.keys())}")
    vector = _EMOTION_STATE.get(conversation_id)
    if vector and emotion_wtc_pb2:
        try:
            emotion_map = {
                label: float(w)
                for label, w in zip(_A2F_EMOTION_LABELS, vector)
                if w > 0
            }
            wtc = emotion_wtc_pb2.EmotionWithTimeCode(time_code=0.0, emotion=emotion_map)
            logger.info(f"[EMOTION] injecting into gRPC convo={conversation_id!r} emotion={emotion_map}")
            return a2f_pb2.AudioWithEmotion(audio_buffer=audio_buffer, emotions=[wtc])
        except Exception as _e:
            logger.warning(f"[EMOTION] Failed to inject emotion for convo={conversation_id}: {_e}")
    elif vector:
        logger.warning(f"[EMOTION] vector set for convo={conversation_id!r} but emotion_wtc_pb2 stub unavailable")
    return a2f_pb2.AudioWithEmotion(audio_buffer=audio_buffer)

def _get_ws_client_info(ws: WebSocket) -> Dict[str, str]:
    info: Dict[str, str] = {}
    try:
        hdrs = ws.headers or {}
        xff = hdrs.get("x-forwarded-for") or hdrs.get("X-Forwarded-For")
        if xff:
            info["x_forwarded_for"] = xff
            info["ip"] = xff.split(",")[0].strip()
        elif ws.client:
            info["ip"] = str(ws.client.host)
            info["port"] = str(ws.client.port)
        ua = hdrs.get("user-agent") or hdrs.get("User-Agent")
        if ua:
            info["user_agent"] = ua
        ch_ua = hdrs.get("sec-ch-ua") or hdrs.get("Sec-CH-UA")
        if ch_ua:
            info["sec_ch_ua"] = ch_ua
        ch_platform = hdrs.get("sec-ch-ua-platform") or hdrs.get("Sec-CH-UA-Platform")
        if ch_platform:
            info["sec_ch_ua_platform"] = ch_platform.strip('"')
    except Exception:
        pass
    return info

async def _add_subscriber(conversation_id: str, ws: WebSocket) -> None:
    async with _SUB_LOCK:
        bucket = _SUBSCRIBERS.setdefault(conversation_id, set())
        bucket.add(ws)
        _WS_CLIENT_INFO[ws] = _get_ws_client_info(ws)

async def _remove_subscriber(conversation_id: str, ws: WebSocket) -> None:
    async with _SUB_LOCK:
        bucket = _SUBSCRIBERS.get(conversation_id)
        if not bucket:
            return
        bucket.discard(ws)
        _WS_CLIENT_INFO.pop(ws, None)
        if not bucket:
            _SUBSCRIBERS.pop(conversation_id, None)

async def _publish(conversation_id: str, text: str) -> None:
    if not conversation_id:
        return
    async with _SUB_LOCK:
        targets = list(_SUBSCRIBERS.get(conversation_id, set()))
    if not targets:
        return
    if _PUB_LOG_EVERY > 0:
        _PUB_COUNTS[conversation_id] = _PUB_COUNTS.get(conversation_id, 0) + 1
        if _PUB_COUNTS[conversation_id] % _PUB_LOG_EVERY == 0:
            try:
                sample = []
                for ws in targets[:3]:
                    info = _WS_CLIENT_INFO.get(ws, {})
                    sample.append({
                        "ip": info.get("ip", ""),
                        "ua": info.get("user_agent", ""),
                        "platform": info.get("sec_ch_ua_platform", "")
                    })
                text_preview = ""
                frame_type = ""
                blendshape_count = None
                try:
                    payload = json.loads(text)
                    frame_type = payload.get("type") or ""
                    # Best-effort blendshape count from payload
                    if "shapes" in payload:
                        shapes = payload.get("shapes")
                        if isinstance(shapes, dict):
                            blendshape_count = len(shapes)
                        elif isinstance(shapes, (list, tuple)):
                            blendshape_count = len(shapes)
                    if blendshape_count is None and "blendshape_weights" in payload:
                        bsw = payload.get("blendshape_weights")
                        if isinstance(bsw, dict):
                            blendshape_count = len(bsw)
                        elif isinstance(bsw, (list, tuple)):
                            blendshape_count = len(bsw)
                    if blendshape_count is None and "weights" in payload and "blend" in str(frame_type).lower():
                        weights = payload.get("weights")
                        if isinstance(weights, (list, tuple)):
                            blendshape_count = len(weights)
                    text_preview = payload.get("text") or _AUDIO_TEXT_BY_CONVO.get(conversation_id, "")
                except Exception:
                    text_preview = _AUDIO_TEXT_BY_CONVO.get(conversation_id, "")
                text_preview = str(text_preview or "").strip()
                if len(text_preview) > 120:
                    text_preview = text_preview[:117] + "..."
                logger.info(
                    "[WS][PUB] conv=%s targets=%d bytes=%d type=%s blendshapes=%s text=%s sample=%s",
                    conversation_id,
                    len(targets),
                    len(text),
                    frame_type or "n/a",
                    blendshape_count if blendshape_count is not None else "n/a",
                    text_preview,
                    sample
                )
            except Exception:
                pass
    dead = []
    for ws in targets:
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    if dead:
        async with _SUB_LOCK:
            bucket = _SUBSCRIBERS.get(conversation_id)
            if not bucket:
                return
            for ws in dead:
                bucket.discard(ws)
            if not bucket:
                _SUBSCRIBERS.pop(conversation_id, None)

def _log_sentence(conversation_id: str, text: Optional[str]) -> None:
    if not conversation_id or not text:
        return
    clean = str(text).strip()
    if not clean:
        return
    if _LAST_LOGGED_SENTENCE_BY_CONVO.get(conversation_id) == clean:
        return
    _LAST_LOGGED_SENTENCE_BY_CONVO[conversation_id] = clean
    logger.info("[A2F][SENTENCE] conversation_id=%s text=%s", conversation_id, clean)

def _extract_conversation_id(payload: dict) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(
        payload.get("conversation_id")
        or payload.get("conv_id")
        or payload.get("conversationId")
        or ""
    ).strip()

def _to_int_millis(raw_value) -> Optional[int]:
    if raw_value is None:
        return None
    try:
        value = str(raw_value).strip()
    except Exception:
        return None
    if not value:
        return None
    try:
        if "." in value:
            return int(float(value))
        return int(value)
    except Exception:
        return None

def _extract_retry_after_ms(init_md: dict, trailing_md: dict) -> Optional[int]:
    merged = {}
    for src in (init_md or {}, trailing_md or {}):
        for k, v in src.items():
            merged[str(k).strip().lower()] = v

    direct_ms_keys = (
        "retry-after-ms",
        "retry_after_ms",
        "x-retry-after-ms",
        "x-ratelimit-reset-ms",
    )
    for key in direct_ms_keys:
        v = _to_int_millis(merged.get(key))
        if v is not None and v > 0:
            return v

    # Standard retry-after header is typically in seconds.
    retry_after_seconds = _to_int_millis(merged.get("retry-after"))
    if retry_after_seconds is not None and retry_after_seconds > 0:
        return retry_after_seconds * 1000
    return None

def _build_capacity_estimate(init_md: dict, trailing_md: dict) -> dict:
    now = time.time()
    global _RESOURCE_EXHAUSTED_TIMES
    _RESOURCE_EXHAUSTED_TIMES = [
        t for t in _RESOURCE_EXHAUSTED_TIMES
        if (now - t) <= _RESOURCE_EXHAUSTED_WINDOW_SEC
    ]
    _RESOURCE_EXHAUSTED_TIMES.append(now)
    failures_in_window = len(_RESOURCE_EXHAUSTED_TIMES)
    queue_position_lb = max(0, failures_in_window - 1)

    retry_after_ms = _extract_retry_after_ms(init_md, trailing_md)
    if retry_after_ms is None:
        # Local fallback estimate (not provider queue depth).
        retry_after_ms = min(
            _RESOURCE_EXHAUSTED_MAX_RETRY_MS,
            _RESOURCE_EXHAUSTED_BASE_RETRY_MS * (2 ** min(queue_position_lb, 5)),
        )

    estimated_wait_ms = retry_after_ms + min(5000, queue_position_lb * 500)
    return {
        "estimated_wait_ms": int(estimated_wait_ms),
        "estimated_position_lower_bound": int(queue_position_lb),
        "events_in_window": int(failures_in_window),
        "window_seconds": int(_RESOURCE_EXHAUSTED_WINDOW_SEC),
        "method": "local_recent_resource_exhausted_events",
    }

def _is_retriable_grpc_status(status_name: str) -> bool:
    return status_name in {"RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "ABORTED"}

def _normalize_outputs(value) -> Set[str]:
    if value is None:
        return {"client"}
    if isinstance(value, (list, tuple)):
        parts = [str(v).strip().lower() for v in value if str(v).strip()]
    else:
        raw = str(value).strip()
        parts = [p.strip().lower() for p in raw.replace(";", ",").split(",") if p.strip()]
    return set(parts) or {"client"}

# --- Secrets Manager support for NVCF API key ---
NVCF_SECRET_ARN = os.getenv(
    "NVCF_SECRET_ARN",
    "arn:aws:secretsmanager:ap-southeast-1:064451089967:secret:UnravelAI/NVDA_API_KEY-gCbo3W",
)
_cached_nvcf_api_key: Optional[str] = None

def get_nvcf_api_key() -> Optional[str]:
    """
    Retrieve NVIDIA Cloud Functions API key from:
    1) Environment variable NVCF_API_KEY (string or JSON like {"NVCF_API_KEY":"..."})
    2) AWS Secrets Manager (JSON or plaintext), SecretId from NVCF_SECRET_ARN
    Caches the value for subsequent calls.
    """
    global _cached_nvcf_api_key
    if _cached_nvcf_api_key:
        return _cached_nvcf_api_key

    # First try env var (may be raw or JSON)
    env_val = os.environ.get("NVCF_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    if env_val:
        val = env_val.strip()
        if val.startswith("{") and val.endswith("}"):
            try:
                data = json.loads(val)
                val = data.get("NVCF_API_KEY") or data.get("token") or data.get("api_key")
            except Exception:
                pass
        if val:
            _cached_nvcf_api_key = val
            return val

    # Fallback to Secrets Manager
    if not NVCF_SECRET_ARN:
        return None
    try:
        sm = boto3.client("secretsmanager", region_name="ap-southeast-1")
        resp = sm.get_secret_value(SecretId=NVCF_SECRET_ARN)
    except ClientError as e:
        logger.error(f"[Secrets] Failed to retrieve NVCF secret: {e}")
        return None

    secret = resp.get("SecretString")
    if secret is None and "SecretBinary" in resp:
        try:
            secret = resp["SecretBinary"].decode("utf-8")
        except Exception:
            secret = None
    if not secret:
        return None

    # Parse possible JSON, else treat as raw
    try:
        parsed = json.loads(secret)
        key = parsed.get("NVCF_API_KEY") or parsed.get("token") or parsed.get("api_key") or secret
    except json.JSONDecodeError:
        key = secret

    _cached_nvcf_api_key = key
    return key

# Default NVIDIA A2F blendshape order (tongue-enabled models, 71 entries).
# Used when the service does not return names in the stream header.
DEFAULT_A2F_NAMES = [
    "EyeBlinkLeft","EyeLookDownLeft","EyeLookInLeft","EyeLookOutLeft","EyeLookUpLeft","EyeSquintLeft","EyeWideLeft",
    "EyeBlinkRight","EyeLookDownRight","EyeLookInRight","EyeLookOutRight","EyeLookUpRight","EyeSquintRight","EyeWideRight",
    "JawForward","JawLeft","JawRight","JawOpen",
    "MouthClose","MouthFunnel","MouthPucker","MouthLeft","MouthRight","MouthSmileLeft","MouthSmileRight","MouthFrownLeft","MouthFrownRight",
    "MouthDimpleLeft","MouthDimpleRight","MouthStretchLeft","MouthStretchRight","MouthRollLower","MouthRollUpper",
    "MouthShrugLower","MouthShrugUpper","MouthPressLeft","MouthPressRight","MouthLowerDownLeft","MouthLowerDownRight",
    "MouthUpperUpLeft","MouthUpperUpRight",
    "BrowDownLeft","BrowDownRight","BrowInnerUp","BrowOuterUpLeft","BrowOuterUpRight",
    "CheekPuff","CheekSquintLeft","CheekSquintRight",
    "NoseSneerLeft","NoseSneerRight",
    "TongueOut","HeadRoll","HeadPitch","HeadYaw",
    "TongueTipUp","TongueTipDown","TongueTipLeft","TongueTipRight",
    "TongueRollUp","TongueRollDown","TongueRollLeft","TongueRollRight",
    "TongueUp","TongueDown","TongueLeft","TongueRight","TongueIn","TongueStretch","TongueWide","TongueNarrow",
]
# Optional: allow override via env (comma-separated)
_env_names = os.getenv("A2F_NAMES_ORDER", "").strip()
if _env_names:
    try:
        env_list = [s.strip() for s in _env_names.split(",") if s.strip()]
        if len(env_list) > 0:
            DEFAULT_A2F_NAMES = env_list
    except Exception:
        pass

@router.websocket("/a2f/visemes")
async def a2f_visemes(ws: WebSocket):
    """
    Pub/Sub consumer: subscribe to viseme/blendshape frames for a conversation_id.

    Client protocol:
      1) First message MUST be text/json:
         { "conversation_id": "..."}
      2) Server pushes frames as JSON (blendshapes_frame/visemes_frame/etc).
    """
    await ws.accept()
    logger.info("[WS][SUB] client connected")
    try:
        header_text = await ws.receive_text()
        cfg = json.loads(header_text or "{}")
    except WebSocketDisconnect:
        return
    except Exception as e:
        await _ws_error(ws, f"Invalid header JSON: {e}")
        return

    conversation_id: Optional[str] = _extract_conversation_id(cfg)
    if not conversation_id:
        await _ws_error(ws, "Missing 'conversation_id' (required).")
        return

    # Always enable pubsub for stream outputs so visemes are broadcast to subscribers.
    outputs = _normalize_outputs(cfg.get("outputs"))
    outputs.add("pubsub")
    send_to_client = True
    if "none" in outputs or "pubsub_only" in outputs:
        send_to_client = False
    elif "pubsub" in outputs and not (outputs & {"client", "ws", "return"}):
        send_to_client = False
    send_to_pubsub = "pubsub" in outputs or "pubsub_only" in outputs

    async def _emit(payload: dict) -> None:
        text = json.dumps(payload, ensure_ascii=False)
        if send_to_client:
            await ws.send_text(text)
        if send_to_pubsub:
            await _publish(conversation_id, text)

    await _add_subscriber(conversation_id, ws)
    sub_info = _WS_CLIENT_INFO.get(ws, {})
    logger.info(
        "[WS][SUB] subscribed conversation_id=%s ip=%s ua=%s platform=%s",
        conversation_id,
        sub_info.get("ip", ""),
        sub_info.get("user_agent", ""),
        sub_info.get("sec_ch_ua_platform", "")
    )
    try:
        while True:
            msg = await ws.receive()
            t = msg.get("type")
            if t == "websocket.disconnect":
                break
            if t == "websocket.receive" and msg.get("text"):
                try:
                    ctrl = json.loads(msg.get("text") or "{}")
                except Exception:
                    ctrl = None
                req_conversation_id = _extract_conversation_id(ctrl or {})
                if req_conversation_id and req_conversation_id != conversation_id:
                    logger.warning(
                        "[WS][SUB] conversation_id change requested on active socket current=%s requested=%s; forcing reconnect",
                        conversation_id,
                        req_conversation_id,
                    )
                    await _ws_error(
                        ws,
                        "conversation_id is immutable per /a2f/visemes connection. "
                        "Reconnect both /a2f/stream and /a2f/visemes with the new conversation_id.",
                    )
                    with contextlib.suppress(Exception):
                        await ws.close(code=1008)
                    break
    except WebSocketDisconnect:
        pass
    finally:
        await _remove_subscriber(conversation_id, ws)
        sub_info = _WS_CLIENT_INFO.get(ws, {})
        logger.info(
            "[WS][SUB] disconnected conversation_id=%s ip=%s ua=%s platform=%s",
            conversation_id,
            sub_info.get("ip", ""),
            sub_info.get("user_agent", ""),
            sub_info.get("sec_ch_ua_platform", "")
        )

@router.websocket("/a2f/stream")
async def a2f_stream(ws: WebSocket):
    """
    WebSocket <-> NVIDIA A2F gRPC bridge.

    Client protocol:
      1) First message MUST be text/json:
         {
           "sr": 16000,          # sample rate
           "ch": 1,              # channels
           "format": "pcm16",    # only pcm16 supported here
           "apikey": "...",      # NGC/NVCF key (optional if running inside NGC)
           "function_id": "...", # model function-id (e.g., Claire/Mark/James)
           "conversation_id": "...", # REQUIRED: conversation/session id (used for routing)
           "outputs": "client,pubsub", # optional: "pubsub" to disable WS replies and only broadcast
         }

      2) Then send binary frames of raw PCM16 audio (little-endian).
      3) Server pushes back viseme frames as JSON:
         { "t": <seconds>, "weights": [ { "name": "<blend>", "weight": <0..1> }, ... ] }

      Close the socket to end.
    """
    await ws.accept()
    logger.info("[WS] client connected")
    print("[WS] accepted")

    # --- 1) Read and validate header ---
    try:
        logger.info("[WS] waiting for header...")
        print("[WS] waiting for header...")
        header_text = await ws.receive_text()
        cfg = json.loads(header_text or "{}")
        logger.info(f"[WS] got header raw={header_text if header_text else '<EMPTY>'}")
        print("[WS] got header")

        # Extra diagnostics on optional header fields
        try:
            _outs = cfg.get("outputs")
            logger.info(f"[WS] header outputs={_outs if _outs is not None else 'None'}")
        except Exception:
            logger.info("[WS] header outputs=<error reading>")

        # --- Optionally extract audio URL for server-driven audio fetch ---
        audio_url: Optional[str] = (cfg.get("audio_url")
                                    or cfg.get("assistant_audio_url")
                                    or cfg.get("url"))
        try:
            logger.info(f"[WS] header audio_url present={bool(audio_url)} len={len(audio_url) if audio_url else 0}")
        except Exception:
            logger.info("[WS] header audio_url present=<err computing length>")
    except Exception as e:
        await _ws_error(ws, f"Invalid header JSON: {e}")
        return
    
    outputs = _normalize_outputs(cfg.get("outputs"))
    outputs.add("pubsub")
    send_to_client = True
    if "none" in outputs or "pubsub_only" in outputs:
        send_to_client = False
    elif "pubsub" in outputs and not (outputs & {"client", "ws", "return"}):
        send_to_client = False
    send_to_pubsub = "pubsub" in outputs or "pubsub_only" in outputs

    AUDIO_CHUNK_TEXT_KEYS = ("text", "audio_text", "chunk_text", "description")
    AUDIO_CHUNK_SENTENCE_KEYS = ("sentence_id", "sentenceId", "sid")
    # Helper to pull any user-provided description that should travel with audio chunks.
    def _pick_audio_chunk_text(src: dict) -> Optional[str]:
        for key in AUDIO_CHUNK_TEXT_KEYS:
            value = src.get(key)
            if value is None:
                continue
            try:
                text = str(value).strip()
            except Exception:
                continue
            if text:
                return text
        return None
    def _pick_audio_chunk_sentence_id(src: dict) -> Optional[str]:
        for key in AUDIO_CHUNK_SENTENCE_KEYS:
            value = src.get(key)
            if value is None:
                continue
            try:
                sid = str(value).strip()
            except Exception:
                continue
            if sid:
                return sid
        return None

    audio_chunk_text = {"value": _pick_audio_chunk_text(cfg)}
    audio_chunk_sentence_id = {"value": _pick_audio_chunk_sentence_id(cfg)}

    def _truthy(v) -> bool:
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

    audio_pubsub = _truthy(
        cfg.get("audio_pubsub")
        or cfg.get("audio_to_pubsub")
        or cfg.get("audio_pub")
        or cfg.get("stream_audio")
    )
    if not audio_pubsub and "audio" in outputs:
        audio_pubsub = True
    if not audio_pubsub and os.getenv("A2F_PUB_AUDIO", "").strip():
        audio_pubsub = _truthy(os.getenv("A2F_PUB_AUDIO"))
    try:
        audio_pub_every = int(cfg.get("audio_pub_every") or os.getenv("A2F_AUDIO_PUB_EVERY", "1"))
    except Exception:
        audio_pub_every = 1
    if audio_pub_every < 1:
        audio_pub_every = 1

    stutter_log = _truthy(os.getenv("A2F_STUTTER_LOG", "1"))
    stutter_gap_ms = int(os.getenv("A2F_STUTTER_GAP_MS", "150"))
    last_send_wall = {"value": None}
    last_t = {"value": None}
    send_count = {"value": 0}
    audio_time_base_bytes = {"value": 0}
    audio_time_reset_pending = {"value": False}
    audio_time_sr_override = {"value": None}
    nested_base_t0 = {"value": None}  # Blendshape timing baseline (reset per turn)
    nested_last_sent = {"value": -1.0}  # Monotonic guard for nested timecodes (reset per turn)
    end_of_audio_sent_at = {"value": None}  # Track when EndOfAudio was sent

    # Track text change markers: (byte_offset, text, sentence_id)
    # When text changes, we record the current audio byte position
    # This allows looking up which text was active at any given timestamp
    audio_text_markers: list = []  # List of (byte_offset, text, sentence_id)

    def _record_text_marker(text: str, sentence_id: str = None, byte_offset: int = None) -> None:
        """Record a text change marker at the specified or current audio byte position."""
        # Use provided byte_offset, or fall back to total_audio_bytes
        if byte_offset is None:
            byte_offset = total_audio_bytes
        audio_text_markers.append((byte_offset, text, sentence_id))
        logger.info(f"[TEXT_MARKER] byte_offset={byte_offset} text={text[:50] if text else 'None'}... sentence_id={sentence_id}")

    def _get_text_for_timestamp(t_seconds: float) -> tuple:
        """Look up the text/sentence_id that corresponds to a given audio timestamp."""
        if not audio_text_markers:
            return None, None
        byte_offset = t_seconds * bytes_per_sec
        # Find the last marker that started before this byte offset
        result_text, result_sentence_id = None, None
        for (marker_offset, text, sentence_id) in audio_text_markers:
            if marker_offset <= byte_offset:
                result_text = text
                result_sentence_id = sentence_id
            else:
                break  # Markers are ordered, so we can stop early
        return result_text, result_sentence_id

    # NOTE: Initial text marker is now recorded in request_gen() when the first audio chunk
    # arrives - this ensures consistent handling and avoids duplicate markers at byte offset 0

    async def _emit(payload: dict) -> None:
        text = json.dumps(payload, ensure_ascii=False)
        if stutter_log and isinstance(payload, dict) and "t" in payload:
            now = time.time()
            t_val = payload.get("t")
            try:
                t_val = float(t_val)
            except Exception:
                t_val = None
            prev_wall = last_send_wall["value"]
            prev_t = last_t["value"]
            if prev_wall is not None and t_val is not None and prev_t is not None:
                wall_gap_ms = int((now - prev_wall) * 1000)
                t_gap_ms = int((t_val - prev_t) * 1000)
                if wall_gap_ms >= stutter_gap_ms:
                    logger.warning(
                        "[WS][STUTTER]"
                        f" type={payload.get('type')}"
                        f" wall_gap_ms={wall_gap_ms}"
                        f" t_gap_ms={t_gap_ms}"
                        f" count={send_count['value']}"
                    )
            last_send_wall["value"] = now
            if t_val is not None:
                last_t["value"] = t_val
            send_count["value"] += 1
        if send_to_client:
            await ws.send_text(text)
        if send_to_pubsub:
            await _publish(conversation_id, text)
    sr = int(cfg.get("sr", 16000))
    ch = int(cfg.get("ch", 1))
    bytes_per_sec = sr * ch * 2
    fmt = (cfg.get("format") or "pcm16").lower()
    if fmt != "pcm16":
        await _ws_error(ws, f"Unsupported format '{fmt}'. Only 'pcm16' is supported.")
        return

    conversation_id: Optional[str] = _extract_conversation_id(cfg)
    if not conversation_id:
        await _ws_error(ws, "Missing 'conversation_id' (required).")
        return

    if audio_pubsub:
        try:
            header_payload = {
                "type": "audio_start",
                "sr": sr,
                "ch": ch,
                "format": fmt,
                "bytes_per_sec": bytes_per_sec,
                "conversation_id": conversation_id,
            }
            text_val = audio_chunk_text.get("value")
            if text_val:
                _AUDIO_TEXT_BY_CONVO[conversation_id] = text_val
                header_payload["text"] = text_val
                _log_sentence(conversation_id, text_val)
            await _publish(conversation_id, json.dumps(header_payload, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"[WS][AUDIO_PUB] failed to publish audio_start: {e}")

    def _resolve_api_key(cfg: dict) -> Optional[str]:
        """
        Resolve NVIDIA Cloud Functions API key from (in order):
        1) WebSocket header JSON field: "apikey"
        2) Environment variables injected by ECS/Copilot: NVCF_API_KEY or NVIDIA_API_KEY
        3) AWS Secrets Manager fallback
        Also supports JSON-formatted secrets such as {"NVCF_API_KEY":"..."}.
        """
        key = (cfg.get("apikey") or os.environ.get("NVCF_API_KEY") or os.environ.get("NVIDIA_API_KEY"))
        if not key:
            # Fallback to Secrets Manager
            return get_nvcf_api_key()
        # If the value looks like JSON, try to extract the token
        k = key.strip()
        if k.startswith("{") and k.endswith("}"):
            try:
                import json as _json
                data = _json.loads(k)
                key = data.get("NVCF_API_KEY") or data.get("token") or data.get("api_key") or key
            except Exception:
                pass
        return key
    apikey: Optional[str] = _resolve_api_key(cfg)
    self_hosted_a2f = _is_self_hosted_a2f()
    function_id: Optional[str] = cfg.get("function_id") or os.environ.get("NVCF_FUNCTION_ID")
    if not self_hosted_a2f and not function_id:
        await _ws_error(ws, "Missing 'function_id' (model function id).")
        return

    upstream_addr = await asyncio.to_thread(_resolve_self_hosted_a2f_addr, function_id) if self_hosted_a2f else NVCF_ENDPOINT
    logger.info(
        f"[WS] header: sr={sr} ch={ch} fmt={fmt} backend={A2F_BACKEND} "
        f"upstream={upstream_addr} function_id={function_id or '-'} "
        f"apikey={'yes' if apikey else 'no'} conversation_id={conversation_id or '-'}"
    )

    metadata = []
    if not self_hosted_a2f:
        if apikey:
            metadata.append(("authorization", f"Bearer {apikey}"))
        metadata.append(("function-id", function_id))

    # Stats for incoming audio
    total_audio_bytes = 0
    total_audio_chunks = 0
    recv_audio_bytes = 0
    recv_audio_chunks = 0
    logged_first_chunk = False

    def _audio_seconds(total_bytes: int) -> float:
        try:
            return float(total_bytes) / float(bytes_per_sec) if bytes_per_sec else 0.0
        except Exception:
            return 0.0
    def _turn_audio_seconds() -> float:
        try:
            base = audio_time_base_bytes["value"] or 0
            turn_bytes = total_audio_bytes - base
            if turn_bytes < 0:
                turn_bytes = 0
            return _audio_seconds(turn_bytes)
        except Exception:
            return 0.0

    # --- 2) Open gRPC channel/stub (async) ---
    if self_hosted_a2f:
        channel = grpc.aio.insecure_channel(upstream_addr)
    else:
        channel = grpc.aio.secure_channel(upstream_addr, grpc.ssl_channel_credentials())
    stub = a2f_ctrl_grpc.A2FControllerServiceStub(channel)

    # Queue for audio chunks coming from WS (chunk, text_snapshot)
    audio_q: asyncio.Queue[Optional[tuple[bytes, Optional[str], Optional[str]]]] = asyncio.Queue()
    # Idle-flush settings: if no audio arrives for FLUSH_IDLE_SEC, auto-send EndOfAudio
    FLUSH_IDLE_SEC = float(os.environ.get("A2F_FLUSH_IDLE_SEC", "1.5"))
    last_audio_ts = asyncio.get_event_loop().time()
    # Verbose WS receive logging controls
    WS_LOG_RECV = os.getenv("A2F_WS_LOG_RECV", "0").lower() in ("1", "true", "yes")
    WS_LOG_EVERY = int(os.getenv("A2F_WS_LOG_EVERY", "25"))

    # Per-sentence tracking
    sentence_audio_bytes = {}  # sentence_id -> total bytes
    sentence_frame_count = {}  # sentence_id -> frame count (estimated by timing)
    total_synced_audio_bytes = 0  # Track NVIDIA's synchronized audio

    # Buffer for retry: store all audio chunks sent to NVIDIA for potential re-request
    # Each entry: (byte_offset, chunk_bytes)
    audio_buffer_for_retry: list = []
    RETRY_COVERAGE_THRESHOLD = float(os.getenv("A2F_RETRY_COVERAGE_THRESHOLD", "0.90"))  # Retry if < 90%
    MAX_RETRY_ATTEMPTS = int(os.getenv("A2F_MAX_RETRY_ATTEMPTS", "2"))
    FIRST_PCM_TIMEOUT_SEC = float(os.getenv("A2F_FIRST_PCM_TIMEOUT_SEC", "2.0"))
    FIRST_PCM_RETRY_MAX_ATTEMPTS = int(os.getenv("A2F_FIRST_PCM_RETRY_MAX_ATTEMPTS", "2"))
    FIRST_PCM_POLL_SEC = float(os.getenv("A2F_FIRST_PCM_POLL_SEC", "0.25"))
    if FIRST_PCM_TIMEOUT_SEC <= 0:
        FIRST_PCM_TIMEOUT_SEC = 2.0
    if FIRST_PCM_POLL_SEC <= 0:
        FIRST_PCM_POLL_SEC = 0.25
    if FIRST_PCM_RETRY_MAX_ATTEMPTS < 0:
        FIRST_PCM_RETRY_MAX_ATTEMPTS = 0
    first_pcm_retry_state = {
        "text": None,
        "sentence_id": None,
        "armed_at": None,
        "attempts": 0,
        "armed_recv_audio_chunks": 0,
    }

    def _arm_first_pcm_retry(
        text_value: Optional[str] = None,
        sentence_id_value: Optional[str] = None,
    ) -> None:
        if use_server_audio:
            return
        text_resolved = str(
            text_value
            or audio_chunk_text.get("value")
            or _AUDIO_TEXT_BY_CONVO.get(conversation_id)
            or ""
        ).strip()
        if not text_resolved:
            return
        sentence_resolved = str(
            sentence_id_value
            or audio_chunk_sentence_id.get("value")
            or _AUDIO_SENTENCE_BY_CONVO.get(conversation_id)
            or ""
        ).strip()
        is_same_target = (
            first_pcm_retry_state["text"] == text_resolved
            and first_pcm_retry_state["sentence_id"] == sentence_resolved
        )
        first_pcm_retry_state["text"] = text_resolved
        first_pcm_retry_state["sentence_id"] = sentence_resolved or None
        first_pcm_retry_state["armed_at"] = time.monotonic()
        first_pcm_retry_state["armed_recv_audio_chunks"] = recv_audio_chunks
        if not is_same_target:
            first_pcm_retry_state["attempts"] = 0
        logger.info(
            "[WS][PCM_RETRY] armed convo=%s sentence_id=%s attempts=%s text=%s",
            conversation_id,
            first_pcm_retry_state["sentence_id"] or "",
            first_pcm_retry_state["attempts"],
            text_resolved[:80],
        )

    def _clear_first_pcm_retry(reason: str) -> None:
        if first_pcm_retry_state["armed_at"] is not None:
            logger.info(
                "[WS][PCM_RETRY] cleared convo=%s reason=%s attempts=%s sentence_id=%s",
                conversation_id,
                reason,
                first_pcm_retry_state["attempts"],
                first_pcm_retry_state["sentence_id"] or "",
            )
        first_pcm_retry_state["text"] = None
        first_pcm_retry_state["sentence_id"] = None
        first_pcm_retry_state["armed_at"] = None
        first_pcm_retry_state["attempts"] = 0
        first_pcm_retry_state["armed_recv_audio_chunks"] = recv_audio_chunks

    async def _request_first_pcm_retry(*, final: bool) -> None:
        text_val = str(first_pcm_retry_state.get("text") or "").strip()
        if not text_val:
            return
        attempt = int(first_pcm_retry_state.get("attempts") or 0) + 1
        sentence_id_val = first_pcm_retry_state.get("sentence_id")
        payload = {
            "type": "retry_audio_request",
            "conversation_id": conversation_id,
            "reason": "no_pcm_after_text",
            "text": text_val,
            "attempt": attempt,
            "max_attempts": FIRST_PCM_RETRY_MAX_ATTEMPTS,
            "retriable": not final,
            "ts_ms": int(time.time() * 1000),
        }
        if sentence_id_val:
            payload["sentence_id"] = sentence_id_val
        if final:
            payload["final"] = True
            payload["error"] = "Timed out waiting for PCM after text control payloads."
        try:
            await ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            logger.warning(f"[WS][PCM_RETRY] failed to notify upstream retry convo={conversation_id}: {exc}")
        try:
            await _publish(conversation_id, json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            logger.warning(f"[WS][PCM_RETRY] failed to publish retry request convo={conversation_id}: {exc}")
        first_pcm_retry_state["attempts"] = attempt
        first_pcm_retry_state["armed_at"] = time.monotonic()
        logger.warning(
            "[WS][PCM_RETRY] convo=%s sentence_id=%s attempt=%s/%s final=%s text=%s",
            conversation_id,
            sentence_id_val or "",
            attempt,
            FIRST_PCM_RETRY_MAX_ATTEMPTS,
            final,
            text_val[:80],
        )

    # --- Helper: ffmpeg audio URL fetcher ---
    async def _pump_ffmpeg_audio(url: str, sr: int, ch: int):
        """
        Use ffmpeg to pull and resample remote audio to PCM16 s16le and enqueue to audio_q.
        This lets the server fetch S3/HTTP audio and stream visemes without the browser sending PCM.
        """
        if not url:
            return
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel", "error",
            "-i", url,
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ac", str(ch),
            "-ar", str(sr),
            "pipe:1",
        ]
        logger.info(f"[WS] ffmpeg start: {' '.join(cmd[:-1])} pipe:1")
        try:
            proc = await asp.create_subprocess_exec(
                *cmd,
                stdout=asp.PIPE,
                stderr=asp.PIPE
            )
            logger.info(f"[WS][DIAG] ffmpeg fetching url_len={len(url)} url_head={url[:120]}")
        except Exception as e:
            logger.error(f"[WS] ffmpeg spawn failed: {e}")
            await _ws_error(ws, f"ffmpeg spawn failed: {e}")
            return
        try:
            first = True
            count = 0
            total = 0

            # --- realtime pacing so A2F doesn't get entire sentence at once ---
            bytes_per_sec = sr * ch * 2  # pcm16 = 2 bytes/sample
            sent_audio_sec = 0.0
            loop = asyncio.get_event_loop()
            t0 = loop.time()
            # Enable ffmpeg realtime pacing to prevent overwhelming NVIDIA
            PACE_REALTIME = os.getenv("A2F_PACE_REALTIME", "0").lower() in ("1", "true", "yes")
            REALTIME_MULTIPLIER = float(os.getenv("A2F_REALTIME_MULTIPLIER", "1.0"))

            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break

                count += 1
                total += len(chunk)

                if first:
                    logger.info(f"[WS] ffmpeg first chunk bytes={len(chunk)}")
                    first = False
                elif count % 50 == 0:
                    logger.info(f"[WS] ffmpeg chunks={count} bytes={total}")

                text_snapshot = audio_chunk_text.get("value") or _AUDIO_TEXT_BY_CONVO.get(conversation_id)
                sentence_snapshot = audio_chunk_sentence_id.get("value") or _AUDIO_SENTENCE_BY_CONVO.get(conversation_id)
                await audio_q.put((chunk, text_snapshot, sentence_snapshot))

                if PACE_REALTIME:
                    sent_audio_sec += len(chunk) / bytes_per_sec
                    # Apply rate multiplier (e.g., 0.5 = half speed, sends slower)
                    target_t = t0 + (sent_audio_sec / REALTIME_MULTIPLIER)
                    now_t = loop.time()
                    sleep_s = target_t - now_t
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)
                    if count % 100 == 0:
                        drift = now_t - target_t
                        logger.info(f"[WS][PACE] sent_audio_sec={sent_audio_sec:.3f} drift={drift:.3f}s multiplier={REALTIME_MULTIPLIER}")

            rc = await proc.wait()
            # Capture stderr for debugging URL/auth/codec issues
            try:
                err_bytes = await proc.stderr.read()
                err_txt = err_bytes.decode(errors="ignore") if err_bytes else ""
            except Exception as _se:
                err_txt = f"<stderr read failed: {_se}>"
            logger.info(f"[WS] ffmpeg finished rc={rc} chunks={count} bytes={total}")
            if err_txt.strip():
                logger.warning(f"[WS] ffmpeg stderr (preview): {err_txt[:400]}")
            if total == 0:
                logger.error(f"[WS][DIAG] ffmpeg produced 0 bytes from url={url!r}")
        except Exception as e:
            logger.error(f"[WS] ffmpeg stream error: {e}")
        finally:
            # signal end-of-audio to request_gen()
            await audio_q.put(None)

    async def request_gen() -> AsyncIterator[ctrl_pb2.AudioStream]:
        """Yield AudioStream messages: header -> many chunks -> end_of_audio."""
        print("[gRPC] request_gen start")
        nonlocal logged_first_chunk, last_audio_ts, total_audio_bytes, total_audio_chunks

        CHUNK_PACING_FPS = 0  # Deprecated
        CHUNK_INTERVAL = 0

        # Calculate fixed chunk size: amount of audio data in CHUNK_INTERVAL seconds
        # bytes = seconds * sample_rate * channels * bytes_per_sample
        FIXED_CHUNK_SIZE = int(CHUNK_INTERVAL * sr * ch * 2) if CHUNK_PACING_FPS > 0 else 0

        # Use field names that are known-good with the published wheel:
        # samples_per_second, channel_count, bits_per_sample, audio_format=AUDIO_FORMAT_PCM
        # (The 'sample_rate_hz/num_channels/PCM16' schema exists in other builds but is not universal.)
        ah = audio_pb2.AudioHeader(
            samples_per_second=sr,
            channel_count=ch,
            bits_per_sample=16,
            audio_format=getattr(audio_pb2.AudioHeader, "AUDIO_FORMAT_PCM", 0)  # fallback to enum 0 if symbol differs
        )
        yield ctrl_pb2.AudioStream(
            audio_stream_header=ctrl_pb2.AudioStreamHeader(audio_header=ah)
        )
        print(f"[gRPC] yielded AudioStreamHeader sr={sr} ch={ch}")

        if CHUNK_PACING_FPS > 0:
            logger.info(f"[gRPC] Chunk pacing enabled: {CHUNK_PACING_FPS} FPS ({CHUNK_INTERVAL*1000:.1f}ms intervals, {FIXED_CHUNK_SIZE} bytes/chunk)")

        # Buffer for accumulating audio to send in fixed-size chunks
        audio_buffer = bytearray()
        buffer_text = None
        buffer_sentence_id = None
        last_chunk_send_time = None

        # Track last recorded text/sentence to detect changes and record markers at correct byte offset
        last_marker_text = None
        last_marker_sentence_id = None

        while True:
            item = await audio_q.get()
            if item is None:
                # Send any remaining buffered data
                if audio_buffer and CHUNK_PACING_FPS > 0:
                    if last_chunk_send_time is not None:
                        elapsed = asyncio.get_event_loop().time() - last_chunk_send_time
                        wait_time = CHUNK_INTERVAL - elapsed
                        if wait_time > 0:
                            await asyncio.sleep(wait_time)

                    last_audio_ts = asyncio.get_event_loop().time()
                    total_audio_bytes += len(audio_buffer)
                    total_audio_chunks += 1

                    msg = ctrl_pb2.AudioStream(
                        audio_stream_data=ctrl_pb2.AudioStreamData(audio_buffer=bytes(audio_buffer))
                    )
                    yield msg
                    logger.info(f"[gRPC][FINAL] sent final buffered chunk: {len(audio_buffer)} bytes")
                break

            chunk, chunk_text, chunk_sentence_id = item

            if CHUNK_PACING_FPS > 0 and FIXED_CHUNK_SIZE > 0:
                # Option A: Accumulate until we have FIXED_CHUNK_SIZE bytes, then send
                # Record text marker when text/sentence changes - BEFORE adding to buffer
                text_changed = (chunk_text != last_marker_text) if chunk_text else False
                sentence_changed = (chunk_sentence_id != last_marker_sentence_id) if chunk_sentence_id else False
                if text_changed or sentence_changed:
                    _record_text_marker(chunk_text, chunk_sentence_id, byte_offset=total_audio_bytes)
                    last_marker_text = chunk_text
                    last_marker_sentence_id = chunk_sentence_id

                audio_buffer.extend(chunk)
                if buffer_text is None:
                    buffer_text = chunk_text
                if buffer_sentence_id is None:
                    buffer_sentence_id = chunk_sentence_id

                # Send fixed-size chunks when buffer is full
                while len(audio_buffer) >= FIXED_CHUNK_SIZE:
                    # Wait for pacing interval
                    if last_chunk_send_time is not None:
                        elapsed = asyncio.get_event_loop().time() - last_chunk_send_time
                        wait_time = CHUNK_INTERVAL - elapsed
                        if wait_time > 0:
                            await asyncio.sleep(wait_time)
                            if total_audio_chunks % 50 == 0:
                                logger.info(f"[gRPC][PACE] waited {wait_time*1000:.1f}ms before chunk#{total_audio_chunks + 1}")

                    # Extract fixed-size chunk from buffer
                    to_send = bytes(audio_buffer[:FIXED_CHUNK_SIZE])
                    audio_buffer = audio_buffer[FIXED_CHUNK_SIZE:]

                    # refresh idle timer
                    last_audio_ts = asyncio.get_event_loop().time()
                    last_chunk_send_time = asyncio.get_event_loop().time()

                    # Buffer for retry
                    audio_buffer_for_retry.append((total_audio_bytes, bytes(to_send)))

                    # Stats
                    total_audio_bytes += len(to_send)
                    total_audio_chunks += 1
                    if total_audio_chunks % 50 == 0:
                        logger.info(f"[WS←Client] audio: chunks={total_audio_chunks} bytes={total_audio_bytes}")
                    if total_audio_chunks % 25 == 0:
                        print(f"[WS] audio chunks={total_audio_chunks} bytes={total_audio_bytes}")

                    # audio_chunk publishing removed - now using synced_audio_chunk from blendshapes

                    # Yield to NVIDIA immediately (can't defer to common code due to while loop)
                    test_msg = _build_awe(to_send, conversation_id)
                    yield ctrl_pb2.AudioStream(audio_with_emotion=test_msg)

            else:
                # Option B / disabled: Send variable-size chunks as they arrive
                # refresh idle timer
                last_audio_ts = asyncio.get_event_loop().time()

                # Record text marker when text/sentence changes - BEFORE incrementing total_audio_bytes
                # This ensures the marker is at the byte offset where this text's audio STARTS
                text_changed = (chunk_text != last_marker_text) if chunk_text else False
                sentence_changed = (chunk_sentence_id != last_marker_sentence_id) if chunk_sentence_id else False
                if text_changed or sentence_changed:
                    _record_text_marker(chunk_text, chunk_sentence_id, byte_offset=total_audio_bytes)
                    last_marker_text = chunk_text
                    last_marker_sentence_id = chunk_sentence_id

                # Buffer for retry
                audio_buffer_for_retry.append((total_audio_bytes, bytes(chunk)))

                # Stats
                total_audio_bytes += len(chunk)
                total_audio_chunks += 1

                # Track per-sentence audio bytes
                if chunk_sentence_id:
                    sentence_audio_bytes[chunk_sentence_id] = sentence_audio_bytes.get(chunk_sentence_id, 0) + len(chunk)

                if total_audio_chunks % 50 == 0:
                    logger.info(f"[WS←Client] audio: chunks={total_audio_chunks} bytes={total_audio_bytes}")

                if total_audio_chunks % 25 == 0:
                    print(f"[WS] audio chunks={total_audio_chunks} bytes={total_audio_bytes}")

                # audio_chunk publishing removed - now using synced_audio_chunk from blendshapes

                # Send chunk to NVIDIA immediately
                test_msg = _build_awe(chunk, conversation_id)
                yield ctrl_pb2.AudioStream(audio_with_emotion=test_msg)

        # End
        logger.info(
            f"[gRPC][DIAG] EndOfAudio: total_chunks={total_audio_chunks} total_bytes={total_audio_bytes} "
            f"audio_sec={_audio_seconds(total_audio_bytes):.3f}"
        )
        print("[gRPC] sending EndOfAudio() to A2F")
        if audio_pubsub:
            try:
                await _publish(
                    conversation_id,
                    json.dumps(
                        {
                            "type": "audio_end",
                            "sr": sr,
                            "ch": ch,
                            "format": "pcm16",
                            "conversation_id": conversation_id,
                            "seq": total_audio_chunks,
                        },
                        ensure_ascii=False,
                    ),
                )
            except Exception as e:
                logger.warning(f"[WS][AUDIO_PUB] failed to publish audio_end: {e}")
        end_of_audio_msg = ctrl_pb2.AudioStream(end_of_audio=ctrl_pb2.AudioStream.EndOfAudio())
        end_of_audio_sent_at["value"] = time.time()
        yield end_of_audio_msg

    # Shared counter to know how many frames we actually emitted back to the client
    frame_counter = {"count": 0}

    async def retry_uncovered_audio(
        covered_seconds: float,
        total_seconds: float,
        attempt: int = 1
    ) -> bool:
        """
        Retry processing for audio that wasn't covered by blendshapes.
        Returns True if retry was successful (or not needed), False if failed.
        """
        nonlocal total_synced_audio_bytes

        if attempt > MAX_RETRY_ATTEMPTS:
            logger.warning(f"[RETRY] Max retry attempts ({MAX_RETRY_ATTEMPTS}) reached, giving up")
            return False

        coverage_ratio = covered_seconds / total_seconds if total_seconds > 0 else 1.0
        if coverage_ratio >= RETRY_COVERAGE_THRESHOLD:
            logger.info(f"[RETRY] Coverage {coverage_ratio:.1%} >= threshold {RETRY_COVERAGE_THRESHOLD:.1%}, no retry needed")
            return True

        # Calculate which bytes to retry
        covered_bytes = int(covered_seconds * bytes_per_sec)
        # Align to PCM16 sample boundary to avoid mid-sample slicing noise.
        if covered_bytes % 2 != 0:
            covered_bytes -= (covered_bytes % 2)
            logger.info(f"[RETRY] Aligning covered_bytes to 2-byte boundary: {covered_bytes}")
        total_bytes_sent = total_audio_bytes
        uncovered_bytes = total_bytes_sent - covered_bytes

        if uncovered_bytes <= 0:
            logger.info(f"[RETRY] No uncovered bytes to retry (covered={covered_bytes}, total={total_bytes_sent})")
            return True

        # Extract uncovered audio from buffer
        retry_audio = bytearray()
        for (offset, chunk_bytes) in audio_buffer_for_retry:
            chunk_end = offset + len(chunk_bytes)
            if chunk_end <= covered_bytes:
                continue  # This chunk was fully covered
            if offset >= covered_bytes:
                # Entire chunk is uncovered
                retry_audio.extend(chunk_bytes)
            else:
                # Partial overlap - take only the uncovered part
                uncovered_start = covered_bytes - offset
                retry_audio.extend(chunk_bytes[uncovered_start:])

        if len(retry_audio) == 0:
            logger.warning(f"[RETRY] No audio bytes found for retry (covered_bytes={covered_bytes})")
            return False

        retry_duration = len(retry_audio) / bytes_per_sec
        logger.info(
            f"[RETRY] Attempt {attempt}: Retrying {len(retry_audio)} bytes ({retry_duration:.3f}s) "
            f"starting from t={covered_seconds:.3f}s"
        )

        # Create a new request generator for retry
        async def retry_request_gen():
            # Send header
            ah = audio_pb2.AudioHeader(
                samples_per_second=sr,
                channel_count=ch,
                bits_per_sample=16,
                audio_format=getattr(audio_pb2.AudioHeader, "AUDIO_FORMAT_PCM", 0)
            )
            yield ctrl_pb2.AudioStream(
                audio_stream_header=ctrl_pb2.AudioStreamHeader(audio_header=ah)
            )

            # Send retry audio in chunks
            RETRY_CHUNK_SIZE = 4096
            for i in range(0, len(retry_audio), RETRY_CHUNK_SIZE):
                chunk = bytes(retry_audio[i:i + RETRY_CHUNK_SIZE])
                msg = a2f_pb2.AudioWithEmotion(audio_buffer=chunk)
                yield ctrl_pb2.AudioStream(audio_with_emotion=msg)

            # Send end of audio
            yield ctrl_pb2.AudioStream(end_of_audio=ctrl_pb2.AudioStream.EndOfAudio())
            logger.info(f"[RETRY] Sent EndOfAudio for retry attempt {attempt}")

        # Process retry responses
        retry_frame_count = 0
        retry_last_t = 0.0
        timestamp_offset = covered_seconds  # Offset to add to retry timestamps

        try:
            async for resp in stub.ProcessAudioStream(retry_request_gen(), metadata=metadata):
                resp_dict = _proto_to_dict(resp)

                # Check for animation data
                if "animation_data" not in resp_dict:
                    continue

                retry_frame_count += 1
                ad = resp_dict.get("animation_data", {})

                # Get timestamp from response
                audio_obj = ad.get("audio") or {}
                raw_t = audio_obj.get("time_code", 0.0)
                try:
                    t_local = float(raw_t)
                except:
                    t_local = 0.0

                # Apply timestamp offset
                t_adjusted = t_local + timestamp_offset
                retry_last_t = max(retry_last_t, t_adjusted)

                # Extract and emit synced audio with adjusted timestamp
                audio_buffer_b64 = audio_obj.get("audio_buffer")
                if audio_buffer_b64 and isinstance(audio_buffer_b64, str):
                    try:
                        import base64
                        synced_bytes = base64.b64decode(audio_buffer_b64)
                        total_synced_audio_bytes += len(synced_bytes)

                        # Look up text for adjusted timestamp
                        retry_text, retry_sentence_id = _get_text_for_timestamp(t_adjusted)

                        synced_payload = {
                            "type": "synced_audio_chunk",
                            "sr": sr,
                            "ch": ch,
                            "format": "pcm16",
                            "pcm": audio_buffer_b64,
                            "t": t_adjusted,
                            "conversation_id": conversation_id,
                            "frame": frame_counter["count"] + retry_frame_count,
                            "retry": True,
                        }
                        if retry_text:
                            synced_payload["text"] = retry_text
                        if retry_sentence_id:
                            synced_payload["sentence_id"] = retry_sentence_id
                        await _publish(conversation_id, json.dumps(synced_payload, ensure_ascii=False))
                    except Exception as e:
                        logger.warning(f"[RETRY] Failed to process synced audio: {e}")

                # Extract and emit blendshapes with adjusted timestamp
                skel = ad.get("skel_animation", {})
                bsw = skel.get("blend_shape_weights", [])
                if bsw:
                    items = bsw if isinstance(bsw, list) else [bsw]
                    for it in items:
                        vals = it.get("values") or it.get("weights") or []
                        if not vals:
                            continue
                        item_t = it.get("time_code", t_local)
                        try:
                            item_t = float(item_t)
                        except:
                            item_t = t_local
                        t_item_adjusted = item_t + timestamp_offset

                        # Look up text for this timestamp
                        item_text, item_sid = _get_text_for_timestamp(t_item_adjusted)

                        # Build shapes dict
                        if blend_names and len(blend_names) == len(vals):
                            shapes = {blend_names[i]: float(vals[i]) for i in range(len(vals))}
                        elif len(DEFAULT_A2F_NAMES) == len(vals):
                            shapes = {DEFAULT_A2F_NAMES[i]: float(vals[i]) for i in range(len(vals))}
                        else:
                            shapes = {str(i): float(v) for i, v in enumerate(vals)}

                        out_bs = {
                            "type": "blendshapes_frame",
                            "t": t_item_adjusted,
                            "shapes": shapes,
                            "conversation_id": conversation_id,
                            "retry": True,
                        }
                        if item_text:
                            out_bs["text"] = item_text
                        if item_sid:
                            out_bs["sentence_id"] = item_sid
                        await _emit(out_bs)

                        # Also emit visemes
                        visemes_out = [{"idx": i, "w": float(w)} for i, w in enumerate(vals)]
                        out_v = {
                            "type": "visemes_frame",
                            "t": t_item_adjusted,
                            "visemes": visemes_out,
                            "conversation_id": conversation_id,
                            "retry": True,
                        }
                        if item_text:
                            out_v["text"] = item_text
                        if item_sid:
                            out_v["sentence_id"] = item_sid
                        await _emit(out_v)

                        frame_counter["last_blendshape_t"] = t_item_adjusted
                        frame_counter["last_viseme_t"] = t_item_adjusted

            logger.info(
                f"[RETRY] Attempt {attempt} completed: {retry_frame_count} frames, "
                f"last_t={retry_last_t:.3f}s (target={total_seconds:.3f}s)"
            )

            # Check if we need another retry
            new_coverage = retry_last_t / total_seconds if total_seconds > 0 else 1.0
            if new_coverage < RETRY_COVERAGE_THRESHOLD:
                logger.warning(
                    f"[RETRY] Coverage still low after attempt {attempt}: {new_coverage:.1%}, "
                    f"trying again..."
                )
                return await retry_uncovered_audio(retry_last_t, total_seconds, attempt + 1)

            return True

        except Exception as e:
            logger.error(f"[RETRY] Error during retry attempt {attempt}: {e}", exc_info=True)
            return False

    # Store blend_names at module level for retry function
    blend_names = None

    async def pump_responses():
        """Forward A2F responses to the WS as compact JSON frames."""
        nonlocal total_synced_audio_bytes
        frame_idx = 0
        # reflect frames outward via shared dict
        frame_counter["count"] = 0
        blend_names = None
        sent_names = False
        # nested_base_t0 is now defined in outer scope for new_turn reset support
        nested_last_sent["value"] = -1.0
        # --- Attempt to fetch blend shape names from DescribeFunction before streaming ---
        pre_names = await fetch_nvidia_blendshape_names(function_id, apikey)
        if pre_names and len(pre_names) > 0:
            blend_names = pre_names
            _payload = {"type": "names", "names": list(blend_names)}
            if conversation_id:
                _payload["conversation_id"] = conversation_id
            # Only log for names frames if requested (not required by instructions)
            await _emit(_payload)
            sent_names = True
            logger.info(f"[DescribeFunction] Sent blend shape names ({len(blend_names)}) to client.")
        else:
            logger.info("[≈DescribeFunction] No blend shape names returned; will use header or default list.")
        try:
            # --- verbose: log call configuration once before starting stream ---
            try:
                _md_preview = [(k, ("***" if k.lower() == "authorization" else v)) for (k, v) in (metadata or [])]
                logger.info(f"[gRPC] invoking ProcessAudioStream; function_id={function_id} metadata={_md_preview}")
            except Exception:
                logger.info("[gRPC] invoking ProcessAudioStream (metadata preview failed)")

            LOG_JSON_PREVIEW = int(os.environ.get("A2F_LOG_JSON_PREVIEW", "8192"))
            LOG_EVERY_N = int(os.environ.get("A2F_LOG_EVERY_N", "1"))  # log each response by default

            logger.info("[gRPC][DIAG] ProcessAudioStream started; awaiting responses...")
            last_frame_at = time.time()
            stream_start_at = time.time()
            NVIDIA_TIMEOUT_AFTER_EOA = float(os.getenv("A2F_NVIDIA_TIMEOUT", "10.0"))

            async for resp in stub.ProcessAudioStream(request_gen(), metadata=metadata):
                now = time.time()
                gap = now - last_frame_at

                # Detect if NVIDIA stopped responding after EndOfAudio
                if end_of_audio_sent_at["value"] and (now - end_of_audio_sent_at["value"]) > NVIDIA_TIMEOUT_AFTER_EOA:
                    turn_audio_sec = _turn_audio_seconds()
                    session_audio_sec = _audio_seconds(total_audio_bytes)
                    logger.error(
                        f"[NVIDIA][TIMEOUT] No frames for {now - end_of_audio_sent_at['value']:.1f}s "
                        f"after EndOfAudio. NVIDIA likely stopped processing. "
                        f"Coverage: {frame_idx} frames, ~{frame_counter.get('last_blendshape_t', 0):.1f}s / "
                        f"{turn_audio_sec:.1f}s turn audio (session {session_audio_sec:.1f}s)"
                    )
                    break  # Stop waiting, NVIDIA isn't sending more

                if gap > 0.5:  # Log gaps > 500ms between frames
                    logger.warning(f"[NVIDIA][RESP] ⚠️ gap={gap:.3f}s since last frame")

                # Track if we're receiving frames after EndOfAudio (turn processing)
                after_eoa = ""
                if end_of_audio_sent_at["value"]:
                    time_since_eoa = now - end_of_audio_sent_at["value"]
                    after_eoa = f" [+{time_since_eoa:.1f}s after EOA]"

                if frame_idx > 0 and frame_idx % 50 == 0:
                    logger.info(f"[NVIDIA][RESP] received frame#{frame_idx + 1} (stream_elapsed={now - stream_start_at:.1f}s){after_eoa}")
                else:
                    logger.info(f"[NVIDIA][RESP] received frame#{frame_idx + 1}{after_eoa}")
                last_frame_at = now
                emitted_nested = False
                # Which oneof is active? (try multiple likely container names)
                oneof_name = None
                for maybe in ("payload", "response", "streaming_response", "result", "message"):
                    if hasattr(resp, "WhichOneof"):
                        try:
                            val = resp.WhichOneof(maybe)
                            if val:
                                oneof_name = maybe + ":" + val
                                break
                        except Exception:
                            pass

                # List explicitly present fields:
                try:
                    present_fields = [f[0].name for f in resp.ListFields()]
                except Exception:
                    present_fields = ["<ListFields failed>"]

                # Convert to dict/JSON for full visibility
                resp_dict = _proto_to_dict(resp)
                # Check for NVIDIA-provided text first (properly timed by NVIDIA)
                frame_text = (
                    resp_dict.get("text")
                    or resp_dict.get("description")
                    or resp_dict.get("audio_text")
                    or resp_dict.get("chunk_text")
                    or resp_dict.get("caption")
                )
                # Note: timestamp-based text lookup happens after base_t is computed below


                # DEBUG ONLY: forward protobuf repr text.
                # IMPORTANT: Only skip further parsing if repr is the *only* content.
                if "_protobuf_repr" in resp_dict and isinstance(resp_dict["_protobuf_repr"], str):
                    out = {
                        "type": "protobuf_repr",
                        "t": 0.0,
                        "text": resp_dict["_protobuf_repr"],
                    }
                    if conversation_id:
                        out["conversation_id"] = conversation_id
                    await _emit(out)

                    # If this dict contains nothing besides repr/error, it’s a true repr-only fallback.
                    other_keys = [k for k in resp_dict.keys() if k not in ("_protobuf_repr", "_protobuf_fallback", "_error")]
                    if not other_keys:
                        continue
                try:
                    raw_json = json.dumps(resp_dict, ensure_ascii=False)
                    preview = (raw_json[:LOG_JSON_PREVIEW] + ("…[truncated]" if len(raw_json) > LOG_JSON_PREVIEW else ""))
                    if VERBOSE_LOGS:
                        logger.info(f"[gRPC] resp#{frame_idx} oneof={oneof_name} fields={present_fields} json_len={len(raw_json)}")
                        logger.info(f"[gRPC] resp#{frame_idx} json={preview}")
                except Exception as e:
                    if VERBOSE_LOGS:
                        logger.warning(f"[gRPC] resp#{frame_idx} json dump failed: {e}; dict_keys={list(resp_dict.keys())}")

                # Heuristics: count frames if we see anything that looks like animation/viseme data
                # This keeps external counters meaningful even if schema differs.
                looks_like_frame = False
                try:
                    # Common patterns we might see:
                    #  - resp.animation / resp.animation_data / resp.blendshape_weights
                    #  - arrays named 'weights', 'visemes', 'blendshapes', 'tracks'
                    j = resp_dict
                    if any(k in j for k in ("animation", "animation_data", "blendshape_weights", "visemes", "tracks", "weights", "frame")):
                        looks_like_frame = True
                except Exception:
                    pass

                if looks_like_frame:
                    frame_idx += 1
                    frame_counter["count"] = frame_idx

                    # --- derive a timestamp 't' (seconds) from NVIDIA-provided time codes ---
                    # Prefer explicit t/timestamp in the dict, then audio.time_code, else 0.0
                    audio_obj = (resp_dict.get("animation_data") or {}).get("audio") or resp_dict.get("audio") or {}
                    audio_tc = audio_obj.get("time_code")
                    base_t = 0.0
                    if audio_tc is not None:
                        try:
                            base_t = float(audio_tc)
                        except Exception:
                            base_t = 0.0

                    # Look up text/sentence_id for this frame's timestamp if not provided by NVIDIA
                    frame_sentence_id = None
                    if not frame_text:
                        looked_up_text, looked_up_sentence_id = _get_text_for_timestamp(base_t)
                        if looked_up_text:
                            frame_text = looked_up_text
                        if looked_up_sentence_id:
                            frame_sentence_id = looked_up_sentence_id
                    # Resolve sentence_id once here so it's available for both synced_audio_chunk and blendshapes_frame
                    current_sentence_id = frame_sentence_id or audio_chunk_sentence_id.get("value") or _AUDIO_SENTENCE_BY_CONVO.get(conversation_id)

                    # --- Extract and publish NVIDIA's synchronized audio ---
                    audio_buffer_b64 = audio_obj.get("audio_buffer")
                    if audio_buffer_b64 and isinstance(audio_buffer_b64, str):
                        try:
                            import base64
                            synced_audio_bytes = base64.b64decode(audio_buffer_b64)
                            synced_audio_b64 = audio_buffer_b64  # Already base64
                            total_synced_audio_bytes += len(synced_audio_bytes)

                            # Publish synchronized audio chunk with subtitle text
                            synced_payload = {
                                "type": "synced_audio_chunk",
                                "sr": sr,
                                "ch": ch,
                                "format": "pcm16",
                                "pcm": synced_audio_b64,
                                "t": base_t,
                                "conversation_id": conversation_id,
                                "frame": frame_idx,
                            }
                            # Include text for subtitles (now properly synced to audio timestamp)
                            if frame_text:
                                synced_payload["text"] = frame_text
                            # Include sentence_id if available (prefer timestamp-based lookup)
                            if current_sentence_id:
                                synced_payload["sentence_id"] = current_sentence_id
                            await _publish(conversation_id, json.dumps(synced_payload, ensure_ascii=False))

                            if frame_idx <= 3:
                                logger.info(
                                    f"[NVIDIA][SYNCED_AUDIO] frame#{frame_idx} t={base_t:.3f}s bytes={len(synced_audio_bytes)} "
                                    f"(NVIDIA's synchronized audio for this frame)"
                                )
                        except Exception as e:
                            logger.warning(f"[NVIDIA][SYNCED_AUDIO] Failed to decode audio_buffer: {e}")

                    raw_t = resp_dict.get("t") or resp_dict.get("timestamp") or base_t
                    try:
                        t_val = float(raw_t)
                    except Exception:
                        t_val = base_t
                        if VERBOSE_LOGS:
                            logger.info(
                                f"[TS][FLAT] frame#{frame_idx} audio_tc={audio_tc if audio_tc is not None else 'None'} raw_t={raw_t} t_val={t_val}"
                            )

                    # --- try to emit BLENDSHAPES in production shape ---
                    # Accept either a dict {"JawOpen":0.3,...} or a list aligned with blend_names
                    shapes = None
                    if "blendshape_weights" in resp_dict:
                        bs = resp_dict["blendshape_weights"]
                        if isinstance(bs, dict):
                            shapes = bs
                        elif isinstance(bs, list):
                            if blend_names and len(blend_names) == len(bs):
                                shapes = {blend_names[i]: float(bs[i]) for i in range(len(bs))}
                            else:
                                # fall back to indexed names
                                shapes = {str(i): float(w) for i, w in enumerate(bs)}
                    elif "animation_data" in resp_dict and isinstance(resp_dict["animation_data"], dict):
                        # Some pipelines put blendshapes under animation_data.shapes
                        shapes = resp_dict["animation_data"].get("shapes")

                    if (not emitted_nested) and shapes:
                        # Apply baseline correction (same as nested path)
                        raw_t_flat = float(t_val)
                        if nested_base_t0["value"] is None:
                            nested_base_t0["value"] = raw_t_flat
                        t_float = raw_t_flat - nested_base_t0["value"]

                        out = {"type": "blendshapes_frame", "t": t_float, "shapes": shapes}
                        frame_counter["last_blendshape_t"] = t_float
                        if conversation_id:
                            out["conversation_id"] = conversation_id
                        if current_sentence_id:
                            out["sentence_id"] = current_sentence_id
                        if frame_text:
                            out["text"] = frame_text
                        logger.info(
                            f"[BLEND][EMIT][FLAT] t={t_float:.3f} raw_t={raw_t_flat:.3f} "
                            f"base_t0={nested_base_t0['value']:.3f} shapes={len(shapes)} frame#{frame_idx} "
                            f"conv_id={conversation_id[:12] if conversation_id else 'NONE'}..."
                        )
                        if VERBOSE_LOGS:
                            logger.info(f"[WS][SEND] type={out['type']} t={out.get('t', 0):.3f} keys={list(out.keys())} bytes={len(json.dumps(out))}")
                        await _emit(out)
                        if frame_idx % 20 == 0:  # Log every 20th frame
                            logger.info(f"[BLEND][EMITTED] pubsub={send_to_pubsub} client={send_to_client} conv={conversation_id[:12] if conversation_id else 'NONE'}...")


                    # --- Fallback: NVIDIA A2F nested structure (animation_data -> skel_animation -> blend_shape_weights -> values) ---
                    if not shapes:
                        try:
                            ad = resp_dict.get("animation_data") or {}
                            skel = ad.get("skel_animation") or ad.get("skeleton_animation") or {}
                            bsw = skel.get("blend_shape_weights") or skel.get("blendshape_weights") or []

                            # Normalize to a list of items { "time_code": ..., "values": [...] }
                            if isinstance(bsw, dict):
                                items = [bsw]
                            elif isinstance(bsw, list):
                                items = bsw
                            else:
                                items = []

                            if items:
                                # Optional: breadcrumb when NVIDIA batches frames
                                if len(items) > 1:
                                    logger.info(f"[WS][BATCH] blend_shape_weights has {len(items)} items")

                                # NOTE: DO NOT reset baseline per response.
                                # We use nested_base_t0 / nested_last_sent defined once per stream above.

                                for it in items:
                                    vals = it.get("values") or it.get("weights") or []
                                    if not isinstance(vals, list) or not vals:
                                        continue

                                    # Prefer NVIDIA's per-item time_code; fallback to derived t_val
                                    t_item = it.get("time_code")
                                    if t_item is None:
                                        t_item = 0.0
                                    try:
                                        t_item = float(t_item)
                                    except Exception:
                                        t_item = float(t_val)

                                    # Initialize stream baseline once, on first seen time_code
                                    if nested_base_t0["value"] is None:
                                        nested_base_t0["value"] = t_item
                                    t_local = t_item - nested_base_t0["value"]

                                    # Debug: see how we're mapping NVIDIA timecodes
                                    if VERBOSE_LOGS:
                                        logger.info(
                                            f"[TS][NESTED] frame#{frame_idx} item_tc={t_item} base_t0={nested_base_t0['value']} t_local={t_local} last_sent={nested_last_sent['value']}"
                                        )

                                    # Monotonic guard (allow tiny jitter)
                                    if t_local < (nested_last_sent["value"] - 0.05):
                                        logger.warning(
                                            f"[WS][DROP] non-monotonic t={t_local:.3f} < last={nested_last_sent['value']:.3f} "
                                            f"item_tc={t_item} base_t0={nested_base_t0['value']} frame#{frame_idx}"
                                        )
                                        continue

                                    # Build shape map with names if available; else default names; else index keys
                                    if blend_names and len(blend_names) == len(vals):
                                        shapes2 = {blend_names[i]: float(vals[i]) for i in range(len(vals))}
                                    elif len(DEFAULT_A2F_NAMES) == len(vals):
                                        shapes2 = {DEFAULT_A2F_NAMES[i]: float(vals[i]) for i, w in enumerate(vals)}
                                        if not sent_names:
                                            _nm = {"type": "names", "names": list(DEFAULT_A2F_NAMES)}
                                            if conversation_id:
                                                _nm["conversation_id"] = conversation_id
                                            await _emit(_nm)
                                            sent_names = True
                                    else:
                                        shapes2 = {str(i): float(w) for i, w in enumerate(vals)}

                                    # Look up text for this item's specific timestamp
                                    item_text, item_sentence_id = _get_text_for_timestamp(t_item)

                                    # Emit blendshapes_frame
                                    _out_bs = {"type": "blendshapes_frame", "t": t_local, "shapes": shapes2}
                                    frame_counter["last_blendshape_t"] = t_local
                                    if conversation_id:
                                        _out_bs["conversation_id"] = conversation_id
                                    if item_text:
                                        _out_bs["text"] = item_text
                                    if item_sentence_id:
                                        _out_bs["sentence_id"] = item_sentence_id
                                    logger.info(
                                        f"[BLEND][EMIT] t={t_local:.3f} item_tc={t_item:.3f} "
                                        f"base_t0={nested_base_t0['value']:.3f} shapes={len(shapes2)} frame#{frame_idx} "
                                        f"conv_id={conversation_id[:12] if conversation_id else 'NONE'}..."
                                    )
                                    if VERBOSE_LOGS:
                                        logger.info(
                                            f"[WS][SEND] type={_out_bs['type']} t={_out_bs.get('t', 0):.3f} "
                                            f"keys={list(_out_bs.keys())} bytes={len(json.dumps(_out_bs))}"
                                        )
                                    await _emit(_out_bs)
                                    if frame_idx % 20 == 0:  # Log every 20th frame
                                        logger.info(f"[BLEND][EMITTED] pubsub={send_to_pubsub} client={send_to_client} conv={conversation_id[:12] if conversation_id else 'NONE'}...")

                                    # Emit visemes_frame (idx/w)
                                    visemes_out = [{"idx": i, "w": float(w)} for i, w in enumerate(vals)]
                                    _out_v = {"type": "visemes_frame", "t": t_local, "visemes": visemes_out}
                                    frame_counter["last_viseme_t"] = t_local
                                    if conversation_id:
                                        _out_v["conversation_id"] = conversation_id
                                    if item_text:
                                        _out_v["text"] = item_text
                                    if item_sentence_id:
                                        _out_v["sentence_id"] = item_sentence_id
                                    if VERBOSE_LOGS:
                                        logger.info(
                                            f"[WS][SEND] type={_out_v['type']} t={_out_v.get('t', 0):.3f} "
                                            f"keys={list(_out_v.keys())} bytes={len(json.dumps(_out_v))}"
                                        )
                                    await _emit(_out_v)

                                    emitted_nested = True
                                    # Update stream-wide last_sent
                                    nested_last_sent["value"] = t_local
                        except Exception as _e:
                            logger.debug(f"[A2F] nested blend_shape_weights parse skipped: {type(_e).__name__}: {_e}")

                    # --- EMOTIONS: emit per-frame emotion data when available ---
                    try:
                        # 1) Explicit per-frame emotions on audio (repeated EmotionWithTimeCode)
                        audio_obj = resp_dict.get("audio") or {}
                        em_list = audio_obj.get("emotions")
                        if isinstance(em_list, list) and em_list:
                            for em in em_list:
                                # Common shapes of the dict after MessageToDict:
                                # em = {"time_code": 12.34, "emotions": {"joy":0.5, ...}}  OR
                                # em = {"time_code": 12.34, "emotion_map": {"joy":0.5, ...}}  OR
                                # em = {"time_code": 12.34, "values": {"joy":0.5, ...}}
                                t_em = em.get("time_code", t_val)
                                try:
                                    t_em = float(t_em)
                                except Exception:
                                    t_em = float(t_val)
                                # prefer 'emotions' then 'emotion_map' then 'values'
                                emo_map = em.get("emotions") or em.get("emotion_map") or em.get("values") or {}
                                if isinstance(emo_map, dict) and emo_map:
                                    out_em = {"type": "emotion_frame", "t": t_em, "emotions": emo_map}
                                    if conversation_id:
                                        out_em["conversation_id"] = conversation_id
                                    if VERBOSE_LOGS:
                                        logger.info(f"[WS][SEND] type={out_em['type']} t={out_em['t']:.3f} keys={list(out_em.keys())} bytes={len(json.dumps(out_em))}")
                                    await _emit(out_em)

                        # 2) Aggregated emotion data in metadata Any (emotion_aggregate)
                        meta = (resp_dict.get("animation_data") or {}).get("metadata") or {}
                        if isinstance(meta, dict) and "emotion_aggregate" in meta:
                            emo_map = meta.get("emotion_aggregate") or {}
                            if isinstance(emo_map, dict) and emo_map:
                                # Try to align timecode with the same item we used for blendshapes; fallback to t_val
                                skel = (resp_dict.get("animation_data") or {}).get("skel_animation") or {}
                                bsw = skel.get("blend_shape_weights") or skel.get("blendshape_weights") or []
                                # derive a representative time_code if present
                                t_meta = None
                                if isinstance(bsw, list) and bsw:
                                    t_meta = bsw[-1].get("time_code") or bsw[0].get("time_code")
                                if t_meta is None:
                                    t_meta = t_val
                                try:
                                    t_meta = float(t_meta)
                                except Exception:
                                    t_meta = float(t_val)
                                out_agg = {"type": "emotion_agg_frame", "t": t_meta, "emotions": emo_map}
                                if conversation_id:
                                    out_agg["conversation_id"] = conversation_id
                                if VERBOSE_LOGS:
                                    logger.info(f"[WS][SEND] type={out_agg['type']} t={out_agg['t']:.3f} keys={list(out_agg.keys())} bytes={len(json.dumps(out_agg))}")
                                await _emit(out_agg)
                    except Exception as _emo_e:
                        logger.debug(f"[A2F] emotion emission skipped: {type(_emo_e).__name__}: {_emo_e}")

                    # --- try to emit VISEMES in production shape ---
                    # Accept either a list of weights aligned to indices, or {"visemes":[{"idx":..,"w":..},...]}
                    visemes_out = None
                    if (not emitted_nested) and "visemes" in resp_dict and isinstance(resp_dict["visemes"], list):
                        # If already objects, pass through; if floats, map to idx/w
                        if resp_dict["visemes"] and isinstance(resp_dict["visemes"][0], dict):
                            visemes_out = resp_dict["visemes"]
                        else:
                            visemes_out = [{"idx": i, "w": float(w)} for i, w in enumerate(resp_dict["visemes"])]
                    elif (not emitted_nested) and "weights" in resp_dict and isinstance(resp_dict["weights"], list):
                        visemes_out = [{"idx": i, "w": float(w)} for i, w in enumerate(resp_dict["weights"])]

                    if visemes_out:
                        t_float = float(t_val)
                        out = {"type": "visemes_frame", "t": t_float, "visemes": visemes_out}
                        frame_counter["last_viseme_t"] = t_float
                        if conversation_id:
                            out["conversation_id"] = conversation_id
                        if frame_text:
                            out["text"] = frame_text
                        if VERBOSE_LOGS:
                            logger.info(f"[WS][SEND] type={out['type']} t={out.get('t', 0):.3f} keys={list(out.keys())} bytes={len(json.dumps(out))}")
                        await _emit(out)

                    # --- optional: keep the debug echo only when enabled ---
                    if os.getenv("A2F_DEBUG_ECHO", "false").lower() in ("1", "true", "yes"):
                        try:
                            echo = {
                                "type": "debug_a2f_resp",
                                "idx": frame_idx,
                                "oneof": oneof_name,
                                "fields": present_fields,
                            }
                            if conversation_id:
                                echo["conversation_id"] = conversation_id
                            await _emit(echo)
                        except Exception:
                            pass

            # Normal stream completion diagnostic
            last_blendshape_val = frame_counter.get("last_blendshape_t")
            last_viseme_t = frame_counter.get("last_viseme_t")
            last_blendshape_t = f"{last_blendshape_val:.3f}" if isinstance(last_blendshape_val, (int, float)) else "n/a"
            last_viseme_t = f"{last_viseme_t:.3f}" if isinstance(last_viseme_t, (int, float)) else "n/a"
            turn_audio_sec = _turn_audio_seconds()
            session_audio_sec = _audio_seconds(total_audio_bytes)
            coverage_pct = (
                (float(last_blendshape_val) / turn_audio_sec * 100)
                if isinstance(last_blendshape_val, (int, float)) and turn_audio_sec > 0
                else 0
            )
            if frame_idx == 0:
                logger.error("[gRPC][DIAG] ProcessAudioStream completed with 0 responses/frames")
            logger.warning(
                f"[gRPC] ⚠️ stream completed; frames_emitted={frame_idx} "
                f"last_blendshape_t={last_blendshape_t} last_viseme_t={last_viseme_t} "
                f"audio_sec(turn)={turn_audio_sec:.3f} session_audio_sec={session_audio_sec:.3f} "
                f"⚠️ COVERAGE={coverage_pct:.1f}%"
            )
            logger.info("[A2F][FRAMES] conversation_id=%s total=%d", conversation_id, frame_idx)
            # Ensure final count is visible outside
            frame_counter["count"] = frame_idx
            # Emit visemes_end so clients know frames finished.
            visemes_end_payload = {
                "type": "visemes_end",
                "conversation_id": conversation_id,
                "frames_emitted": frame_idx,
            }
            last_viseme_val = frame_counter.get("last_viseme_t")
            if isinstance(last_viseme_val, (int, float)):
                visemes_end_payload["last_viseme_t"] = float(last_viseme_val)
            elif isinstance(last_viseme_val, str) and last_viseme_val.lower() != "n/a":
                try:
                    visemes_end_payload["last_viseme_t"] = float(last_viseme_val)
                except Exception:
                    visemes_end_payload["last_viseme_t"] = last_viseme_val
            try:
                await _emit(visemes_end_payload)
            except Exception:
                logger.warning("[WS] failed to emit visemes_end payload", exc_info=True)
        except asyncio.CancelledError:
            frame_counter["count"] = frame_idx
            return  # expected on shutdown
        except grpc_aio.AioRpcError as e:
            frame_counter["count"] = frame_idx
            status_name = e.code().name if e.code() else "UNKNOWN"
            # Clear the Vast.ai address cache on UNAVAILABLE so the next request
            # re-discovers the gRPC port (handles worker restarts and slow NIM startup).
            if status_name == "UNAVAILABLE" and _is_self_hosted_a2f():
                _VAST_ROUTE_CACHE.update({"addr": "", "route_url": "", "expires_at": 0.0})
                logger.info("[VAST] cleared route cache due to UNAVAILABLE — will re-discover on next request")
            details_text = e.details() or ""
            try:
                init_md = dict(e.initial_metadata() or [])
            except Exception:
                init_md = {}
            try:
                trailing_md = dict(e.trailing_metadata() or [])
            except Exception:
                trailing_md = {}
            try:
                dbg = e.debug_error_string() or ""
            except Exception:
                dbg = ""
            logger.error(
                "[gRPC] error: code=%s details=%s debug=%s initial_md=%s trailing_md=%s",
                status_name,
                details_text,
                dbg,
                init_md,
                trailing_md,
            )
            try:
                _err = {
                    "type": "error",
                    "error_type": "grpc_error",
                    "error": f"gRPC status={status_name}, details={details_text}",
                    "grpc_status": status_name,
                    "grpc_details": details_text,
                    "grpc_debug": dbg,
                    "grpc_initial_md": init_md,
                    "grpc_trailing_md": trailing_md,
                    "retriable": _is_retriable_grpc_status(status_name),
                    "ts_ms": int(time.time() * 1000),
                }
                if status_name == "RESOURCE_EXHAUSTED":
                    estimate = _build_capacity_estimate(init_md, trailing_md)
                    _err["error_type"] = "capacity_exhausted"
                    _err["retry_after_ms"] = int(estimate.get("estimated_wait_ms", 0))
                    _err["queue_estimate"] = estimate
                if conversation_id:
                    _err["conversation_id"] = conversation_id
                await _emit(_err)
            except Exception:
                pass
        except Exception as e:
            frame_counter["count"] = frame_idx
            logger.exception(f"[gRPC] unexpected error: {e}")
            try:
                _err = {
                    "type": "error",
                    "error_type": "grpc_unexpected_error",
                    "error": f"gRPC: {str(e)}",
                    "retriable": False,
                    "ts_ms": int(time.time() * 1000),
                }
                if conversation_id:
                    _err["conversation_id"] = conversation_id
                await _emit(_err)
            except Exception:
                pass
    # --- 3) Start response pump and read WS incoming audio ---
    # Decide if using server-driven audio fetch
    use_server_audio = bool(audio_url)
    logger.info(f"[WS] use_server_audio={use_server_audio}")
    try:
        logger.info(f"[WS] client requested outputs={cfg.get('outputs', None)} use_server_audio={use_server_audio} sr={sr} ch={ch}")
    except Exception:
        pass
    pump_task = asyncio.create_task(pump_responses())
    ff_task = None
    if use_server_audio:
        logger.info("[WS] Using server-side audio fetch from URL (ffmpeg)…")
        ff_task = asyncio.create_task(_pump_ffmpeg_audio(audio_url, sr, ch))
    else:
        logger.info("[WS] Expecting client-sent PCM frames via WS (no audio_url).")
        _arm_first_pcm_retry()
    ended = False
    hold_turn_open = False
    recv_counter = 0
    try:
        while True:
            # In server audio mode, just wait for ffmpeg to finish and break
            if use_server_audio:
                # In URL-driven mode, wait for ffmpeg to finish and then allow gRPC to flush.
                try:
                    if ff_task:
                        await ff_task
                    ended = True
                except Exception:
                    ended = True
                break
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=FIRST_PCM_POLL_SEC)
            except asyncio.TimeoutError:
                if (
                    not ended
                    and first_pcm_retry_state["armed_at"] is not None
                    and recv_audio_chunks <= int(first_pcm_retry_state["armed_recv_audio_chunks"] or 0)
                ):
                    wait_s = time.monotonic() - float(first_pcm_retry_state["armed_at"] or 0.0)
                    if wait_s >= FIRST_PCM_TIMEOUT_SEC:
                        if int(first_pcm_retry_state["attempts"] or 0) < FIRST_PCM_RETRY_MAX_ATTEMPTS:
                            await _request_first_pcm_retry(final=False)
                        else:
                            await _request_first_pcm_retry(final=True)
                            logger.error(
                                "[WS][PCM_RETRY] giving up convo=%s sentence_id=%s after %s attempts without PCM",
                                conversation_id,
                                first_pcm_retry_state["sentence_id"] or "",
                                first_pcm_retry_state["attempts"],
                            )
                            ended = True
                            break
                now_ts = asyncio.get_event_loop().time()
                if (
                    not hold_turn_open
                    and not ended
                    and recv_audio_chunks > 0
                    and (now_ts - last_audio_ts) >= FLUSH_IDLE_SEC
                ):
                    print(f"[WS] idle {now_ts - last_audio_ts:.2f}s → auto EndOfAudio()")
                    ended = True
                    break
                continue
            # Debug breadcrumb for message types
            recv_counter += 1
            if WS_LOG_RECV and (recv_counter <= 5 or recv_counter % WS_LOG_EVERY == 0):
                print(f"[WS] recv type={msg.get('type')} has_bytes={msg.get('bytes') is not None} has_text={msg.get('text') is not None}")
            t = msg.get("type")
            if t == "websocket.receive":
                if "bytes" in msg and msg["bytes"] is not None:
                    b = msg["bytes"]
                    text_snapshot = audio_chunk_text.get("value") or _AUDIO_TEXT_BY_CONVO.get(conversation_id)
                    sentence_snapshot = audio_chunk_sentence_id.get("value") or _AUDIO_SENTENCE_BY_CONVO.get(conversation_id)
                    await audio_q.put((b, text_snapshot, sentence_snapshot))
                    recv_audio_bytes += len(b)
                    recv_audio_chunks += 1
                    if recv_audio_chunks % 25 == 0:
                        print(f"[WS] audio (enqueue) chunks={recv_audio_chunks} bytes={recv_audio_bytes}")
                    last_audio_ts = asyncio.get_event_loop().time()
                    if first_pcm_retry_state["armed_at"] is not None:
                        _clear_first_pcm_retry("first_pcm_received")
                elif "text" in msg and msg["text"] is not None:
                    # Optional: handle control JSON mid-stream
                    # e.g., {"flush":true} or {"end":true}
                    ctrl_msg = (msg["text"] or "").strip()
                    if ctrl_msg:
                        print(f"[WS] control text len={len(ctrl_msg)} head={ctrl_msg[:80]}")
                        ctrl_text_for_chunks = None
                        try:
                            data = json.loads(ctrl_msg)
                        except Exception:
                            ctrl_text_for_chunks = ctrl_msg
                        else:
                            requested_conversation_id = _extract_conversation_id(data)
                            if requested_conversation_id and requested_conversation_id != conversation_id:
                                logger.warning(
                                    "[WS] conversation_id change requested on active stream current=%s requested=%s; forcing reconnect",
                                    conversation_id,
                                    requested_conversation_id,
                                )
                                await _ws_error(
                                    ws,
                                    "conversation_id is immutable per /a2f/stream connection. "
                                    "Reconnect both /a2f/stream and /a2f/visemes with the new conversation_id.",
                                )
                                ended = True
                                break
                            if _truthy(data.get("tts_end")):
                                logger.info("[WS] received tts_end=true; finalizing explicit turn")
                                ended = True
                                break
                            if _truthy(data.get("tts_chunk_only")):
                                hold_turn_open = True
                            turn_boundary = str(data.get("turn_boundary") or "").strip().lower()
                            if turn_boundary in {"start", "continue"}:
                                hold_turn_open = True
                                logger.info(
                                    "[WS] turn_boundary=%s hold_turn_open=%s conversation_id=%s",
                                    turn_boundary,
                                    hold_turn_open,
                                    conversation_id,
                                )
                            if isinstance(data, dict) and data.get("type") == "tts_word_alignment":
                                words = data.get("words")
                                if isinstance(words, list) and words:
                                    out = {
                                        "type": "tts_word_alignment",
                                        "words": words,
                                        "conversation_id": conversation_id,
                                    }
                                    sentence_id = data.get("sentence_id")
                                    if sentence_id:
                                        out["sentence_id"] = sentence_id
                                    logger.info(
                                        "[A2F][ALIGN] conversation_id=%s words=%d sentence_id=%s",
                                        conversation_id,
                                        len(words),
                                        sentence_id or "",
                                    )
                                    await _publish(conversation_id, json.dumps(out, ensure_ascii=False))
                                continue
                            new_turn_flag = _truthy(data.get("new_turn") or data.get("newTurn"))
                            if new_turn_flag:
                                audio_time_base_bytes["value"] = total_audio_bytes
                                audio_time_reset_pending["value"] = True
                                nested_base_t0["value"] = None  # Reset blendshape timing baseline
                                nested_last_sent["value"] = -1.0  # Reset monotonic guard for new turn
                                logger.info("[WS] new_turn: reset audio and blendshape timing baselines")
                            bps = (
                                data.get("bytes_per_sec")
                                or data.get("bytes_per_second")
                                or data.get("bytesPerSec")
                            )
                            if bps:
                                try:
                                    bps_val = float(bps)
                                    if bps_val > 0 and ch > 0:
                                        sr_override = bps_val / (ch * 2.0)
                                        audio_time_sr_override["value"] = sr_override
                                        logger.info(
                                            f"[WS] audio_time sr_override={sr_override:.1f} bytes_per_sec={bps_val}"
                                        )
                                except Exception:
                                    pass
                            text_override = _pick_audio_chunk_text(data)
                            if text_override:
                                ctrl_text_for_chunks = text_override
                            if data.get("end"):
                                print("[WS] received end=true (client finished)")
                                hold_turn_open = False
                                ended = True
                                break
                        if ctrl_text_for_chunks:
                            audio_chunk_text["value"] = ctrl_text_for_chunks
                            _AUDIO_TEXT_BY_CONVO[conversation_id] = ctrl_text_for_chunks
                            _log_sentence(conversation_id, ctrl_text_for_chunks)
                        sentence_override = _pick_audio_chunk_sentence_id(data) if isinstance(data, dict) else None
                        if sentence_override:
                            audio_chunk_sentence_id["value"] = sentence_override
                            _AUDIO_SENTENCE_BY_CONVO[conversation_id] = sentence_override
                        if ctrl_text_for_chunks or sentence_override:
                            _arm_first_pcm_retry(ctrl_text_for_chunks, sentence_override)
                        # NOTE: Text markers are now recorded in request_gen() when audio chunks
                        # arrive with changed text - this ensures markers are at the correct byte
                        # offset where that sentence's audio actually starts, not when the text
                        # update message is received (which can be before the audio arrives)
                # Auto-flush if idle and we haven't explicitly ended
                now_ts = asyncio.get_event_loop().time()
                if (
                    not hold_turn_open
                    and not ended
                    and recv_audio_chunks > 0
                    and (now_ts - last_audio_ts) >= FLUSH_IDLE_SEC
                ):
                    print(f"[WS] idle {now_ts - last_audio_ts:.2f}s → auto EndOfAudio()")
                    ended = True
                    break
            elif t == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        logger.info(
            f"[WS] closing: audio_chunks={total_audio_chunks} audio_bytes={total_audio_bytes} "
            f"audio_sec={_audio_seconds(total_audio_bytes):.3f}"
        )
        print(
            f"[WS] closing: audio_chunks={total_audio_chunks} audio_bytes={total_audio_bytes} "
            f"audio_sec={_audio_seconds(total_audio_bytes):.3f}"
        )
        # Signal request generator to finish (this triggers EndOfAudio() yield)
        await audio_q.put(None)
        # If the client sent end=true, allow gRPC to flush animation frames
        if pump_task:
            if ended:
                print("[WS] expecting A2F frames after EndOfAudio (waiting up to 60s)...")
                print("[WS] waiting for gRPC flush after EndOfAudio...")
                FLUSH_TOTAL = 60.0
                FLUSH_STEP = 5.0
                waited = 0.0
                while True:
                    try:
                        await asyncio.wait_for(pump_task, timeout=FLUSH_STEP)
                        break  # pump finished
                    except asyncio.TimeoutError:
                        waited += FLUSH_STEP
                        logger.info(
                            f"[WS] waiting for A2F frames… waited={waited:.1f}s frames={frame_counter.get('count', 0)}")
                        if waited >= FLUSH_TOTAL:
                            logger.warning("[WS] gRPC flush timed out; cancelling pump task.")
                            with contextlib.suppress(Exception):
                                pump_task.cancel()
                                await pump_task
                            break
                # After flush attempt, if no frames ever arrived, emit a strong breadcrumb
                if frame_counter.get("count", 0) == 0:
                    logger.warning("[WS] No viseme frames received before close (frame_idx=0).")
            else:
                # No explicit end; cancel the pump
                with contextlib.suppress(Exception):
                    pump_task.cancel()
                    await pump_task

        # Always log final coverage stats, even if pump was cancelled
        if pump_task:
            last_blendshape_t = frame_counter.get("last_blendshape_t", 0)
            turn_audio_seconds = _turn_audio_seconds()
            session_audio_seconds = _audio_seconds(total_audio_bytes)
            coverage_pct = (last_blendshape_t / turn_audio_seconds * 100) if turn_audio_seconds > 0 else 0
            logger.warning(
                f"[WS][SESSION_END] Frames: {frame_counter.get('count', 0)} | "
                f"Blendshapes: {last_blendshape_t:.3f}s | "
                f"Audio(turn): {turn_audio_seconds:.3f}s | SessionAudio: {session_audio_seconds:.3f}s | "
                f"Coverage: {coverage_pct:.1f}%"
            )

            # Retry if coverage is below threshold
            if isinstance(last_blendshape_t, (int, float)) and session_audio_seconds > 0:
                coverage_ratio = last_blendshape_t / session_audio_seconds
                if coverage_ratio < RETRY_COVERAGE_THRESHOLD and audio_buffer_for_retry:
                    logger.info(
                        f"[RETRY] Coverage {coverage_ratio:.1%} < threshold {RETRY_COVERAGE_THRESHOLD:.1%}, "
                        f"attempting retry for uncovered audio..."
                    )
                    try:
                        retry_success = await retry_uncovered_audio(
                            covered_seconds=float(last_blendshape_t),
                            total_seconds=session_audio_seconds
                        )
                        if retry_success:
                            # Update coverage stats after retry
                            new_last_t = frame_counter.get("last_blendshape_t", last_blendshape_t)
                            new_coverage = (new_last_t / session_audio_seconds * 100) if session_audio_seconds > 0 else 0
                            logger.info(
                                f"[RETRY] Final coverage after retry: {new_coverage:.1f}% "
                                f"(was {coverage_pct:.1f}%)"
                            )
                        else:
                            logger.warning("[RETRY] Retry failed to improve coverage")
                    except Exception as e:
                        logger.error(f"[RETRY] Error during retry: {e}", exc_info=True)

            # Log per-sentence stats
            if sentence_audio_bytes:
                bytes_per_second = sr * ch * 2  # PCM16
                total_expected_frames = 0
                logger.info("[SENTENCE_STATS] Per-sentence audio and expected frames:")
                for sent_id, sent_bytes in sentence_audio_bytes.items():
                    sent_duration = sent_bytes / bytes_per_second
                    expected_frames = int(sent_duration * 30)  # 30 FPS
                    total_expected_frames += expected_frames
                    logger.info(
                        f"  sentence_id={sent_id}: {sent_bytes} bytes, {sent_duration:.3f}s audio, "
                        f"~{expected_frames} expected frames"
                    )
                actual_frames = frame_counter.get('count', 0)
                logger.warning(
                    f"[SENTENCE_STATS] TOTAL: Expected ~{total_expected_frames} frames, "
                    f"Received {actual_frames} frames, "
                    f"Missing ~{total_expected_frames - actual_frames} frames ({(actual_frames/total_expected_frames*100 if total_expected_frames > 0 else 0):.1f}% received)"
                )

                # Log synchronized audio stats
                if total_synced_audio_bytes > 0:
                    synced_duration = total_synced_audio_bytes / bytes_per_second
                    original_duration = session_audio_seconds
                    logger.warning(
                        f"[SYNCED_AUDIO_STATS] NVIDIA returned {total_synced_audio_bytes} bytes of synced audio "
                        f"({synced_duration:.3f}s) vs {session_audio_seconds:.3f}s original audio sent | "
                        f"Synced audio coverage: {(synced_duration/original_duration*100 if original_duration > 0 else 0):.1f}%"
                    )
        # Wait for ffmpeg task if present and not already done
        if 'ff_task' in locals() and ff_task:
            with contextlib.suppress(Exception):
                await ff_task
        with contextlib.suppress(Exception):
            await channel.close()
        with contextlib.suppress(Exception):
            await ws.close()



async def _ws_error(ws: WebSocket, message: str):
    try:
        if getattr(ws, "application_state", None) != WebSocketState.CONNECTED:
            return
        await ws.send_text(json.dumps({"error": message}))
    except Exception:
        pass
    finally:
        with contextlib.suppress(Exception):
            await ws.close()

            
