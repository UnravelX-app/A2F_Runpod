import os
import logging
from typing import List, Dict, Any
import importlib, pkgutil
import sys
import io
import wave
import tempfile
import subprocess
import csv
import glob
from dotenv import load_dotenv


load_dotenv()

A2F_MODE = os.getenv("A2F_MODE", "stub").lower().strip()

# --- NVCF (NVIDIA hosted) ---
NVCF_ADDR = os.getenv("NVCF_ADDR", "grpc.nvcf.nvidia.com:443")
NVCF_FUNCTION_ID = os.getenv("NVCF_FUNCTION_ID", "")
NVCF_API_KEY = os.getenv("NVCF_API_KEY", "")

# --- Local/self-hosted A2F microservice (optional, later step) ---
A2F_GRPC_ADDR = os.getenv("A2F_GRPC_ADDR", "localhost:52000")

# Optional explicit module path overrides for proto stubs (useful when wheel layout differs)
A2F_PB2_MODULE = os.getenv("A2F_PB2_MODULE", "").strip()
A2F_PB2_GRPC_MODULE = os.getenv("A2F_PB2_GRPC_MODULE", "").strip()

# Audio params (A2F expects PCM16 WAV; if you send raw bytes, set correct sample_rate)
A2F_SAMPLE_RATE = int(os.getenv("A2F_SAMPLE_RATE", "16000"))
A2F_CHUNK_BYTES = int(os.getenv("A2F_CHUNK_BYTES", str(32_000)))  # ~1/2s at 16kHz mono PCM16

# Optional start/config (some A2F endpoints expect an initial start/config frame)
A2F_SEND_START = os.getenv("A2F_SEND_START", "1") in ("1", "true", "True")
A2F_LANGUAGE = os.getenv("A2F_LANGUAGE", "en-US")
A2F_SPEAKER = os.getenv("A2F_SPEAKER", "default")


# --- CLI fallback using NVIDIA's sample client (subprocess) ---
A2F_CLIENT_DIR = os.getenv("A2F_CLIENT_DIR", "Audio2Face-3D-Samples/scripts/audio2face_3d_api_client").strip()
A2F_CONFIG_YAML = os.getenv("A2F_CONFIG_YAML", "config/config_claire.yml").strip()

# Optional: map numeric indices ("0","1",...) to human-readable names (e.g., ARKit visemes)
# Set A2F_NAME_MAP as a comma-separated list, e.g.:
# A2F_NAME_MAP=viseme_sil,viseme_PP,viseme_FF,viseme_TH,viseme_DD,viseme_kk,viseme_CH,viseme_SS,viseme_nn,viseme_RR,viseme_aa,viseme_E,viseme_I,viseme_O,viseme_U
A2F_NAME_MAP = [s.strip() for s in os.getenv("A2F_NAME_MAP", "").split(",")] if os.getenv("A2F_NAME_MAP") else None

# Enable debug logging via env
if os.getenv("A2F_DEBUG", "0") in ("1","true","True"):
    logging.basicConfig(level=logging.DEBUG)


def _process_audio_stub(audio_bytes: bytes) -> List[Dict[str, Any]]:
    return [
        {
            "timestamp": 0.0,
            "weights": [
                {"name": "viseme_AA", "weight": 0.55},
                {"name": "viseme_IH", "weight": 0.15},
                {"name": "viseme_UH", "weight": 0.05},
            ],
        }
    ]

def _load_a2f_stubs():
    """
    Dynamically import A2F gRPC stub modules from the nvidia_ace wheel.
    Order:
      1) Explicit env overrides: A2F_PB2_MODULE / A2F_PB2_GRPC_MODULE
      2) Known candidates
      3) Walk installed nvidia_ace submodules
    """
    tried = []

    # 1) Explicit overrides
    if A2F_PB2_MODULE and A2F_PB2_GRPC_MODULE:
        try:
            pb2 = importlib.import_module(A2F_PB2_MODULE)
            pb2_grpc = importlib.import_module(A2F_PB2_GRPC_MODULE)
            if hasattr(pb2_grpc, "A2FControllerServiceStub") and hasattr(pb2, "AudioStream"):
                return pb2, pb2_grpc
            tried.append(f"overrides {A2F_PB2_MODULE}/{A2F_PB2_GRPC_MODULE}: missing A2FControllerServiceStub or AudioStream")
        except Exception as e:
            tried.append(f"overrides {A2F_PB2_MODULE}/{A2F_PB2_GRPC_MODULE}: {e}")

    # 2) Quick known candidates (different wheels expose different paths)
    candidates = [
        ("nvidia_ace.controller.v1.a2f_controller_pb2", "nvidia_ace.controller.v1.a2f_controller_pb2_grpc"),
        ("nvidia_ace.controller.v1.A2FControllerService_pb2", "nvidia_ace.controller.v1.A2FControllerService_pb2_grpc"),
        ("nvidia_ace.a2f_controller.v1.a2f_controller_pb2", "nvidia_ace.a2f_controller.v1.a2f_controller_pb2_grpc"),
        ("nvidia_ace.a2f_controller.v1.A2FControllerService_pb2", "nvidia_ace.a2f_controller.v1.A2FControllerService_pb2_grpc"),
        ("nvidia_ace.controller.a2f.v1.a2f_controller_pb2", "nvidia_ace.controller.a2f.v1.a2f_controller_pb2_grpc"),
    ]
    for mod_pb2, mod_grpc in candidates:
        try:
            pb2 = importlib.import_module(mod_pb2)
            pb2_grpc = importlib.import_module(mod_grpc)
            if hasattr(pb2_grpc, "A2FControllerServiceStub") and hasattr(pb2, "AudioStream"):
                return pb2, pb2_grpc
            tried.append(f"{mod_pb2}: missing symbols")
        except Exception as e:
            tried.append(f"{mod_pb2}: {e}")

    # 3) Walk all submodules in nvidia_ace
    try:
        import nvidia_ace
    except Exception as e:
        tried.append(f"import nvidia_ace failed: {e}")
        raise ImportError("Could not import A2F gRPC stubs. Tried: " + " | ".join(tried))

    discovered = []
    for modinfo in pkgutil.walk_packages(nvidia_ace.__path__, nvidia_ace.__name__ + "."):
        name = modinfo.name
        discovered.append(name)
        # Only attempt to import likely grpc/proto modules to keep it fast
        if not (name.endswith("_pb2") or name.endswith("_pb2_grpc") or "controller" in name.lower() or "a2f" in name.lower()):
            continue
        try:
            m = importlib.import_module(name)
        except Exception as e:
            tried.append(f"{name}: {e}")
            continue

        # Find a grpc module exposing the Stub
        if name.endswith("_pb2_grpc") and hasattr(m, "A2FControllerServiceStub"):
            # Find a matching *_pb2 module that contains AudioStream
            guesses = [
                name.replace("_pb2_grpc", "_pb2"),
                name.replace("A2FControllerService_pb2_grpc", "A2FControllerService_pb2"),
            ]
            for pb2_name in guesses:
                try:
                    pb2 = importlib.import_module(pb2_name)
                    if hasattr(pb2, "AudioStream"):
                        return pb2, m
                    else:
                        tried.append(f"{pb2_name}: no AudioStream")
                except Exception as e:
                    tried.append(f"{pb2_name}: {e}")

    # Emit a succinct diagnostic to aid configuration
    diag = []
    diag.append("Could not import A2F gRPC stubs from nvidia_ace wheel.")
    if discovered:
        diag.append("Discovered submodules:")
        # print only a subset to keep error short
        preview = discovered[:50]
        diag.extend(["  - " + s for s in preview])
        if len(discovered) > 50:
            diag.append(f"  ... (+{len(discovered)-50} more)")
    diag.append("Tried: " + " | ".join(tried))
    raise ImportError("\n".join(diag))
def _process_audio_nvcf_cli(audio_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Fallback path that shells out to NVIDIA's official sample client:
      python nim_a2f_3d_client.py <wav> <config.yml> --apikey ... --function-id ...
    Then parses the produced CSV of keyframes into our segments shape.
    """
    if not NVCF_API_KEY or not NVCF_FUNCTION_ID:
        raise RuntimeError("NVCF mode requires NVCF_API_KEY and NVCF_FUNCTION_ID to be set.")
    client_dir = A2F_CLIENT_DIR
    config_yaml = os.path.join(client_dir, A2F_CONFIG_YAML) if not os.path.isabs(A2F_CONFIG_YAML) else A2F_CONFIG_YAML
    if not os.path.isdir(client_dir):
        raise RuntimeError(f"A2F client directory not found: {client_dir}")
    if not os.path.isfile(config_yaml):
        raise RuntimeError(f"A2F config YAML not found: {config_yaml}")

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "input.wav")
        # If it's a WAV file already, just dump bytes; otherwise attempt to write a PCM16 WAV
        try:
            # Try to read as WAV
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wf_in:
                params = wf_in.getparams()
                frames = wf_in.readframes(wf_in.getnframes())
            # Write back as-is
            with wave.open(wav_path, 'wb') as wf_out:
                wf_out.setparams(params)
                wf_out.writeframes(frames)
        except wave.Error:
            # Not a WAV; assume raw PCM16 mono at env sample rate
            with wave.open(wav_path, 'wb') as wf_out:
                wf_out.setnchannels(1)
                wf_out.setsampwidth(2)
                wf_out.setframerate(A2F_SAMPLE_RATE)
                wf_out.writeframes(audio_bytes)

        # When using cwd=client_dir, call the script by its filename only.
        cmd = [
            sys.executable,
            "nim_a2f_3d_client.py",
            wav_path,
            config_yaml,
            "--apikey", NVCF_API_KEY,
            "--function-id", NVCF_FUNCTION_ID,
        ]
        proc = subprocess.run(
            cmd,
            cwd=client_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "A2F CLI client failed (exit {}). CWD: {} CMD: {}\nSTDERR:\n{}".format(
                    proc.returncode, os.path.abspath(client_dir), " ".join(cmd), proc.stderr.strip()
                )
            )

        # Find most recent CSV produced by the client
        csv_candidates = glob.glob(os.path.join(client_dir, "*.csv")) + glob.glob(os.path.join(td, "*.csv"))
        if not csv_candidates:
            # Some versions write under a subfolder; scan recursively in client_dir
            csv_candidates = glob.glob(os.path.join(client_dir, "**", "*.csv"), recursive=True)
        if not csv_candidates:
            raise RuntimeError("A2F CLI client produced no CSV output to parse.")
        csv_path = max(csv_candidates, key=os.path.getmtime)

        # Parse CSV expecting columns: name, value, timecode (order may vary)
        segments_by_ts: Dict[float, List[Dict[str, float]]] = {}
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            name_idx = value_idx = time_idx = None
            if header:
                h = [c.strip().lower() for c in header]
                for i, col in enumerate(h):
                    if "name" in col and name_idx is None: name_idx = i
                    if "value" in col and value_idx is None: value_idx = i
                    if "time" in col and time_idx is None: time_idx = i
            for row in reader:
                try:
                    if name_idx is None or value_idx is None or time_idx is None:
                        # Assume default order
                        n, v, t = row[0], float(row[1]), float(row[2])
                    else:
                        n = row[name_idx]
                        v = float(row[value_idx])
                        t = float(row[time_idx])
                    # If the CSV uses numeric indices as 'name', optionally map them to friendly labels
                    mapped_name = n
                    if A2F_NAME_MAP and n.isdigit():
                        idx = int(n)
                        if 0 <= idx < len(A2F_NAME_MAP):
                            mapped_name = A2F_NAME_MAP[idx] or n
                    segments_by_ts.setdefault(t, []).append({"name": mapped_name, "weight": v})
                except Exception:
                    continue

        # Convert to sorted segments list
        segments: List[Dict[str, Any]] = [
            {"timestamp": ts, "weights": weights}
            for ts, weights in sorted(segments_by_ts.items(), key=lambda kv: kv[0])
        ]
        return segments

def _process_audio_nvcf(audio_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Calls NVIDIA's hosted A2F service over gRPC:
      host: grpc.nvcf.nvidia.com:443
      metadata: ("authorization", f"Bearer {NVCF_API_KEY}"), ("function-id", NVCF_FUNCTION_ID)
    Streams audio and collects viseme/blendshape frames.
    """
    import grpc

    try:
        # Stub lives under services.a2f_controller
        from nvidia_ace.services.a2f_controller import v1_pb2_grpc as controller_pb2_grpc
        # Request/response message *types* live under controller, audio and a2f namespaces
        from nvidia_ace.controller import v1_pb2 as controller_pb2
        from nvidia_ace.audio import v1_pb2 as audio_pb2
        from nvidia_ace.a2f import v1_pb2 as a2f_pb2
        from nvidia_ace.animation_data import v1_pb2 as anim_pb2
    except Exception as e:
        raise ImportError(f"Failed to import NVIDIA A2F stubs explicitly: {e}")

    # TLS channel + larger message limits for audio
    options = [
        ("grpc.max_send_message_length", -1),
        ("grpc.max_receive_message_length", -1),
    ]
    channel = grpc.secure_channel(NVCF_ADDR, grpc.ssl_channel_credentials(), options=options)
    stub = controller_pb2_grpc.A2FControllerServiceStub(channel)

    # --- Try to parse WAV and extract raw PCM16 frames (A2F expects PCM16) ---
    raw_pcm = audio_bytes
    sample_rate = A2F_SAMPLE_RATE
    try:
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
            sample_rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            nchannels = wf.getnchannels()
            nframes = wf.getnframes()
            # A2F requires 16-bit PCM
            if sampwidth != 2:
                raise ValueError(f"Unsupported sample width: {sampwidth*8} bits. Please use 16-bit PCM WAV.")
            raw_pcm = wf.readframes(nframes)
    except wave.Error:
        # Not a WAV; assume caller already provided raw PCM16 bytes and env sample rate
        pass
    except Exception as e:
        # If parsing fails for another reason, surface a clear error
        raise RuntimeError(f"Failed to parse audio as WAV: {e}")

    # Build request stream generator using NVIDIA's documented proto:
    def stream_gen():
        """
        Build a stream of nvidia_ace.controller.v1.AudioStream messages:
          1) AudioStream(audio_stream_header=AudioStreamHeader(...AudioHeader...))
          2) many AudioStream(audio_with_emotion=AudioWithEmotion(audio_buffer=<PCM>))
          3) AudioStream(end_of_audio=EndOfAudio())
        Spec: https://docs.nvidia.com/ace/audio2face-3d-microservice/1.0/text/interacting/a2f-controller-rpc.html
        """
        # 1) Send header first (required)
        hdr = audio_pb2.AudioHeader(
            audio_format=audio_pb2.AudioHeader.AUDIO_FORMAT_PCM,  # PCM16
            channel_count=1,
            samples_per_second=sample_rate,
            bits_per_sample=16,
        )
        ctrl_hdr = controller_pb2.AudioStreamHeader(
            audio_header=hdr
            # face_params / emotion_post_processing_params / blendshape_params optional
        )
        yield controller_pb2.AudioStream(audio_stream_header=ctrl_hdr)

        # 2) Stream raw PCM chunks as AudioWithEmotion (emotions optional)
        for i in range(0, len(raw_pcm), A2F_CHUNK_BYTES):
            chunk = raw_pcm[i : i + A2F_CHUNK_BYTES]
            awe = a2f_pb2.AudioWithEmotion(
                audio_buffer=chunk,
                # emotions=[]  # optional
            )
            yield controller_pb2.AudioStream(audio_with_emotion=awe)

        # 3) End marker
        yield controller_pb2.AudioStream(end_of_audio=controller_pb2.AudioStream.EndOfAudio())

    # Required metadata from NVIDIA docs
    metadata = [
        ("authorization", f"Bearer {NVCF_API_KEY}"),
        ("function-id", NVCF_FUNCTION_ID),
    ]
    # Some deployments require model selection via header (rare). Enable via env if needed.
    if os.getenv("NVCF_MODEL", "").strip():
        metadata.append(("nv-ai-model", os.getenv("NVCF_MODEL").strip()))

    try:
        responses = stub.ProcessAudioStream(stream_gen(), metadata=metadata)

        blendshape_names: List[str] = []
        segments: List[Dict[str, Any]] = []

        for msg in responses:
            if msg is None:
                continue

            part_name = msg.WhichOneof("stream_part")
            logging.debug(f"[A2F] Incoming part: {part_name}")

            # --- HEADER: capture blendshape names if provided ---
            if part_name in ("animation_data_stream_header", "anim_data_stream_header", "animation_header"):
                hdr = getattr(msg, part_name)
                # Try a few common header layouts
                skel_hdr = getattr(hdr, "skel_animation_header", None) or getattr(hdr, "skeleton_animation_header", None)
                if skel_hdr:
                    # Names might be 'blend_shapes' or 'blend_shape_names'
                    names = getattr(skel_hdr, "blend_shapes", None) or getattr(skel_hdr, "blend_shape_names", None)
                    if names:
                        try:
                            blendshape_names = list(names)
                            logging.debug(f"[A2F] Blendshape names captured: {len(blendshape_names)}")
                        except Exception:
                            pass
                continue

            # --- ANIMATION DATA FRAMES ---
            if part_name in ("animation_data", "anim_data"):
                ad = getattr(msg, part_name)
                # Common container: skel_animation (or skeleton_animation)
                skel = getattr(ad, "skel_animation", None) or getattr(ad, "skeleton_animation", None)
                if skel:
                    # Each frame entry may be 'blend_shape_weights' or 'blendshape_weights'
                    frame_list = getattr(skel, "blend_shape_weights", None) or getattr(skel, "blendshape_weights", None)
                    if frame_list:
                        for fa in frame_list:
                            # Time field may be time_code or timecode
                            ts = 0.0
                            if hasattr(fa, "time_code"):
                                try: ts = float(fa.time_code)
                                except Exception: pass
                            elif hasattr(fa, "timecode"):
                                try: ts = float(fa.timecode)
                                except Exception: pass
                            # Values field might be 'values' or 'weights'
                            vals = []
                            if hasattr(fa, "values"):
                                try: vals = [float(v) for v in fa.values]
                                except Exception: vals = []
                            elif hasattr(fa, "weights"):
                                try: vals = [float(v) for v in fa.weights]
                                except Exception: vals = []

                            if vals:
                                if blendshape_names and len(blendshape_names) == len(vals):
                                    weights = [{"name": n, "weight": v} for n, v in zip(blendshape_names, vals)]
                                else:
                                    weights = [{"name": str(i), "weight": v} for i, v in enumerate(vals)]
                                    # Optional user-provided mapping if no names available
                                    if A2F_NAME_MAP and (not blendshape_names):
                                        for w in weights:
                                            if w["name"].isdigit():
                                                idx = int(w["name"])
                                                if 0 <= idx < len(A2F_NAME_MAP):
                                                    w["name"] = A2F_NAME_MAP[idx] or w["name"]
                                segments.append({"timestamp": ts, "weights": weights})
                # Some deployments may send a list of frames directly under ad.frames
                elif hasattr(ad, "frames"):
                    for fr in ad.frames:
                        ts = float(getattr(fr, "timestamp", 0.0))
                        n2 = list(getattr(fr, "names", []))
                        v2 = [float(v) for v in getattr(fr, "values", [])]
                        if n2 and v2:
                            weights = [{"name": n, "weight": v} for n, v in zip(n2, v2)]
                            segments.append({"timestamp": ts, "weights": weights})
                continue

            # Ignore other parts (status/events) for JSON output

        # If no frames parsed, escalate so caller can fall back to CLI (known-good)
        if not segments:
            raise RuntimeError("A2F native gRPC returned no animation frames.")

        return segments
    except Exception as e:
        raise RuntimeError(f"A2F gRPC call failed: {e}")


def process_audio_to_visemes(audio_bytes: bytes) -> List[Dict[str, Any]]:
    mode = A2F_MODE

    if mode == "stub":
        return _process_audio_stub(audio_bytes)

    if mode == "nvcf":
        if not NVCF_API_KEY or not NVCF_FUNCTION_ID:
            raise RuntimeError(
                "NVCF mode requires NVCF_API_KEY and NVCF_FUNCTION_ID to be set."
            )
        try:
            return _process_audio_nvcf(audio_bytes)
        except Exception as e:
            logging.warning(f"A2F native gRPC failed, falling back to CLI client: {e}")
            return _process_audio_nvcf_cli(audio_bytes)

    if mode == "nvcf_cli":
        return _process_audio_nvcf_cli(audio_bytes)

    if mode == "local":
        # We'll add local/self-hosted microservice support in a later step
        raise NotImplementedError("Local A2F mode not implemented yet.")

    # Fallback to stub
    return _process_audio_stub(audio_bytes)