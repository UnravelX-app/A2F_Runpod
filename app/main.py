from fastapi import FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import base64
import json
from typing import Optional
from app.gprc_client import process_audio_to_visemes
from app.ws_stream import router as ws_stream_router
from app.ws_stream import fetch_nvidia_blendshape_names, set_emotion
app = FastAPI(title="Audio2Face Wrapper API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(ws_stream_router)


async def _extract_audio_bytes(request: Request) -> bytes:
    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in content_type:
        form = await request.form()
        audio = form.get("audio") or form.get("file")
        if isinstance(audio, UploadFile):
            return await audio.read()
        if isinstance(audio, (bytes, bytearray)):
            return bytes(audio)

    body = await request.body()
    if not body:
        return b""

    if "application/json" in content_type:
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"Invalid JSON body: {e}")

        chunks = (
            payload.get("chunks")
            or payload.get("audio_chunks")
            or payload.get("chunk")
            or payload.get("audio_b64")
        )
        if isinstance(chunks, str):
            chunks = [chunks]
        if not chunks:
            raise ValueError("Missing audio chunks in JSON body.")

        encoding = (payload.get("encoding") or "base64").lower()
        if encoding not in ("base64", "b64"):
            raise ValueError(f"Unsupported encoding '{encoding}'. Use base64.")
        try:
            return b"".join(base64.b64decode(c) for c in chunks)
        except Exception as e:
            raise ValueError(f"Invalid base64 audio chunk: {e}")

    return body


@app.post("/a2f/viseme")
async def generate_visemes(request: Request):
    try:
        audio_bytes = await _extract_audio_bytes(request)
        if not audio_bytes:
            return JSONResponse(status_code=400, content={"error": "No audio provided."})
        segments = await asyncio.to_thread(process_audio_to_visemes, audio_bytes)
        return JSONResponse(content={"segments": segments})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/a2f")
def root():
    return {"message": "Audio2Face ECS Wrapper is running"}

@app.get("/a2f/function-info/{function_id}")
async def function_info(function_id: str, apikey: Optional[str] = None):
    names = await fetch_nvidia_blendshape_names(function_id, apikey)
    if names:
        return {"function_id": function_id, "count": len(names), "names": names}
    return {"function_id": function_id, "count": 0, "names": []}

@app.post("/a2f/emotion")
async def set_a2f_emotion(request: Request):
    try:
        body = await request.json()
        conversation_id = body.get("conversation_id", "")
        emotion = body.get("emotion", [])
        if not isinstance(emotion, list) or len(emotion) != 10:
            return JSONResponse(status_code=400, content={"error": "emotion must be a list of 10 floats"})
        set_emotion(conversation_id, [float(v) for v in emotion])
        return JSONResponse(content={"ok": True})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/a2f/health")
async def a2f_health():
    return {"status": "ok"}
