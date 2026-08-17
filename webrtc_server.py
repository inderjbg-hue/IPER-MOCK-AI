import os
import json
import time
import hmac
import hashlib
import base64
import asyncio
from pathlib import Path
from collections import OrderedDict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

APP_SECRET = os.getenv("GD_ENGINE_SECRET", "CHANGE_THIS_SECRET")
MAX_PARTICIPANTS = 7
MAX_MINUTES = 10
RECORDINGS_DIR = Path(os.getenv("GD_RECORDINGS_DIR", "recordings"))
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="IPER Inbuilt GD Engine")
app.mount("/static", StaticFiles(directory="static"), name="static")

rooms = {}


def sign(payload: dict) -> str:
    raw = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    sig = hmac.new(APP_SECRET.encode(), raw.encode(), hashlib.sha256).hexdigest()
    return raw + "." + sig


def verify(token: str) -> dict:
    try:
        raw, sig = token.rsplit(".", 1)
        expected = hmac.new(APP_SECRET.encode(), raw.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise ValueError("bad signature")
        padded = raw + "=" * (-len(raw) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode())
        if payload.get("exp", 0) < time.time():
            raise ValueError("expired")
        return payload
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid GD access token: {exc}")


def room_state(code: str):
    return rooms.setdefault(code, {
        "participants": OrderedDict(),
        "started": False,
        "started_at": None,
        "ended": False,
        "host_id": None,
    })


async def send_json(ws: WebSocket, message: dict):
    try:
        await ws.send_text(json.dumps(message))
    except Exception:
        pass


async def broadcast(code: str, message: dict, exclude: str | None = None):
    room = rooms.get(code)
    if not room:
        return
    for pid, info in list(room["participants"].items()):
        if pid == exclude:
            continue
        await send_json(info["ws"], message)


@app.get("/health")
async def health():
    return {"ok": True, "service": "iper-inbuilt-gd-engine"}


@app.get("/")
async def index():
    return HTMLResponse('<h2>IPER Inbuilt GD Engine</h2><p>Use the IPER portal to enter a GD room.</p>')


@app.get("/room")
async def room_page():
    return HTMLResponse(Path("static/gd_room.html").read_text())


@app.websocket("/ws/{room_code}")
async def websocket_endpoint(websocket: WebSocket, room_code: str):
    await websocket.accept()
    token = websocket.query_params.get("token", "")
    try:
        user = verify(token)
    except HTTPException as exc:
        await send_json(websocket, {"type": "error", "message": exc.detail})
        await websocket.close(code=1008)
        return

    room_code = room_code.upper()
    if user.get("room", "").upper() != room_code:
        await send_json(websocket, {"type": "error", "message": "This access token is not valid for this GD room."})
        await websocket.close(code=1008)
        return
    room = room_state(room_code)
    pid = user["scholar_id"]
    if room["ended"]:
        await send_json(websocket, {"type": "error", "message": "This GD session has ended."})
        await websocket.close(code=1008)
        return
    if pid not in room["participants"] and len(room["participants"]) >= MAX_PARTICIPANTS:
        await send_json(websocket, {"type": "error", "message": "GD room is full. Maximum 7 students."})
        await websocket.close(code=1008)
        return

    if pid not in room["participants"]:
        room["participants"][pid] = {
            "ws": websocket,
            "name": user["display_name"],
            "joined_at": time.time(),
        }
        if room["host_id"] is None:
            room["host_id"] = pid
    else:
        room["participants"][pid]["ws"] = websocket

    existing = [
        {"id": other_id, "name": info["name"]}
        for other_id, info in room["participants"].items() if other_id != pid
    ]
    await send_json(websocket, {
        "type": "joined",
        "self_id": pid,
        "display_name": user["display_name"],
        "is_host": room["host_id"] == pid,
        "participants": [{"id": k, "name": v["name"]} for k, v in room["participants"].items()],
        "existing_peers": existing,
        "started": room["started"],
        "started_at": room["started_at"],
    })
    await broadcast(room_code.upper(), {"type": "peer_joined", "id": pid, "name": user["display_name"]}, exclude=pid)

    try:
        while True:
            data = json.loads(await websocket.receive_text())
            kind = data.get("type")
            if kind in {"offer", "answer", "ice"}:
                target = data.get("target")
                if target in room["participants"]:
                    await send_json(room["participants"][target]["ws"], {
                        **data,
                        "from": pid,
                        "from_name": user["display_name"],
                    })
            elif kind == "start_gd":
                if room["host_id"] == pid and not room["started"]:
                    room["started"] = True
                    room["started_at"] = time.time()
                    await broadcast(room_code.upper(), {"type": "gd_started", "started_at": room["started_at"]})
            elif kind == "end_gd":
                if room["host_id"] == pid:
                    room["ended"] = True
                    await broadcast(room_code.upper(), {"type": "gd_ended"})
            elif kind == "ping":
                await send_json(websocket, {"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if pid in room["participants"]:
            del room["participants"][pid]
        if room["host_id"] == pid and room["participants"]:
            room["host_id"] = next(iter(room["participants"]))
            await broadcast(room_code.upper(), {"type": "host_changed", "host_id": room["host_id"]})
        else:
            await broadcast(room_code.upper(), {"type": "peer_left", "id": pid})
        if not room["participants"]:
            rooms.pop(room_code.upper(), None)


@app.post("/recording")
async def upload_recording(token: str = Form(...), room_code: str = Form(...), recording: UploadFile = File(...)):
    user = verify(token)
    code = room_code.upper()
    room = rooms.get(code)
    if not room:
        raise HTTPException(status_code=404, detail="GD room not found")
    if room.get("host_id") != user["scholar_id"]:
        raise HTTPException(status_code=403, detail="Only the GD host can upload the recording")
    data = await recording.read()
    if len(data) > 250 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Recording is too large")
    filename = f"GD_{code}_{int(time.time())}.webm"
    path = RECORDINGS_DIR / filename
    path.write_bytes(data)
    return {"ok": True, "filename": filename, "bytes": len(data)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
