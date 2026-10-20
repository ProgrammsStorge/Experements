import asyncio
import json
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from trainer import Trainer

BASE     = Path(__file__).resolve().parent
FRONTEND = BASE.parent / "frontend"

trainer = Trainer(model_dir=BASE / "saved_models")
clients: set = set()


async def sim_loop():
    while True:
        if trainer.running and clients:
            n = max(1, int(round(trainer.speed)))
            for _ in range(n):
                try:
                    trainer.step()
                except Exception:
                    traceback.print_exc()
                    try:
                        trainer.hard_reset()
                    except Exception:
                        traceback.print_exc()
        try:
            payload = json.dumps(trainer.snapshot())
        except Exception:
            traceback.print_exc()
            payload = json.dumps({"type": "frame", "error": "snapshot"})
        dead = []
        for ws in list(clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)
        await asyncio.sleep(1 / 30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(sim_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="CCM Biped", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(FRONTEND / "index.html"))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        await ws.send_text(json.dumps(trainer.snapshot()))
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            cmd = msg.get("cmd")
            try:
                if cmd == "start":
                    trainer.running = True
                elif cmd == "pause":
                    trainer.running = False
                elif cmd == "reset":
                    trainer.hard_reset()
                elif cmd == "reset_brain":
                    trainer.reset_brains()
                elif cmd == "speed":
                    trainer.speed = float(msg.get("value", 1))
                elif cmd == "save":
                    trainer.save()
                elif cmd == "load":
                    trainer.load()
                elif cmd == "watch_best":
                    trainer.watch_best = bool(msg.get("value", False))
                    trainer.hard_reset()
                elif cmd == "update_env":
                    trainer.update_env(msg.get("patch", {}))
                elif cmd == "reset_env":
                    trainer.reset_env()
                elif cmd == "set_mode":
                    trainer.set_mode(msg.get("mode", "stand"), msg.get("target_x"))
                elif cmd == "set_target":
                    trainer.set_target(msg.get("target_x", 0.0))
                elif cmd == "load_best_into_current":
                    if trainer.best_models is not None:
                        for kind, m in trainer.best_models.items():
                            cur = trainer.models[kind]
                            cur.wordset = {w: set(s) for w, s in m.wordset.items()}
                            new_q = defaultdict(lambda: defaultdict(float))
                            for w, d in m.quality.items():
                                new_q[w] = defaultdict(float, dict(d))
                            cur.quality = new_q
                        trainer.hard_reset()
            except Exception:
                traceback.print_exc()
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)