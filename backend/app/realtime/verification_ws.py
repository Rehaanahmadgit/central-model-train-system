"""WebSocket stream for verification progress [doc 05 §3]."""

from datetime import datetime
from typing import cast

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import Settings
from app.realtime.broadcaster import EventBroadcaster
from app.services import verifications as verifications_service

router = APIRouter()


@router.websocket("/ws/verifications/{verification_id}")
async def verification_stream(websocket: WebSocket, verification_id: str) -> None:
    await websocket.accept()
    settings = cast(Settings, websocket.app.state.settings)
    broadcaster = cast(EventBroadcaster, websocket.app.state.broadcaster)
    # Subscribe first, then snapshot, so no event is lost in between.
    queue = broadcaster.subscribe(verification_id)
    try:
        try:
            verification = verifications_service.get_verification(
                settings.home, verification_id
            )
        except Exception:  # unknown id → empty snapshot; stream stays idle
            verification = None
        checks = verification["checks"] if verification else []
        await websocket.send_json(
            {
                "type": "snapshot",
                "run_id": verification_id,
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "seq": None,
                "data": {"verification": verification, "checks": checks},
            }
        )
        status = verification.get("status") if verification else None
        if status in ("passed", "passed_with_warnings", "failed"):
            # Finished: replay (snapshot) then close [doc 05 §8].
            return
        while True:
            event = await queue.get()
            await websocket.send_json({**event, "run_id": verification_id})
            if event["type"] == "state":
                break
    except WebSocketDisconnect:
        pass
    finally:
        broadcaster.unsubscribe(verification_id, queue)
