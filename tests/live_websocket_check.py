"""Manual smoke test for a running Uvicorn server."""

import asyncio
import json
import sys
import urllib.request
from urllib.parse import urlsplit, urlunsplit

import websockets


async def main() -> None:
    base_url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip(
        "/"
    )
    request = urllib.request.Request(
        f"{base_url}/api/v1/chat/rooms",
        data=json.dumps({"device_id": "live-check-device"}).encode(),
        headers={
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
        },
    )
    room = json.load(urllib.request.urlopen(request))
    parsed = urlsplit(base_url)
    socket_scheme = "wss" if parsed.scheme == "https" else "ws"
    socket_url = urlunsplit(
        (
            socket_scheme,
            parsed.netloc,
            f"/api/v1/chat/ws/{room['room_id']}",
            f"device_id=live-check-device&member_token={room['member_token']}",
            "",
        )
    )
    async with websockets.connect(socket_url) as socket:
        event = json.loads(await socket.recv())
        assert event["type"] == "history"
        print(f"websocket_upgrade=ok room={room['room_id']}")


if __name__ == "__main__":
    asyncio.run(main())
