import unittest

from fastapi.testclient import TestClient

from main import app


class ChatRoomTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def _create_room(self):
        response = self.client.post(
            "/api/v1/chat/rooms", json={"device_id": "device-a-12345678"}
        )
        self.assertEqual(response.status_code, 201)
        return response.json()

    def test_room_allows_two_members_and_rejects_a_third(self):
        room = self._create_room()
        second = self.client.post(
            "/api/v1/chat/rooms/join",
            json={"device_id": "device-b-12345678", "room_id": room["room_id"]},
        )
        self.assertEqual(second.status_code, 200)

        third = self.client.post(
            "/api/v1/chat/rooms/join",
            json={"device_id": "device-c-12345678", "room_id": room["room_id"]},
        )
        self.assertEqual(third.status_code, 409)

    def test_invalid_qr_invitation_token_is_rejected(self):
        room = self._create_room()
        response = self.client.post(
            "/api/v1/chat/rooms/join",
            json={
                "device_id": "device-b-12345678",
                "room_id": room["room_id"],
                "join_token": "invalid-token",
            },
        )
        self.assertEqual(response.status_code, 403)

    def test_websocket_broadcasts_message_to_both_devices(self):
        room = self._create_room()
        second = self.client.post(
            "/api/v1/chat/rooms/join",
            json={
                "device_id": "device-b-12345678",
                "room_id": room["room_id"],
                "join_token": room["join_token"],
            },
        ).json()
        first_url = (
            f"/api/v1/chat/ws/{room['room_id']}?device_id=device-a-12345678"
            f"&member_token={room['member_token']}"
        )
        second_url = (
            f"/api/v1/chat/ws/{room['room_id']}?device_id=device-b-12345678"
            f"&member_token={second['member_token']}"
        )
        with self.client.websocket_connect(first_url) as first_socket:
            self.assertEqual(first_socket.receive_json()["type"], "history")
            first_socket.receive_json()  # first presence update
            with self.client.websocket_connect(second_url) as second_socket:
                self.assertEqual(second_socket.receive_json()["type"], "history")
                first_socket.receive_json()  # both devices are now online
                second_socket.receive_json()
                first_socket.send_json(
                    {
                        "type": "message",
                        "text": "請問服務台在哪裡",
                        "source": "text_to_sign",
                        "tsl": "服務台 哪裡",
                        "animation_url": "/static/example.glb",
                    }
                )
                first_message = first_socket.receive_json()
                second_message = second_socket.receive_json()
                self.assertEqual(first_message["text"], "請問服務台在哪裡")
                self.assertEqual(second_message["tsl"], "服務台 哪裡")
                self.assertEqual(
                    second_message["animation_url"], "/static/example.glb"
                )
                self.assertEqual(first_message["id"], second_message["id"])


if __name__ == "__main__":
    unittest.main()
