import asyncio
import httpx
import json

async def run_tests():
    print("Testing '50k' query...")
    async with httpx.AsyncClient(timeout=120.0) as client:
        chat_payload = {
            "session_id": "test_qa_456",
            "message": "50k"
        }
        resp = await client.post("http://localhost:8000/api/v1/chat", json=chat_payload)
        print(f"Status: {resp.status_code}")
        try:
            print(f"Response: {json.dumps(resp.json(), indent=2)}")
        except Exception as e:
            print(f"Failed to parse JSON: {e}, text: {resp.text}")

if __name__ == "__main__":
    asyncio.run(run_tests())
