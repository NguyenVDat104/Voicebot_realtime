import json
import logging
from typing import Optional
import httpx

logger = logging.getLogger("voicebot.n8n")

DEFAULT_ERROR_MESSAGE = "Hệ thống đang bận hoặc chưa thể kết nối đến máy chủ xử lý. Bạn vui lòng thử lại sau nhé."

class N8NClient:
    """
    HTTP client for calling n8n Webhooks with session handling,
    timeouts, and resilient fallback responses.
    """

    def __init__(
        self,
        webhook_url: str = "http://localhost:5678/webhook/voicebot",
        timeout: float = 10.0,
    ):
        self.webhook_url = webhook_url
        self.timeout = timeout

    async def stream_message(
        self,
        session_id: str,
        text: str,
        custom_url: Optional[str] = None,
    ):
        """
        Streams response chunks from n8n webhook (supporting NDJSON and SSE events).
        Yields text chunks as they arrive from the server in real-time.
        """
        target_url = custom_url or self.webhook_url
        payload = {
            "session_id": session_id,
            "sessionId": session_id,
            "text": text,
            "chatInput": text,
            "message": text,
        }

        logger.info(f"Streaming request to n8n webhook ({target_url}) for session '{session_id}': '{text}'")
        timeout = httpx.Timeout(self.timeout, connect=10.0)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    target_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status_code != 200:
                        logger.error(f"n8n stream error status {response.status_code}")
                        yield DEFAULT_ERROR_MESSAGE
                        return

                    buffer = ""
                    yielded_any = False

                    async for chunk in response.aiter_text():
                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                                if isinstance(data, dict):
                                    if data.get("type") == "item" and "content" in data:
                                        content = data["content"]
                                        if content:
                                            yielded_any = True
                                            yield content
                                    elif "output" in data:
                                        yielded_any = True
                                        yield str(data["output"])
                                    elif "text" in data:
                                        yielded_any = True
                                        yield str(data["text"])
                            except Exception:
                                pass

                    # If not NDJSON or single JSON response
                    if buffer.strip() and not yielded_any:
                        try:
                            data = json.loads(buffer.strip())
                            if isinstance(data, dict):
                                for key in ["text", "output", "response", "message", "answer"]:
                                    if key in data and isinstance(data[key], str):
                                        yield data[key].strip()
                                        return
                            elif isinstance(data, list) and len(data) > 0:
                                item = data[0]
                                if isinstance(item, dict):
                                    for key in ["text", "output", "response", "message", "answer"]:
                                        if key in item and isinstance(item[key], str):
                                            yield item[key].strip()
                                            return
                        except Exception:
                            pass
                        yield buffer.strip()

        except httpx.TimeoutException:
            logger.warning(f"Timeout ({self.timeout}s) waiting for n8n streaming response.")
            yield "Xin lỗi bạn, thời gian phản hồi quá lâu. Bạn vui lòng thử lại nhé."
        except httpx.ConnectError:
            logger.warning(f"Could not connect to n8n webhook at {target_url}.")
            yield f'Tôi đã nghe thấy: "{text}". Hiện tại n8n chưa được kết nối, vui lòng kiểm tra webhook.'
        except Exception as e:
            logger.error(f"Unexpected error while streaming from n8n: {e}", exc_info=True)
            yield DEFAULT_ERROR_MESSAGE

    async def send_message(
        self,
        session_id: str,
        text: str,
        custom_url: Optional[str] = None,
    ) -> str:
        """
        Sends a request to n8n and returns the complete assembled response string.
        """
        chunks = []
        async for chunk in self.stream_message(session_id, text, custom_url):
            chunks.append(chunk)
        return "".join(chunks).strip() or "Tôi đã ghi nhận thông tin của bạn."
