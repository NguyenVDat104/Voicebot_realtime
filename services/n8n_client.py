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

    async def send_message(
        self,
        session_id: str,
        text: str,
        custom_url: Optional[str] = None,
    ) -> str:
        """
        Sends a POST request with {"session_id": "...", "text": "..."} to the n8n webhook.
        Returns the text response from n8n or an error fallback message.
        """
        target_url = custom_url or self.webhook_url
        payload = {
            "session_id": session_id,
            "sessionId": session_id,
            "text": text,
            "chatInput": text,
            "message": text,
        }

        logger.info(f"Sending request to n8n webhook ({target_url}) for session '{session_id}': '{text}'")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    target_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

            if response.status_code == 200:
                raw_text = response.text.strip()
                if not raw_text:
                    return "Tôi đã ghi nhận thông tin của bạn."

                # 1. Support n8n AI Agent / Chat Trigger streaming NDJSON events
                if '{"type":' in raw_text:
                    ndjson_chunks = []
                    for line in raw_text.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item_obj = json.loads(line)
                            if isinstance(item_obj, dict):
                                if item_obj.get("type") == "item" and "content" in item_obj:
                                    ndjson_chunks.append(item_obj["content"])
                                elif "output" in item_obj:
                                    ndjson_chunks.append(str(item_obj["output"]))
                                elif "text" in item_obj:
                                    ndjson_chunks.append(str(item_obj["text"]))
                        except Exception:
                            pass
                    if ndjson_chunks:
                        assembled = "".join(ndjson_chunks).strip()
                        if assembled:
                            return assembled

                # 2. Try parsing as standard single JSON
                try:
                    data = json.loads(raw_text)
                    # Support n8n "All Incoming Items" list response: [{"output": "..."}]
                    if isinstance(data, list):
                        if len(data) > 0:
                            item = data[0]
                            if isinstance(item, dict):
                                data = item
                            elif isinstance(item, str):
                                return item.strip()
                        else:
                            return "Tôi đã ghi nhận thông tin của bạn."

                    if isinstance(data, dict):
                        # Support common n8n return keys: text, output, response, message, answer
                        for key in ["text", "output", "response", "message", "answer"]:
                            if key in data and isinstance(data[key], str):
                                return data[key].strip()
                        # Fallback to any non-empty string value in the dict
                        for val in data.values():
                            if isinstance(val, str) and val.strip():
                                return val.strip()
                        return str(data)
                    elif isinstance(data, str):
                        return data.strip()
                except Exception:
                    # Not standard JSON, return raw text
                    if raw_text:
                        return raw_text

                return "Tôi đã ghi nhận thông tin của bạn."
            else:
                logger.error(
                    f"n8n webhook returned error status {response.status_code}: {response.text[:200]}"
                )
                return DEFAULT_ERROR_MESSAGE

        except httpx.TimeoutException:
            logger.warning(f"Timeout ({self.timeout}s) waiting for n8n webhook response.")
            return "Xin lỗi bạn, thời gian phản hồi quá lâu. Bạn vui lòng thử lại nhé."

        except httpx.ConnectError:
            logger.warning(
                f"Could not connect to n8n webhook at {target_url}. (Is n8n running?). Providing contextual fallback."
            )
            return f"Tôi đã nghe thấy: \"{text}\". Hiện tại n8n chưa được kết nối, vui lòng kiểm tra webhook."

        except Exception as e:
            logger.error(f"Unexpected error while communicating with n8n: {e}", exc_info=True)
            return DEFAULT_ERROR_MESSAGE
