import json
from collections import deque
from typing import Deque

MAX_BUFFER_SIZE = 1024 * 1024  # 1MB
BUFFER_CLEANUP_THRESHOLD = MAX_BUFFER_SIZE // 2


class BufferError(Exception):
    """Raised when buffer operations fail"""

    pass


class StreamBuffer:
    def __init__(self) -> None:
        self.chunks: Deque[str] = deque()
        self.total_bytes: int = 0
        self.last_valid_json_pos: int = 0

    def write(self, content: str) -> None:
        chunk_bytes = len(content.encode("utf-8"))
        if self.total_bytes + chunk_bytes > MAX_BUFFER_SIZE:
            raise BufferError(
                f"Buffer size limit ({MAX_BUFFER_SIZE} bytes) " "exceeded"
            )
        self.chunks.append(content)
        self.total_bytes += chunk_bytes

    def getvalue(self) -> str:
        return "".join(self.chunks)

    def cleanup(self) -> None:
        """Find and preserve the last complete JSON object"""
        content = self.getvalue()
        try:
            last_brace = content.rstrip().rfind("}")
            if last_brace > self.last_valid_json_pos:
                json.loads(content[: last_brace + 1])  # Validate JSON
                self.last_valid_json_pos = last_brace + 1
                new_content = content[self.last_valid_json_pos :]
                self.chunks.clear()
                if new_content:
                    self.chunks.append(new_content)
                self.total_bytes = len(new_content.encode("utf-8"))
        except json.JSONDecodeError:
            pass

    def close(self) -> None:
        """Reset the buffer state"""
        self.chunks.clear()
        self.total_bytes = 0
        self.last_valid_json_pos = 0
