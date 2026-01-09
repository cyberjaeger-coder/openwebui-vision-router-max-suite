"""
title: Vision Router MAX - Vision Follow-up Pipe (marker-driven, reuse last image + new context)
author: cyberjaeger
version: 0.4.1
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import base64
import json
import re
import asyncio
import urllib.parse
import urllib.request
import urllib.error


class Pipe:
    def __init__(self):
        self.valves = self.Valves()
        self._sem_limit = int(getattr(self.valves, "max_concurrent_sidecalls", 1) or 1)
        self._sem = asyncio.Semaphore(max(1, self._sem_limit))

    class Valves(BaseModel):
        ollama_base_url: str = Field(default="http://localhost:11434")
        vision_model_id: str = Field(default="hf.co/openbmb/MiniCPM-V-2_6-gguf:Q4_K_M")
        timeout_s: int = Field(default=120)

        # low-hardware guards
        max_output_tokens: int = Field(default=650, description="Ollama num_predict cap for follow-up call.")
        max_concurrent_sidecalls: int = Field(default=1, description="Limit concurrent Ollama side-calls.")

        # language
        language_preset: str = Field(default="auto", description="auto|en|pl|de|fr|es|it|... (headers stay in English)")

        # anti-loop
        max_followups_per_chat: int = Field(default=2)
        send_only_last_image: bool = Field(default=True)
        resolve_image_urls: bool = Field(
            default=True,
            description="Attempt to download http(s) image URLs and convert them to base64.",
        )
        openwebui_base_url: str = Field(
            default="http://localhost:8080",
            description="Base URL for OpenWebUI to resolve file IDs into image bytes.",
        )
        openwebui_file_content_path: str = Field(
            default="/api/files/{id}/content",
            description="Path template to fetch file bytes from OpenWebUI.",
        )
        openwebui_file_metadata_path: str = Field(
            default="/api/files/{id}",
            description="Fallback path template when file content endpoint is unavailable.",
        )
        resolve_image_timeout_s: int = Field(
            default=10, description="Timeout for fetching image URLs in seconds."
        )
        resolve_image_max_bytes: int = Field(
            default=5_000_000,
            description="Max bytes to read when resolving image URLs to base64.",
        )

        # marker used by Filter
        followup_marker: str = Field(default="VISION_NEEDS_FOLLOWUP: YES")

    def _urlopen_text(self, req: urllib.request.Request, timeout_s: int) -> str:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.read().decode("utf-8", errors="replace")

    # ---------- helpers ----------
    def _get_sem(self) -> asyncio.Semaphore:
        limit = int(getattr(self.valves, "max_concurrent_sidecalls", 1) or 1)
        limit = max(1, limit)
        if getattr(self, "_sem_limit", None) != limit:
            self._sem_limit = limit
            self._sem = asyncio.Semaphore(limit)
        return self._sem

    def _lang_hint(self, user_text: str) -> str:
        preset = (getattr(self.valves, "language_preset", "auto") or "auto").strip().lower()
        if preset == "auto":
            return "Use the same language as the USER NEW CONTEXT for the content. If unclear/mixed, use English. Keep section headers in English."
        return f"Use language '{preset}' for the content. Keep section headers in English."

    def _normalize_base64(self, value: str) -> str:
        return re.sub(r"\s+", "", value or "")

    def _is_base64_image(self, value: str) -> bool:
        if not value or not isinstance(value, str):
            return False
        cleaned = self._normalize_base64(value)
        if not cleaned:
            return False
        try:
            base64.b64decode(cleaned, validate=True)
        except Exception:
            return False
        return True

    def _is_uuid_like(self, value: str) -> bool:
        if not value or not isinstance(value, str):
            return False
        return bool(
            re.fullmatch(
                r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                value.strip(),
            )
        ) or bool(re.fullmatch(r"[0-9a-fA-F]{32}", value.strip()))

    def _extract_b64_from_json(self, payload: Any) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        candidates = [
            payload.get("data"),
            payload.get("base64"),
            payload.get("b64"),
            payload.get("content"),
        ]
        file_obj = payload.get("file")
        if isinstance(file_obj, dict):
            candidates.extend(
                [
                    file_obj.get("data"),
                    file_obj.get("base64"),
                    file_obj.get("b64"),
                    file_obj.get("content"),
                ]
            )
        for item in candidates:
            if isinstance(item, str) and item.startswith("data:image") and "base64," in item:
                return item.split("base64,", 1)[1]
            if isinstance(item, str) and self._is_base64_image(item):
                return self._normalize_base64(item)
        return None

    def _candidate_openwebui_urls(self, file_id: str) -> List[str]:
        base = (self.valves.openwebui_base_url or "").rstrip("/")
        if not base:
            return []
        return [
            base + self.valves.openwebui_file_content_path.format(id=file_id),
            base + self.valves.openwebui_file_metadata_path.format(id=file_id),
        ]

    def _fetch_url_as_base64(self, url: str) -> Optional[str]:
        if not url or not isinstance(url, str):
            return None
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return None
        try:
            req = urllib.request.Request(
                url, headers={"Accept": "application/octet-stream,application/json"}
            )
            with urllib.request.urlopen(
                req, timeout=int(self.valves.resolve_image_timeout_s)
            ) as resp:
                chunk = resp.read(int(self.valves.resolve_image_max_bytes) + 1)
                content_type = resp.headers.get("Content-Type", "")
            if len(chunk) > int(self.valves.resolve_image_max_bytes):
                return None
            if "application/json" in content_type:
                try:
                    payload = json.loads(chunk.decode("utf-8", errors="replace"))
                except Exception:
                    payload = None
                extracted = self._extract_b64_from_json(payload)
                if extracted:
                    return extracted
            return base64.b64encode(chunk).decode("utf-8")
        except Exception:
            return None

    async def _resolve_images(self, images: List[str]) -> List[str]:
        resolved: List[str] = []
        for img in images or []:
            if not isinstance(img, str):
                continue
            if img.startswith("data:image") and "base64," in img:
                resolved.append(img.split("base64,", 1)[1])
                continue
            if self._is_base64_image(img):
                resolved.append(self._normalize_base64(img))
                continue
            if not self.valves.resolve_image_urls:
                continue
            fetched = None
            parsed = urllib.parse.urlparse(img)
            if parsed.scheme in {"http", "https"}:
                fetched = await asyncio.to_thread(self._fetch_url_as_base64, img)
            elif img.startswith("/") and self.valves.openwebui_base_url:
                candidate = self.valves.openwebui_base_url.rstrip("/") + img
                fetched = await asyncio.to_thread(self._fetch_url_as_base64, candidate)
            elif self._is_uuid_like(img):
                for candidate in self._candidate_openwebui_urls(img):
                    fetched = await asyncio.to_thread(self._fetch_url_as_base64, candidate)
                    if fetched:
                        break
            if fetched:
                resolved.append(fetched)
        return resolved


    def _count_followups(self, messages: List[Dict[str, Any]]) -> int:
        return sum(
            1
            for m in messages
            if isinstance(m.get("content"), str)
            and "VISION FOLLOW-UP RESULT" in m["content"]
        )

    def _find_last_marker_block(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        for m in reversed(messages):
            c = m.get("content")
            if m.get("role") in ("system", "developer") and isinstance(c, str):
                if self.valves.followup_marker in c:
                    return c
        return None

    def _extract_missing(self, marker_block: str) -> str:
        """
        Optional: allow Filter to pass a list of missing items in a block:
        VERIFIER_MISSING:
        - ...
        - ...
        """
        if not marker_block:
            return ""
        m = re.search(
            r"(?im)^\s*VERIFIER_MISSING\s*:\s*\n(.*?)(\n[A-Z_ ]+:\s*|\Z)",
            marker_block,
            re.S,
        )
        if not m:
            return ""
        lines = [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]
        # keep first ~10 lines
        return "\n".join(lines[:10])

    def _last_user_msg(
        self, messages: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        for m in reversed(messages):
            if m.get("role") == "user":
                return m
        return None

    def _user_has_images(self, user_msg: Dict[str, Any]) -> bool:
        if not user_msg:
            return False
        raw = user_msg.get("images")
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    return True
                if isinstance(item, dict):
                    if item.get("data") or item.get("base64") or item.get("b64"):
                        return True
                    image_url = item.get("image_url") or {}
                    if isinstance(image_url, dict) and image_url.get("url"):
                        return True
                    if item.get("url"):
                        return True
        elif isinstance(raw, str):
            return bool(raw.strip())
        elif raw:
            return True
        c = user_msg.get("content")
        if isinstance(c, list):
            return any(
                isinstance(it, dict) and it.get("type") == "image_url" for it in c
            )
        files = user_msg.get("files")
        if isinstance(files, list):
            for item in files:
                if isinstance(item, str) and item.strip():
                    return True
                if isinstance(item, dict):
                    if item.get("data") or item.get("base64") or item.get("b64"):
                        return True
                    if item.get("url"):
                        return True
                    file_obj = item.get("file")
                    if isinstance(file_obj, dict):
                        if file_obj.get("data") or file_obj.get("base64") or file_obj.get("b64"):
                            return True
                        if file_obj.get("url") or file_obj.get("id"):
                            return True
        return False

    def _get_user_text(self, user_msg: Dict[str, Any]) -> str:
        c = user_msg.get("content")
        if isinstance(c, str):
            return c.strip()
        if isinstance(c, list):
            parts = []
            for it in c:
                if isinstance(it, dict) and it.get("type") == "text":
                    parts.append(it.get("text", ""))
            return "\n".join([p for p in parts if p]).strip()
        return ""

    def _find_last_user_with_image(
        self, messages: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        for m in reversed(messages):
            if m.get("role") != "user":
                continue
            raw = m.get("images")
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, str) and item.strip():
                        return m
                    if isinstance(item, dict):
                        if item.get("data") or item.get("base64") or item.get("b64"):
                            return m
                        image_url = item.get("image_url") or {}
                        if isinstance(image_url, dict) and image_url.get("url"):
                            return m
                        if item.get("url"):
                            return m
            elif isinstance(raw, str) and raw.strip():
                return m
            c = m.get("content")
            if isinstance(c, list) and any(
                isinstance(it, dict) and it.get("type") == "image_url" for it in c
            ):
                return m
            files = m.get("files")
            if isinstance(files, list):
                for item in files:
                    if isinstance(item, str) and item.strip():
                        return m
                    if isinstance(item, dict):
                        if item.get("data") or item.get("base64") or item.get("b64"):
                            return m
                        if item.get("url"):
                            return m
                        file_obj = item.get("file")
                        if isinstance(file_obj, dict):
                            if file_obj.get("data") or file_obj.get("base64") or file_obj.get("b64"):
                                return m
                            if file_obj.get("url") or file_obj.get("id"):
                                return m
        return None

    def _extract_images_b64(self, user_msg: Dict[str, Any]) -> List[str]:
        imgs: List[str] = []

        def _normalize_image(value: Any) -> Optional[str]:
            if not value:
                return None
            if isinstance(value, str):
                if value.startswith("data:image") and "base64," in value:
                    return value.split("base64,", 1)[1]
                return value.strip() or None
            if isinstance(value, dict):
                data = value.get("data") or value.get("base64") or value.get("b64")
                if isinstance(data, str) and data.strip():
                    return data.strip()
                url = value.get("url")
                if not isinstance(url, str):
                    image_url = value.get("image_url") or {}
                    if isinstance(image_url, dict):
                        url = image_url.get("url")
                if isinstance(url, str) and url.strip():
                    if url.startswith("data:image") and "base64," in url:
                        return url.split("base64,", 1)[1]
                    return url.strip()
                file_obj = value.get("file")
                if isinstance(file_obj, dict):
                    data = file_obj.get("data") or file_obj.get("base64") or file_obj.get("b64")
                    if isinstance(data, str) and data.strip():
                        return data.strip()
                    url = file_obj.get("url") or file_obj.get("id")
                    if isinstance(url, str) and url.strip():
                        if url.startswith("data:image") and "base64," in url:
                            return url.split("base64,", 1)[1]
                        return url.strip()
            return None

        raw = user_msg.get("images")
        if isinstance(raw, list):
            for it in raw:
                normalized = _normalize_image(it)
                if normalized:
                    imgs.append(normalized)

        content = user_msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = (item.get("image_url") or {}).get("url")
                    normalized = _normalize_image(url)
                    if normalized:
                        imgs.append(normalized)

        files = user_msg.get("files")
        if isinstance(files, list):
            for item in files:
                normalized = _normalize_image(item)
                if normalized:
                    imgs.append(normalized)

        imgs = list(dict.fromkeys(imgs))
        if self.valves.send_only_last_image and len(imgs) > 1:
            imgs = imgs[-1:]
        return imgs

    async def pipe(self, body: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        messages = body.get("messages", [])
        if not messages:
            return body

        # safety limiter
        if self._count_followups(messages) >= self.valves.max_followups_per_chat:
            return body

        # Pipe runs BEFORE assistant generates => last msg is user
        last_user = self._last_user_msg(messages)
        if not last_user:
            return body

        # If user attached a new image, let Filter handle it (do not do follow-up here)
        if self._user_has_images(last_user):
            return body

        # Need marker from previous Filter run
        marker_block = self._find_last_marker_block(messages)
        if not marker_block:
            return body

        # Reuse the last image from history
        user_with_img = self._find_last_user_with_image(messages)
        if not user_with_img:
            return body

        images_b64 = self._extract_images_b64(user_with_img)
        if not images_b64:
            return body

        images_b64 = await self._resolve_images(images_b64)
        if not images_b64:
            return body

        new_context = self._get_user_text(last_user)
        missing = self._extract_missing(marker_block)

        lang_hint = self._lang_hint(new_context)

        prompt = (
            "FOLLOW-UP VISION ANALYSIS (reuse same image; incorporate user's new context).\n\n"
            f"LANGUAGE:\n{lang_hint}\n\n"
            f"USER NEW CONTEXT:\n{new_context}\n\n"
        )
        if missing:
            prompt += f"FOCUS ON THESE MISSING ITEMS:\n{missing}\n\n"

        prompt += (
            "Rules:\n"
            "- Be concrete and grounded.\n"
            "- If diagram/flowchart present: output LABELS (exact OCR), NODES, EDGES (X -> Y).\n"
            "- If text unclear, use [ILLEGIBLE].\n"
            "- Do NOT invent.\n"
            "Return short structured output.\n"
        )

        payload = {
            "model": self.valves.vision_model_id,
            "stream": False,
            "messages": [{"role": "user", "content": prompt, "images": images_b64}],
            "options": {"temperature": 0.2, "num_predict": int(getattr(self.valves, "max_output_tokens", 650) or 650)},
        }

        req = urllib.request.Request(
            self.valves.ollama_base_url.rstrip("/") + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            async with self._get_sem():
                raw = await asyncio.to_thread(self._urlopen_text, req, self.valves.timeout_s)
            data = json.loads(raw)
            followup = data.get("message", {}).get("content", "")
        except Exception as e:
            followup = f"VISION FOLLOW-UP ERROR: {repr(e)}"

        messages.append(
            {
                "role": "system",
                "content": (
                    "VISION FOLLOW-UP RESULT:\n"
                    f"(model={self.valves.vision_model_id})\n"
                    f"{followup}\n\n"
                    "VISION_NEEDS_FOLLOWUP: RESOLVED (one follow-up injected)"
                ),
            }
        )

        body["messages"] = messages
        return body
