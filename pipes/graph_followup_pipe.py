"""
title: Vision Router MAX - Graph Follow-up Pipe (uses verifier MISSING list)
author: cyberjaeger
version: 0.4.1
required_open_webui_version: 0.3.8
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple
import base64
import json
import re
import time
import asyncio
import urllib.parse
import urllib.request
import urllib.error


class Pipe:
    def __init__(self):
        self.valves = self.Valves()
        self._sem_limit = int(getattr(self.valves, "max_concurrent_sidecalls", 1) or 1)
        self._sem = asyncio.Semaphore(max(1, self._sem_limit))
        self._last_run_ts = 0.0

    class Valves(BaseModel):
        ollama_base_url: str = Field(default="http://localhost:11434")
        vision_model_id: str = Field(default="hf.co/openbmb/MiniCPM-V-2_6-gguf:Q4_K_M")

        timeout_s: int = Field(
            default=240, description="Timeout for graph follow-up call."
        )
        temperature: float = Field(default=0.2)

        # low-hardware guards
        max_output_tokens: int = Field(default=650, description="Ollama num_predict cap for follow-up call.")
        max_concurrent_sidecalls: int = Field(default=1, description="Limit concurrent Ollama side-calls.")
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

        # language / i18n
        language_preset: str = Field(default="auto", description="auto|en|pl|de|fr|es|it|... (headers stay in English)")
        trigger_language_pack: str = Field(default="en", description="Trigger keyword pack: en|pl|de|fr|es|it|auto")
        additional_trigger_keywords: List[str] = Field(default_factory=list, description="Extra trigger keywords (any language).")

        # Triggering
        enable: bool = Field(default=True)
        trigger_keywords: List[str] = Field(
            default_factory=lambda: [
                "unclear",
                "not sure",
                "missing",
                "insufficient",
                "can't tell",
                "please crop",
                "crop",
                "zoom",
            ],
            description="If assistant response contains any of these, pipe may run.",
        )

        # Only run when vision meta says graph
        require_graph_type: bool = Field(default=True)

        # Safety
        max_missing_items: int = Field(
            default=8, description="Cap MISSING bullets included in follow-up."
        )
        cooldown_s: int = Field(
            default=12,
            description="Avoid repeated follow-ups too frequently in one conversation.",
        )

        # Output role
        inject_as_role: str = Field(default="system")

        # Follow-up prompt (graph-only)
        graph_followup_prompt: str = Field(
            default=(
                "You are a graph/diagram follow-up extractor.\n"
                "You will re-check the SAME image.\n"
                "Focus ONLY on the requested missing/unclear items.\n"
                "Do NOT interpret meaning. Do NOT invent nodes/edges.\n"
                "If the image does not contain enough detail, specify EXACT crop/zoom needed.\n\n"
                "Return EXACTLY:\n"
                "ADDED_FINDINGS (bullets):\n- ...\n\n"
                "CONFIDENCE:\nLOW/MEDIUM/HIGH\n\n"
                "NEEDED_CROP_OR_ZOOM:\n- ...\n\n"
                "UNCERTAINTY:\n<...>"
            )
        )


    def _urlopen_text(self, req: urllib.request.Request, timeout_s: int) -> str:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.read().decode("utf-8", errors="replace")

    # -------- helpers --------

    def _get_sem(self) -> asyncio.Semaphore:
        limit = int(getattr(self.valves, "max_concurrent_sidecalls", 1) or 1)
        limit = max(1, limit)
        if getattr(self, "_sem_limit", None) != limit:
            self._sem_limit = limit
            self._sem = asyncio.Semaphore(limit)
        return self._sem

    def _trigger_keywords(self) -> List[str]:
        packs = {
            "en": [
                "unclear",
                "not sure",
                "missing",
                "insufficient",
                "can't tell",
                "please crop",
                "crop",
                "zoom",
            ],
            "pl": [
                "niejasne",
                "brakuje",
                "nie widać",
                "zrób zbliżenie",
                "wyślij wycinek",
                "zoom",
            ],
            "de": ["unklar", "fehlt", "nicht sicher", "bitte zuschneiden", "zoom"],
            "fr": ["flou", "manque", "pas sûr", "recadrer", "zoom"],
            "es": ["no está claro", "falta", "no estoy seguro", "recorta", "zoom"],
            "it": ["non chiaro", "manca", "non sono sicuro", "ritaglia", "zoom"],
        }
        pack = (getattr(self.valves, "trigger_language_pack", "en") or "en").strip().lower()
        base = packs.get(pack, packs["en"]) if pack != "auto" else []
        extra = list(self.valves.trigger_keywords or [])
        addl = list(getattr(self.valves, "additional_trigger_keywords", []) or [])
        if pack == "auto":
            base = packs["en"] + packs["pl"] + packs["de"] + packs["fr"] + packs["es"] + packs["it"]
        return list(dict.fromkeys(base + extra + addl))

    def _lang_hint(self, user_text: str) -> str:
        preset = (getattr(self.valves, "language_preset", "auto") or "auto").strip().lower()
        if preset == "auto":
            return "Use the same language as the user context for the content. If unclear/mixed, use English. Keep section headers in English."
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

    async def _ollama_chat(self, prompt: str, images_b64: List[str]) -> str:
        payload: Dict[str, Any] = {
            "model": self.valves.vision_model_id,
            "stream": False,
            "messages": [{"role": "user", "content": prompt, "images": images_b64}],
            "options": {"temperature": self.valves.temperature, "num_predict": int(getattr(self.valves, "max_output_tokens", 650) or 650)},
        }

        url = self.valves.ollama_base_url.rstrip("/") + "/api/chat"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )

        try:
            async with self._get_sem():
                raw = await asyncio.to_thread(
                    lambda: urllib.request.urlopen(req, timeout=self.valves.timeout_s)
                    .read()
                    .decode("utf-8", errors="replace")
                )
            j = json.loads(raw)
            return j.get("message", {}).get(
                "content", f"VISION ERROR: Unexpected response: {j}"
            )
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            return f"VISION ERROR: HTTP {e.code} {e.reason}\n{body}"
        except Exception as e:
            return f"VISION ERROR: {repr(e)}"

    def _has_images(self, msg: Dict[str, Any]) -> bool:
        if not msg:
            return False
        raw = msg.get("images", None)
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
            return False
        if isinstance(raw, str):
            return bool(raw.strip())
        if raw is not None:
            return True
        c = msg.get("content")
        if isinstance(c, list):
            return any(
                isinstance(item, dict) and item.get("type") == "image_url" for item in c
            )
        files = msg.get("files")
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

    def _extract_base64_images(self, msg: Dict[str, Any]) -> List[str]:
        imgs: List[str] = []

        def _normalize_image(value: Any) -> Optional[str]:
            if not value:
                return None
            if isinstance(value, str):
                if value.startswith("data:image") and "base64," in value:
                    try:
                        return value.split("base64,", 1)[1]
                    except Exception:
                        return None
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
                        try:
                            return url.split("base64,", 1)[1]
                        except Exception:
                            return None
                    return url.strip()
                file_obj = value.get("file")
                if isinstance(file_obj, dict):
                    data = file_obj.get("data") or file_obj.get("base64") or file_obj.get("b64")
                    if isinstance(data, str) and data.strip():
                        return data.strip()
                    url = file_obj.get("url") or file_obj.get("id")
                    if isinstance(url, str) and url.strip():
                        if url.startswith("data:image") and "base64," in url:
                            try:
                                return url.split("base64,", 1)[1]
                            except Exception:
                                return None
                        return url.strip()
            return None

        raw = msg.get("images")
        if isinstance(raw, list):
            for it in raw:
                normalized = _normalize_image(it)
                if normalized:
                    imgs.append(normalized)

        c = msg.get("content")
        if isinstance(c, list):
            for item in c:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = (item.get("image_url") or {}).get("url")
                    normalized = _normalize_image(url)
                    if normalized:
                        imgs.append(normalized)

        files = msg.get("files")
        if isinstance(files, list):
            for item in files:
                normalized = _normalize_image(item)
                if normalized:
                    imgs.append(normalized)

        # de-dupe
        return list(dict.fromkeys(imgs))

    def _count_followups(self, messages: List[Dict[str, Any]]) -> int:
        return sum(1 for m in messages or [] if isinstance(m.get("content"), str) and "GRAPH_FOLLOWUP_PIPE_RESULT" in m["content"])

    def _last_assistant_text(self, messages: List[Dict[str, Any]]) -> str:
        for m in reversed(messages or []):
            if m.get("role") == "assistant":
                c = m.get("content")
                if isinstance(c, str):
                    return c
        return ""

    def _triggered(self, assistant_text: str) -> bool:
        t = (assistant_text or "").lower()
        kws = self._trigger_keywords()
        return any((k or "").lower() in t for k in kws)

    def _extract_vision_meta_type(
        self, messages: List[Dict[str, Any]]
    ) -> Optional[str]:
        # Parse from injected VISION_META: "- type: graph"
        for m in reversed(messages or []):
            c = m.get("content") or ""
            if isinstance(c, str) and "VISION_META:" in c and "- type:" in c:
                mm = re.search(r"(?im)^\s*-\s*type\s*:\s*([a-z]+)\s*$", c)
                if mm:
                    return mm.group(1).strip().lower()
        return None

    def _extract_verifier_missing(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract missing/unclear items from the Router Suite injection.

        Priority:
        1) VERIFIER_MISSING: bullet list (suite marker contract)
        2) VERIFIER RESULT ... then parse MISSING: bullets inside the verifier text
        """
        text = ""
        for m in reversed(messages or []):
            c = m.get("content") or ""
            if not isinstance(c, str):
                continue
            if "VERIFIER_MISSING:" in c or "VERIFIER RESULT" in c:
                text = c
                break
        if not text:
            return []

        # 1) VERIFIER_MISSING block
        m1 = re.search(
            r"(?is)\bVERIFIER_MISSING\s*:\s*\n(.*?)(?:\n\s*[A-Z_ ]+\s*:\s*|\Z)",
            text,
        )
        if m1:
            bullets = re.findall(r"(?m)^\s*-\s+(.*)$", m1.group(1))
            clean = [b.strip() for b in bullets if b.strip()]
            return clean[: self.valves.max_missing_items]

        # 2) parse MISSING inside verifier result
        # try to isolate verifier text after 'VERIFIER RESULT'
        m2 = re.search(r"(?is)\bVERIFIER RESULT\b.*?:\s*\n(.*)$", text)
        verifier_txt = m2.group(1).strip() if m2 else text

        m3 = re.search(
            r"(?is)\bMISSING\s*:\s*(.*?)(?:\n\s*HALLUCINATION_RISK\s*:|\Z)",
            verifier_txt,
        )
        block = m3.group(1).strip() if m3 else ""
        bullets = re.findall(r"(?m)^\s*-\s+(.*)$", block)
        clean = [b.strip() for b in bullets if b.strip()]
        return clean[: self.valves.max_missing_items]


    def _find_last_user_with_images(
        self, messages: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        for m in reversed(messages or []):
            if m.get("role") == "user" and self._has_images(m):
                return m
        return None

    # -------- main pipe --------

    async def pipe(self, body: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if not self.valves.enable:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        if self._count_followups(messages) >= int(self.valves.max_followups_per_chat):
            return body

        # cooldown
        now = time.time()
        if (now - self._last_run_ts) < float(self.valves.cooldown_s):
            return body

        assistant_text = self._last_assistant_text(messages)
        meta_type = self._extract_vision_meta_type(messages)
        keyword_trigger = self._triggered(assistant_text)
        meta_trigger = bool(self.valves.run_on_meta_type and meta_type == "graph")
        if not (keyword_trigger or meta_trigger):
            return body

        # require graph type?
        if self.valves.require_graph_type:
            vtype = self._extract_vision_meta_type(messages)
            if vtype is None or vtype != "graph":
                return body

        missing = self._extract_verifier_missing(messages)
        if not missing:
            # If we can't find verifier missing list, do nothing (avoid random follow-ups)
            return body

        user_msg = self._find_last_user_with_images(messages)
        if not user_msg:
            return body

        images_b64 = self._extract_base64_images(user_msg)
        if not images_b64:
            return body
        images_b64 = await self._resolve_images(images_b64)
        if not images_b64:
            return body

        missing_txt = "\n".join([f"- {x}" for x in missing])
        lang_hint = self._lang_hint(missing_txt)
        prompt = f"{self.valves.graph_followup_prompt}\n\nLANGUAGE:\n{lang_hint}\n\nMISSING ITEMS:\n{missing_txt}"

        followup = await self._ollama_chat(prompt, images_b64)

        messages.append(
            {
                "role": self.valves.inject_as_role,
                "content": (
                    "GRAPH_FOLLOWUP_PIPE_RESULT:\n"
                    f"(model={self.valves.vision_model_id})\n\n"
                    f"MISSING_ITEMS_USED:\n{missing_txt}\n\n"
                    f"{followup}"
                ),
            }
        )

        body["messages"] = messages
        self._last_run_ts = now
        return body
