"""
title: Vision Router MAX - Filter (Ollama side-call + Multi-Image Strategy + Quality Gate + Retry + Verifier + Clarify) [stdlib]
author: cyberjaeger
credits: iamg30, open-webui, atgehrhardt (base), ChatGPT (OpenAI)
version: 0.4.1
required_open_webui_version: 0.3.8
"""

from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any, Optional, Tuple, Literal, Dict, List
import base64
import urllib.parse
import re
import json
import asyncio
import urllib.request
import urllib.error

from open_webui.utils.misc import get_last_user_message_item


class Filter:
    HARD_MAX_IMAGES = 10  # hard-cap for safety/stability

    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )

        # --- Vision model config (Ollama) ---
        vision_model_id: str = Field(
            default="hf.co/openbmb/MiniCPM-V-2_6-gguf:Q4_K_M",
            description="Ollama vision model NAME (from `ollama list`), not the digest.",
        )
        ollama_base_url: str = Field(
            default="http://localhost:11434",
            description="Ollama base URL reachable from OpenWebUI container.",
        )
        vision_timeout_s: int = Field(
            default=120, description="Timeout for vision model analysis in seconds."
        )
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
        temperature: float = Field(
            default=0.2, description="Ollama temperature for vision calls."
        )

        # --- Multi-image strategy ---
        multi_image_strategy: Literal["last", "sequential", "block"] = Field(
            default="last",
            description=(
                "How to handle multiple images in a single user message:\n"
                "- last: send only last image (stable default)\n"
                "- sequential: run full triple-pass per image and inject all results (power-user)\n"
                "- block: if too many images, ask user to resend fewer"
            ),
        )

        max_images_queue: int = Field(
            default=6,
            description=(
                "Maximum number of images allowed in a single user message before blocking. "
                "This is a user-configurable 'queue' limit. Hard-cap is 10."
            ),
        )

        max_images_sequential: int = Field(
            default=6,
            description=(
                "In sequential mode: maximum number of images to process sequentially. "
                "Hard-cap is 10."
            ),
        )

        sequential_include_summary: bool = Field(
            default=True,
            description="In sequential mode: append a short merged summary at the end of injected vision results.",
        )

        per_image_timeout_s: int = Field(
            default=90,
            description="Sequential mode: timeout per image vision call (seconds). Helps avoid queue stalls.",
        )

        # --- Controls / exclusions ---
        skip_reroute_models: list[str] = Field(
            default_factory=list, description="Models that bypass this filter."
        )
        enabled_for_admins: bool = Field(
            default=True, description="Enable for admin users."
        )
        enabled_for_users: bool = Field(
            default=True, description="Enable for regular users."
        )

        # --- UX ---
        status: bool = Field(default=True, description="Emit status indicator events.")

        # --- Performance / low-hardware guards ---
        max_output_tokens: int = Field(
            default=650,
            description="Ollama num_predict cap for side-calls (lower = faster, less CPU/RAM).",
        )
        max_concurrent_sidecalls: int = Field(
            default=1,
            description="Limit concurrent Ollama side-calls to avoid overwhelming small machines.",
        )
        max_injected_chars: int = Field(
            default=12000,
            description="Trim injected vision bundle to this many characters to protect small main LLM context.",
        )

        # --- Language / i18n ---
        language_preset: str = Field(
            default="auto",
            description="Prompt/content language: auto (match user) or a fixed code like en, pl, de, fr, es, it. Headers remain in English.",
        )
        text_request_language_pack: str = Field(
            default="en",
            description="Keyword pack for detecting 'read the text' requests: en | pl | de | fr | es | it | auto.",
        )

        # --- Verifier compute policy ---
        verifier_mode: Literal["always", "on_low_quality", "never"] = Field(
            default="on_low_quality",
            description="When to run verifier pass. 'on_low_quality' saves compute on limited hardware.",
        )

        # --- Injection behavior ---
        inject_as_role: str = Field(
            default="system",
            description="Role used for injected vision analysis message.",
        )

        # --- Vision prompts (generic) ---
        vision_prompt: str = Field(
            default=(
                "You are a vision analysis engine.\n"
                "Task:\n"
                "1) Describe concretely what is in the image (no generic filler).\n"
                "2) If there is text, do OCR and output it (use [ILLEGIBLE] where unclear).\n"
                "3) Provide DETAILS as bullet points.\n"
                "4) State uncertainty explicitly.\n\n"
                "Return EXACTLY:\n"
                "SUMMARY:\n<...>\n\n"
                "DETAILS (bullets):\n- ...\n\n"
                "OCR:\n<...>\n\n"
                "UNCERTAINTY:\n<...>"
            ),
            description="First-pass prompt.",
        )

        repair_prompt: str = Field(
            default=(
                "Improve the previous vision answer.\n"
                "Rules:\n"
                "- Be more specific and grounded.\n"
                "- Expand DETAILS with more bullets.\n"
                "- OCR all visible text (even partial; use [ILLEGIBLE]).\n"
                "- Do NOT hallucinate. If unsure, say what is unclear and what would resolve it.\n\n"
                "Return EXACTLY:\n"
                "SUMMARY:\n<...>\n\n"
                "DETAILS (bullets):\n- ...\n\n"
                "OCR:\n<...>\n\n"
                "UNCERTAINTY:\n<...>"
            ),
            description="Retry prompt when first output is weak.",
        )

        ocr_focus_prompt: str = Field(
            default=(
                "You are doing OCR-only extraction.\n"
                "Extract ALL visible text as faithfully as possible.\n"
                "Keep line breaks. Use [ILLEGIBLE] where unreadable.\n\n"
                "Return EXACTLY:\n"
                "OCR:\n<...>"
            ),
            description="Optional OCR-only pass if OCR seems missing.",
        )

        # --- Catch-all verifier pass (self-consistency) ---
        enable_verifier_pass: bool = Field(
            default=True,
            description="Run a verification call to detect low grounding / hallucination risk (domain-agnostic).",
        )
        verifier_prompt: str = Field(
            default=(
                "You are a strict verifier.\n"
                "You will receive (A) the user's question and (B) a candidate vision analysis.\n"
                "You can look at the image again.\n"
                "Your job: judge whether the analysis is grounded in the image and useful.\n"
                "Do NOT add new details to the candidate. Only judge quality.\n\n"
                "Output EXACTLY this format:\n"
                "VERDICT: PASS or FAIL\n"
                "REASON:\n<one short paragraph>\n"
                "MISSING:\n- <bullet list of what is missing/unclear>\n"
                "HALLUCINATION_RISK: LOW/MEDIUM/HIGH"
            ),
            description="Prompt for verification pass.",
        )
        verifier_fail_on_high_risk: bool = Field(
            default=True, description="If hallucination risk is HIGH, treat as FAIL."
        )

        main_model_instruction: str = Field(
            default=(
                "You have been provided VISION ANALYSIS from a vision model.\n"
                "Use it as context to answer the user.\n"
                "Sanity-check it.\n"
                "If the vision analysis is ambiguous/low-confidence/insufficient, ask 1–2 targeted clarifying questions.\n"
            ),
            description="Instruction to the main model.",
        )

        clarify_instruction: str = Field(
            default=(
                "QUALITY-GATE TRIPPED: The vision result looks unreliable or too generic.\n"
                "Ask 1–2 targeted clarifying questions and/or request a higher-resolution image or a crop focusing on relevant area/text.\n"
                "Then proceed only with what is safe."
            ),
            description="Injected if vision remains weak or fails verifier.",
        )

        # --- Quality-gate settings ---
        enable_quality_gate: bool = Field(
            default=True, description="Enable quality scoring."
        )
        enable_vision_retry: bool = Field(
            default=True, description="Retry once when weak."
        )
        enable_ocr_focus_pass: bool = Field(
            default=True, description="OCR-only pass when needed."
        )

        quality_threshold: int = Field(default=75, description="Score threshold 0–100.")
        min_chars: int = Field(default=220, description="Min characters expected.")
        min_bullets: int = Field(default=3, description="Min bullet lines expected.")
        generic_phrases: list[str] = Field(
            default_factory=lambda: [
                "not provided in the image",
                "cannot determine",
                "hard to tell",
                "it appears to be",
                "diagram illustrates",
                "flowchart",
                "process map",
                "various stages",
                "relationship between",
                "connection between",
                "some kind of",
                "various components",
                "the direction of the arrows indicates",
            ],
            description="Phrases that often indicate low-signal generic output.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._sem_limit = int(getattr(self.valves, "max_concurrent_sidecalls", 1) or 1)
        self._sem = asyncio.Semaphore(max(1, self._sem_limit))

    
    def _get_sem(self) -> asyncio.Semaphore:
        limit = int(getattr(self.valves, "max_concurrent_sidecalls", 1) or 1)
        limit = max(1, limit)
        if getattr(self, "_sem_limit", None) != limit:
            self._sem_limit = limit
            self._sem = asyncio.Semaphore(limit)
        return self._sem

    def _trim(self, s: str, limit: int) -> str:
        if not s:
            return ""
        if limit <= 0 or len(s) <= limit:
            return s
        return s[:limit] + "\n\n[TRIMMED]"

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

    def _candidate_openwebui_urls(self, file_id: str) -> List[str]:
        base = (self.valves.openwebui_base_url or "").rstrip("/")
        if not base:
            return []
        return [
            base + self.valves.openwebui_file_content_path.format(id=file_id),
            base + self.valves.openwebui_file_metadata_path.format(id=file_id),
        ]

    def _fetch_image_as_base64(self, url: str) -> Optional[str]:
        if not url or not isinstance(url, str):
            return None
        if url.startswith("data:image") and "base64," in url:
            return url.split("base64,", 1)[1]
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme in {"http", "https"}:
            return self._fetch_url_as_base64(url)
        if url.startswith("/") and self.valves.openwebui_base_url:
            candidate = self.valves.openwebui_base_url.rstrip("/") + url
            return self._fetch_url_as_base64(candidate)
        if self._is_uuid_like(url):
            for candidate in self._candidate_openwebui_urls(url):
                fetched = self._fetch_url_as_base64(candidate)
                if fetched:
                    return fetched
        return None

    def _detect_lang(self, text: str) -> str:
        # lightweight heuristic, no extra deps (best-effort)
        t = (text or "").lower()

        # Polish
        if any(ch in t for ch in ["ą", "ć", "ę", "ł", "ń", "ó", "ś", "ż", "ź"]):
            return "pl"
        if re.search(r"\b(co|jak|dlaczego|proszę|czy|jest|to|nie)\b", t) and re.search(r"\b(jest|nie|co)\b", t):
            return "pl"

        # Spanish
        if any(ch in t for ch in ["¿", "¡", "ñ", "á", "é", "í", "ó", "ú"]) or re.search(r"\b(que\s+dice|lee|texto)\b", t):
            return "es"

        # German
        if any(ch in t for ch in ["ä", "ö", "ü", "ß"]) or re.search(r"\b(bitte|was\s+steht|lies|text)\b", t):
            return "de"

        # French
        if any(ch in t for ch in ["ç", "œ"]) or re.search(r"\b(qu['’]est-ce|lire|texte)\b", t):
            return "fr"

        # Italian
        if re.search(r"\b(cosa\s+c['’]è\s+scritto|leggi|testo)\b", t):
            return "it"

        return "en"

    def _lang_hint(self, user_text: str) -> str:
        preset = (getattr(self.valves, "language_preset", "auto") or "auto").strip().lower()
        if preset == "auto":
            return "Use the same language as the USER QUESTION for the content. If unclear/mixed, use English. Keep section headers in English."
        return f"Use language '{preset}' for the content. Keep section headers in English."


# ---------- detect images ----------
    def _has_images(self, user_message: dict) -> bool:
        if not user_message:
            return False

        raw = user_message.get("images", None)
        # Open WebUI may include "images": [] even when no images are attached
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
            # some other truthy representation
            return True

        files = user_message.get("files")
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

        content = user_message.get("content")
        if isinstance(content, list):
            return any(
                isinstance(item, dict) and item.get("type") == "image_url"
                for item in content
            )
        return False

    # ---------- extract base64 images ----------
    def _extract_base64_images(self, user_message: dict) -> list[str]:
        imgs: list[str] = []

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

        raw = user_message.get("images")
        if isinstance(raw, list):
            for it in raw:
                normalized = _normalize_image(it)
                if normalized:
                    imgs.append(normalized)

        content = user_message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = (item.get("image_url") or {}).get("url")
                    normalized = _normalize_image(url)
                    if normalized:
                        imgs.append(normalized)

        files = user_message.get("files")
        if isinstance(files, list):
            for item in files:
                normalized = _normalize_image(item)
                if normalized:
                    imgs.append(normalized)

        # de-dupe preserve order
        imgs = list(dict.fromkeys(imgs))
        return imgs

    # ---------- user text only ----------
    def _get_user_text_only(self, user_message: dict) -> str:
        c = user_message.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = []
            for item in c:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "\n".join([p for p in parts if p])
        return ""

    # ---------- format helpers ----------
    def _has_required_sections(self, txt: str) -> bool:
        low = (txt or "").lower()
        return (
            ("summary:" in low)
            and ("details" in low)
            and ("ocr:" in low)
            and ("uncertainty" in low)
        )

    def _extract_section(self, txt: str, header: str) -> str:
        if not txt:
            return ""
        pat = rf"(?is){re.escape(header)}\s*:\s*(.*?)(?=\n[A-Z][A-Z ]+\s*:|\Z)"
        m = re.search(pat, txt)
        return m.group(1).strip() if m else ""

    # ---------- quality gate ----------
    def _quality_score(self, txt: str) -> Tuple[int, list[str]]:
        reasons: list[str] = []
        if not txt or not isinstance(txt, str):
            return 0, ["empty"]

        t = txt.strip()
        low = t.lower()
        score = 100

        if not self._has_required_sections(t):
            score -= 20
            reasons.append("bad_format")

        if len(t) < self.valves.min_chars:
            score -= 30
            reasons.append("too_short")

        hits = sum(1 for p in self.valves.generic_phrases if p in low)
        if hits >= 2:
            score -= 20
            reasons.append("too_generic")

        bullet_count = len(re.findall(r"(?m)^\s*[-•]\s+", t))
        if bullet_count < self.valves.min_bullets:
            score -= 15
            reasons.append("low_structure")

        ocr = self._extract_section(t, "OCR")
        if len((ocr or "").strip()) < 10:
            score -= 8
            reasons.append("ocr_empty_or_tiny")

        unc = self._extract_section(t, "UNCERTAINTY")
        if len((unc or "").strip()) == 0:
            score -= 5
            reasons.append("no_uncertainty")

        score = max(0, min(100, score))
        return score, reasons

    def _likely_text_request(self, user_text: str) -> bool:
        u = (user_text or "").lower()
        packs = {
            "en": [
                "ocr",
                "read",
                "text",
                "written",
                "label",
                "what does it say",
                "transcribe",
                "extract text",
            ],
            "pl": [
                "co jest napisane",
                "odczytaj",
                "napis",
                "tekst",
                "przepisz",
            ],
            "de": [
                "was steht",
                "lies",
                "text",
                "beschriftung",
                "ocr",
            ],
            "fr": [
                "qu'est-ce qui est écrit",
                "qu'est ce qui est écrit",
                "lis",
                "texte",
                "ocr",
            ],
            "es": [
                "qué dice",
                "que dice",
                "lee",
                "texto",
                "ocr",
            ],
            "it": [
                "cosa c'è scritto",
                "cosa e scritto",
                "leggi",
                "testo",
                "ocr",
            ],
        }

        pack = (getattr(self.valves, "text_request_language_pack", "en") or "en").strip().lower()
        if pack == "auto":
            pack = self._detect_lang(user_text)
        if pack not in packs:
            pack = "en"

        return any(x in u for x in packs[pack])

    # ---------- multi-image selection ----------
    def _clamp_hard(self, n: int) -> int:
        try:
            n = int(n)
        except Exception:
            n = 1
        return max(1, min(self.HARD_MAX_IMAGES, n))

    def _queue_limit(self) -> int:
        # user-configurable, but hard-capped
        return self._clamp_hard(self.valves.max_images_queue)

    def _sequential_limit(self) -> int:
        return self._clamp_hard(self.valves.max_images_sequential)

    def _select_images_by_strategy(
        self, imgs: list[str]
    ) -> Tuple[list[str], Optional[str]]:
        if not imgs:
            return [], None

        qlim = self._queue_limit()
        if len(imgs) > qlim:
            return [], f"BLOCK: received {len(imgs)} images > max_images_queue={qlim}"

        strat = self.valves.multi_image_strategy

        if strat == "last":
            return [imgs[-1]], None

        if strat == "block":
            # already checked queue limit above, so allow remaining
            return imgs, None

        # sequential
        n = min(len(imgs), self._sequential_limit())
        return imgs[:n], None

    
    def _urlopen_text(self, req: urllib.request.Request, timeout_s: int) -> str:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.read().decode("utf-8", errors="replace")

# ---------- stdlib ollama call ----------
    async def _call_vision_ollama(
        self,
        user_message: dict,
        prompt_override: Optional[str] = None,
        extra_context: Optional[str] = None,
        images_override: Optional[list[str]] = None,
        timeout_override_s: Optional[int] = None,
    ) -> str:
        if not self.valves.vision_model_id:
            return "VISION ERROR: No vision_model_id configured."

        images_raw = (
            images_override
            if images_override is not None
            else self._extract_base64_images(user_message)
        )
        if not images_raw:
            return "VISION ERROR: Image detected but no base64 image payload was found to send to Ollama."

        images_b64: list[str] = []
        non_base64: list[str] = []
        for img in images_raw:
            if self._is_base64_image(img):
                images_b64.append(self._normalize_base64(img))
            else:
                fetched = None
                if self.valves.resolve_image_urls:
                    fetched = await asyncio.to_thread(
                        self._fetch_image_as_base64, str(img)
                    )
                if fetched:
                    images_b64.append(fetched)
                else:
                    non_base64.append(img)

        if not images_b64:
            examples = ", ".join([str(v)[:48] for v in non_base64[:3]])
            hint = ""
            if non_base64 and not self.valves.resolve_image_urls:
                hint = " URL resolving is disabled; enable resolve_image_urls to fetch http(s) images."
            return (
                "VISION ERROR: Image detected but no base64 image payload was found to send to Ollama. "
                "Non-base64 image entries were provided. "
                f"Examples: {examples}.{hint}"
            )

        user_text = self._get_user_text_only(user_message)
        prompt_base = prompt_override if prompt_override else self.valves.vision_prompt

        lang_hint = self._lang_hint(user_text)
        prompt = f"{prompt_base}\n\nLANGUAGE:\n{lang_hint}\n\nUSER QUESTION:\n{user_text}"
        if extra_context:
            prompt += f"\n\nEXTRA CONTEXT:\n{extra_context}"

        payload = {
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

        timeout_s = (
            int(timeout_override_s)
            if timeout_override_s is not None
            else int(self.valves.vision_timeout_s)
        )

        try:
            async with self._get_sem():
                raw = await asyncio.to_thread(self._urlopen_text, req, timeout_s)
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

    # ---------- verifier parsing (hard) ----------
    def _parse_verifier(self, txt: str) -> Tuple[bool, str, str]:
        if not txt:
            return False, "verifier_empty", "UNKNOWN"

        verdict_m = re.search(r"(?im)^\s*VERDICT\s*:\s*(PASS|FAIL)\s*$", txt)
        risk_m = re.search(
            r"(?im)^\s*HALLUCINATION_RISK\s*:\s*(LOW|MEDIUM|HIGH)\s*$", txt
        )

        verdict = verdict_m.group(1) if verdict_m else "UNKNOWN"
        risk = risk_m.group(1) if risk_m else "UNKNOWN"

        if self.valves.verifier_fail_on_high_risk and risk == "HIGH":
            return False, "verifier_high_hallucination_risk", risk

        if verdict == "PASS":
            return True, "verifier_pass", risk
        if verdict == "FAIL":
            return False, "verifier_fail", risk

        return False, "verifier_ambiguous", risk

    

    def _extract_missing_bullets(self, verifier_txt: str) -> List[str]:
        """Extracts the bullet list under MISSING: from the verifier output."""
        if not verifier_txt:
            return []
        m = re.search(
            r"(?is)\bMISSING\s*:\s*(.*?)(?:\n\s*HALLUCINATION_RISK\s*:|\Z)",
            verifier_txt,
        )
        block = m.group(1).strip() if m else ""
        bullets = re.findall(r"(?m)^\s*-\s+(.*)$", block)
        clean = [b.strip() for b in bullets if b.strip()]
        return clean[:10]

    def _guess_meta_type(
        self, user_text: str, vision_text: str, ocr_attach: str
    ) -> str:
        """Best-effort classification for follow-up pipes (graph vs document vs photo).

        Keep this lightweight and language-agnostic to support EU languages without hardcoding
        mixed-language keywords in defaults.
        """
        ut = (user_text or "").lower()
        vt = (vision_text or "").lower()

        # Graph/diagram cues (mostly language-agnostic)
        graph_cues = [
            "graph",
            "chart",
            "plot",
            "diagram",
            "flowchart",
            "schematic",
            "x-axis",
            "y-axis",
            "axis",
            "legend",
            "nodes",
            "edges",
            "->",
        ]
        if any(k in ut for k in graph_cues) or any(k in vt for k in graph_cues):
            return "graph"

        # Document/text cues
        doc_cues = [
            "ocr",
            "read",
            "text",
            "transcribe",
            "extract text",
            "what does it say",
        ]
        if any(k in ut for k in doc_cues):
            return "document"

        # If OCR-only attach exists or OCR section is non-empty, likely document
        if (ocr_attach or "").strip():
            return "document"

        return "photo"

    def _suite_marker(
        self,
        meta_type: str,
        needs_followup: bool,
        verifier_missing: List[str],
        num_images: int,
        ocr_present: bool,
    ) -> str:
        """Stable marker contract for the Router Suite follow-up pipes."""
        lines = []
        lines.append("VISION_META:")
        lines.append(f"- type: {meta_type}")
        lines.append(f"- images: {int(num_images)}")
        lines.append(f"- ocr_present: {'yes' if ocr_present else 'no'}")
        lines.append("")
        lines.append(f"VISION_NEEDS_FOLLOWUP: {'YES' if needs_followup else 'NO'}")
        if verifier_missing:
            lines.append("")
            lines.append("VERIFIER_MISSING:")
            for item in verifier_missing[:10]:
                lines.append(f"- {item}")
        return "\n".join(lines)
# ---------- per-image triple-pass ----------
    async def _analyze_one_image(
        self,
        user_message: dict,
        img_b64: str,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        label: str = "",
    ) -> Dict[str, Any]:
        # status helper
        async def st(msg: str, done: bool = False):
            if self.valves.status:
                await __event_emitter__(
                    {"type": "status", "data": {"description": msg, "done": done}}
                )

        timeout_img = (
            self.valves.per_image_timeout_s
            if self.valves.multi_image_strategy == "sequential"
            else self.valves.vision_timeout_s
        )

        await st(f"{label}Vision pass #1…", done=False)
        vision_text = await self._call_vision_ollama(
            user_message,
            images_override=[img_b64],
            timeout_override_s=timeout_img,
        )

        score, reasons = (100, ["quality_gate_disabled"])
        if self.valves.enable_quality_gate:
            score, reasons = self._quality_score(vision_text)

        # retry if weak
        if (
            self.valves.enable_quality_gate
            and self.valves.enable_vision_retry
            and score < self.valves.quality_threshold
        ):
            await st(f"{label}Weak (score={score}) → retry…", done=False)
            retry_text = await self._call_vision_ollama(
                user_message,
                prompt_override=self.valves.repair_prompt,
                images_override=[img_b64],
                timeout_override_s=timeout_img,
            )
            score2, reasons2 = self._quality_score(retry_text)
            if score2 > score:
                vision_text, score, reasons = retry_text, score2, reasons2

        # optional OCR-only pass
        user_text = self._get_user_text_only(user_message)
        ocr_attach = ""
        if self.valves.enable_ocr_focus_pass:
            ocr_section = self._extract_section(vision_text, "OCR")
            if (len((ocr_section or "").strip()) < 10) and self._likely_text_request(
                user_text
            ):
                await st(f"{label}OCR missing → OCR-only pass…", done=False)
                ocr_only = await self._call_vision_ollama(
                    user_message,
                    prompt_override=self.valves.ocr_focus_prompt,
                    images_override=[img_b64],
                    timeout_override_s=timeout_img,
                )
                ocr_attach = f"\n\nOCR-ONLY PASS RESULT:\n{ocr_only}"

        # verifier pass
        verifier_text = ""
        verifier_ok = True
        verifier_reason = "verifier_disabled"
        verifier_risk = "UNKNOWN"

        should_verifier = self.valves.enable_verifier_pass and (self.valves.verifier_mode == "always" or (self.valves.verifier_mode == "on_low_quality" and (score < self.valves.quality_threshold or ("bad_format" in reasons) or ("too_generic" in reasons))))

        if should_verifier:
            await st(f"{label}Verifier pass…", done=False)
            verifier_text = await self._call_vision_ollama(
                user_message,
                prompt_override=self.valves.verifier_prompt,
                extra_context=f"CANDIDATE VISION ANALYSIS:\n{vision_text}",
                images_override=[img_b64],
                timeout_override_s=timeout_img,
            )
            verifier_ok, verifier_reason, verifier_risk = self._parse_verifier(
                verifier_text
            )

        vision_still_weak = self.valves.enable_quality_gate and (
            score < self.valves.quality_threshold
        )
        gate_tripped = vision_still_weak or (not verifier_ok)

        verifier_missing = self._extract_missing_bullets(verifier_text)
        base_ocr = self._extract_section(vision_text, "OCR")
        ocr_present = bool((base_ocr or "").strip()) or bool((ocr_attach or "").strip())

        await st(
            f"{label}Done. score={score}, verifier={verifier_reason}, risk={verifier_risk}",
            done=False,
        )

        return {
            "vision_text": vision_text,
            "score": score,
            "reasons": reasons,
            "ocr_attach": ocr_attach,
            "verifier_text": verifier_text,
            "verifier_ok": verifier_ok,
            "verifier_reason": verifier_reason,
            "verifier_risk": verifier_risk,
            "gate_tripped": gate_tripped,
            "verifier_missing": verifier_missing,
            "ocr_present": ocr_present,
        }

    def _make_short_summary(self, blocks: List[Dict[str, Any]]) -> str:
        # very light summarizer without another LLM call:
        # extract SUMMARY section if present, else first 200 chars.
        lines = []
        for i, b in enumerate(blocks, start=1):
            txt = b.get("vision_text") or ""
            summ = self._extract_section(txt, "SUMMARY")
            if not summ:
                summ = (
                    (txt.strip()[:200] + "…") if len(txt.strip()) > 200 else txt.strip()
                )
            lines.append(f"- Image {i}: {summ}")
        return "\n".join(lines)
    # ---------- main inlet ----------
    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __model__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        # bypass
        if __model__ and __model__.get("id") in self.valves.skip_reroute_models:
            return body
        if __model__ and __model__.get("id") == self.valves.vision_model_id:
            return body

        # role gating
        if __user__ is not None:
            role = __user__.get("role")
            if role == "admin" and not self.valves.enabled_for_admins:
                return body
            if role == "user" and not self.valves.enabled_for_users:
                return body

        messages = body.get("messages") or []
        if not messages:
            return body

        user_message = get_last_user_message_item(messages)
        if not user_message or not self._has_images(user_message):
            return body

        # helper: insert injected message right before last user msg
        def _insert(messages_list: list, injected_msg: dict) -> None:
            try:
                idx = len(messages_list) - 1
                while idx >= 0 and messages_list[idx].get("role") != "user":
                    idx -= 1
                if idx >= 0:
                    messages_list.insert(idx, injected_msg)
                else:
                    messages_list.append(injected_msg)
            except Exception:
                messages_list.append(injected_msg)

        async def st(msg: str, done: bool = False):
            if self.valves.status:
                await __event_emitter__({"type": "status", "data": {"description": msg, "done": done}})

        # Extract / select images
        all_imgs = self._extract_base64_images(user_message)
        selected_imgs, block_reason = self._select_images_by_strategy(all_imgs)

        # If too many images, block with a clear instruction (no crashes)
        if block_reason:
            qlim = self._queue_limit()
            content = (
                f"{self.valves.main_model_instruction}\n"
                f"VISION ROUTER NOTICE: {block_reason}.\n"
                f"Please resend **{qlim} or fewer** images (or crop to the relevant region).\n"
                "If you need multi-image reasoning, set multi_image_strategy to 'sequential' (power-user).\n"
            )
            suite_marker = self._suite_marker("photo", False, [], len(all_imgs) or 0, False)
            content = self._trim(content + "\n\n" + suite_marker, int(self.valves.max_injected_chars))
            injected = {"role": self.valves.inject_as_role, "content": content}
            _insert(messages, injected)
            body["messages"] = messages
            await st("Too many images → asked user to resend fewer.", done=True)
            return body

        strat = self.valves.multi_image_strategy
        user_text = self._get_user_text_only(user_message)

        # --- SEQUENTIAL MODE (per-image triple-pass) ---
        if strat == "sequential":
            n = len(selected_imgs)
            await st(f"Sequential vision: processing {n} image(s)…", done=False)

            blocks: List[Dict[str, Any]] = []
            parts: List[str] = []
            any_gate = False
            missing_all: List[str] = []
            any_ocr = False

            for i, img in enumerate(selected_imgs, start=1):
                b = await self._analyze_one_image(user_message, img, __event_emitter__, label=f"[{i}/{n}] ")
                blocks.append(b)
                any_gate = any_gate or bool(b.get("gate_tripped"))
                any_ocr = any_ocr or bool(b.get("ocr_present"))
                missing_all.extend(b.get("verifier_missing") or [])

                chunk = (
                    f"VISION ANALYSIS RESULT (Image {i}, score={b.get('score')}, reasons={b.get('reasons')}):\n"
                    f"{b.get('vision_text','')}"
                )
                if b.get("ocr_attach"):
                    chunk += f"\n{b['ocr_attach']}"
                if b.get("verifier_text"):
                    chunk += (
                        f"\n\nVERIFIER RESULT ({b.get('verifier_reason')}, risk={b.get('verifier_risk')}):\n"
                        f"{b.get('verifier_text','')}"
                    )
                parts.append(chunk)

            merged_summary = ""
            if self.valves.sequential_include_summary:
                merged_summary = "\n\nMERGED SUMMARY:\n" + self._make_short_summary(blocks)

            first_v = blocks[0].get("vision_text", "") if blocks else ""
            first_o = blocks[0].get("ocr_attach", "") if blocks else ""
            meta_type = self._guess_meta_type(user_text, first_v, first_o)

            missing_dedup = list(dict.fromkeys([m for m in missing_all if isinstance(m, str) and m.strip()]))

            suite_marker = self._suite_marker(meta_type, any_gate, missing_dedup, n, any_ocr)

            extra = f"\n\n{self.valves.clarify_instruction}" if any_gate else ""
            content = (
                f"{self.valves.main_model_instruction}{extra}\n\n"
                f"SEQUENTIAL MODE: {n} image(s) processed (queue_limit={self._queue_limit()}, hard_cap={self.HARD_MAX_IMAGES}).\n\n"
                + "\n\n---\n\n".join(parts)
                + merged_summary
                + "\n\n" + suite_marker
            )
            content = self._trim(content, int(self.valves.max_injected_chars))
            injected = {"role": self.valves.inject_as_role, "content": content}
            _insert(messages, injected)
            body["messages"] = messages
            await st("Sequential vision injected. Continuing with main model.", done=True)
            return body

        # --- LAST MODE (single image) ---
        if strat == "last":
            await st("Vision analysis (last image)…", done=False)
            b = await self._analyze_one_image(user_message, selected_imgs[-1], __event_emitter__, label="")
            meta_type = self._guess_meta_type(user_text, b.get("vision_text",""), b.get("ocr_attach",""))
            suite_marker = self._suite_marker(
                meta_type,
                bool(b.get("gate_tripped")),
                b.get("verifier_missing") or [],
                1,
                bool(b.get("ocr_present")),
            )

            extra = f"\n\n{self.valves.clarify_instruction}" if b.get("gate_tripped") else ""
            content = (
                f"{self.valves.main_model_instruction}{extra}\n\n"
                f"VISION ANALYSIS (model={self.valves.vision_model_id}, score={b.get('score')}, reasons={b.get('reasons')}):\n"
                f"{b.get('vision_text','')}\n"
            )
            if b.get("ocr_attach"):
                content += f"{b['ocr_attach']}\n"
            if b.get("verifier_text"):
                content += (
                    f"\nVERIFIER RESULT ({b.get('verifier_reason')}, risk={b.get('verifier_risk')}):\n"
                    f"{b.get('verifier_text','')}\n"
                )
            content += f"\n{suite_marker}"
            content = self._trim(content, int(self.valves.max_injected_chars))
            injected = {"role": self.valves.inject_as_role, "content": content}
            _insert(messages, injected)
            body["messages"] = messages
            await st("Vision injected. Continuing with main model.", done=True)
            return body

        # --- BLOCK MODE (send selected images as one pack; power-user) ---
        await st(f"Block-mode vision (pack of {len(selected_imgs)} image(s))…", done=False)

        vision_text = await self._call_vision_ollama(
            user_message,
            images_override=selected_imgs,
            timeout_override_s=int(self.valves.vision_timeout_s),
        )
        score, reasons = (100, ["quality_gate_disabled"])
        if self.valves.enable_quality_gate:
            score, reasons = self._quality_score(vision_text)

        if self.valves.enable_quality_gate and self.valves.enable_vision_retry and score < self.valves.quality_threshold:
            retry_text = await self._call_vision_ollama(
                user_message,
                prompt_override=self.valves.repair_prompt,
                images_override=selected_imgs,
                timeout_override_s=int(self.valves.vision_timeout_s),
            )
            score2, reasons2 = self._quality_score(retry_text)
            if score2 > score:
                vision_text, score, reasons = retry_text, score2, reasons2

        ocr_attach = ""
        if self.valves.enable_ocr_focus_pass:
            ocr_section = self._extract_section(vision_text, "OCR")
            if (len((ocr_section or "").strip()) < 10) and self._likely_text_request(user_text):
                ocr_only = await self._call_vision_ollama(
                    user_message,
                    prompt_override=self.valves.ocr_focus_prompt,
                    images_override=selected_imgs,
                    timeout_override_s=int(self.valves.vision_timeout_s),
                )
                ocr_attach = f"\n\nOCR-ONLY PASS RESULT:\n{ocr_only}"

        verifier_text = ""
        verifier_ok = True
        verifier_reason = "verifier_disabled"
        verifier_risk = "UNKNOWN"

        should_verifier = self.valves.enable_verifier_pass and (
            self.valves.verifier_mode == "always"
            or (self.valves.verifier_mode == "on_low_quality" and (score < self.valves.quality_threshold or "bad_format" in reasons or "too_generic" in reasons))
        )

        if should_verifier:
            verifier_text = await self._call_vision_ollama(
                user_message,
                prompt_override=self.valves.verifier_prompt,
                extra_context=f"CANDIDATE VISION ANALYSIS:\n{vision_text}",
                images_override=selected_imgs,
                timeout_override_s=int(self.valves.vision_timeout_s),
            )
            verifier_ok, verifier_reason, verifier_risk = self._parse_verifier(verifier_text)

        gate_tripped = (self.valves.enable_quality_gate and score < self.valves.quality_threshold) or (not verifier_ok)
        verifier_missing = self._extract_missing_bullets(verifier_text)
        base_ocr = self._extract_section(vision_text, "OCR")
        ocr_present = bool((base_ocr or "").strip()) or bool((ocr_attach or "").strip())

        meta_type = self._guess_meta_type(user_text, vision_text, ocr_attach)
        suite_marker = self._suite_marker(
            meta_type,
            bool(gate_tripped),
            verifier_missing,
            len(selected_imgs),
            ocr_present,
        )

        extra = f"\n\n{self.valves.clarify_instruction}" if gate_tripped else ""
        content = (
            f"{self.valves.main_model_instruction}{extra}\n\n"
            f"BLOCK MODE (multi-image pack): count={len(selected_imgs)} (queue_limit={self._queue_limit()}, hard_cap={self.HARD_MAX_IMAGES})\n\n"
            f"VISION ANALYSIS (model={self.valves.vision_model_id}, score={score}, reasons={reasons}):\n{vision_text}\n"
        )
        if ocr_attach:
            content += f"{ocr_attach}\n"
        if verifier_text:
            content += f"\nVERIFIER RESULT ({verifier_reason}, risk={verifier_risk}):\n{verifier_text}\n"
        content += f"\n{suite_marker}"

        content = self._trim(content, int(self.valves.max_injected_chars))
        injected = {"role": self.valves.inject_as_role, "content": content}
        _insert(messages, injected)
        body["messages"] = messages
        await st("Block-mode vision injected. Continuing with main model.", done=True)
        return body
