<p align="center">
  <a href="https://github.com/cyberjaeger-coder/openwebui-vision-router-max-suite">
    <img src="https://img.shields.io/badge/Open%20WebUI-Vision%20Router%20MAX%20Suite-1f6feb" alt="Repo">
  </a>
  </a>
  <a href="https://github.com/cyberjaeger-coder/openwebui-vision-router-max-suite/releases">
    <img src="https://img.shields.io/badge/Releases-latest-blue" alt="Releases">
  </a>
  <a href="https://github.com/cyberjaeger-coder/openwebui-vision-router-max-suite/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MPL--2.0-purple" alt="License">
  </a>
</p>

<h1 align="center">Vision Quality Router Suite (Open WebUI)</h1>

<p align="center">
  A practical Open WebUI suite that adds <b>robust image handling</b> to any text LLM using an Ollama vision side-call,
  quality gating, optional OCR and verifier checks, and optional follow-up pipes — optimized for low hardware.
</p>

---

## What this suite does

This suite improves reliability for image+text chats by preferring **accuracy and clarification** over confident guessing:

- **Vision side-call (Ollama VLM)** without changing your main text LLM
- **Quality gate + one retry** (optional) when the vision output is weak
- **Optional OCR-only pass** when the user is clearly asking “what does it say?” and OCR is missing
- **Optional verifier pass** for grounding / hallucination risk estimation
- **Clear “ask questions instead of hallucinating” behavior** when evidence is insufficient
- Optional follow-up pipes for:
  - **Vision Follow-up** (targeted missing-details query)
  - **Graph/Diagram extraction** (nodes/edges/labels/legend with minimal interpretation)

---

## Components

### 1) Filter — Dynamic Vision Router Max (Vision Quality Router)
**What it does**
- Detects images in the last user message
- Calls an Ollama vision model (side-call)
- Enforces structured output: `SUMMARY / DETAILS / OCR / UNCERTAINTY`
- Scores quality; retries once if weak (optional)
- If OCR seems missing and the user likely asked “what does it say”, runs an OCR-only pass (optional)
- Runs a verifier pass to estimate grounding / hallucination risk (optional / conditional)
- Injects the final vision bundle into the conversation as a **system** message
- If quality gate trips, instructs the main model to ask **1–2 targeted questions** instead of hallucinating

**Output injection**
- Adds a `VISION ANALYSIS` block and a `VERIFIER RESULT` block into context.
- Does **not** change the main model. Your text LLM remains the answering model.

### 2) Pipe — Vision Follow-up Pipe
**What it does**
- When the vision output is unclear / missing details, this pipe runs a second targeted vision query focusing only on missing elements.
- Appends a `VISION_FOLLOWUP_RESULT` message into the chat.

### 3) Pipe — Graph Follow-up Pipe
**What it does**
- When the image is a diagram/flowchart/graph, this pipe requests **structure extraction**
  (nodes, edges, labels, legend) with minimal interpretation.
- Appends a `GRAPH_FOLLOWUP_RESULT` message into the chat.

---

## Compatibility / Tested models

This suite has been **tested only** with the **openbmb MiniCPM vision models** and they are **highly recommended**:

- openbmb (organization): https://huggingface.co/openbmb  
- MiniCPM collections: https://huggingface.co/collections/openbmb/minicpm-o-and-minicpm-v  

Other VLMs may work, but are currently **not verified** by this project.

---

## EU language support (i18n)

This suite supports a **language preset** for the *content* of vision output.

- `language_preset = auto` (default): follows the user language when possible
- Or force a language: `en`, `pl`, `de`, `fr`, `es`, `it`

For stable parsing and interoperability, section headers remain consistent:
`SUMMARY / DETAILS / OCR / UNCERTAINTY`

---

## Architecture (high-level)

Goal: add reliable vision to any text LLM without switching the main answering model.

### Data flow (happy path)

User message (image)  
→ Filter: Dynamic Vision Router Max  
→ detect images  
→ side-call to Ollama VLM (bounded tokens)  
→ parse structured output: `SUMMARY / DETAILS / OCR / UNCERTAINTY`  
→ quality score  
→ optional retry (only if weak)  
→ optional OCR-only pass (only if user asks for text and OCR missing)  
→ optional verifier pass (recommended default: only on low-quality/risky outputs)  
→ classify meta type: `photo | document | graph`  
→ inject into context (system):
- `VISION_META`
- `VISION ANALYSIS`
- `VERIFIER RESULT`
- `VISION_NEEDS_FOLLOWUP: YES/NO`
- `VERIFIER_MISSING` (optional)

Main text LLM (unchanged)  
→ produces final answer (prefers clarification over guessing)

### Optional follow-ups (pipes)

- Vision Follow-up Pipe:
  triggers when `VISION_NEEDS_FOLLOWUP=YES` (or missing list present),
  runs a targeted VLM query for missing elements and appends `VISION_FOLLOWUP_RESULT`.

- Graph Follow-up Pipe:
  triggers only when `VISION_META type=graph`,
  requests structure extraction (nodes, edges, labels, legend) with minimal interpretation,
  appends `GRAPH_FOLLOWUP_RESULT`.

### Safety rails (LowHW)

- `max_concurrent_sidecalls` (default 1)
- `max_output_tokens` cap (`num_predict`)
- injection trimming (`max_injected_chars`)
- follow-up cooldown + per-chat limit (anti-loop)

---

## Requirements

- Open WebUI with Functions enabled
- Ollama running locally or on LAN
- A vision-capable model available in Ollama (a VLM)

---

## Installation (recommended: suite approach)

### 1) Install the Filter
Open WebUI → Admin → Functions:
- Add Filter script: `filters/openwebui-vision-router-max-filter.py`
- Enable it and attach it to the model(s) you want to enhance

### 2) Install optional Pipes
Add:
- `pipes/vision_followup_pipe.py`
- `pipes/graph_followup_pipe.py`

Pipes appear as separate selectable “models” in the UI.

### 3) Configure Valves (important)
At minimum:
- `ollama_base_url` (default: `http://localhost:11434`)
- `vision_model_id` (your VLM in Ollama)

Recommended VLM baseline: **openbmb MiniCPM** (see Compatibility section).

---

## Example valve configuration (starter)

> Names may vary slightly by version — use this as guidance.

```yaml
ollama_base_url: "http://localhost:11434"
vision_model_id: "minicpm-v-2_6"   # example
language_preset: "auto"           # or "en", "pl", "de", "fr", "es", "it"

# Optional: resolve image URLs / file IDs
resolve_image_urls: true
openwebui_base_url: "http://localhost:8080"

multi_image_strategy: "last"      # recommended for LowHW
max_concurrent_sidecalls: 1
max_output_tokens: 450            # LowHW: 350–450, Balanced: 650
max_injected_chars: 12000

enable_vision_retry: true
enable_ocr_focus_pass: true
verifier_mode: "on_low_quality"   # "never" | "on_low_quality" | "always"
vision_timeout_s: 90
Recommended valve presets
Preset A — LowHW (consumer-grade, small models)
Use this if you run small LLM/VLMs and want to avoid overload.

max_concurrent_sidecalls = 1

vision_timeout_s = 60–90

max_output_tokens = 350–450

multi_image_strategy = "last"

enable_vision_retry = false (or true if you accept extra compute)

enable_ocr_focus_pass = true (only triggers when user asks for text + OCR missing)

verifier_mode = "on_low_quality" (or never for max speed)

max_injected_chars = 8000–12000

Preset B — Balanced (recommended default)
max_concurrent_sidecalls = 1

vision_timeout_s = 90

max_output_tokens = 650

multi_image_strategy = "last"

enable_vision_retry = true

enable_ocr_focus_pass = true

verifier_mode = "on_low_quality"

max_injected_chars = 12000

Preset C — MaxAccuracy (power users)
max_concurrent_sidecalls = 1–2 (only if your hardware can handle it)

vision_timeout_s = 120

max_output_tokens = 900–1200

multi_image_strategy = "sequential" (only if you routinely send multiple relevant images)

enable_vision_retry = true

enable_ocr_focus_pass = true

verifier_mode = "always"

max_injected_chars = 16000–24000

Troubleshooting
“Nothing happens when I upload images”
Make sure the Filter is enabled and attached to the model you are using.

Confirm your message actually contains images (some clients can strip image fields).

“Timeout / slow responses”
Increase timeout_s

Lower max_output_tokens

Use Preset A (LowHW)

Confirm Ollama/VLM is running and not memory-swapping

“My main LLM gets slow after injection”
Reduce max_injected_chars

Set verifier_mode = on_low_quality or never

“Graph pipe runs on non-graphs”
Ensure the Graph Follow-up Pipe triggers only when VISION_META type=graph

Keep graph classification conservative in the Filter

Provenance
This project is inspired by the Open WebUI community concept of dynamic vision routing (image detection + routing).
This suite is an independent implementation that extends the idea with quality gating, OCR-only fallback, verifier pass,
EU language presets, and follow-up pipes.

Credits
cyberjaeger (author)

ChatGPT (OpenAI) — implementation assistance

Inspired by the Open WebUI community concept of dynamic vision routing

Support
If this suite helps you, you can support development on Ko-fi:
https://ko-fi.com/cyberjaeger

License
MPL-2.0 (see LICENSE)

Suggested repo layout
text
Skopiuj kod
/filters/
  openwebui-vision-router-max-filter.py
/pipes/
  vision_followup_pipe.py
  graph_followup_pipe.py
.github/
  FUNDING.yml
  ISSUE_TEMPLATE/
README.md
LICENSE
NOTICE
CHANGELOG.md
VERSION
makefile
Skopiuj kod
::contentReference[oaicite:0]{index=0}
