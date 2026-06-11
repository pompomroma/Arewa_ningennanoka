import os
import json
import time
import random
import subprocess
import requests
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import socket
import logging
from urllib.parse import urlparse
import re

# ==========================================
# 0. SETUP AND CONFIGURATION
# ==========================================
WORKSPACE_DIR = "nexus_workspace"
LOG_FILE = os.path.join(WORKSPACE_DIR, "nexus_activity.log")

# Ensure directories exist before setting up logging
if not os.path.exists(WORKSPACE_DIR):
    os.makedirs(WORKSPACE_DIR)
if not os.path.exists(os.path.join(WORKSPACE_DIR, "assets")):
    os.makedirs(os.path.join(WORKSPACE_DIR, "assets"))

# --- SET UP LOGGING ---
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("Nexus-G System Initialized.")
print("[System] Nexus-G Initializing...")

# --- MAIN ONLINE MODEL (NVIDIA API Catalog) ---
# Qwen3.5 397B A17B: a 397B-parameter MoE model (17B active) with a
# 262,144-token context window and state-of-the-art agentic coding skill,
# hosted on the SAME NVIDIA endpoint as before, so the existing API key keeps
# working with no key switching. It is a THINKING model: it reasons inside
# <think>...</think> before answering; the pipeline streams that reasoning to
# the console live and strips it from everything that gets parsed or saved.
NVIDIA_MODEL = "qwen/qwen3.5-397b-a17b"
MODEL_DISPLAY = "Qwen3.5 397B A17B (NVIDIA)"

# --- QUALITY PIPELINE TUNING ---
# "max"  = full agentic pipeline: plan -> code -> verify -> self-review -> integration review
# "fast" = skip the self-review and integration passes (fewer API requests per build)
QUALITY_MODE = os.environ.get("NEXUS_QUALITY", "max").strip().lower()

MAX_CONTINUATIONS = 3           # auto-continue rounds when output hits the token limit
MAX_FIX_ITERATIONS = 2          # surgical syntax-fix attempts per file
MAX_PLAN_FILES = 8              # hard cap on files a plan may contain
MAX_TOTAL_CHARS = 200_000       # runaway-output guard across continuations
CONTEXT_BUDGET_ONLINE = 120_000   # chars of project context per request (online)
CONTEXT_BUDGET_OFFLINE = 6_000    # chars of project context per request (offline llama)
PER_FILE_CONTEXT_CAP = 30_000     # chars of any single file included as context
MIN_CALL_GAP_SECONDS = 1.5      # global throttle between API calls (rate-limit friendly)

# Per-role sampling parameters. Qwen3.5 runs in thinking mode, and its
# official recommendation for thinking-mode coding is temperature 0.6 /
# top_p 0.95 for every role — running it colder causes repetition and
# degeneration. max_tokens is sized so the private reasoning never crowds
# out the actual answer that follows it.
ROLE_PARAMS = {
    "planner":     {"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192},
    "coder":       {"temperature": 0.6, "top_p": 0.95, "max_tokens": 16384},
    "fixer":       {"temperature": 0.6, "top_p": 0.95, "max_tokens": 16384},
    "reviewer":    {"temperature": 0.6, "top_p": 0.95, "max_tokens": 16384},
    "integration": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 16384},
    "update":      {"temperature": 0.6, "top_p": 0.95, "max_tokens": 16384},
    "json_repair": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192},
}

# ==========================================
# 1. CONNECTIVITY & DEPENDENCY CHECK
# ==========================================
def is_connected():
    """Returns True if the device has an active internet connection."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=1.5)
        return True
    except OSError:
        return False

# --- API KEYS SETUP ---
def _load_nvidia_key_1():
    """Resolves the main LLM key: env var, then gitignored nexus_key.txt, then the built-in key."""
    env_key = os.environ.get("NVIDIA_API_KEY_1")
    if env_key and env_key.strip():
        return env_key.strip()
    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nexus_key.txt")
    try:
        with open(key_file, "r") as f:
            file_key = f.read().strip()
        if file_key:
            return file_key
    except OSError:
        pass
    # No key baked into the git copy: paste your nvapi-... key into a file named
    # nexus_key.txt next to this script (gitignored), or export NVIDIA_API_KEY_1.
    return ""

NVIDIA_API_KEY_1 = _load_nvidia_key_1()

raw_nvidia_key_2 = os.environ.get("NVIDIA_API_KEY_2")
NVIDIA_API_KEY_2 = raw_nvidia_key_2.strip() if raw_nvidia_key_2 else None

LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "models/gpt-oss-120b.gguf")

# Cloud Client Check (NVIDIA API via OpenAI SDK)
try:
    from openai import OpenAI
    if NVIDIA_API_KEY_1:
        cloud_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY_1
        )
        print(f"[System] ✅ NVIDIA API Key 1 (LLM) detected. Online brain: {MODEL_DISPLAY}.")
    else:
        cloud_client = None
        print("[System] ⚠️ No NVIDIA API Key 1 found. (Put it in nexus_key.txt next to main.py, or set NVIDIA_API_KEY_1)")
except ImportError:
    cloud_client = None
    print("[System] ❌ OpenAI library not installed. Run: pip install openai")

# Local Client (Direct in-memory inference via llama-cpp-python)
try:
    from llama_cpp import Llama
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"[System] ⏳ Loading massive local model from {LOCAL_MODEL_PATH}...")
        local_llm = Llama(model_path=LOCAL_MODEL_PATH, n_ctx=8192, n_gpu_layers=-1, verbose=False)
        print(f"[System] ✅ Local AI model loaded into memory directly.")
    else:
        local_llm = None
        print(f"[System] ⚠️ Offline model not found at {LOCAL_MODEL_PATH}. Create a 'models' folder and add the .gguf file.")
except ImportError:
    local_llm = None
    print("[System] ❌ llama-cpp-python not installed. Run: pip install llama-cpp-python")

# ==========================================
# 2. VOICE I/O SETUP & FALLBACK
# ==========================================
try:
    import speech_recognition as sr
    import pyttsx3
    tts_engine = pyttsx3.init()   # can raise RuntimeError on headless systems
    tts_engine.setProperty('rate', 175)
    VOICE_ENABLED = True
except Exception:
    VOICE_ENABLED = False

def speak(text):
    print(f"\n[Nexus-G]: {text}")
    if VOICE_ENABLED:
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception:
            pass

def listen_command():
    if not VOICE_ENABLED:
        return input("\n[Text Mode] Command the AI: ")

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            online = is_connected()
            mode_str = f"ONLINE ({MODEL_DISPLAY})" if online else "OFFLINE (Native 120B Engine)"
            speak(f"I am listening in {mode_str} mode. What are we building?")
            print(f"[🎙️ Mic active - {mode_str}]")

            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=15)

            if online:
                try:
                    text = recognizer.recognize_google(audio)
                except Exception:
                    text = recognizer.recognize_sphinx(audio)
            else:
                text = recognizer.recognize_sphinx(audio)

            print(f"[You Said]: {text}")
            logging.info(f"User command recognized: {text}")
            return text
    except Exception:
        speak("Voice recognition failed or not supported in this environment. Switching to text input.")
        return input("\n[Text Mode] Command: ")

# ==========================================
# 3. STATE MANAGEMENT
# ==========================================
STATE_FILE = os.path.join(WORKSPACE_DIR, "state.json")
latest_telemetry = {"fps": None}

DEFAULT_STATE = {
    "status": "idle",
    "prompt": "",
    "plan": None,
    "completed_assets": [],
    "completed_files": [],
    "reviewed_files": [],
    "integration_done": False,
    "history": [],
    "model": NVIDIA_MODEL,
}

def _fresh_state():
    return json.loads(json.dumps(DEFAULT_STATE))

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                # Merge over defaults so state files from older versions resume cleanly.
                return {**_fresh_state(), **loaded}
        except Exception:
            pass
    return _fresh_state()

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

# ==========================================
# 4. SYSTEM PROMPTS (Claude-Code-style agentic quality pipeline)
# ==========================================
# TECH_SPEC is the single source of truth for runtime rules. It is injected
# into every code-facing prompt so the rules can never drift between roles.
TECH_SPEC = """RUNTIME CONTRACT — every generated file MUST satisfy ALL of these rules:

1. RENDERING STACK
- Three.js r128 GLOBAL build only. index.html loads it EXACTLY as:
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  followed by plain <script src="game.js"></script> tags (never type="module").
- `THREE` is a global object. FORBIDDEN in all .js files: `import`, `export`, top-level `await`.
- FORBIDDEN: Three.js examples/jsm addons (OrbitControls, GLTFLoader, EffectComposer, ...) — they do not exist in the global bundle. Write your own small camera/control logic instead.

2. ASSETS — PROCEDURAL-ONLY POLICY
- Files under ./assets/ are text metadata stubs, NEVER loadable 3D models.
- FORBIDDEN: loading .usd/.glb/.gltf/.obj/.fbx files, external images/textures/audio/fonts, or ANY network resource except the single Three.js CDN script tag above.
- Build ALL visuals from procedural Three.js geometry (BoxGeometry, SphereGeometry, CylinderGeometry, ConeGeometry, TorusGeometry, PlaneGeometry, custom BufferGeometry), THREE.Group hierarchies, and MeshStandardMaterial/MeshPhongMaterial colors.
- Runtime textures only via 2D canvas + THREE.CanvasTexture. Sound (optional) only via inline WebAudio synthesis.
- The game must boot and play perfectly with the assets directory empty.

3. RESPONSIVE + DUAL INPUT (PC and mobile are BOTH first-class)
- index.html has <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">.
- The renderer canvas fills the window; on window resize update camera.aspect, call camera.updateProjectionMatrix(), and renderer.setSize(window.innerWidth, window.innerHeight).
- KEYBOARD+MOUSE: WASD/arrow keys move; space and/or mouse for the primary action.
- TOUCH: a visible on-screen virtual joystick (left side) plus action button(s) (right side) using pointer/touch events, with CSS touch-action: none on control surfaces and preventDefault() to stop page scrolling.
- Both input schemes drive the SAME movement/action functions.

4. GAME LOOP AND FEEL
- requestAnimationFrame loop; delta time from THREE.Clock; ALL movement and timers scale by delta (frame-rate independent).
- Lighting: at least one AmbientLight plus one DirectionalLight; renderer.shadowMap.enabled = true; key objects cast/receive shadows.
- DOM HUD overlay: score/status plus a one-line controls hint.
- Explicit game states: START SCREEN -> PLAYING -> GAME OVER -> RESTART. Restart fully resets the game WITHOUT reloading the page.

5. TELEMETRY
- Every 2 seconds: fetch('./telemetry', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ fps: measuredFps }) }).catch(function () {});
- Telemetry must never break the game if the endpoint is unavailable.

6. COMPLETENESS BAR (non-negotiable)
- Output COMPLETE, immediately runnable files: no placeholders, no TODOs, no "..." elisions, no "rest of the code unchanged" comments.
- Zero console errors on load is the standard. Every referenced function/variable is defined; every element ID used in JS exists in the HTML; every file referenced by a tag exists in the plan."""

# Multi-file response protocol shared by the integration and update roles.
FILE_BLOCK_SPEC = """MULTI-FILE OUTPUT PROTOCOL (when you must return files):
For EACH file you change, output exactly:
===FILE: filename===
<complete file content>
===END FILE===
The marker lines must start at column 0, exactly as shown. Output nothing outside the blocks."""

PLANNER_PROMPT = """You are the Nexus-G Architect — a principal game engineer producing a one-shot technical design for a browser WebGL game. You think ahead like a senior engineer: you name every public interface up front so independently generated files integrate perfectly.

THE RUNTIME YOUR DESIGN MUST TARGET:
__TECH_SPEC__

YOUR TASK:
From the user's game request, design a minimal, fully buildable file plan.

DESIGN RULES:
1. 2 to 5 files total. ALWAYS include "index.html" (listed FIRST) and "game.js". Add "style.css" or one extra .js file ONLY if genuinely needed.
2. Each file's "description" is a build contract: state precisely WHAT the file implements AND its PUBLIC INTERFACE — the exact global function names, global variable names, and HTML element IDs that other files reference. All files must agree on these names.
3. "assets_needed": 0-4 OPTIONAL decorative asset searches. They only produce metadata stubs; the game never loads them — list them purely as thematic inspiration.
4. "advanced_mechanics": 3-6 concrete mechanics implementable with procedural geometry, ALWAYS including both "Touch joystick + buttons (mobile)" and "Keyboard + mouse (PC)".
5. Never plan anything that needs a build tool, server-side code, module syntax, or external assets.

THINKING: Reason as deeply as you need inside your private thinking section first (game design, interfaces, risks). Everything AFTER your thinking must be ONLY the JSON object.

CRITICAL OUTPUT RULES:
1. After your thinking, output ONLY one valid JSON object. Its first character MUST be { and its last character MUST be }.
2. No conversational text, no explanations, no markdown fences.

EXPECTED JSON SHAPE:
{
  "game_name": "Name",
  "files": [
    {"filename": "index.html", "description": "Viewport meta; three.js r128 CDN tag then game.js tag; HUD elements #hud, #score; overlay screens #overlay, #overlay-title, #overlay-msg, #overlay-btn; touch controls #joystick-zone, #btn-action; inline CSS"},
    {"filename": "game.js", "description": "Defines initGame(), startGame(), endGame(), restartGame(); reads the element IDs above; scene/camera/renderer; keyboard+mouse and touch joystick input; delta-time loop; telemetry POST"}
  ],
  "assets_needed": [{"filename": "short_name", "prompt": "3D search description"}],
  "advanced_mechanics": ["Touch joystick + buttons (mobile)", "Keyboard + mouse (PC)", "..."]
}""".replace("__TECH_SPEC__", TECH_SPEC)

CODER_PROMPT = """You are the Nexus-G Lead Engineer — a staff-level WebGL/Three.js programmer. You write production code that ships to players unmodified, in one pass. You are meticulous: every identifier you reference exists, every code path terminates, both input schemes work, and the file is complete from its first line to its last.

__TECH_SPEC__

CONSISTENCY RULES (when other project files are provided in the request):
- They are the source of truth. Match their EXACT global function names, variable names, element IDs and CSS classes. Consistency with the existing files outranks your personal preferences.
- If the plan's description of this file names an interface (function/ID), implement it with exactly that name.

BEFORE YOU EMIT, SILENTLY SELF-CHECK:
- Does the file run from a blank page with zero console errors?
- Do BOTH keyboard/mouse and touch controls work?
- Does resizing keep the canvas full-window and undistorted?
- Is game over reachable, and does restart work without a page reload?
- Is every visual procedural (no asset/file/URL loads beyond the pinned three.js tag)?
- Is the file COMPLETE (no truncation, no placeholders)?

THINKING: Use your private thinking section to architect the file (structures, edge cases, both input schemes) as deeply as you need. Everything AFTER your thinking must be ONLY the raw file.

CRITICAL OUTPUT RULES:
1. After your thinking, output ONLY the raw content of the requested file.
2. The first character after your thinking is the first character of the file (for example `<` for HTML).
3. NO conversational text, NO explanations, NO markdown fences.""".replace("__TECH_SPEC__", TECH_SPEC)

FIXER_PROMPT = """You are the Nexus-G Debug Surgeon. You receive ONE file and ONE concrete error or validation report. Fix it with the minimum change necessary.

RULES:
1. Change only what the error requires. Preserve all other behavior, names and structure.
2. If the file is TRUNCATED (cut off mid-statement), complete it consistently with its own style.
3. The corrected file must satisfy this contract:
__TECH_SPEC__

THINKING: Diagnose the error in your private thinking section first. Everything AFTER your thinking must be ONLY the corrected file.

CRITICAL OUTPUT RULES:
1. After your thinking, output the COMPLETE corrected file — every line, top to bottom.
2. Raw content only: no commentary, no markdown fences.""".replace("__TECH_SPEC__", TECH_SPEC)

REVIEWER_PROMPT = """You are the Nexus-G Adversarial Reviewer — a hostile senior engineer paid to find what is genuinely BROKEN. You receive one file under review (plus sibling project files for cross-reference). Hunt ONLY for real defects:

- Runtime errors: undefined variables/functions, references to element IDs that exist in no provided file, syntax slips, use-before-define across script load order.
- Truncated or unreachable logic; event listeners never attached; game states that cannot be reached or exited; restart that does not reset state.
- Contract violations (below): missing touch OR keyboard input path, non-responsive canvas, forbidden module syntax/addons/asset loads, missing telemetry, missing lighting/HUD/game-over.

__TECH_SPEC__

THINKING: Trace the code path by path in your private thinking section as deeply as you need. Everything AFTER your thinking must be ONLY your verdict per the protocol below.

DECISION PROTOCOL (follow exactly):
- If the file would ship as-is (no genuine defects), reply after your thinking with EXACTLY this single line and nothing else:
APPROVED
- Otherwise reply with the COMPLETE corrected file: raw content only, no commentary, no markdown fences, no diff — the whole file.

HARD RULE: never rewrite working code for style, taste, or "improvement". Fix defects only. When in doubt, reply APPROVED.""".replace("__TECH_SPEC__", TECH_SPEC)

INTEGRATION_PROMPT = """You are the Nexus-G Release Integrator. You receive ALL files of a finished build. Verify ONLY the cross-file contracts:

- index.html loads three.js r128 BEFORE the game script tag(s); every planned .js/.css file has a matching tag with the exact filename.
- Every element ID referenced in JS exists in the HTML; every CSS selector targets real elements.
- Every function/global called in one file is defined in a file loaded before the call runs.
- No colliding duplicate global declarations across files; no forbidden module syntax or external asset loads anywhere.

__TECH_SPEC__

THINKING: Cross-check every contract in your private thinking section first. Everything AFTER your thinking must be ONLY your verdict per the protocol below.

DECISION PROTOCOL (follow exactly):
- If the build is coherent and shippable, reply after your thinking with EXACTLY this single line and nothing else:
NO_CHANGES_NEEDED
- Otherwise return ONLY the files that must change (complete content for each) using the protocol below. Do NOT invent new files. Do NOT restyle working code — minimal cross-file repairs only.

__FILE_BLOCK_SPEC__""".replace("__TECH_SPEC__", TECH_SPEC).replace("__FILE_BLOCK_SPEC__", FILE_BLOCK_SPEC)

UPDATE_PROMPT = """You are the Nexus-G Live-Ops Engineer. You receive a working game's full file set plus ONE change request. Apply the request the way a careful senior engineer edits production code:

- The CURRENT BUILD files are the live truth. Build on top of them: changes accumulate across requests. NEVER regenerate the game from scratch and NEVER drop features the request does not name.
- Preserve ALL existing behavior not named by the request.
- Keep every existing public name (functions, globals, element IDs) stable unless the request requires renaming.
- Changed files must remain complete and satisfy the runtime contract:
__TECH_SPEC__

OUTPUT:
- Plan the change in your private thinking section first; everything AFTER your thinking must be ONLY the file blocks.
- Return ONLY the files you changed (complete content for each), via the protocol below.
- You may add at most 2 NEW files if the request truly requires them (and you must wire them into index.html in the same response).
- No commentary outside the blocks.

__FILE_BLOCK_SPEC__""".replace("__TECH_SPEC__", TECH_SPEC).replace("__FILE_BLOCK_SPEC__", FILE_BLOCK_SPEC)

JSON_REPAIR_PROMPT = """You are a strict JSON repair machine. The user gives you text that was MEANT to be one valid JSON object but is malformed (stray prose, markdown fences, bad quotes/commas, truncation). Reconstruct the intended object, preserving all of its content.

OUTPUT RULES:
1. After any private thinking, output ONLY the corrected JSON object. First character {, last character }.
2. No fences, no commentary."""

CONTINUE_INSTRUCTION = (
    "Your previous message hit the length limit mid-output. Continue EXACTLY where you stopped, "
    "starting with the very next character of the file. Do not repeat any earlier text, do not "
    "summarize, do not add any preamble or markdown fences, and do not open a new <think> block — "
    "continue the raw output directly."
)

# ==========================================
# 5. SERVER & HYBRID GENERATION UTILS
# ==========================================
class TelemetryHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WORKSPACE_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_POST(self):
        global latest_telemetry
        if urlparse(self.path).path == "/telemetry":
            try:
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])).decode('utf-8'))
                latest_telemetry["fps"] = data.get("fps", 0)
            except: pass
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

def serve_game():
    server_address = ("0.0.0.0", 8080)
    try:
        with ThreadingHTTPServer(server_address, TelemetryHandler) as httpd:
            logging.info("Local threaded web server started on port 8080.")
            print("[System] Web Server active! You will see output in the preview panel.")
            httpd.serve_forever()
    except OSError as e:
        logging.warning(f"Web server could not bind to port 8080: {e}")
        print(f"[System] ⚠️ Web server not started (is port 8080 already in use?): {e}")

# ==========================================
# 6. LLM CORE (streaming, retries, auto-continuation)
# ==========================================
_last_api_call_ts = [0.0]

def _throttle():
    """Keeps a polite global gap between API calls so bursts never trip rate limits."""
    wait = MIN_CALL_GAP_SECONDS - (time.time() - _last_api_call_ts[0])
    if wait > 0:
        time.sleep(wait)
    _last_api_call_ts[0] = time.time()

def _stream_chat_once(messages, params, max_retries=5):
    """Runs ONE streamed chat completion against the NVIDIA endpoint.

    Returns (text, finish_reason). Returns ("", None) after unrecoverable errors.
    A failure mid-stream discards the partial text and retries the whole segment.
    """
    base_delay = 3
    max_tokens = params["max_tokens"]

    for attempt in range(max_retries):
        _throttle()
        try:
            full_response = ""
            finish_reason = None
            completion = cloud_client.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=messages,
                temperature=params["temperature"],
                top_p=params["top_p"],
                max_tokens=max_tokens,
                stream=True
            )
            for chunk in completion:
                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.delta:
                        # Some deployments stream the model's private reasoning in a
                        # separate field: show it live, but never keep it in the answer.
                        reasoning = getattr(choice.delta, "reasoning_content", None)
                        if reasoning:
                            print(reasoning, end="", flush=True)
                        if choice.delta.content is not None:
                            content = choice.delta.content
                            print(content, end="", flush=True)
                            full_response += content
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
            print()
            return full_response, (finish_reason or "stop")
        except Exception as e:
            error_message = str(e).lower()
            if "401" in error_message or "403" in error_message or "unauthorized" in error_message or "forbidden" in error_message:
                logging.error(f"API Authentication Error: {e}")
                print(f"\n[API Auth Error]: NVIDIA rejected the API key. Please ensure your NVIDIA_API_KEY_1 is correctly set. Details: {e}")
                return "", None
            elif "404" in error_message or "not found" in error_message:
                logging.error(f"API Model Error: {e}")
                print(f"\n[API Model Error]: The endpoint could not find model '{NVIDIA_MODEL}'. Details: {e}")
                return "", None
            elif "429" in error_message or "rate limit" in error_message or "too many requests" in error_message:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                logging.warning(f"API Rate limit hit (Attempt {attempt + 1}). Retrying in {delay:.1f}s.")
                speak(f"API Rate limit hit. Retrying in {int(delay)} seconds...")
                time.sleep(delay)
            elif "400" in error_message:
                if "max_token" in error_message and max_tokens > 1024:
                    max_tokens = max(1024, max_tokens // 2)
                    logging.warning(f"API rejected max_tokens; retrying with {max_tokens}.")
                    continue
                logging.error(f"API 400 Bad Request: {e}")
                print(f"\n[API 400 Error]: The API rejected the payload. Details: {e}")
                return "", None
            else:
                delay = 2 * (attempt + 1)
                logging.warning(f"Transient API error (Attempt {attempt + 1}): {e}. Retrying in {delay}s.")
                print(f"\n[System] ⚠️ Connection hiccup with the AI ({e}). Retrying in {delay}s...")
                time.sleep(delay)

    logging.error("Max retries reached. The system is overloaded.")
    speak("Max retries reached. The AI is too busy.")
    return "", None

def dedup_overlap(accumulated, new_chunk, window=200, min_overlap=20):
    """Trims text the model repeated at a continuation boundary.

    Finds the longest suffix of `accumulated` (within `window` chars) that is
    also a prefix of `new_chunk` and removes it from the chunk.
    """
    tail = accumulated[-window:]
    limit = min(len(tail), len(new_chunk))
    for size in range(limit, min_overlap - 1, -1):
        if tail[-size:] == new_chunk[:size]:
            return new_chunk[size:]
    return new_chunk

def ask_model(system_prompt, user_prompt=None, messages=None, role="coder", max_retries=5, allow_continuation=True):
    """Asks the active model (online Qwen3.5 397B A17B, or local fallback) for a response.

    Online responses that hit the token limit are automatically continued and
    stitched, so long files are never silently truncated. The model's private
    <think> reasoning streams to the console but is stripped from the returned
    text. Returns "" on failure.
    """
    online = is_connected()
    params = ROLE_PARAMS.get(role, ROLE_PARAMS["coder"])

    if messages is None:
        safe_user_prompt = str(user_prompt if user_prompt is not None else "").strip()
        if not safe_user_prompt:
            safe_user_prompt = "Please proceed with the current task."
        safe_system_prompt = str(system_prompt or "").strip()
        messages = []
        if safe_system_prompt:
            messages.append({"role": "system", "content": safe_system_prompt})
        messages.append({"role": "user", "content": safe_user_prompt})

    if online and cloud_client:
        print(f"\n[Using Cloud API - {MODEL_DISPLAY} | {NVIDIA_MODEL} | role={role}]")
        accumulated = ""
        convo = list(messages)
        continuations = 0
        while True:
            segment, finish_reason = _stream_chat_once(convo, params, max_retries=max_retries)
            if not segment:
                # Total failure on the first call, or a dead continuation: keep what we have.
                return strip_reasoning(accumulated)
            if accumulated:
                segment = re.sub(r'^\s*```[a-zA-Z0-9]*\n', '', segment)
                head = segment[:80].strip()
                if len(head) >= 20 and head in accumulated[:200]:
                    logging.warning("Continuation restarted from the top; discarding it and stopping.")
                    break
                segment = dedup_overlap(accumulated, segment)
            accumulated += segment
            if finish_reason != "length" or not allow_continuation:
                break
            if continuations >= MAX_CONTINUATIONS or len(accumulated) >= MAX_TOTAL_CHARS:
                logging.warning("Output still truncated after maximum continuations.")
                print(f"\n[System] ⚠️ Output still truncated after {MAX_CONTINUATIONS} continuations; proceeding with what we have.")
                break
            continuations += 1
            print(f"\n[System] ↩️ Output hit the token limit. Requesting continuation {continuations}/{MAX_CONTINUATIONS}...")
            convo = list(messages) + [
                {"role": "assistant", "content": accumulated},
                {"role": "user", "content": CONTINUE_INSTRUCTION},
            ]
        # Continuations stitched on the raw text; reasoning is removed only at the end.
        return strip_reasoning(accumulated)

    elif local_llm:
        print(f"\n[Using Native Local Engine - Offline Mode (Direct Memory 120B)]")
        local_messages = []
        for m in messages:
            content = m["content"]
            if len(content) > CONTEXT_BUDGET_OFFLINE:
                half = CONTEXT_BUDGET_OFFLINE // 2
                content = content[:half] + "\n/* ...truncated to fit the local context window... */\n" + content[-half:]
            local_messages.append({"role": m["role"], "content": content})
        try:
            full_response = ""
            completion = local_llm.create_chat_completion(
                messages=local_messages,
                temperature=0.7,
                max_tokens=2048,
                stream=True
            )
            for chunk in completion:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        print(content, end="", flush=True)
                        full_response += content
            print()
            return strip_reasoning(full_response)
        except Exception as e:
            logging.error(f"Error communicating with local AI: {e}")
            print(f"\n[Error communicating with local AI]: {e}")
            return ""

    else:
        logging.error("Critical Error: No AI clients available.")
        print("\n[Critical Error]: No AI clients available. Please connect to internet or ensure your .gguf file is loaded.")
        return ""

# ==========================================
# 7. PARSING & VALIDATION UTILITIES
# ==========================================
def strip_reasoning(text):
    """Removes the model's private <think>...</think> reasoning from a response.

    Balanced blocks are removed wherever they appear. If the output was cut
    off inside an unterminated <think> block, everything from that tag onward
    is reasoning and is dropped. If a stray closing tag remains (the opening
    tag was streamed as a separate reasoning field), everything before it is
    reasoning and is dropped.
    """
    text = text or ""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    lowered = text.lower()
    open_pos = lowered.rfind('<think>')
    if open_pos != -1:
        text = text[:open_pos]
        lowered = lowered[:open_pos]
    close_pos = lowered.rfind('</think>')
    if close_pos != -1:
        text = text[close_pos + len('</think>'):]
    return text.strip()

def clean_code(text):
    """Extracts raw code/JSON from a model response without mangling code that merely CONTAINS fences."""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    match = re.search(r'```[a-zA-Z0-9]*\n?(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def extract_json(text):
    """Finds and parses the first complete JSON object in text. Raises ValueError on failure."""
    text = clean_code(text)
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    raise ValueError("Unbalanced JSON object in model output.")

def safe_filename(name):
    """Returns a sanitized relative filename, or None if the name is unsafe."""
    name = str(name).strip().strip('"').strip("'").strip("`").strip()
    while name.startswith("./"):
        name = name[2:]
    if not name or len(name) > 100:
        return None
    if name.startswith("/") or "\\" in name or ".." in name:
        return None
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", name):
        return None
    return name

def parse_plan(raw, allow_repair=True):
    """Parses planner output into a plan dict, with one model-assisted JSON repair attempt."""
    try:
        plan = extract_json(raw)
    except ValueError:
        if not allow_repair:
            return None
        logging.warning("Plan JSON malformed; requesting model-side repair.")
        speak("The plan came back malformed. Repairing it...")
        repaired = ask_model(JSON_REPAIR_PROMPT, f"Repair this into one valid JSON object:\n\n{raw}", role="json_repair")
        if not repaired:
            return None
        try:
            plan = extract_json(repaired)
        except ValueError:
            return None

    if not isinstance(plan, dict):
        return None
    files = []
    seen = set()
    for f in plan.get("files", []):
        if not isinstance(f, dict):
            continue
        fname = safe_filename(f.get("filename", ""))
        if not fname or fname in seen:
            continue
        seen.add(fname)
        files.append({"filename": fname, "description": str(f.get("description", "")).strip()})
    if not files:
        return None
    if "index.html" not in seen:
        files.insert(0, {
            "filename": "index.html",
            "description": ("HTML shell: viewport meta; three.js r128 CDN script tag, then the game script tag(s); "
                            "HUD and touch-control elements exactly as referenced by the game code.")
        })
    plan["files"] = files[:MAX_PLAN_FILES]
    if not isinstance(plan.get("assets_needed"), list):
        plan["assets_needed"] = []
    if not isinstance(plan.get("advanced_mechanics"), list):
        plan["advanced_mechanics"] = []
    return plan

FILE_BLOCK_RE = re.compile(r'^===FILE:\s*(.+?)\s*===\s*\n(.*?)\n?^===END FILE===\s*$', re.MULTILINE | re.DOTALL)
FILE_HEADER_RE = re.compile(r'^===FILE:\s*(.+?)\s*===\s*\n', re.MULTILINE)

def parse_file_blocks(text):
    """Parses ===FILE: name=== blocks. Returns (verdict, {filename: content}).

    verdict is "approved", "no_changes", or None.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None, {}

    def _first_line(s):
        return s.split("\n", 1)[0].strip()

    if _first_line(stripped).startswith("APPROVED"):
        return "approved", {}
    if _first_line(stripped).startswith("NO_CHANGES_NEEDED"):
        return "no_changes", {}

    body = stripped
    if body.startswith("```"):
        body = clean_code(body)
        if _first_line(body).startswith("APPROVED"):
            return "approved", {}
        if _first_line(body).startswith("NO_CHANGES_NEEDED"):
            return "no_changes", {}

    blocks = {}
    for m in FILE_BLOCK_RE.finditer(body):
        name = safe_filename(m.group(1))
        if not name:
            logging.warning(f"Skipping file block with unsafe name: {m.group(1)!r}")
            continue
        content = m.group(2)
        if content.strip().startswith("```"):
            content = clean_code(content)
        else:
            content = content.strip("\n")
        if not content.strip():
            continue
        if name in blocks:
            logging.warning(f"Duplicate file block for {name}; keeping the last one.")
        blocks[name] = content

    # Salvage a final un-terminated block (output truncated before ===END FILE===).
    last_header = None
    for m in FILE_HEADER_RE.finditer(body):
        last_header = m
    if last_header:
        name = safe_filename(last_header.group(1))
        tail = body[last_header.end():]
        if name and name not in blocks and "===END FILE===" not in tail:
            tail = tail.strip("\n")
            if tail.strip():
                logging.warning(f"Salvaging un-terminated file block for {name} (subject to validation gates).")
                blocks[name] = tail

    return None, blocks

def test_syntax(filename):
    """Checks syntax for JS and JSON files to catch model hallucinations."""
    filepath = os.path.join(WORKSPACE_DIR, filename)

    # 1. Check JSON files natively in Python
    if filename.endswith(".json"):
        try:
            with open(filepath, 'r') as f:
                json.load(f)
            return True, ""
        except json.JSONDecodeError as e:
            return False, f"JSON Syntax Error: {str(e)}"

    # 2. Check JS files using Node (if installed)
    if filename.endswith(".js"):
        try:
            subprocess.run(["node", "-v"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            res = subprocess.run(["node", "--check", filepath], capture_output=True, text=True)
            return (res.returncode == 0, res.stderr)
        except FileNotFoundError:
            logging.warning("Node.js not found in PATH; skipping JavaScript syntax validation.")
            return True, ""
        except Exception as e:
            return False, str(e)

    # Ignore other file types (HTML, CSS, GLSL, etc.) for strict syntax checking
    return True, ""

def accept_candidate(fname, old_code, new_code):
    """Anti-regression gate for model rewrites (review/integration/update passes).

    A candidate replaces the existing code only if it is non-empty, not
    suspiciously shorter than the original, and passes the syntax check.
    Returns the code that should be kept.
    """
    new_code = (new_code or "").strip()
    if not new_code:
        return old_code
    if old_code and len(new_code) < len(old_code) * 0.5:
        logging.warning(f"Rejected candidate for {fname}: suspiciously short ({len(new_code)} vs {len(old_code)} chars).")
        return old_code
    candidate_name = f".candidate.{os.path.basename(fname)}"
    candidate_path = os.path.join(WORKSPACE_DIR, candidate_name)
    try:
        with open(candidate_path, 'w') as f:
            f.write(new_code)
        passed, err = test_syntax(candidate_name)
        if not passed:
            logging.warning(f"Rejected candidate for {fname}: syntax check failed: {err}")
            return old_code
    finally:
        try:
            os.remove(candidate_path)
        except OSError:
            pass
    return new_code

# ==========================================
# 8. NVIDIA OMNIVERSE ASSET FETCH
# ==========================================
def fetch_nvidia_usd_asset(prompt, output_path, max_retries=4):
    if not is_connected():
        speak("Offline Mode Active. Creating placeholder USD asset instead of calling NVIDIA Omniverse.")
        with open(output_path, 'w') as f: f.write("DUMMY_OFFLINE_ASSET")
        return

    if not NVIDIA_API_KEY_2:
        logging.error("NVIDIA_API_KEY_2 is missing. Cannot fetch 3D assets.")
        speak("Missing API Key 2 for 3D assets. Creating placeholder.")
        with open(output_path, 'w') as f: f.write("DUMMY_MISSING_KEY")
        return

    base_delay = 5

    for attempt in range(max_retries):
        try:
            speak(f"Searching NVIDIA Omniverse for asset: {prompt}...")

            response = requests.post(
                url='https://ai.api.nvidia.com/v1/omniverse/nvidia/usdsearch',
                headers={
                    'Authorization': f'Bearer {NVIDIA_API_KEY_2}',
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                },
                data=json.dumps(
                    dict(
                        description = prompt,
                        file_extension_include ='usd*',
                        return_images ='true',
                        return_metadata = 'true',
                        return_vision_generated_metadata = 'true',
                        cutoff_threshold = '1.05',
                        limit = '50'
                    )
                )
            )

            response.raise_for_status()
            data = response.json()

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

            speak("Omniverse asset search complete. Metadata saved.")
            return

        except requests.exceptions.RequestException as e:
            error_message = str(e).lower()
            if "429" in error_message or "too many requests" in error_message:
                delay = base_delay * (2 ** attempt)
                logging.warning(f"NVIDIA API Rate limit hit (Attempt {attempt + 1}). Retrying in {delay}s.")
                speak(f"API Rate limit hit. Retrying in {delay} seconds...")
                time.sleep(delay)
            elif "401" in error_message or "403" in error_message:
                 logging.error(f"Omniverse Auth Error: {e}")
                 speak("Omniverse API Key 2 rejected. Falling back to dummy asset.")
                 with open(output_path, 'w') as f: f.write("DUMMY")
                 return
            else:
                 logging.error(f"NVIDIA API Error: {e}")
                 speak("Error reaching NVIDIA API. Falling back to dummy asset.")
                 with open(output_path, 'w') as f: f.write("DUMMY")
                 return

    logging.error("Max retries reached for NVIDIA API. Falling back to dummy.")
    speak("Max retries reached. Falling back to dummy asset.")
    with open(output_path, 'w') as f: f.write("DUMMY")

# ==========================================
# 9. CORE AGENTIC PIPELINE
# ==========================================
def build_context(state, current_file=None, include_files=True):
    """Assembles the shared project context handed to the model with a request.

    Online, this includes the plan plus the full content of already generated
    files (budgeted), which is what keeps every file consistent with its
    siblings. Offline it stays tiny to fit the local context window.
    """
    online = is_connected() and cloud_client is not None
    budget = CONTEXT_BUDGET_ONLINE if online else CONTEXT_BUDGET_OFFLINE
    plan = state.get("plan") or {}
    try:
        plan_json = json.dumps(plan, indent=2)
    except (TypeError, ValueError):
        plan_json = str(plan)

    parts = [
        f"USER REQUEST:\n{state.get('prompt', '')}",
        f"PROJECT PLAN (source of truth for filenames and public interfaces):\n{plan_json}",
        "ASSET POLICY REMINDER: anything under ./assets/ is a metadata stub, never a loadable model — all visuals must be procedural.",
    ]
    used = sum(len(p) for p in parts)

    if include_files and online:
        for file_info in plan.get("files", []):
            fname = file_info.get("filename")
            if not fname or fname == current_file:
                continue
            if fname not in state.get("completed_files", []):
                continue
            fpath = os.path.join(WORKSPACE_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, 'r') as f:
                    content = f.read()
            except OSError:
                continue
            if len(content) > PER_FILE_CONTEXT_CAP:
                content = content[:PER_FILE_CONTEXT_CAP] + "\n/* ...truncated... */"
            section = f"CURRENT CONTENT OF {fname} (match its names exactly):\n{content}"
            if used + len(section) > budget:
                break
            parts.append(section)
            used += len(section)

    return "\n\n".join(parts)

def plan_project(state):
    """Asks the architect role for a build plan and validates it."""
    speak("Planning architecture...")
    raw = ask_model(PLANNER_PROMPT, state["prompt"], role="planner")
    if not raw:
        return False
    plan = parse_plan(raw)
    if not plan:
        logging.error("Failed to parse JSON plan from the AI.")
        speak("Failed to parse plan. Please try again.")
        return False
    state["plan"] = plan
    state["status"] = "coding"
    save_state(state)
    logging.info(f"Plan generated successfully with {len(plan.get('files', []))} files.")
    return True

def fetch_assets(state):
    for asset in state["plan"].get("assets_needed", []):
        if not isinstance(asset, dict):
            continue
        asset_name = safe_filename(asset.get("filename", ""))
        if not asset_name or asset_name in state["completed_assets"]:
            continue
        asset_path = os.path.join(WORKSPACE_DIR, "assets", f"{asset_name}.usd")
        os.makedirs(os.path.dirname(asset_path), exist_ok=True)
        fetch_nvidia_usd_asset(asset.get("prompt", asset_name), asset_path)
        state["completed_assets"].append(asset_name)
        save_state(state)

def generate_file(state, file_info):
    """Generates one file through the full quality gauntlet:

    generate (with cross-file context) -> syntax gate -> surgical fix loop ->
    adversarial self-review -> save. Returns True on success.
    """
    fname = file_info["filename"]
    speak(f"Generating {fname}...")
    fpath = os.path.join(WORKSPACE_DIR, fname)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)

    context = build_context(state, current_file=fname)
    request = (
        f"{context}\n\n"
        f"YOUR TASK: write the COMPLETE content of `{fname}`.\n"
        f"FILE CONTRACT: {file_info.get('description', 'See the project plan.')}\n"
        f"MECHANICS TO COVER ACROSS THE PROJECT: {state['plan'].get('advanced_mechanics')}\n"
        f"Output the raw file content only."
    )
    code = clean_code(ask_model(CODER_PROMPT, request, role="coder"))
    if not code:
        return False

    with open(fpath, 'w') as f:
        f.write(code)

    # --- Syntax gate + surgical fix loop ---
    passed, err = test_syntax(fname)
    fixes = 0
    while not passed and fixes < MAX_FIX_ITERATIONS:
        fixes += 1
        speak(f"Fixing a syntax error in {fname} (attempt {fixes})...")
        logging.warning(f"Syntax error in {fname} (fix attempt {fixes}): {err}")
        fix_request = (
            f"FILE: {fname}\n"
            f"VALIDATION ERROR:\n{err}\n\n"
            f"CURRENT CONTENT:\n{code}\n\n"
            f"Return the complete corrected file."
        )
        fixed = clean_code(ask_model(FIXER_PROMPT, fix_request, role="fixer"))
        if not fixed:
            break
        code = fixed
        with open(fpath, 'w') as f:
            f.write(code)
        passed, err = test_syntax(fname)
    if not passed:
        logging.warning(f"{fname} still failing validation after {fixes} fix attempts; keeping best effort.")

    # --- Adversarial self-review pass (max quality mode, online only) ---
    if QUALITY_MODE != "fast" and is_connected() and cloud_client and fname not in state.get("reviewed_files", []):
        speak(f"Reviewing {fname} for defects...")
        review_request = (
            f"{build_context(state, current_file=fname)}\n\n"
            f"FILE UNDER REVIEW: `{fname}`\n\n"
            f"CONTENT:\n{code}\n\n"
            f"Reply APPROVED, or reply with the complete corrected file."
        )
        verdict_raw = ask_model(REVIEWER_PROMPT, review_request, role="reviewer")
        if verdict_raw:
            first_line = verdict_raw.strip().split("\n", 1)[0].strip()
            if first_line.startswith("APPROVED"):
                logging.info(f"Reviewer approved {fname}.")
            else:
                accepted = accept_candidate(fname, code, clean_code(verdict_raw))
                if accepted != code:
                    code = accepted
                    with open(fpath, 'w') as f:
                        f.write(code)
                    logging.info(f"Reviewer corrections applied to {fname}.")
                    speak(f"Review found and fixed defects in {fname}.")
        state.setdefault("reviewed_files", []).append(fname)

    state["completed_files"].append(fname)
    save_state(state)
    logging.info(f"Successfully generated file: {fname}")
    return True

def apply_file_blocks(state, blocks, allow_new, max_new=3):
    """Validates and writes model-returned file blocks. Returns the list of files actually changed."""
    plan_files = {f.get("filename") for f in (state.get("plan") or {}).get("files", [])}
    known = set(state.get("completed_files", [])) | plan_files
    changed = []
    new_count = 0
    for fname, content in blocks.items():
        fpath = os.path.join(WORKSPACE_DIR, fname)
        is_new = fname not in known and not os.path.isfile(fpath)
        if is_new:
            if not allow_new or new_count >= max_new:
                logging.warning(f"Skipping unexpected new file from model: {fname}")
                continue
            new_count += 1
        old_code = ""
        if os.path.isfile(fpath):
            try:
                with open(fpath, 'r') as f:
                    old_code = f.read()
            except OSError:
                pass
        accepted = accept_candidate(fname, old_code, content)
        if accepted == old_code:
            continue
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, 'w') as f:
            f.write(accepted)
        if fname not in state.get("completed_files", []):
            state.setdefault("completed_files", []).append(fname)
        changed.append(fname)
        logging.info(f"Applied model-provided update to {fname}")
    if changed:
        save_state(state)
    return changed

def integration_review(state):
    """One final pass with every file in view, fixing cross-file contract breaks."""
    if QUALITY_MODE == "fast" or state.get("integration_done"):
        return
    if not (is_connected() and cloud_client):
        return
    files = [f["filename"] for f in state["plan"].get("files", []) if f.get("filename") in state.get("completed_files", [])]
    if len(files) < 2:
        state["integration_done"] = True
        save_state(state)
        return

    speak("Running the final integration review across all files...")
    sections = []
    for fname in files:
        fpath = os.path.join(WORKSPACE_DIR, fname)
        try:
            with open(fpath, 'r') as f:
                content = f.read()
        except OSError:
            continue
        if len(content) > PER_FILE_CONTEXT_CAP:
            content = content[:PER_FILE_CONTEXT_CAP] + "\n/* ...truncated... */"
        sections.append(f"===FILE: {fname}===\n{content}\n===END FILE===")

    request = (
        f"USER REQUEST:\n{state.get('prompt', '')}\n\n"
        f"PROJECT PLAN:\n{json.dumps(state.get('plan') or {}, indent=2)}\n\n"
        f"COMPLETE BUILD:\n\n" + "\n\n".join(sections) +
        "\n\nReply NO_CHANGES_NEEDED, or return only the corrected files via the protocol."
    )
    raw = ask_model(INTEGRATION_PROMPT, request, role="integration")
    if raw:
        verdict, blocks = parse_file_blocks(raw)
        if verdict:
            logging.info("Integration review: no changes needed.")
        elif blocks:
            changed = apply_file_blocks(state, blocks, allow_new=False)
            if changed:
                speak(f"Integration fixes applied to: {', '.join(changed)}.")
        else:
            logging.warning("Integration review returned nothing parseable; skipping it.")
    state["integration_done"] = True
    save_state(state)

# --- Build stacking helpers -------------------------------------------------
# A first description builds a game; every later instruction STACKS onto the
# living build instead of recreating a different game. Starting over is the
# explicit exception, and the previous game is archived rather than destroyed.
FRESH_START_KEYWORDS = ("new game", "new project", "start over", "from scratch", "reset")
FRESH_START_PREFIXES = ("new game", "new project")

def _fresh_start_request(user_prompt):
    """Detects an explicit fresh-start command.

    Returns the new game description for "new game <idea>" style commands,
    "" for a bare fresh-start keyword, or None when this is NOT a fresh start
    (so phrases like "reset the score when falling" still stack as changes).
    """
    command = user_prompt.strip()
    lowered = command.lower()
    if lowered in FRESH_START_KEYWORDS:
        return ""
    for prefix in FRESH_START_PREFIXES:
        if lowered.startswith(prefix + " ") or lowered.startswith(prefix + ":"):
            return command[len(prefix):].lstrip(" :,-")
    return None

def _known_build_files(state):
    """Every filename the current build could own, in stable order."""
    fnames = []
    for f in (state.get("plan") or {}).get("files", []):
        if isinstance(f, dict) and f.get("filename"):
            fnames.append(f["filename"])
    fnames += state.get("completed_files", [])
    fnames += ["index.html", "game.js", "style.css"]
    seen = set()
    ordered = []
    for fname in fnames:
        if fname and fname not in seen:
            seen.add(fname)
            ordered.append(fname)
    return ordered

def _has_existing_build(state):
    """True when a generated game already lives in the workspace."""
    return any(os.path.isfile(os.path.join(WORKSPACE_DIR, f)) for f in _known_build_files(state))

def _archive_current_build(state):
    """Moves the current build's files into nexus_workspace/archive/<timestamp>/
    so starting a new game never destroys the previous one."""
    to_move = [f for f in _known_build_files(state) if os.path.isfile(os.path.join(WORKSPACE_DIR, f))]
    if not to_move:
        return
    archive_dir = os.path.join(WORKSPACE_DIR, "archive", str(int(time.time())))
    os.makedirs(archive_dir, exist_ok=True)
    for fname in to_move:
        src = os.path.join(WORKSPACE_DIR, fname)
        dst = os.path.join(archive_dir, fname.replace("/", "__"))
        try:
            os.replace(src, dst)
        except OSError as e:
            logging.warning(f"Could not archive {fname}: {e}")
    speak(f"Archived the previous game ({len(to_move)} files) into archive/{os.path.basename(archive_dir)}.")
    logging.info(f"Archived previous build files to {archive_dir}: {to_move}")

def _register_new_plan_files(state, added_files, request):
    """Folds files the model added during a stacked change into the plan, so
    future context windows, updates and integration reviews treat them as
    first-class project files."""
    plan = state.get("plan")
    if not isinstance(plan, dict) or not added_files:
        return
    files = plan.setdefault("files", [])
    known = {f.get("filename") for f in files if isinstance(f, dict)}
    for fname in added_files:
        if fname in known or len(files) >= 12:
            continue
        snippet = " ".join(str(request).split())[:80]
        files.append({"filename": fname, "description": f"Added by change request: {snippet}"})
        known.add(fname)

def run_update(state, user_prompt):
    """Stacks a change request onto the WHOLE current build (not just game.js)."""
    speak("Stacking your change onto the current build...")
    plan_files = [f.get("filename") for f in (state.get("plan") or {}).get("files", [])]
    candidates = [fn for fn in plan_files if fn] + state.get("completed_files", []) + ["index.html", "game.js", "style.css"]
    current_files = {}
    for fname in candidates:
        if not fname or fname in current_files:
            continue
        fpath = os.path.join(WORKSPACE_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, 'r') as f:
                content = f.read()
        except OSError:
            continue
        if len(content) > PER_FILE_CONTEXT_CAP:
            content = content[:PER_FILE_CONTEXT_CAP] + "\n/* ...truncated... */"
        current_files[fname] = content

    if not current_files:
        speak("There is no existing build to update. Describe a new game instead.")
        return

    sections = [f"===FILE: {fname}===\n{content}\n===END FILE===" for fname, content in current_files.items()]
    request = (
        f"CHANGE REQUEST:\n{user_prompt}\n\n"
        f"PROJECT PLAN:\n{json.dumps(state.get('plan') or {}, indent=2)}\n\n"
        f"CURRENT BUILD:\n\n" + "\n\n".join(sections) +
        "\n\nReturn ONLY the changed files via the protocol."
    )
    raw = ask_model(UPDATE_PROMPT, request, role="update")
    if not raw:
        speak("The update request failed. Please try again.")
        return

    verdict, blocks = parse_file_blocks(raw)
    if blocks:
        changed = apply_file_blocks(state, blocks, allow_new=True, max_new=3)
        if changed:
            added = [f for f in changed if f not in current_files]
            _register_new_plan_files(state, added, user_prompt)
            state.setdefault("history", []).append({"type": "update", "request": user_prompt, "changed": changed})
            save_state(state)
            # A change that adds files or touches several at once gets one
            # cross-file integration pass so the stack stays coherent.
            if added or len(changed) >= 2:
                state["integration_done"] = False
                save_state(state)
                integration_review(state)
            speak(f"Change stacked onto the build — touched: {', '.join(changed)}. Refresh the preview!")
        else:
            speak("The change produced no valid file edits, so the build was left untouched.")
        return

    # Legacy fallback: a raw single-file response that validates as JS becomes game.js.
    legacy = clean_code(raw)
    if legacy and "game.js" in current_files:
        accepted = accept_candidate("game.js", current_files["game.js"], legacy)
        if accepted != current_files["game.js"]:
            with open(os.path.join(WORKSPACE_DIR, "game.js"), 'w') as f:
                f.write(accepted)
            state.setdefault("history", []).append({"type": "update", "request": user_prompt, "changed": ["game.js"]})
            save_state(state)
            speak("Change stacked onto game.js. Refresh the preview!")
            logging.info("Successfully updated game.js (legacy single-file path).")
            return
    speak("The change response could not be applied safely, so the build was left untouched.")

def run_nexus_g(user_prompt):
    state = load_state()
    command = user_prompt.strip()
    if not command:
        return

    if command.lower().startswith("update"):
        run_update(state, command)
        return

    fresh_request = _fresh_start_request(command)
    if fresh_request is not None:
        if not fresh_request:
            speak("Tell me what new game to build — for example: 'new game a neon snake arena'.")
            return
        _archive_current_build(state)
        command = fresh_request
    elif command.lower() != "resume" and _has_existing_build(state):
        # Default behavior: stack the instruction onto the existing build
        # instead of recreating a whole different game in the workspace.
        run_update(state, command)
        return

    if command.lower() != "resume":
        state = _fresh_state()
        state["status"] = "planning"
        state["prompt"] = command
        state["history"].append({"type": "build", "request": command})

    if not state.get("plan"):
        if not plan_project(state):
            return

    fetch_assets(state)

    for file_info in state["plan"].get("files", []):
        fname = file_info.get("filename")
        if not fname or fname in state["completed_files"]:
            continue
        if not generate_file(state, file_info):
            speak(f"Generation failed for {fname}. Say 'resume' to retry from where we stopped.")
            return
        time.sleep(2)

    integration_review(state)

    state["status"] = "done"
    save_state(state)
    speak("Build finished. Check your preview screen now!")
    logging.info("Build Finished successfully.")

if __name__ == "__main__":
    print(f"[System] 🚀 Nexus-G online | Brain: {MODEL_DISPLAY} | Quality mode: {QUALITY_MODE.upper()}")
    print("[System] Commands: describe a game to build it • any further instruction STACKS changes onto the current build")
    print("[System]           'new game <idea>' starts fresh (previous build is archived) • 'resume' continues an interrupted build • 'exit' quits")
    threading.Thread(target=serve_game, daemon=True).start()
    while True:
        try:
            cmd = listen_command()
        except KeyboardInterrupt:
            print("\nExiting...")
            logging.info("System manually exited via KeyboardInterrupt.")
            break

        if cmd.lower() in ['exit', 'stop', 'quit']:
            logging.info("System exited via user command.")
            break

        run_nexus_g(cmd)
