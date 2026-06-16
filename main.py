import os
import json
import time
import random
import queue
import shutil
import tempfile
import subprocess
import requests
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import socket
import logging
from urllib.parse import urlparse
import re

# readline (when available) keeps the typed command line intact and editable
# even if a background task prints to the console while you are typing.
try:
    import readline  # noqa: F401
except ImportError:
    readline = None

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

# Qwen3.5's official recommended inference settings for thinking-mode coding are
# temperature 0.6 / top_p 0.95 / top_k 20. The OpenAI SDK has no native top_k
# field, so it is passed via extra_body; if the NVIDIA endpoint rejects it the
# call falls back to standard sampling automatically and stops sending it for the
# rest of the session (so quality is maximized when supported, never broken when not).
QWEN_TOP_K = 20
_extra_body_supported = [True]

# --- QUALITY PIPELINE TUNING ---
# "max"  = full agentic pipeline: plan -> code -> verify -> self-review -> integration review
# "fast" = skip the self-review and integration passes (fewer API requests per build)
QUALITY_MODE = os.environ.get("NEXUS_QUALITY", "max").strip().lower()

# Stream raw model tokens to the console? OFF by default keeps the console calm
# so the command prompt stays usable while a build runs — the AI no longer floods
# the terminal with sentences as it writes. The full output is still saved to the
# generated files and the activity log. Set NEXUS_STREAM=1 to watch it think live.
CONSOLE_STREAM = os.environ.get("NEXUS_STREAM", "0").strip().lower() in ("1", "true", "yes", "on")

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
    "update_planner": {"temperature": 0.5, "top_p": 0.95, "max_tokens": 4096},
    "deep_fixer":     {"temperature": 0.6, "top_p": 0.95, "max_tokens": 16384},
    "playtest_fixer": {"temperature": 0.5, "top_p": 0.95, "max_tokens": 16384},
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
    """Prints a Nexus-G message and voices it when TTS is available."""
    print(f"\n[Nexus-G]: {text}")
    if VOICE_ENABLED:
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception:
            pass

def listen_command():
    """Captures the next user command by microphone, falling back to text input."""
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
    """Returns a deep copy of the default build state."""
    return json.loads(json.dumps(DEFAULT_STATE))

def load_state():
    """Loads the persisted build state, merged over defaults for forward compatibility."""
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
    """Persists the build state atomically so a crash can never truncate it."""
    tmp_path = f"{STATE_FILE}.tmp"
    with open(tmp_path, 'w') as f:
        json.dump(state, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, STATE_FILE)

# ==========================================
# 4. SYSTEM PROMPTS (Claude-Code-style agentic quality pipeline)
# ==========================================
# TECH_SPEC is the single source of truth for runtime rules. It is injected
# into every code-facing prompt so the rules can never drift between roles.
# TECH_SPEC_BASE holds the mode-independent runtime contract. Per-render-style
# recipes live in TECH_SPEC_ADDENDA and are composed in by tech_spec_for() so the
# coder gets exactly the right high-detail recipe for 2D, HD-2D, or 3D.
TECH_SPEC_BASE = """RUNTIME CONTRACT — every generated file MUST satisfy ALL of these rules:

1. RENDERING STACK
- Three.js r128 GLOBAL build only. index.html loads it EXACTLY as:
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  followed by plain <script src="game.js"></script> tags (never type="module").
- `THREE` is a global object. FORBIDDEN in all .js files: `import`, `export`, top-level `await`.
- FORBIDDEN: Three.js examples/jsm addons (OrbitControls, GLTFLoader, EffectComposer, UnrealBloomPass, RenderPass, ShaderPass, any post-processing pass) — they do not exist in the global bundle. Write your own small camera/control/effect logic instead.

2. ASSETS — PROCEDURAL-ONLY POLICY
- Files under ./assets/ are text metadata stubs, NEVER loadable models.
- FORBIDDEN: loading .usd/.glb/.gltf/.obj/.fbx files, external images/textures/audio/fonts, or ANY network resource except the single Three.js CDN script tag above and the ./telemetry POST.
- Build ALL visuals procedurally per the RENDER STYLE recipe appended below. Runtime textures only via 2D canvas + THREE.CanvasTexture. Custom shaders only via inline GLSL written in JS template strings + THREE.ShaderMaterial (never import a shader). Sound (optional) only via inline WebAudio synthesis.
- The game must boot and play perfectly with the assets directory empty.

3. RESPONSIVE + DUAL INPUT (PC and mobile are BOTH first-class)
- index.html has <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">.
- The renderer canvas fills the window; on resize update the camera in use (perspective: set .aspect; orthographic: recompute left/right/top/bottom frustum bounds), call updateProjectionMatrix(), and renderer.setSize(window.innerWidth, window.innerHeight).
- KEYBOARD+MOUSE: WASD/arrow keys move; space and/or mouse for the primary action.
- TOUCH: a visible on-screen virtual joystick (left side) plus action button(s) (right side) using pointer/touch events, with CSS touch-action: none on control surfaces and preventDefault() to stop page scrolling.
- Both input schemes drive the SAME movement/action functions.

4. GAME LOOP, HUD, STATES, QUALITY
- requestAnimationFrame loop; delta time from THREE.Clock; ALL movement and timers scale by delta (frame-rate independent).
- DOM HUD overlay: score/status plus a one-line controls hint.
- Explicit game states: START SCREEN -> PLAYING -> GAME OVER -> RESTART. Restart fully resets the game WITHOUT reloading the page.
- HIGH VISUAL QUALITY IS REQUIRED. Follow the RENDER STYLE recipe appended below for cameras, sprites/meshes, backgrounds, lighting, animation, and effects, and honor the build's ART DIRECTION (palette, mood, lighting, detail).

5. TELEMETRY
- Every 2 seconds: fetch('./telemetry', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ fps: measuredFps }) }).catch(function () {});
- Telemetry must never break the game if the endpoint is unavailable.

6. COMPLETENESS BAR (non-negotiable)
- Output COMPLETE, immediately runnable files: no placeholders, no TODOs, no "..." elisions, no "rest of the code unchanged" comments.
- Zero console errors on load is the standard. Every referenced function/variable is defined; every element ID used in JS exists in the HTML; every file referenced by a tag exists in the plan."""

TECH_SPEC_ADDENDA = {
    "2d": """RENDER STYLE: 2D — crisp high-detail sprites with layered parallax. Build it with Three.js in an orthographic 2D setup:
- CAMERA: THREE.OrthographicCamera mapping ~1 world unit to 1 logical pixel. On resize recompute left/right/top/bottom from the viewport (NOT .aspect), then updateProjectionMatrix(). renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)). Camera looks down -Z; layer objects on stepped Z planes.
- SPRITES (procedural, high detail): define a reusable makeSpriteTexture(opts) that draws into an offscreen canvas in MULTIPLE layers — base silhouette, gradient body shading (createLinearGradient/createRadialGradient), 2-3 discrete cel-shade bands, a rim highlight on the lit edge, a core shadow on the unlit side, a MANDATORY dark outline stroke drawn last, and small detail accents — wrapped to a THREE.CanvasTexture (NearestFilter for pixel-art, LinearFilter for painterly). Use THREE.Sprite/SpriteMaterial for actors & pickups; textured PlaneGeometry + MeshBasicMaterial for tiles & parallax strips.
- BACKGROUNDS: a full-screen gradient-sky quad behind everything, plus 3-5 parallax PlaneGeometry strips at increasing -Z (gradient/silhouette canvas textures) scrolled at fractional camera speed via material.map.offset or position. Optional THREE.Points starfield.
- LIGHTING: unlit MeshBasic/Sprite materials are fine (no DirectionalLight needed). Ground shadows are FAUX — a soft dark ellipse sprite under each actor.
- ANIMATION: spritesheet frames in one canvas grid; animate with texture.repeat.set(1/cols,1/rows) + stepping texture.offset on a Clock-delta accumulator; plus layered sprite-part transforms.
- EFFECTS: additive glow sprites (AdditiveBlending, depthWrite:false), THREE.Points particles (AdditiveBlending, soft-circle CanvasTexture), screen shake, full-screen flash and radial vignette quads.""",

    "hd2d": """RENDER STYLE: HD-2D — 2D sprites composited into a lit 3D world (the Octopath Traveler look):
- CAMERA: THREE.PerspectiveCamera, low fov ~35-45, raised on Y and tilted down 25-40°, lookAt the play plane. Standard resize (.aspect + updateProjectionMatrix + setSize).
- ACTORS: billboarded textured PlaneGeometry (NOT THREE.Sprite, so they light and cast shadows). MeshStandardMaterial({ map, transparent:true, alphaTest:0.5 }); castShadow and receiveShadow; billboard by copying the camera's yaw to the plane each frame (grounded, upright). Build sprite textures with the SAME multi-layer Canvas2D factory as 2D (outline mandatory).
- ENVIRONMENT: a real 3D ground PlaneGeometry with a CanvasTexture tile pattern (receiveShadow); procedural box/cylinder props (castShadow) as buildings/trees; THREE.Fog or FogExp2 for depth haze; gradient sky via scene.background color or a large inverted gradient sphere.
- LIGHTING: AmbientLight + DirectionalLight; renderer.shadowMap.enabled = true with THREE.PCFSoftShadowMap — this lit, soft-shadowed stack is the core of the look.
- ANIMATION: spritesheet offset on the billboard + bob/squash via plane scale; environment props sway via transforms; all Clock-delta scaled.
- EFFECTS: faux-bloom via additive glow quads over emissive points; THREE.Points particles; optional custom THREE.ShaderMaterial with inline GLSL template strings (dissolve/scanline/water/hit-flash); screen shake, flashes, trails.""",

    "3d": """RENDER STYLE: 3D — full procedural meshes:
- CAMERA: THREE.PerspectiveCamera fov ~50-60; write your OWN follow/orbit camera (OrbitControls is a banned addon). Standard resize.
- ACTORS: build as THREE.Group part-hierarchies (torso/limbs/head as separate primitives) so they read as detailed and animate. MeshStandardMaterial with metalness/roughness; emissive + emissiveIntensity accents for glow; combine Box/Sphere/Cylinder/Cone/TorusGeometry plus LatheGeometry/ExtrudeGeometry (both in core r128) for bevels and curves.
- WORLD: scene.fog; gradient sky via a large SphereGeometry rendered with side: THREE.BackSide and a canvas-gradient texture; procedural ground (PlaneGeometry, or displaced BufferGeometry for hills); scattered procedural props.
- LIGHTING: AmbientLight + DirectionalLight; renderer.shadowMap.enabled = true (THREE.PCFSoftShadowMap); key objects cast and receive shadows.
- ANIMATION: transform/sine-driven animation of THREE.Group children (e.g. walk cycle = sine limb rotations), delta-scaled. No external clips/GLTF.
- EFFECTS: additive glow sprites/quads, THREE.Points particle systems (AdditiveBlending), custom THREE.ShaderMaterial with inline GLSL template strings, screen shake, flashes, trails.""",
}

def tech_spec_for(render_style):
    """Composes the base runtime contract with the per-mode rendering addendum."""
    style = render_style if render_style in TECH_SPEC_ADDENDA else "3d"
    return TECH_SPEC_BASE + "\n\n" + TECH_SPEC_ADDENDA[style]

def fill_spec(prompt_template, render_style):
    """Injects the per-mode runtime contract into a prompt template at call time."""
    return prompt_template.replace("__TECH_SPEC__", tech_spec_for(render_style))

# Back-compat default used by prompts that bake the contract at import time.
TECH_SPEC = tech_spec_for("3d")

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
6. Choose "render_style" from the user's request: if the user names a style obey it (pixel/sprite/side-scroller/retro/2D -> "2d"; HD-2D/2.5D/Octopath/diorama/tactics -> "hd2d"; 3D/first-person/racer/voxel/mesh -> "3d"); otherwise infer from genre (platformer/shmup/puzzle -> "2d"; top-down RPG/adventure/tactical with depth -> "hd2d"; FPS/racer/flight/sandbox -> "3d"). If genuinely ambiguous, default "3d".
7. Provide "art_direction": a palette (hex list), mood, lighting, and a detail_bar describing the high-detail look. Reflect the chosen render_style consistently in each file's description.

THINKING: Reason as deeply as you need inside your private thinking section first (game design, render style, interfaces, risks). Everything AFTER your thinking must be ONLY the JSON object.

CRITICAL OUTPUT RULES:
1. After your thinking, output ONLY one valid JSON object. Its first character MUST be { and its last character MUST be }.
2. No conversational text, no explanations, no markdown fences.

EXPECTED JSON SHAPE:
{
  "game_name": "Name",
  "render_style": "2d" | "hd2d" | "3d",
  "art_direction": {"palette": ["#hex", "..."], "mood": "...", "lighting": "...", "detail_bar": "..."},
  "files": [
    {"filename": "index.html", "description": "Viewport meta; three.js r128 CDN tag then game.js tag; HUD elements #hud, #score; overlay screens #overlay, #overlay-title, #overlay-msg, #overlay-btn; touch controls #joystick-zone, #btn-action; inline CSS"},
    {"filename": "game.js", "description": "Defines initGame(), startGame(), endGame(), restartGame(); reads the element IDs above; scene/camera/renderer; keyboard+mouse and touch joystick input; delta-time loop; telemetry POST"}
  ],
  "assets_needed": [{"filename": "short_name", "prompt": "thematic search description"}],
  "advanced_mechanics": ["Touch joystick + buttons (mobile)", "Keyboard + mouse (PC)", "..."]
}""".replace("__TECH_SPEC__", TECH_SPEC_BASE)

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
3. NO conversational text, NO explanations, NO markdown fences."""

FIXER_PROMPT = """You are the Nexus-G Debug Surgeon. You receive ONE file and ONE concrete error or validation report. Fix it with the minimum change necessary.

RULES:
1. Change only what the error requires. Preserve all other behavior, names and structure.
2. If the file is TRUNCATED (cut off mid-statement), complete it consistently with its own style.
3. The corrected file must satisfy this contract:
__TECH_SPEC__

THINKING (in your private section, in this order): 1) name the ROOT CAUSE of the error in one line; 2) state the MINIMAL change that fixes it; 3) then write the file. Everything AFTER your thinking must be ONLY the corrected file.

CRITICAL OUTPUT RULES:
1. After your thinking, output the COMPLETE corrected file — every line, top to bottom.
2. Raw content only: no commentary, no markdown fences."""

REVIEWER_PROMPT = """You are the Nexus-G Adversarial Reviewer — a hostile senior engineer paid to find what is genuinely BROKEN. You receive one file under review (plus sibling project files for cross-reference). Hunt ONLY for real defects:

- Runtime errors: undefined variables/functions, references to element IDs that exist in no provided file, syntax slips, use-before-define across script load order.
- Truncated or unreachable logic; event listeners never attached; game states that cannot be reached or exited; restart that does not reset state.
- PLAYABILITY (the game must be playable, not just load): the player/avatar is actually created, added to the scene, and within the camera's view; input handlers are wired to the movement/action logic so controls really move the player; the animation loop is STARTED (requestAnimationFrame is actually called) and calls renderer.render every frame; the camera is positioned/oriented to see the playfield (not stuck at the origin looking at nothing); scoring/win/lose actually triggers and the game is neither instantly over nor impossible to lose; nothing is referenced before it is defined or under a misspelled name.
- Contract violations (below): missing touch OR keyboard input path, non-responsive canvas, forbidden module syntax/addons/asset loads, missing telemetry, missing lighting/HUD/game-over.

__TECH_SPEC__

THINKING: Trace the code path by path in your private thinking section as deeply as you need. Everything AFTER your thinking must be ONLY your verdict per the protocol below.

DECISION PROTOCOL (follow exactly):
- If the file would ship as-is (no genuine defects), reply after your thinking with EXACTLY this single line and nothing else:
APPROVED
- Otherwise reply with the COMPLETE corrected file: raw content only, no commentary, no markdown fences, no diff — the whole file.

HARD RULE: never rewrite working code for style, taste, or "improvement". Fix defects only. When in doubt, reply APPROVED."""

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

__FILE_BLOCK_SPEC__""".replace("__FILE_BLOCK_SPEC__", FILE_BLOCK_SPEC)

UPDATE_PROMPT = """You are the Nexus-G Live-Ops Engineer. You receive a working game's full file set plus ONE change request. Apply the request the way a careful senior engineer edits production code:

- The CURRENT BUILD files are the live truth. Build on top of them: changes accumulate across requests. NEVER regenerate the game from scratch and NEVER drop features the request does not name.
- Preserve ALL existing behavior not named by the request.
- Keep every existing public name (functions, globals, element IDs) stable unless the request requires renaming.
- Changed files must remain complete and satisfy the runtime contract:
__TECH_SPEC__

OUTPUT:
- Plan the change in your private thinking section first; everything AFTER your thinking must be ONLY the file blocks.
- If a CHANGE BLUEPRINT is provided, treat it as the authoritative scope: edit exactly its target_files, honor every item in preserve, guard against its risks, and add only its new_files.
- Return ONLY the files you changed (complete content for each), via the protocol below.
- You may add at most 2 NEW files if the request truly requires them (and you must wire them into index.html in the same response).
- No commentary outside the blocks.

__FILE_BLOCK_SPEC__""".replace("__FILE_BLOCK_SPEC__", FILE_BLOCK_SPEC)

JSON_REPAIR_PROMPT = """You are a strict JSON repair machine. The user gives you text that was MEANT to be one valid JSON object but is malformed (stray prose, markdown fences, bad quotes/commas, truncation). Reconstruct the intended object, preserving all of its content.

OUTPUT RULES:
1. After any private thinking, output ONLY the corrected JSON object. First character {, last character }.
2. No fences, no commentary."""

UPDATE_BLUEPRINT_PROMPT = """You are the Nexus-G Change Architect. Given a change request for an existing browser game plus its file list and plan, produce a SHORT JSON blueprint that scopes the work precisely. Do NOT write code.

THINKING: Reason privately about the smallest correct change — which existing files it touches, what must NOT regress, and what could break. Everything AFTER your thinking must be ONLY the JSON object.

OUTPUT (JSON only, first character {, last character }):
{
  "target_files": ["existing files you will edit"],
  "new_files": ["only if truly required, else empty"],
  "edits": ["concrete, specific edits to make"],
  "preserve": ["existing behavior/features that must NOT change or regress"],
  "risks": ["what could break; how to guard against it"]
}
No prose, no markdown fences."""

DEEP_FIXER_PROMPT = """You are the Nexus-G Senior Fix Engineer. Two quick fixes have already FAILED to make this file valid. Reconsider it holistically with the full project context provided.

__TECH_SPEC__

You MAY restructure the file, but you MUST preserve its public interface (the global function names, variable names and element IDs other files rely on) and satisfy the runtime contract above.

THINKING: Privately diagnose the REAL root cause the quick fixes missed, then plan the corrected file. Everything AFTER your thinking must be ONLY the complete corrected file.

OUTPUT: the COMPLETE corrected file — raw content only, no commentary, no markdown fences."""

PLAYTEST_FIXER_PROMPT = """You are the Nexus-G Playtest Fix Engineer. The game PARSES and LOADS but CRASHES AT RUNTIME, which makes it unplayable even though the server serves it. You are given the exact runtime error (with its stack) and the file to fix.

__TECH_SPEC__

Fix the ROOT CAUSE of the runtime crash so the game actually runs and is playable. The most common causes are: a misspelled or undefined function/variable name; calling something before it is defined; reading a property of an undefined object; a value that is never initialized; an animation-loop function that throws on a frame. Preserve the file's public interface and ALL working behavior — change only what is needed to stop the crash and make the game playable.

THINKING: privately identify EXACTLY which name is undefined or which value is null/undefined and why, trace how it reaches the reported error, then write the corrected file. Everything AFTER your thinking must be ONLY the complete corrected file.

OUTPUT: the COMPLETE corrected file — raw content only, no commentary, no markdown fences."""

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
    """Serves the generated game from nexus_workspace and accepts FPS telemetry POSTs."""

    def __init__(self, *args, **kwargs):
        """Anchors the file server to the workspace directory."""
        super().__init__(*args, directory=WORKSPACE_DIR, **kwargs)

    def end_headers(self):
        """Adds no-cache and CORS headers so edits show up on refresh."""
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, fmt, *args):
        """Keeps HTTP access logs OFF the console so server traffic never
        scrambles the command prompt; telemetry heartbeats are dropped and
        other requests go to the activity log file instead."""
        message = fmt % args
        if "/telemetry" not in message:
            logging.info(f"HTTP {self.address_string()} {message}")

    def do_POST(self):
        """Receives {"fps": ...} telemetry beacons from the running game."""
        global latest_telemetry
        if urlparse(self.path).path != "/telemetry":
            self.send_response(404)
            self.end_headers()
            return

        # Telemetry beacons are tiny; reject anything oversized or malformed.
        max_body = 4096
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            content_length = 0
        if content_length <= 0 or content_length > max_body:
            self.send_response(413)
            self.end_headers()
            return

        try:
            data = json.loads(self.rfile.read(content_length).decode("utf-8"))
            latest_telemetry["fps"] = data.get("fps", 0)
        except (UnicodeDecodeError, ValueError, AttributeError):
            self.send_response(400)
            self.end_headers()
            return

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

def serve_game():
    """Runs the threaded preview web server for the generated game on port 8080."""
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
            last_pulse = time.time()
            pulsed = False
            create_kwargs = dict(
                model=NVIDIA_MODEL,
                messages=messages,
                temperature=params["temperature"],
                top_p=params["top_p"],
                max_tokens=max_tokens,
                stream=True,
            )
            if _extra_body_supported[0]:
                create_kwargs["extra_body"] = {"top_k": QWEN_TOP_K}
            completion = cloud_client.chat.completions.create(**create_kwargs)
            for chunk in completion:
                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.delta:
                        # Some deployments stream the model's private reasoning in a
                        # separate field: show it live only in stream mode, never keep it.
                        reasoning = getattr(choice.delta, "reasoning_content", None)
                        if reasoning and CONSOLE_STREAM:
                            print(reasoning, end="", flush=True)
                        if choice.delta.content is not None:
                            content = choice.delta.content
                            if CONSOLE_STREAM:
                                print(content, end="", flush=True)
                            full_response += content
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                # Calm mode: a single dot every few seconds shows liveness without
                # flooding the console (so you can keep typing the next command).
                if not CONSOLE_STREAM and time.time() - last_pulse >= 4:
                    last_pulse = time.time()
                    pulsed = True
                    print(".", end="", flush=True)
            if CONSOLE_STREAM or pulsed:
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
                if _extra_body_supported[0]:
                    _extra_body_supported[0] = False
                    logging.warning("Endpoint rejected extra params (top_k); retrying with standard sampling only.")
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
        status = f"[Cloud API - {MODEL_DISPLAY} | role={role}]"
        if CONSOLE_STREAM:
            print(f"\n{status}")
        else:
            logging.info(status)
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
        status = "[Native Local Engine - Offline Mode (Direct Memory 120B)]"
        if CONSOLE_STREAM:
            print(f"\n{status}")
        else:
            logging.info(status)
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
                        if CONSOLE_STREAM:
                            print(content, end="", flush=True)
                        full_response += content
            if CONSOLE_STREAM:
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
    files = files[:MAX_PLAN_FILES]
    present = {f["filename"] for f in files}
    if "index.html" not in present:
        files.insert(0, {
            "filename": "index.html",
            "description": ("HTML shell: viewport meta; three.js r128 CDN script tag, then the game script tag(s); "
                            "HUD and touch-control elements exactly as referenced by the game code.")
        })
    if "game.js" not in present:
        files.append({
            "filename": "game.js",
            "description": ("Core game runtime: scene/camera/renderer setup, keyboard+mouse and touch input, "
                            "delta-time game loop and state transitions, HUD updates, telemetry POST.")
        })
    plan["files"] = files
    if not isinstance(plan.get("assets_needed"), list):
        plan["assets_needed"] = []
    if not isinstance(plan.get("advanced_mechanics"), list):
        plan["advanced_mechanics"] = []
    style = plan.get("render_style")
    plan["render_style"] = style if style in ("2d", "hd2d", "3d") else "3d"
    if not isinstance(plan.get("art_direction"), dict):
        plan["art_direction"] = {}
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
        """Returns the first non-blank-trimmed line of a response."""
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

# ==========================================
# 7b. HEADLESS PLAYTEST (runtime crash detection)
# ==========================================
# `node --check` only proves a file PARSES; it never runs it. A game can parse,
# load, and still throw at runtime (a misspelled function, an undefined value in
# the animation loop) — which freezes the screen and makes it unplayable even
# though the server serves it. This stage EXECUTES the assembled game under a
# stubbed browser/WebGL/THREE environment in Node and reports genuine runtime
# crashes (a ReferenceError = a real undefined-reference bug in any browser) so a
# targeted fixer can repair them. It is best-effort: if Node is missing or the
# result is ambiguous, the build proceeds exactly as before.
PLAYTEST_HARNESS_JS = r'''
'use strict';
var fs = require('fs');
var vm = require('vm');

function makeStub() {
  var target = function stub() {};
  return new Proxy(target, {
    get: function (t, prop) {
      if (prop === Symbol.toPrimitive) return function () { return 0; };
      if (prop === Symbol.iterator) return function () { return [][Symbol.iterator](); };
      if (prop === Symbol.toStringTag) return 'Stub';
      if (prop === 'then') return undefined;
      if (prop === 'length') return 0;
      if (prop === 'nodeType') return 1;
      if (prop === 'prototype') { if (!t.__p) t.__p = {}; return t.__p; }
      if (typeof prop === 'symbol') return undefined;
      if (Object.prototype.hasOwnProperty.call(t, prop)) return t[prop];
      var s = makeStub(); t[prop] = s; return s;
    },
    set: function (t, prop, val) { t[prop] = val; return true; },
    has: function () { return true; },
    apply: function () { return makeStub(); },
    construct: function () { return makeStub(); }
  });
}

var rafQueue = [];
var box = {};
box.globalThis = box; box.window = box; box.self = box; box.top = box; box.parent = box;
box.console = console;
box.Math = Math; box.JSON = JSON; box.Date = Date; box.Array = Array; box.Object = Object;
box.String = String; box.Number = Number; box.Boolean = Boolean; box.Symbol = Symbol;
box.Map = Map; box.Set = Set; box.WeakMap = WeakMap; box.WeakSet = WeakSet;
box.Promise = Promise; box.RegExp = RegExp; box.Function = Function;
box.Error = Error; box.TypeError = TypeError; box.RangeError = RangeError; box.SyntaxError = SyntaxError;
box.Float32Array = Float32Array; box.Float64Array = Float64Array;
box.Uint8Array = Uint8Array; box.Uint16Array = Uint16Array; box.Uint32Array = Uint32Array;
box.Int8Array = Int8Array; box.Int16Array = Int16Array; box.Int32Array = Int32Array;
box.Uint8ClampedArray = Uint8ClampedArray; box.ArrayBuffer = ArrayBuffer; box.DataView = DataView;
box.parseInt = parseInt; box.parseFloat = parseFloat; box.isNaN = isNaN; box.isFinite = isFinite;
box.encodeURIComponent = encodeURIComponent; box.decodeURIComponent = decodeURIComponent;
box.performance = { now: function () { return Date.now(); } };
box.requestAnimationFrame = function (cb) { rafQueue.push(cb); return rafQueue.length; };
box.cancelAnimationFrame = function () {};
box.setTimeout = function () { return 0; }; box.clearTimeout = function () {};
box.setInterval = function () { return 0; }; box.clearInterval = function () {};
box.requestIdleCallback = function () { return 0; }; box.cancelIdleCallback = function () {};
box.queueMicrotask = function () {};
box.navigator = { userAgent: 'node', platform: 'node', maxTouchPoints: 0, language: 'en', vendor: '', getGamepads: function () { return []; } };
box.devicePixelRatio = 1; box.innerWidth = 1280; box.innerHeight = 720; box.outerWidth = 1280; box.outerHeight = 720;
box.scrollX = 0; box.scrollY = 0; box.pageXOffset = 0; box.pageYOffset = 0;
box.location = { href: 'http://localhost/', protocol: 'http:', host: 'localhost', hostname: 'localhost', port: '', pathname: '/', search: '', hash: '', origin: 'http://localhost', reload: function () {}, replace: function () {}, assign: function () {} };
box.history = { pushState: function () {}, replaceState: function () {}, back: function () {}, forward: function () {}, go: function () {} };
box.localStorage = { getItem: function () { return null; }, setItem: function () {}, removeItem: function () {}, clear: function () {} };
box.sessionStorage = box.localStorage;
box.alert = function () {}; box.confirm = function () { return true; }; box.prompt = function () { return ''; };
box.addEventListener = function () {}; box.removeEventListener = function () {}; box.dispatchEvent = function () { return true; };
box.matchMedia = function () { return { matches: false, media: '', addListener: function () {}, removeListener: function () {}, addEventListener: function () {}, removeEventListener: function () {} }; };
box.getComputedStyle = function () { return makeStub(); };
box.screen = { width: 1280, height: 720, availWidth: 1280, availHeight: 720, orientation: { type: 'landscape-primary', angle: 0, addEventListener: function () {}, lock: function () { return Promise.resolve(); } } };
box.fetch = function () { return Promise.resolve({ ok: true, status: 200, json: function () { return Promise.resolve({}); }, text: function () { return Promise.resolve(''); } }); };
box.AudioContext = function () { return makeStub(); }; box.webkitAudioContext = box.AudioContext;
box.Audio = function () { return makeStub(); }; box.Image = function () { return makeStub(); };
box.Worker = function () { return makeStub(); }; box.SharedWorker = function () { return makeStub(); };
box.WebSocket = function () { return makeStub(); }; box.XMLHttpRequest = function () { return makeStub(); };
box.URL = function () { return makeStub(); }; box.URLSearchParams = function () { return makeStub(); };
box.Blob = function () { return makeStub(); }; box.File = function () { return makeStub(); }; box.FileReader = function () { return makeStub(); };
box.FormData = function () { return makeStub(); }; box.Headers = function () { return makeStub(); };
box.ResizeObserver = function () { return makeStub(); }; box.IntersectionObserver = function () { return makeStub(); };
box.MutationObserver = function () { return makeStub(); }; box.PerformanceObserver = function () { return makeStub(); };
box.OffscreenCanvas = function () { return makeStub(); }; box.Path2D = function () { return makeStub(); };
box.DOMParser = function () { return makeStub(); }; box.XMLSerializer = function () { return makeStub(); };
box.Event = function () { return makeStub(); }; box.CustomEvent = function () { return makeStub(); };
box.KeyboardEvent = function () { return makeStub(); }; box.MouseEvent = function () { return makeStub(); };
box.PointerEvent = function () { return makeStub(); }; box.TouchEvent = function () { return makeStub(); };
box.WheelEvent = function () { return makeStub(); }; box.DragEvent = function () { return makeStub(); };
box.WebGLRenderingContext = function () {}; box.WebGL2RenderingContext = function () {};
box.THREE = makeStub();

function makeEl() { return makeStub(); }
var docTarget = makeStub();
box.document = new Proxy(docTarget, {
  get: function (t, prop) {
    if (prop === 'createElement' || prop === 'createElementNS') return function () { return makeEl(); };
    if (prop === 'createTextNode' || prop === 'createDocumentFragment') return function () { return makeEl(); };
    if (prop === 'getElementById') return function () { return makeEl(); };
    if (prop === 'querySelector') return function () { return makeEl(); };
    if (prop === 'querySelectorAll' || prop === 'getElementsByTagName' || prop === 'getElementsByClassName' || prop === 'getElementsByName') return function () { return []; };
    if (prop === 'addEventListener' || prop === 'removeEventListener') return function () {};
    if (prop === 'body') { if (!t.__body) t.__body = makeEl(); return t.__body; }
    if (prop === 'head') { if (!t.__head) t.__head = makeEl(); return t.__head; }
    if (prop === 'documentElement') { if (!t.__de) t.__de = makeEl(); return t.__de; }
    if (prop === 'readyState') return 'complete';
    if (prop === 'hidden') return false;
    if (prop === 'visibilityState') return 'visible';
    if (prop === 'fullscreenElement' || prop === 'pointerLockElement') return null;
    if (prop === 'cookie') return '';
    if (typeof prop === 'symbol') return undefined;
    if (Object.prototype.hasOwnProperty.call(t, prop)) return t[prop];
    var s = makeStub(); t[prop] = s; return s;
  },
  set: function (t, prop, val) { t[prop] = val; return true; },
  has: function () { return true; }
});

var ctx = vm.createContext(box);
var firstError = null;
function record(phase, e) {
  if (firstError) return;
  firstError = { phase: phase, name: (e && e.name) || 'Error', message: String((e && e.message) || e), stack: String((e && e.stack) || '') };
}

var files = process.argv.slice(2);
for (var i = 0; i < files.length; i++) {
  var src = '';
  try { src = fs.readFileSync(files[i], 'utf8'); } catch (e) { continue; }
  try { vm.runInContext(src, ctx, { filename: files[i], timeout: 5000 }); }
  catch (e) { record('load:' + files[i], e); }
  if (firstError) break;
}

if (!firstError) {
  var entries = ['init', 'initGame', 'setup', 'main', 'boot', 'start', 'startGame', 'beginGame', 'run', 'play'];
  for (var k = 0; k < entries.length; k++) {
    try { if (typeof box[entries[k]] === 'function') box[entries[k]](); }
    catch (e) { record('call:' + entries[k], e); break; }
  }
}

if (!firstError) {
  var frames = 0;
  try {
    while (rafQueue.length && frames < 45) {
      var cb = rafQueue.shift();
      frames++;
      if (typeof cb === 'function') cb(1000 + frames * 16);
    }
  } catch (e) { record('frame', e); }
}

process.stdout.write(firstError ? JSON.stringify(firstError) : 'OK');
'''

SCRIPT_TAG_RE = re.compile(r'<script\b([^>]*)>(.*?)</script>', re.IGNORECASE | re.DOTALL)
SCRIPT_SRC_RE = re.compile(r'\bsrc\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)

def _assemble_runtime_sources(state):
    """Returns ordered (label, source) JS chunks that the browser would execute,
    reconstructed from index.html's script tags (external libs are skipped — they
    are stubbed), falling back to the build's .js files."""
    sources = []
    html_path = os.path.join(WORKSPACE_DIR, "index.html")
    html = ""
    if os.path.isfile(html_path):
        try:
            with open(html_path, 'r') as f:
                html = f.read()
        except OSError:
            html = ""
    if html:
        inline_idx = 0
        for m in SCRIPT_TAG_RE.finditer(html):
            attrs, body = m.group(1), m.group(2)
            if re.search(r'type\s*=\s*["\']module["\']', attrs, re.IGNORECASE):
                continue
            src_m = SCRIPT_SRC_RE.search(attrs)
            if src_m:
                src = src_m.group(1)
                if src.startswith("http") or src.startswith("//"):
                    continue  # external library (e.g. the THREE CDN) — already stubbed
                local = safe_filename(src)
                p = os.path.join(WORKSPACE_DIR, local) if local else ""
                if local and os.path.isfile(p):
                    try:
                        with open(p, 'r') as f:
                            sources.append((local, f.read()))
                    except OSError:
                        pass
            elif body.strip():
                inline_idx += 1
                sources.append((f"index.html#inline{inline_idx}", body))
    if not sources:
        for fname in ["game.js"] + [f for f in state.get("completed_files", []) if f.endswith(".js")]:
            if any(lbl == fname for lbl, _ in sources):
                continue
            p = os.path.join(WORKSPACE_DIR, fname)
            if os.path.isfile(p):
                try:
                    with open(p, 'r') as f:
                        sources.append((fname, f.read()))
                except OSError:
                    pass
    return sources

def playtest_runtime(state):
    """Executes the assembled game headlessly in Node and returns a crash dict
    {name, message, phase, stack, hard, file} or None when it runs clean / cannot
    be tested. `hard` is True for a ReferenceError (a real undefined-reference bug).
    """
    try:
        subprocess.run(["node", "-v"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        logging.warning("Node.js not found; skipping headless playtest.")
        return None
    except Exception:
        return None

    sources = _assemble_runtime_sources(state)
    if not sources:
        return None

    tmp = tempfile.mkdtemp(prefix="nexus_playtest_")
    try:
        harness_path = os.path.join(tmp, "_harness.js")
        with open(harness_path, 'w') as f:
            f.write(PLAYTEST_HARNESS_JS)
        temp_to_label = {}
        arg_paths = []
        for n, (label, source) in enumerate(sources):
            safe = re.sub(r'[^A-Za-z0-9_.]', "_", label)
            tp = os.path.join(tmp, f"{n:02d}__{safe}.js")
            with open(tp, 'w') as f:
                f.write(source)
            temp_to_label[os.path.basename(tp)] = label
            arg_paths.append(tp)
        try:
            res = subprocess.run(["node", harness_path] + arg_paths,
                                 capture_output=True, text=True, timeout=40)
        except subprocess.TimeoutExpired:
            logging.warning("Headless playtest timed out (possible infinite loop).")
            return {"name": "Timeout", "message": "playtest timed out", "phase": "run",
                    "stack": "", "hard": False, "file": "game.js"}
        out = (res.stdout or "").strip()
        if out == "OK" or not out:
            return None
        try:
            crash = json.loads(out)
        except (ValueError, TypeError):
            logging.info(f"Playtest produced unparseable output; ignoring: {out[:200]}")
            return None
        crash["hard"] = crash.get("name") == "ReferenceError"
        # Map the crash back to an editable build file via the stack frames.
        label = None
        stack = crash.get("stack", "") + " " + crash.get("phase", "")
        for base, lbl in temp_to_label.items():
            if base in stack:
                label = lbl
                break
        if label is None and sources:
            label = sources[-1][0]
        if label and label.startswith("index.html"):
            crash["file"] = "index.html"
        elif label and label.endswith(".js"):
            crash["file"] = label
        else:
            crash["file"] = "game.js"
        return crash
    except Exception as e:
        logging.warning(f"Headless playtest error (skipped): {e}")
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def playtest_and_fix(state, max_iters=2):
    """Runs the headless playtest and, when it finds a real runtime crash, asks a
    targeted fixer to repair the offending file so the game becomes playable."""
    if not (is_connected() and cloud_client):
        crash = playtest_runtime(state)
        if crash and crash.get("hard"):
            logging.warning(f"Playtest found a runtime crash (offline, not fixed): {crash.get('message')}")
        return

    style = (state.get("plan") or {}).get("render_style", "3d")
    for attempt in range(max_iters):
        crash = playtest_runtime(state)
        if not crash:
            if attempt:
                speak("Runtime crash fixed — the game is playable now.")
            return
        if not crash.get("hard"):
            logging.info(f"Playtest advisory ({crash.get('name')}: {crash.get('message')}); not auto-fixing.")
            return
        target = crash.get("file") or "game.js"
        fpath = os.path.join(WORKSPACE_DIR, target)
        if not os.path.isfile(fpath):
            target = "game.js"
            fpath = os.path.join(WORKSPACE_DIR, target)
            if not os.path.isfile(fpath):
                return
        try:
            with open(fpath, 'r') as f:
                code = f.read()
        except OSError:
            return
        speak(f"The game loads but crashes at runtime ({crash.get('name')}). Fixing {target}...")
        logging.warning(f"Playtest crash in {target}: {crash.get('name')}: {crash.get('message')}")
        report = (
            f"{build_context(state, current_file=target)}\n\n"
            f"RUNTIME CRASH (the game loads but is unplayable):\n"
            f"  error: {crash.get('name')}: {crash.get('message')}\n"
            f"  phase: {crash.get('phase', '')}\n"
            f"  stack:\n{crash.get('stack', '')[:1500]}\n\n"
            f"FILE TO FIX: `{target}`\n\n"
            f"CURRENT CONTENT:\n{code}\n\n"
            f"Return the complete corrected file."
        )
        fixed = clean_code(ask_model(fill_spec(PLAYTEST_FIXER_PROMPT, style), report, role="playtest_fixer"))
        accepted = accept_candidate(target, code, _autofix_file(target, fixed))
        if accepted == code:
            logging.warning(f"Playtest fix for {target} was rejected or unchanged; stopping.")
            return
        with open(fpath, 'w') as f:
            f.write(accepted)
        logging.info(f"Applied playtest fix to {target}.")

    final = playtest_runtime(state)
    if final and final.get("hard"):
        speak("I applied fixes but the game may still have a runtime issue — check the preview console.")
        logging.warning(f"Playtest still failing after {max_iters} fixes: {final.get('message')}")

def _strip_module_type(html):
    """Removes type="module" from <script> tags (the one safe mechanical auto-fix)."""
    return re.sub(r'(<script\b[^>]*?)\s+type\s*=\s*["\']module["\']', r'\1', html, flags=re.IGNORECASE)

def validate_static(filename, content):
    """Cheap regex checks for the most common procedural-contract violations.

    Returns (hard, soft): `hard` are high-confidence violations worth a model fix;
    `soft` are heuristic and logged only. No external dependencies.
    """
    hard, soft = [], []
    is_html = filename.endswith((".html", ".htm"))
    is_js = filename.endswith(".js")

    if is_html and re.search(r'<script\b[^>]*\btype\s*=\s*["\']module["\']', content, re.IGNORECASE):
        hard.append('index.html uses <script type="module"> — use a plain classic <script> tag (no ES modules).')
    if re.search(r'examples/jsm|OrbitControls|GLTFLoader|EffectComposer|UnrealBloomPass|ShaderPass|RenderPass|OutlinePass', content):
        hard.append("Uses a Three.js examples/jsm addon that is absent from the r128 global build — replace it with hand-written logic.")
    if re.search(r'\.(glb|gltf|obj|fbx|usd)\b', content, re.IGNORECASE):
        hard.append("References an external 3D model file — all visuals must be procedural (no model loading).")
    if re.search(r'TextureLoader\s*\([^)]*\)\s*\.\s*load\s*\(', content) or re.search(r'\bnew\s+Audio\s*\(', content):
        hard.append("Loads an external texture/audio asset — use CanvasTexture / inline WebAudio instead.")
    if re.search(r'<img\b[^>]*\bsrc\s*=\s*["\']https?:', content, re.IGNORECASE) or re.search(r'url\(\s*["\']?https?:', content, re.IGNORECASE):
        hard.append("Loads an external image URL — all visuals must be procedural.")
    for m in re.finditer(r'fetch\(\s*["\']([^"\']+)["\']', content):
        target = m.group(1)
        if target.startswith("http") and "/telemetry" not in target:
            hard.append(f"Network fetch to {target} — only the local ./telemetry POST is allowed.")
            break

    if is_js and "THREE." in content and "requestAnimationFrame" in content and "/telemetry" not in content:
        soft.append("This game script may be missing the ./telemetry POST.")
    if is_js and "THREE." in content:
        if "requestAnimationFrame" not in content:
            soft.append("No requestAnimationFrame loop found — the game may never animate or update.")
        if ".render(" not in content:
            soft.append("No renderer.render(...) call found — the screen may stay blank.")

    return hard, soft

def _autofix_file(fname, code):
    """Applies the safe mechanical auto-fixes (currently: strip <script type=module> in HTML)."""
    if fname.endswith((".html", ".htm")):
        return _strip_module_type(code)
    return code

def accept_candidate(fname, old_code, new_code):
    """Anti-regression gate for model rewrites (review/integration/update passes).

    A candidate replaces the existing code only if it is non-empty, not
    suspiciously shorter than the original, passes the syntax check, and (for
    HTML) does not introduce a procedural-contract violation the old file lacked.
    Returns the code that should be kept.
    """
    new_code = (new_code or "").strip()
    if not new_code:
        return old_code
    if old_code and len(new_code) < len(old_code) * 0.5:
        logging.warning(f"Rejected candidate for {fname}: suspiciously short ({len(new_code)} vs {len(old_code)} chars).")
        return old_code
    if fname.endswith((".html", ".htm")):
        new_code = _strip_module_type(new_code)
        new_hard, _ = validate_static(fname, new_code)
        old_hard, _ = validate_static(fname, old_code) if old_code else ([], [])
        if new_hard and not old_hard:
            logging.warning(f"Rejected HTML candidate for {fname}: introduces contract violations {new_hard}")
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
    """Fetches NVIDIA Omniverse USD search metadata for an asset prompt.

    Writes placeholder stubs when offline or unauthorized; generated games
    never load these files (the runtime contract keeps all visuals procedural).
    """
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
                ),
                timeout=(5, 30)
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

    render_style = plan.get("render_style", "3d")
    art = plan.get("art_direction") if isinstance(plan.get("art_direction"), dict) else {}
    art_summary = "; ".join(f"{k}: {v}" for k, v in art.items() if v)
    style_line = f"RENDER STYLE: {render_style} — follow that style's recipe in the runtime contract exactly."
    if art_summary:
        style_line += f"\nART DIRECTION: {art_summary}"

    parts = [
        f"USER REQUEST:\n{state.get('prompt', '')}",
        style_line,
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
    """Fetches any not-yet-completed plan assets, checkpointing progress in state."""
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
    style = (state.get("plan") or {}).get("render_style", "3d")
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
    code = clean_code(ask_model(fill_spec(CODER_PROMPT, style), request, role="coder"))
    if not code:
        return False

    code = _autofix_file(fname, code)
    with open(fpath, 'w') as f:
        f.write(code)

    # --- Validation gate (syntax + static contract checks) + diagnose/fix loop ---
    passed, err = test_syntax(fname)
    hard, soft = validate_static(fname, code)
    for note in soft:
        logging.info(f"{fname} static note: {note}")
    fixes = 0
    while (not passed or hard) and fixes < MAX_FIX_ITERATIONS:
        fixes += 1
        problem = err if not passed else ""
        if hard:
            problem += ("\n" if problem else "") + "CONTRACT ISSUES:\n- " + "\n- ".join(hard)
        speak(f"Fixing {fname} (attempt {fixes})...")
        logging.warning(f"Validation issue in {fname} (fix attempt {fixes}): {problem}")
        fix_request = (
            f"FILE: {fname}\n"
            f"VALIDATION REPORT:\n{problem}\n\n"
            f"CURRENT CONTENT:\n{code}\n\n"
            f"Return the complete corrected file."
        )
        fixed = clean_code(ask_model(fill_spec(FIXER_PROMPT, style), fix_request, role="fixer"))
        if not fixed:
            break
        code = _autofix_file(fname, fixed)
        with open(fpath, 'w') as f:
            f.write(code)
        passed, err = test_syntax(fname)
        hard, soft = validate_static(fname, code)

    # --- Deep-fix escalation: one holistic pass with full project context ---
    if not passed:
        speak(f"Escalating to a deep fix for {fname}...")
        logging.warning(f"{fname} still failing syntax after {fixes} quick fixes; deep-fixing.")
        deep_request = (
            f"{build_context(state, current_file=fname)}\n\n"
            f"FILE: {fname}\n"
            f"The quick fixes did not resolve this. Latest error:\n{err}\n\n"
            f"CURRENT CONTENT:\n{code}\n\n"
            f"Return the complete corrected file."
        )
        deep = clean_code(ask_model(fill_spec(DEEP_FIXER_PROMPT, style), deep_request, role="deep_fixer"))
        if deep:
            code = _autofix_file(fname, deep)
            with open(fpath, 'w') as f:
                f.write(code)
            passed, err = test_syntax(fname)
        if not passed:
            logging.warning(f"{fname} still failing validation after deep fix; keeping best effort.")

    # --- Adversarial self-review pass (max quality mode, online only) ---
    if QUALITY_MODE != "fast" and is_connected() and cloud_client and fname not in state.get("reviewed_files", []):
        speak(f"Reviewing {fname} for defects...")
        review_request = (
            f"{build_context(state, current_file=fname)}\n\n"
            f"FILE UNDER REVIEW: `{fname}`\n\n"
            f"CONTENT:\n{code}\n\n"
            f"Reply APPROVED, or reply with the complete corrected file."
        )
        verdict_raw = ask_model(fill_spec(REVIEWER_PROMPT, style), review_request, role="reviewer")
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
    style = (state.get("plan") or {}).get("render_style", "3d")
    raw = ask_model(fill_spec(INTEGRATION_PROMPT, style), request, role="integration")
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

_BLUEPRINT_KEYWORDS = (
    "add", "new ", "system", "mode", "level", "screen", "menu", "enemy", "enemies",
    "boss", "inventory", "weapon", "multiple", "feature", "rework", "redesign",
    "overhaul", "mechanic", "stage", "also", "as well", "plus ", "replace",
)

def _needs_blueprint(user_prompt):
    """True when an adjustment is structural enough to warrant a planning pass first."""
    text = " ".join(str(user_prompt).split())
    body = text[len("update"):] if text.lower().startswith("update") else text
    if len(body.strip()) > 80:
        return True
    low = body.lower()
    return any(kw in low for kw in _BLUEPRINT_KEYWORDS)

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

    style = (state.get("plan") or {}).get("render_style", "3d")

    # --- Blueprint stage: scope structural changes before editing (one cheap call) ---
    blueprint_block = ""
    if _needs_blueprint(user_prompt):
        speak("Blueprinting the change first...")
        bp_request = (
            f"CHANGE REQUEST:\n{user_prompt}\n\n"
            f"RENDER STYLE: {style}\n"
            f"PROJECT PLAN:\n{json.dumps(state.get('plan') or {}, indent=2)}\n\n"
            f"EXISTING FILES: {', '.join(current_files.keys())}\n\n"
            "Produce the JSON change blueprint."
        )
        bp_raw = ask_model(UPDATE_BLUEPRINT_PROMPT, bp_request, role="update_planner")
        try:
            blueprint = extract_json(bp_raw)
            blueprint_block = (
                "CHANGE BLUEPRINT (authoritative scope — edit exactly target_files, "
                "honor preserve, guard risks, add only new_files):\n"
                + json.dumps(blueprint, indent=2) + "\n\n"
            )
        except (ValueError, TypeError):
            logging.warning("Update blueprint was unparseable; proceeding without it.")

    sections = [f"===FILE: {fname}===\n{content}\n===END FILE===" for fname, content in current_files.items()]
    request = (
        f"CHANGE REQUEST:\n{user_prompt}\n\n"
        f"{blueprint_block}"
        f"PROJECT PLAN:\n{json.dumps(state.get('plan') or {}, indent=2)}\n\n"
        f"CURRENT BUILD:\n\n" + "\n\n".join(sections) +
        "\n\nReturn ONLY the changed files via the protocol."
    )
    raw = ask_model(fill_spec(UPDATE_PROMPT, style), request, role="update")
    if not raw:
        speak("The update request failed. Please try again.")
        return

    verdict, blocks = parse_file_blocks(raw)
    if verdict in ("no_changes", "approved"):
        speak("The model judged that no file changes were needed for that request.")
        return
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
            playtest_and_fix(state)
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
    """Routes a user command: build a new game, stack a change, resume, or start fresh.

    The first description builds a game in nexus_workspace/. Every later
    instruction stacks adjustments onto that build; explicit fresh-start
    commands ('new game <idea>') archive the old build and plan a new one.
    """
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
    playtest_and_fix(state)

    state["status"] = "done"
    save_state(state)
    speak("Build finished. Check your preview screen now!")
    logging.info("Build Finished successfully.")

# ==========================================
# 10. COMMAND QUEUE + BACKGROUND BUILD WORKER
# ==========================================
# Builds run on a worker thread so the input prompt on the main thread is NEVER
# blocked while the AI is working. You can type the next adjustment at any time;
# if a build is in progress it is queued and runs right after the current one.
command_queue = queue.Queue()
worker_busy = threading.Event()

def build_worker():
    """Processes queued commands one at a time on a background thread."""
    while True:
        cmd = command_queue.get()
        if cmd is None:               # sentinel: shut the worker down
            command_queue.task_done()
            break
        worker_busy.set()
        try:
            run_nexus_g(cmd)
        except Exception as e:
            logging.error(f"Build failed for command {cmd!r}: {e}")
            speak(f"Something went wrong while working on that: {e}")
        finally:
            worker_busy.clear()
            command_queue.task_done()
        if command_queue.empty():
            print("\n[Nexus-G] ✅ Ready — type your next game idea or adjustment.")

if __name__ == "__main__":
    print(f"[System] 🚀 Nexus-G online | Brain: {MODEL_DISPLAY} | Quality mode: {QUALITY_MODE.upper()}")
    print("[System] Commands: describe a game to build it • any further instruction STACKS changes onto the current build")
    print("[System]           'new game <idea>' starts fresh (previous build is archived) • 'resume' continues an interrupted build • 'exit' quits")
    print("[System] You can type a new request at ANY time — even while it is working; it will be queued and run next.")
    if not CONSOLE_STREAM:
        print("[System] (Console is in calm mode so you can type freely. Set NEXUS_STREAM=1 to watch the AI write live.)")
    threading.Thread(target=serve_game, daemon=True).start()
    threading.Thread(target=build_worker, daemon=True).start()
    time.sleep(0.75)  # let the server print its startup line before the first prompt
    while True:
        try:
            cmd = listen_command()
        except KeyboardInterrupt:
            print("\nExiting...")
            logging.info("System manually exited via KeyboardInterrupt.")
            break

        cmd = cmd.strip()
        if not cmd:
            continue
        if cmd.lower() in ['exit', 'stop', 'quit']:
            logging.info("System exited via user command.")
            break

        was_busy = worker_busy.is_set() or not command_queue.empty()
        command_queue.put(cmd)
        if was_busy:
            speak("Got it — I'm busy right now, so I queued that. It runs as soon as the current task finishes.")
