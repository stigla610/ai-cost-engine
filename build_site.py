import urllib.request
import json
import os
import re
import sys
import html
import shutil
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DIST_DIR = os.path.join(BASE_DIR, "dist")

PROVIDER_NAMES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "deepseek": "DeepSeek",
    "meta-llama": "Meta Llama",
    "meta": "Meta",
    "qwen": "Qwen",
    "mistralai": "Mistral AI",
    "x-ai": "xAI",
    "cohere": "Cohere",
    "microsoft": "Microsoft",
    "nvidia": "NVIDIA",
    "amazon": "Amazon",
    "perplexity": "Perplexity",
    "minimax": "MiniMax",
    "moonshotai": "Moonshot AI",
    "z-ai": "Z.AI (GLM)",
    "thudm": "Zhipu GLM",
    "01-ai": "01.AI",
    "nousresearch": "Nous Research",
    "allenai": "AllenAI",
    "ibm-granite": "IBM Granite",
    "baidu": "Baidu",
    "bytedance": "ByteDance",
    "bytedance-seed": "ByteDance Seed",
    "tencent": "Tencent",
    "xiaomi": "Xiaomi",
    "liquid": "Liquid AI",
    "cognitivecomputations": "Cognitive Computations",
    "sao10k": "Sao10K",
    "thedrummer": "TheDrummer",
    "gryphe": "Gryphe",
    "aion-labs": "Aion Labs",
    "inflection": "Inflection",
    "ai21": "AI21 Labs",
}

def slugify(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def format_price(val):
    if round(val, 2) == round(val, 4):
        return f"${val:,.2f}"
    if round(val, 3) == round(val, 4):
        return f"${val:,.3f}"
    return f"${val:,.4f}"

def clean_provider(model_id):
    if "/" not in model_id:
        return "AI"
    raw = model_id.split("/")[0].lstrip("~").strip().lower()
    if raw in PROVIDER_NAMES:
        return PROVIDER_NAMES[raw]
    return raw.replace("-", " ").title()

def clean_model_name(name, model_id):
    cleaned = name.strip().lstrip("~").strip()
    # Fix awkward double spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned

def is_open_weight(model_id, name):
    mid = model_id.lower().lstrip("~")
    combined = f"{mid} {name.lower()}"

    # Batch, pure image, audio, or realtime endpoints are API-specific
    if any(k in mid for k in [":batch", "-image", "-audio", "-realtime"]):
        return False

    # OpenAI gpt-oss models are explicitly open-weight
    if "gpt-oss" in mid:
        return True

    # Closed proprietary API providers
    if mid.startswith(("openai/", "anthropic/", "x-ai/", "amazon/", "perplexity/", "ai21/", "inflection/", "openrouter/")):
        return False

    # Google: Gemma is open-weight, Gemini is closed proprietary API
    if mid.startswith("google/"):
        return "gemma" in mid

    # Meta: meta-llama is open-weight; meta/muse-spark-open or glimmer-30b are open
    if mid.startswith("meta/"):
        return "open" in mid or "30b" in mid or "llama" in mid
    if mid.startswith("meta-llama/"):
        return True

    # Qwen: models with explicit parameter sizes or open series (qwq, coder, vl, qwen-2.5, qwen3-14b/32b/etc.) are open-weight;
    # commercial API tiers (max, plus, turbo, flash without parameter counts) are hosted APIs
    if mid.startswith("qwen/"):
        if re.search(r"\b\d+(?:\.\d+)?[bt]\b", combined) or "qwq" in mid or "qwen-2.5" in mid or "qwen2.5" in mid:
            return True
        if any(k in mid for k in ["max", "plus", "turbo", "flash"]):
            return False
        return True

    # Mistral: mistral-medium is closed API; Mixtral, Mistral Small/Large/Nemo/Saba, Ministral, Codestral, Devstral, Voxtral, Pixtral are open-weight
    if mid.startswith("mistralai/"):
        if "mistral-medium" in mid:
            return False
        return True

    # DeepSeek: V3, R1, Chat, Reasoner, Coder, Prover are open-weight; synthetic flash-latest/pro-latest aliases are API tiers
    if mid.startswith("deepseek/"):
        if model_id.startswith("~") or "flash" in mid or "v4-pro" in mid or "pro-latest" in mid:
            return False
        return True

    # Other open-weight providers (including Kimi K2, Z.AI GLM, Cohere Command R/A, Microsoft Phi, NVIDIA Nemotron)
    if mid.startswith((
        "microsoft/phi", "microsoft/wizard", "nvidia/", "nousresearch/",
        "cognitivecomputations/", "sao10k/", "thedrummer/", "gryphe/",
        "allenai/", "ibm-granite/", "liquid/", "01-ai/", "moonshotai/kimi",
        "z-ai/glm", "thudm/glm", "cohere/command-r", "cohere/command-a"
    )):
        return True

    # If model explicitly states a parameter count in billions (e.g. 7b, 8b, 14b, 27b, 30b, 32b, 70b, 72b, 120b, 235b, 405b), it is typically open-weight
    if re.search(r"\b\d+(?:\.\d+)?b\b", combined):
        return True

    return False

def get_deploy_commands(m, base_url):
    """Map an open-weight model to Ollama, vLLM, Docker, and TokenBench CLI launch commands."""
    mid = m["id"].lower().lstrip("~")
    params_b = m["params_b"]
    tail = m["id"].split("/")[-1].split(":")[0]

    ollama_tag = tail
    hf_repo = m["id"].split(":")[0].lstrip("~")

    if mid.startswith("meta-llama/"):
        if "llama-4-maverick" in mid:
            ollama_tag, hf_repo = "llama4:maverick", "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
        elif "llama-4-scout" in mid:
            ollama_tag, hf_repo = "llama4:scout", "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        elif "llama-3.3-70b" in mid:
            ollama_tag, hf_repo = "llama3.3:70b", "meta-llama/Llama-3.3-70B-Instruct"
        elif "llama-3.1-70b" in mid:
            ollama_tag, hf_repo = "llama3.1:70b", "meta-llama/Llama-3.1-70B-Instruct"
        elif "llama-3.1-8b" in mid:
            ollama_tag, hf_repo = "llama3.1:8b", "meta-llama/Llama-3.1-8B-Instruct"
        elif "llama-3.2-3b" in mid:
            ollama_tag, hf_repo = "llama3.2:3b", "meta-llama/Llama-3.2-3B-Instruct"
        elif "llama-3.2-1b" in mid:
            ollama_tag, hf_repo = "llama3.2:1b", "meta-llama/Llama-3.2-1B-Instruct"
        else:
            ml = re.search(r"llama-(\d+(?:\.\d+)?)-(\d+b)", mid)
            if ml:
                ver, sz = ml.group(1), ml.group(2)
                ollama_tag = f"llama{ver}:{sz}"
                hf_repo = f"meta-llama/Llama-{ver}-{sz.upper()}-Instruct"
    elif mid.startswith("qwen/"):
        if "coder" in mid and "32b" in mid:
            ollama_tag, hf_repo = "qwen2.5-coder:32b", "Qwen/Qwen2.5-Coder-32B-Instruct"
        elif "coder" in mid and "30b" in mid:
            ollama_tag, hf_repo = "qwen3-coder:30b-a3b", "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        elif "coder" in mid:
            ollama_tag, hf_repo = "qwen3-coder:480b", "Qwen/Qwen3-Coder-480B-A35B-Instruct"
        elif "qwq" in mid:
            ollama_tag, hf_repo = "qwq:32b", "Qwen/QwQ-32B"
        elif "2.5-vl-72b" in mid:
            ollama_tag, hf_repo = "qwen2.5vl:72b", "Qwen/Qwen2.5-VL-72B-Instruct"
        elif "2.5-72b" in mid:
            ollama_tag, hf_repo = "qwen2.5:72b", "Qwen/Qwen2.5-72B-Instruct"
        else:
            mq = re.search(r"qwen(\d+(?:\.\d+)?)-(\d+b)(?:-(a\d+b))?", mid)
            if mq:
                ver, sz, act = mq.group(1), mq.group(2), mq.group(3)
                ollama_tag = f"qwen{ver}:{sz}" + (f"-{act}" if act else "")
                hf_repo = f"Qwen/Qwen{ver}-{sz.upper()}" + (f"-{act.upper()}" if act else "")
    elif mid.startswith("deepseek/"):
        if "r1" in mid:
            sz = "671b" if params_b >= 100 else f"{int(round(params_b))}b"
            ollama_tag, hf_repo = f"deepseek-r1:{sz}", "deepseek-ai/DeepSeek-R1-0528"
        elif "v3" in mid or "chat" in mid:
            ollama_tag, hf_repo = "deepseek-v3", "deepseek-ai/DeepSeek-V3"
    elif mid.startswith("google/gemma"):
        mg = re.search(r"gemma-(\d+)-(\d+b)", mid)
        if mg:
            ollama_tag = f"gemma{mg.group(1)}:{mg.group(2)}"
            hf_repo = f"google/{tail}"
    elif mid.startswith("microsoft/phi"):
        if "mini" in mid or "multimodal" in mid:
            ollama_tag, hf_repo = "phi4-mini", "microsoft/Phi-4-mini-instruct"
        else:
            ollama_tag, hf_repo = "phi4:14b", "microsoft/phi-4"
    elif mid.startswith("mistralai/"):
        if "mistral-large" in mid:
            ollama_tag, hf_repo = "mistral-large:123b", "mistralai/Mistral-Large-Instruct-2411"
        elif "mistral-small" in mid:
            ollama_tag, hf_repo = "mistral-small3.2:24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
        elif "codestral" in mid:
            ollama_tag, hf_repo = "codestral:22b", "mistralai/Codestral-22B-v0.1"
        elif "devstral" in mid:
            ollama_tag, hf_repo = "devstral:24b", "mistralai/Devstral-Small-2505"
        elif "mixtral-8x22b" in mid:
            ollama_tag, hf_repo = "mixtral:8x22b", "mistralai/Mixtral-8x22B-Instruct-v0.1"
        elif "mixtral-8x7b" in mid:
            ollama_tag, hf_repo = "mixtral:8x7b", "mistralai/Mixtral-8x7B-Instruct-v0.1"
        elif "ministral-8b" in mid:
            ollama_tag, hf_repo = "ministral-3:8b", "mistralai/Ministral-8B-Instruct-2410"
        elif "mistral-nemo" in mid:
            ollama_tag, hf_repo = "mistral-nemo:12b", "mistralai/Mistral-Nemo-Instruct-2407"
    elif mid.startswith("openai/gpt-oss"):
        mo = re.search(r"gpt-oss-(\d+b)", mid)
        if mo:
            ollama_tag, hf_repo = f"gpt-oss:{mo.group(1)}", f"openai/gpt-oss-{mo.group(1)}"
    elif mid.startswith("cohere/"):
        if "command-a" in mid:
            ollama_tag, hf_repo = "command-a:111b", "CohereForAI/c4ai-command-a-03-2025"
        elif "command-r-plus" in mid:
            ollama_tag, hf_repo = "command-r-plus:104b", "CohereForAI/c4ai-command-r-plus"
        elif "command-r" in mid:
            ollama_tag, hf_repo = "command-r:35b", "CohereForAI/c4ai-command-r-v01"
    elif mid.startswith("nvidia/"):
        sz = f"{int(round(params_b))}b"
        ollama_tag, hf_repo = f"nemotron:{sz}", m["id"].split(":")[0]
    elif mid.startswith(("z-ai/", "thudm/")):
        ollama_tag = "glm4:9b" if params_b <= 15 else "glm-4.5"
        hf_repo = f"zai-org/{tail}"
    elif mid.startswith("moonshotai/"):
        ollama_tag, hf_repo = "kimi-k2", "moonshotai/Kimi-K2-Instruct"

    q4 = m["vram_int4"]
    tp_gpus = 1 if q4 <= 75 else (2 if q4 <= 155 else (4 if q4 <= 310 else 8))
    max_len = min(int(m["context_length"]), 32768)

    return {
        "ollama_cmd": f"ollama run {ollama_tag}",
        "vllm_cmd": f"vllm serve {hf_repo} --max-model-len {max_len} --tensor-parallel-size {tp_gpus} --gpu-memory-utilization 0.92",
        "docker_cmd": f"docker run --gpus all -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model {hf_repo} --tensor-parallel-size {tp_gpus}",
        "cli_cmd": f"curl -sL {base_url}/tokenbench.py | python3 -",
        "tp_gpus": tp_gpus,
    }

def render_cmd_box(label, cmd):
    esc_cmd = html.escape(cmd)
    return f"""
      <div class="cmd-item">
        <div class="cmd-label">{html.escape(label)}</div>
        <div class="cmd-box">
          <code>{esc_cmd}</code>
          <button type="button" class="btn-ghost cmd-copy" data-cmd="{esc_cmd}" onclick="copyCmd(this)">📋 Copy</button>
        </div>
      </div>"""

def render_deploy_card(m, config):
    if not m["is_open_weight"]:
        return ""
    base_url = config["base_url"].rstrip("/")
    cmds = get_deploy_commands(m, base_url)
    aff_runpod = config["affiliates"]["runpod"]
    aff_vast = config["affiliates"]["vastai"]
    return f"""
  <div class="card" style="margin-top:24px;">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:8px;">
      <h2 style="margin-bottom:0;">🚀 Deploy &amp; Run Commands (Ollama, vLLM &amp; Cloud GPU Docker)</h2>
      <span class="badge badge-green" style="margin-bottom:0;">Recommended TP: {cmds['tp_gpus']}x GPU</span>
    </div>
    <p style="color:var(--muted);font-size:0.9rem;margin-bottom:14px;">Copy and paste to run <strong>{html.escape(m['name'])}</strong> locally or launch an OpenAI-compatible API server on a Cloud GPU pod:</p>
    {render_cmd_box("1. Ollama (Local 4-Bit GGUF)", cmds["ollama_cmd"])}
    {render_cmd_box("2. vLLM (OpenAI-Compatible Production API)", cmds["vllm_cmd"])}
    {render_cmd_box("3. Docker / Cloud GPU Pod (RunPod & Vast.ai)", cmds["docker_cmd"])}
    {render_cmd_box("4. Local GPU Compatibility CLI Check (Detects nvidia-smi / Apple Silicon)", cmds["cli_cmd"])}
    <div style="margin-top:16px;display:flex;gap:12px;flex-wrap:wrap;">
      <a href="{html.escape(aff_runpod['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta">Launch vLLM / Ollama Template on RunPod →</a>
      <a href="{html.escape(aff_vast['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta btn-green">Rent Cheap GPUs on Vast.ai →</a>
    </div>
  </div>"""

def render_cli_card(config):
    base_url = config["base_url"].rstrip("/")
    unix_cmd = f"curl -sL {base_url}/tokenbench.py | python3 -"
    win_cmd = f"irm {base_url}/tokenbench.py | python -"
    return f"""
  <div class="card" style="margin:24px 0;border-color:rgba(16,185,129,0.4);">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:6px;">
      <h2 style="margin-bottom:0;">🖥️ Check Your Local GPU in Terminal (Zero-Install CLI)</h2>
      <span class="badge badge-green" style="margin-bottom:0;">Zero Dependencies · Python 3</span>
    </div>
    <p style="color:var(--muted);font-size:0.9rem;margin-bottom:14px;">Auto-detects your NVIDIA GPU (<code>nvidia-smi</code>) or Apple Silicon Unified Memory and prints which Llama, Qwen, DeepSeek, Gemma, and Mistral models fit—with exact <code>ollama run</code> and <code>vllm serve</code> commands. Supports <code>--vram 24 --ctx 32768 --live</code>.</p>
    <div class="grid-2" style="margin:0;gap:14px;">
      {render_cmd_box("Linux / macOS (Terminal)", unix_cmd)}
      {render_cmd_box("Windows (PowerShell)", win_cmd)}
    </div>
  </div>"""

def get_hardware_url(tier, config):
    url = f"https://www.amazon.com/s?k={tier['query']}"
    tag = (config.get("amazon_associate_tag") or "").strip()
    if tag:
        url += f"&tag={tag}"
    return url

def pick_local_hardware_tier(vram_needed, config):
    tiers = config.get("local_hardware_tiers", [])
    for t in tiers:
        if t["max_vram_gb"] >= vram_needed:
            return t, True
    return (tiers[-1] if tiers else None), False

def render_hardware_buy_card(m, config):
    tiers = config.get("local_hardware_tiers", [])
    if not tiers:
        return ""
    hw, fits_single_box = pick_local_hardware_tier(m["vram_int4"], config)
    if not hw:
        return ""
    hw_url = get_hardware_url(hw, config)
    gpu = config["gpu_tiers"][m["gpu_tier_idx"]]
    aff_cloud = config["affiliates"][gpu["provider_key"]]
    cloud_monthly = round(gpu["hourly_usd"] * 24 * 30, 2)
    elec_monthly = round((hw["watts"] / 1000.0) * 24 * 30 * 0.18, 2)

    if fits_single_box:
        if m["is_open_weight"]:
            headline = f"Buy Local Hardware vs. Rent Cloud GPU for {html.escape(m['name'])}"
            summary = (
                f"Running <strong>{html.escape(m['name'])}</strong> (~{m['vram_int4']} GB in 4-bit) fits locally on "
                f"<strong>{html.escape(hw['name'])}</strong> (~${hw['approx_price_usd']:,} upfront, ~${elec_monthly:.2f}/mo electricity at $0.18/kWh). "
                f"Or rent a <strong>{html.escape(gpu['name'])}</strong> on-demand for <strong>${gpu['hourly_usd']:.2f}/hr</strong> with zero hardware CapEx."
            )
        else:
            headline = f"Buy Local Hardware vs. Rent Cloud GPU (Open-Weight Alternative to {html.escape(m['name'])})"
            summary = (
                f"While {html.escape(m['name'])} is a proprietary API, running an <strong>open-weight ~{m['params_b']:g}B equivalent</strong> (~{m['vram_int4']} GB in 4-bit) fits locally on "
                f"<strong>{html.escape(hw['name'])}</strong> (~${hw['approx_price_usd']:,} upfront, ~${elec_monthly:.2f}/mo electricity at $0.18/kWh). "
                f"Or rent a <strong>{html.escape(gpu['name'])}</strong> on-demand for <strong>${gpu['hourly_usd']:.2f}/hr</strong> with zero hardware CapEx."
            )
        badge_txt = f"Fits Locally on {hw['max_vram_gb']}GB Tier"
    else:
        rtx24 = tiers[1] if len(tiers) > 1 else hw
        hw_url = get_hardware_url(rtx24, config)
        hw = rtx24
        elec_monthly = round((hw["watts"] / 1000.0) * 24 * 30 * 0.18, 2)
        headline = f"Cluster Hardware vs. Cloud GPU Rental (~{m['vram_int4']} GB Q4)"
        summary = (
            f"At <strong>{m['vram_int4']} GB</strong> (4-bit), this model exceeds single-GPU consumer hardware and requires a multi-GPU node or Cloud cluster "
            f"(<strong>{html.escape(gpu['name'])}</strong> at <strong>${gpu['hourly_usd']:.2f}/hr</strong>). "
            f"Want to run smaller 7B–32B distilled models locally instead? Check out a 24GB <strong>{html.escape(hw['name'])}</strong> (~${hw['approx_price_usd']:,})."
        )
        badge_txt = "Multi-GPU / Cloud Cluster Recommended"

    payback_months = round(hw["approx_price_usd"] / max(cloud_monthly - elec_monthly, 1.0), 1)
    payback_str = f"~{payback_months} months (24/7 use)" if fits_single_box and cloud_monthly > elec_monthly else "Cloud pay-as-you-go wins for burst traffic"

    return f"""
  <div class="card" style="margin-top:24px;">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:8px;">
      <h2 style="margin-bottom:0;">🛒 {headline}</h2>
      <span class="badge" style="margin-bottom:0;">{badge_txt}</span>
    </div>
    <p style="color:var(--muted);font-size:0.92rem;margin-bottom:14px;">{summary}</p>
    <div class="grid-2" style="margin:10px 0 0;">
      <div class="card" style="background:#0b0f19;padding:16px;">
        <span class="badge badge-green">Option A: Own Local Hardware</span>
        <h3 style="font-size:1.05rem;">{html.escape(hw['name'])}</h3>
        <div class="stat-row"><span>Typical Retail / Refurb Price</span><strong>~${hw['approx_price_usd']:,}</strong></div>
        <div class="stat-row"><span>24/7 Power Draw ({hw['watts']}W @ $0.18/kWh)</span><strong>~${elec_monthly:.2f}/mo</strong></div>
        <div class="stat-row"><span>Hardware Payback vs 24/7 Cloud</span><strong>{payback_str}</strong></div>
        <div style="margin-top:14px;">
          <a href="{html.escape(hw_url)}" target="_blank" rel="nofollow sponsored" data-track-cta="amazon_{hw['id']}" class="btn-cta btn-green">Check GPU Price on Amazon →</a>
        </div>
      </div>
      <div class="card" style="background:#0b0f19;padding:16px;">
        <span class="badge">Option B: On-Demand Cloud GPU</span>
        <h3 style="font-size:1.05rem;">{html.escape(gpu['name'])}</h3>
        <div class="stat-row"><span>Upfront Hardware Cost</span><strong>$0</strong></div>
        <div class="stat-row"><span>Hourly On-Demand Rate</span><strong>${gpu['hourly_usd']:.2f}/hr (${cloud_monthly:,.0f}/mo 24/7)</strong></div>
        <div class="stat-row"><span>Best For</span><strong>Burst workloads, zero maintenance</strong></div>
        <div style="margin-top:14px;">
          <a href="{html.escape(aff_cloud['url'])}" target="_blank" rel="nofollow sponsored" data-track-cta="cloud_{gpu['id']}" class="btn-cta">{html.escape(aff_cloud['cta_text'])} →</a>
        </div>
      </div>
    </div>
  </div>"""

def render_hardware_hub_section(config):
    tiers = config.get("local_hardware_tiers", [])
    if not tiers:
        return ""
    cards = ""
    for t in tiers:
        url = get_hardware_url(t, config)
        elec = round((t["watts"] / 1000.0) * 24 * 30 * 0.18, 2)
        cards += f"""
        <div class="card">
          <span class="badge badge-green">Up to {t['max_vram_gb']} GB VRAM · {t['watts']}W</span>
          <h3>{html.escape(t['name'])}</h3>
          <p style="color:var(--muted);font-size:0.88rem;margin:8px 0 12px;">{html.escape(t['best_for'])}</p>
          <div class="stat-row"><span>Est. Hardware Price</span><strong>~${t['approx_price_usd']:,}</strong></div>
          <div class="stat-row"><span>24/7 Electricity ($0.18/kWh)</span><strong>~${elec:.2f}/mo</strong></div>
          <div style="margin-top:14px;">
            <a href="{html.escape(url)}" target="_blank" rel="nofollow sponsored" data-track-cta="amazon_{t['id']}" class="btn-cta btn-green">Check Current Price on Amazon →</a>
          </div>
        </div>"""
    return f"""
  <h2 style="color:#fff;margin-top:36px;margin-bottom:8px;">🛒 Local LLM Hardware Buying Guide (Best GPUs &amp; Unified Memory Rigs)</h2>
  <p style="color:var(--muted);margin-bottom:16px;">Compare local GPU &amp; Apple Silicon tiers by usable VRAM, 24/7 electricity draw, and model capacity before buying hardware.</p>
  <div class="grid-2">
    {cards}
  </div>"""

def estimate_params_b(model_id, name):
    combined = f"{model_id} {name}".lower()

    # 1. Check MoE NxMB pattern first (e.g. 8x7b -> 47B, 8x22b -> 141B)
    m_moe = re.search(r"\b(\d+)x(\d+(?:\.\d+)?)b\b", combined)
    if m_moe:
        n_exp = float(m_moe.group(1))
        b_exp = float(m_moe.group(2))
        return round(n_exp * b_exp * 0.8, 1)

    # 2. Check Trillion parameter pattern (e.g. 2.4t -> 2400B)
    m_tril = re.search(r"\b(\d+(?:\.\d+)?)t\b", combined)
    if m_tril:
        val_t = float(m_tril.group(1))
        if 0.5 <= val_t <= 10:
            return round(val_t * 1000.0, 1)

    # 3. Check total Billion parameter pattern (ignoring active parameter tags like a3b, a17b, a22b)
    m_bil = re.search(r"(?<![a-z0-9])(\d+(?:\.\d+)?)b\b", combined)
    if m_bil:
        val = float(m_bil.group(1))
        if 0.5 <= val <= 2000:
            return val

    # 4. Known specific model families without NNb in their display name
    if "deepseek" in combined:
        if re.search(r"\b(flash|lite|mini|nano|small)\b", combined):
            return 14.0
        if any(k in combined for k in ["v3", "r1", "chat", "reasoner", "prover", "v4"]):
            return 671.0

    if "phi-4" in combined or "phi-3" in combined:
        if re.search(r"\b(mini|small|multimodal)\b", combined):
            return 3.8
        return 14.0
    if "llama-4-maverick" in combined:
        return 400.0
    if "llama-4-scout" in combined:
        return 109.0
    if "mistral-large" in combined or "pixtral-large" in combined:
        return 123.0
    if "mistral-medium" in combined:
        return 70.0
    if any(k in combined for k in ["mistral-small", "devstral", "voxtral-small"]):
        return 24.0
    if "codestral" in combined:
        return 22.0
    if "mistral-nemo" in combined or "mistral-saba" in combined:
        return 12.0
    if "command-a" in combined:
        return 111.0
    if "command-r-plus" in combined:
        return 104.0
    if "command-r" in combined:
        return 35.0
    if "kimi-k2" in combined:
        return 1000.0
    if "glm-4.5-air" in combined:
        return 106.0
    if "glm-4.5" in combined or "glm-5" in combined:
        return 355.0

    # 5. Check Nano / Lite / Micro / Mini / Flash / Haiku / Small with word boundaries BEFORE generic provider names
    if re.search(r"\b(nano|micro|lite)\b", combined):
        return 4.0
    if re.search(r"\b(mini|flash|haiku|small)\b", combined):
        return 8.0

    # 6. Flagship / Large / Pro tiers (for open-weight equivalent comparison)
    if re.search(r"\b(opus|ultra|pro|max|large|gpt-5|gpt-4|claude|o1|o3|grok)\b", combined):
        return 70.0

    return 32.0

def fetch_models():
    url = "https://openrouter.ai/api/v1/models"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "TokenBench-AutoBuilder/2.1"}
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    models = []
    seen_slugs = set()
    for item in raw.get("data", []):
        model_id = item.get("id", "")
        raw_name = item.get("name", model_id)
        name = clean_model_name(raw_name, model_id)
        pricing = item.get("pricing") or {}
        try:
            prompt_per_tok = float(pricing.get("prompt", 0) or 0)
            completion_per_tok = float(pricing.get("completion", 0) or 0)
        except (ValueError, TypeError):
            continue

        input_1m = round(prompt_per_tok * 1_000_000, 4)
        output_1m = round(completion_per_tok * 1_000_000, 4)
        if input_1m < 0 or output_1m < 0:
            continue

        context_length = int(item.get("context_length") or 4096)
        slug = slugify(model_id)
        if not slug or slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        provider = clean_provider(model_id)
        raw_prov = model_id.split("/")[0].lstrip("~").lower() if "/" in model_id else "other"
        params_b = estimate_params_b(model_id, name)
        open_weight = is_open_weight(model_id, name)
        is_secondary = any(k in model_id.lower() for k in [":batch", ":free", "-image", "-audio", "-realtime", "guard", "safeguard"]) or model_id.startswith("~")

        if params_b >= 100:
            gpu_tier_idx = 3
        elif params_b >= 65:
            gpu_tier_idx = 2
        elif params_b >= 25:
            gpu_tier_idx = 1
        else:
            gpu_tier_idx = 0

        models.append({
            "id": model_id,
            "name": name,
            "slug": slug,
            "provider": provider,
            "raw_provider": raw_prov,
            "is_open_weight": open_weight,
            "is_secondary": is_secondary,
            "params_b": params_b,
            "vram_fp16": round(params_b * 2.2, 1),
            "vram_int8": round(params_b * 1.15, 1),
            "vram_int4": round(params_b * 0.65, 1),
            "input_1m": input_1m,
            "output_1m": output_1m,
            "blended_1m": round((input_1m * 0.7) + (output_1m * 0.3), 4),
            "context_length": context_length,
            "description": (item.get("description") or f"Live API token pricing, VRAM requirements, and Cloud GPU break-even analysis for {name}.")[:280],
            "gpu_tier_idx": gpu_tier_idx,
        })

    paid_models = [m for m in models if m["input_1m"] > 0 or m["output_1m"] > 0]

    # Interleave across major families so the top models represent OpenAI, Anthropic, Google, DeepSeek, Llama, Qwen, Mistral, xAI, etc.
    priority_families = [
        "openai", "anthropic", "google", "deepseek",
        "meta-llama", "qwen", "mistralai", "x-ai",
        "nvidia", "cohere", "z-ai", "moonshotai", "meta", "microsoft"
    ]

    flagship_exact = {
        "openai/gpt-5": 0, "openai/gpt-4o": 1, "openai/gpt-4.1": 2, "openai/gpt-4o-mini": 3,
        "openai/o3": 4, "openai/gpt-oss-120b": 5, "openai/gpt-5-mini": 6, "openai/gpt-4": 7,
        "anthropic/claude-sonnet-4.5": 0, "anthropic/claude-opus-4.5": 1, "anthropic/claude-haiku-4.5": 2,
        "anthropic/claude-sonnet-4.6": 3, "anthropic/claude-opus-4.6": 4, "anthropic/claude-sonnet-5": 5,
        "google/gemini-2.5-pro": 0, "google/gemini-2.5-flash": 1, "google/gemma-4-31b-it": 2,
        "google/gemma-3-27b-it": 3, "google/gemma-3-12b-it": 4,
        "deepseek/deepseek-chat": 0, "deepseek/deepseek-r1-0528": 1, "deepseek/deepseek-v3.2": 2,
        "deepseek/deepseek-chat-v3.1": 3,
        "meta-llama/llama-4-maverick": 0, "meta-llama/llama-4-scout": 1,
        "meta-llama/llama-3.3-70b-instruct": 2, "meta-llama/llama-3.1-70b-instruct": 3,
        "meta-llama/llama-3.1-8b-instruct": 4,
        "qwen/qwen3-32b": 0, "qwen/qwen-2.5-72b-instruct": 1, "qwen/qwen3-235b-a22b-2507": 2,
        "qwen/qwen-2.5-coder-32b-instruct": 3, "qwen/qwen3-14b": 4,
        "mistralai/mistral-large": 0, "mistralai/mistral-small-3.2-24b-instruct": 1,
        "mistralai/codestral-2508": 2, "mistralai/mixtral-8x22b-instruct": 3,
    }

    def within_family_key(m):
        mid = m["id"].lower()
        dated = bool(re.search(r"-\d{4}-\d{2}-\d{2}", mid))
        old_gen = "gpt-3.5" in mid or "0613" in mid
        flagship_rank = flagship_exact.get(mid, 100)
        return (
            1 if m["is_secondary"] else 0,
            flagship_rank,
            1 if old_gen else 0,
            1 if dated else 0,
            m["name"]
        )

    buckets = {fam: [] for fam in priority_families}
    other_bucket = []
    for m in paid_models:
        rp = m["raw_provider"]
        if rp in buckets:
            buckets[rp].append(m)
        else:
            other_bucket.append(m)

    for fam in buckets:
        buckets[fam].sort(key=within_family_key)
    other_bucket.sort(key=within_family_key)

    # Separate primary vs secondary (:batch / ~ alias) so primary models always come first
    interleaved_primary = []
    interleaved_secondary = []

    primary_buckets = [[m for m in buckets[f] if not m["is_secondary"]] for f in priority_families]
    primary_buckets.append([m for m in other_bucket if not m["is_secondary"]])

    max_len = max((len(b) for b in primary_buckets), default=0)
    for idx in range(max_len):
        for b in primary_buckets:
            if idx < len(b):
                interleaved_primary.append(b[idx])

    for f in priority_families:
        interleaved_secondary.extend([m for m in buckets[f] if m["is_secondary"]])
    interleaved_secondary.extend([m for m in other_bucket if m["is_secondary"]])

    return interleaved_primary + interleaved_secondary

def shared_head(title, description, canonical_url, config, json_ld=None):
    base_url = config["base_url"].rstrip("/")
    schema_tag = ""
    if json_ld:
        schema_tag = f'<script type="application/ld+json">{json.dumps(json_ld)}</script>'
    gsc = config.get("google_site_verification", "").strip()
    gsc_tag = f'<meta name="google-site-verification" content="{html.escape(gsc)}" />' if gsc else ""

    analytics = config.get("analytics") or {}
    analytics_tags = []
    gc_code = (analytics.get("goatcounter_code") or "").strip()
    if gc_code:
        analytics_tags.append(
            f'<script data-goatcounter="https://{html.escape(gc_code)}.goatcounter.com/count" async src="//gc.zgo.at/count.js"></script>'
        )
    cf_tok = (analytics.get("cloudflare_token") or "").strip()
    if cf_tok:
        analytics_tags.append(
            f'<script defer src="https://static.cloudflareinsights.com/beacon.min.js" data-cf-beacon=\'{json.dumps({"token": cf_tok})}\'></script>'
        )
    ga4_id = (analytics.get("ga4_measurement_id") or "").strip()
    if ga4_id:
        analytics_tags.append(
            f'<script async src="https://www.googletagmanager.com/gtag/js?id={html.escape(ga4_id)}"></script>'
            f'<script>window.dataLayer=window.dataLayer||[];function gtag(){{dataLayer.push(arguments);}}gtag("js",new Date());gtag("config","{html.escape(ga4_id)}");</script>'
        )
    analytics_html = "\n  ".join(analytics_tags)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html.escape(title)}</title>
  <meta name="description" content="{html.escape(description)}" />
  <link rel="canonical" href="{html.escape(canonical_url)}" />
  <link rel="icon" type="image/svg+xml" href="{base_url}/favicon.svg" />
  <meta property="og:site_name" content="{html.escape(config['site_name'])}" />
  <meta property="og:title" content="{html.escape(title)}" />
  <meta property="og:description" content="{html.escape(description)}" />
  <meta property="og:type" content="website" />
  <meta property="og:url" content="{html.escape(canonical_url)}" />
  <meta property="og:image" content="{base_url}/og-card.svg" />
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="{html.escape(title)}" />
  <meta name="twitter:description" content="{html.escape(description)}" />
  <meta name="twitter:image" content="{base_url}/og-card.svg" />
  {gsc_tag}
  {schema_tag}
  {analytics_html}
  <style>
    :root {{
      --bg: #0b0f19;
      --card: #131b2e;
      --card-hover: #1a2540;
      --border: #23304f;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-green: #10b981;
      --accent-purple: #818cf8;
      --warn: #f59e0b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{
      max-width: 100%;
      overflow-x: hidden;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .container {{ max-width: 1180px; margin: 0 auto; padding: 0 20px; width: 100%; }}
    header {{
      border-bottom: 1px solid var(--border);
      background: rgba(11, 15, 25, 0.92);
      position: sticky;
      top: 0;
      z-index: 50;
      backdrop-filter: blur(8px);
    }}
    .nav {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 14px 0;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .brand {{
      font-size: 1.25rem;
      font-weight: 800;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .brand span {{ color: var(--accent); }}
    .nav-links {{ display: flex; gap: 16px; font-size: 0.92rem; align-items: center; flex-wrap: wrap; }}
    .btn-cta {{
      background: linear-gradient(135deg, #0284c7, #4f46e5);
      color: #fff !important;
      padding: 9px 16px;
      border-radius: 8px;
      font-weight: 600;
      display: inline-block;
      transition: transform 0.15s ease, opacity 0.15s ease;
      text-decoration: none !important;
      white-space: nowrap;
      border: none;
      cursor: pointer;
    }}
    .btn-cta:hover {{ transform: translateY(-1px); opacity: 0.95; }}
    .btn-green {{
      background: linear-gradient(135deg, #059669, #10b981);
    }}
    .btn-ghost {{
      background: rgba(56, 189, 248, 0.08);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 7px 12px;
      border-radius: 8px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .btn-ghost:hover {{
      border-color: var(--accent);
      color: #fff;
      background: rgba(56, 189, 248, 0.18);
    }}
    .share-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px dashed var(--border);
    }}
    .hero {{ padding: 38px 0 24px; }}
    .hero h1 {{ font-size: 2.15rem; line-height: 1.25; margin-bottom: 12px; color: #fff; word-break: break-word; }}
    .hero p {{ color: var(--muted); font-size: 1.05rem; max-width: 840px; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 340px), 1fr)); gap: 20px; margin: 24px 0; }}
    .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr)); gap: 18px; margin: 24px 0; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 22px;
      min-width: 0;
    }}
    .card h2, .card h3 {{ color: #fff; margin-bottom: 10px; }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.76rem;
      font-weight: 600;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      margin-bottom: 10px;
    }}
    .badge-green {{
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent-green);
    }}
    .badge-warn {{
      background: rgba(245, 158, 11, 0.15);
      color: var(--warn);
    }}
    .stat-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 10px 0;
      border-bottom: 1px solid var(--border);
      font-size: 0.94rem;
      flex-wrap: wrap;
    }}
    .stat-row:last-child {{ border-bottom: none; }}
    .price-big {{ font-size: 1.75rem; font-weight: 800; color: var(--accent-green); }}
    .slider-group {{ margin-bottom: 16px; }}
    .slider-group label {{ display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 6px; font-size: 0.92rem; gap: 8px; flex-wrap: wrap; }}
    input[type="range"] {{ width: 100%; accent-color: var(--accent); cursor: pointer; }}
    input[type="text"], input[type="number"], select {{
      width: 100%;
      padding: 10px 14px;
      background: #0b0f19;
      border: 1px solid var(--border);
      color: #fff;
      border-radius: 8px;
      font-size: 0.95rem;
    }}
    .table-wrap {{
      width: 100%;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
      margin-top: 12px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.91rem; min-width: 640px; }}
    th, td {{ padding: 12px 10px; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 600; background: rgba(255,255,255,0.02); white-space: nowrap; }}
    th.sortable {{ cursor: pointer; user-select: none; transition: color 0.15s ease; }}
    th.sortable:hover {{ color: var(--accent); }}
    tr:hover td {{ background: rgba(255,255,255,0.02); }}
    .filter-bar {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }}
    .filter-pill {{
      background: #0b0f19;
      border: 1px solid var(--border);
      color: var(--muted);
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .filter-pill:hover, .filter-pill.active {{
      border-color: var(--accent);
      color: #fff;
      background: rgba(56, 189, 248, 0.15);
    }}
    .affiliate-box {{
      background: linear-gradient(135deg, rgba(14, 165, 233, 0.12), rgba(79, 70, 229, 0.12));
      border: 1px solid var(--accent);
      border-radius: 12px;
      padding: 20px;
      margin: 24px 0;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }}
    .notice-box {{
      background: rgba(245, 158, 11, 0.1);
      border: 1px solid rgba(245, 158, 11, 0.4);
      border-radius: 10px;
      padding: 14px 18px;
      margin-bottom: 20px;
      font-size: 0.92rem;
      color: #fde68a;
    }}
    .cmd-item {{ margin-bottom: 12px; }}
    .cmd-item:last-child {{ margin-bottom: 0; }}
    .cmd-label {{ font-size: 0.8rem; font-weight: 600; color: var(--muted); margin-bottom: 4px; }}
    .cmd-box {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      background: #070a12;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 9px 12px;
    }}
    .cmd-box code {{
      color: #a5f3fc;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.84rem;
      overflow-x: auto;
      white-space: nowrap;
    }}
    .cmd-copy {{ flex-shrink: 0; padding: 5px 10px; font-size: 0.78rem; }}
    footer {{
      margin-top: 60px;
      border-top: 1px solid var(--border);
      padding: 32px 0;
      color: var(--muted);
      font-size: 0.88rem;
    }}
    @media (max-width: 768px) {{
      .container {{ padding: 0 14px; }}
      .hero {{ padding: 24px 0 16px; }}
      .hero h1 {{ font-size: 1.55rem; }}
      .hero p {{ font-size: 0.95rem; }}
      .nav {{ flex-direction: column; align-items: flex-start; gap: 10px; }}
      .nav-links {{ width: 100%; gap: 8px 12px; font-size: 0.84rem; }}
      .btn-cta {{ padding: 8px 12px; font-size: 0.84rem; white-space: normal; text-align: center; }}
      .card {{ padding: 16px; }}
      .grid-2, .grid-3 {{ grid-template-columns: 1fr; gap: 14px; }}
      table {{ min-width: 540px; font-size: 0.84rem; }}
      th, td {{ padding: 9px 8px; }}
    }}
  </style>
</head>
"""

def shared_nav(prefix="", config=None):
    runpod_url = config["affiliates"]["runpod"]["url"] if config else "#"
    return f"""<body>
<header>
  <div class="container nav">
    <a href="{prefix}index.html" class="brand">⚡ Token<span>Bench</span></a>
    <nav class="nav-links">
      <a href="{prefix}index.html">API Cost Calculator</a>
      <a href="{prefix}vram-electricity-calculator.html">VRAM &amp; Electricity Calculator</a>
      <a href="{prefix}gpu-cloud-pricing.html">Cloud GPU Rates</a>
      <a href="{prefix}benchmarks.html">2026 Benchmarks &amp; Guides</a>
      <a href="{prefix}index.html#comparisons">Model Comparisons</a>
      <a href="{html.escape(runpod_url)}" target="_blank" rel="nofollow sponsored" data-track-cta="nav_runpod" class="btn-cta">Rent Cloud GPUs (From $0.20/hr)</a>
    </nav>
  </div>
</header>
"""

def shared_footer(config, updated_str, prefix=""):
    sponsor_url = (config.get("sponsor_inquiry_url") or "https://github.com/stigla610/ai-cost-engine/issues").strip()
    return f"""<footer>
  <div class="container">
    <p><strong>{html.escape(config['site_name'])}</strong> — Automatically updated daily with live API pricing, VRAM calculators &amp; cloud GPU benchmarks. Last auto-sync: {updated_str}.</p>
    <p style="margin-top:8px;">Links to cloud GPU &amp; hardware providers include partner referral links at zero extra cost to you, keeping our calculators 100% free.</p>
    <p style="margin-top:8px;">
      <a href="{prefix}index.html">All AI Models</a> ·
      <a href="{prefix}vram-electricity-calculator.html">VRAM &amp; Local GPU Electricity Calculator</a> ·
      <a href="{prefix}gpu-cloud-pricing.html">Cloud GPU Price Directory</a> ·
      <a href="{prefix}benchmarks.html">2026 Benchmarks &amp; Guides</a> ·
      <a href="{prefix}tokenbench.py">Terminal GPU Checker CLI</a> ·
      <a href="{html.escape(sponsor_url)}" target="_blank" rel="noopener">📣 Sponsor a GPU Tier / Advertise</a> ·
      <a href="{prefix}sitemap.xml">XML Sitemap</a>
    </p>
  </div>
</footer>
<script>
  function copyCmd(btn) {{
    const cmd = btn.getAttribute('data-cmd') || '';
    if (!cmd) return;
    navigator.clipboard.writeText(cmd);
    const orig = btn.textContent;
    btn.textContent = '✅ Copied!';
    setTimeout(() => btn.textContent = orig, 1800);
  }}
  (function() {{
    document.addEventListener('click', function(e) {{
      const link = e.target && e.target.closest ? e.target.closest('a[rel*="sponsored"], a[data-track-cta]') : null;
      if (!link) return;
      const ctaId = link.getAttribute('data-track-cta') || link.hostname || 'affiliate';
      try {{
        const store = JSON.parse(localStorage.getItem('tb_cta_clicks') || '{{}}');
        store[ctaId] = (store[ctaId] || 0) + 1;
        localStorage.setItem('tb_cta_clicks', JSON.stringify(store));
      }} catch (_) {{}}
      if (window.goatcounter && typeof window.goatcounter.count === 'function') {{
        window.goatcounter.count({{ path: 'cta/' + ctaId, title: 'CTA: ' + ctaId, event: true }});
      }}
      if (typeof window.gtag === 'function') {{
        window.gtag('event', 'affiliate_click', {{ event_category: 'outbound', event_label: ctaId }});
      }}
    }});
  }})();
</script>
</body>
</html>
"""

def build_model_page(m, all_models, config, updated_str):
    base_url = config["base_url"].rstrip("/")
    canonical = f"{base_url}/models/{m['slug']}.html"
    gpu = config["gpu_tiers"][m["gpu_tier_idx"]]
    aff = config["affiliates"][gpu["provider_key"]]
    alt_key = "vastai" if gpu["provider_key"] != "vastai" else "runpod"
    alt_aff = config["affiliates"][alt_key]

    daily_gpu_cost = round(gpu["hourly_usd"] * 24, 2)
    blended = max(m["blended_1m"], 0.05)
    breakeven_m_tokens_day = round(daily_gpu_cost / blended, 1)
    in_max = min(500_000_000, max(50_000_000, int(((breakeven_m_tokens_day * 1.4) // 10 + 1) * 10_000_000)))
    out_max = max(20_000_000, int(in_max * 0.4))

    in_fmt = format_price(m["input_1m"])
    out_fmt = format_price(m["output_1m"])

    if m["is_open_weight"]:
        title = f"{m['name']} API Cost Calculator & Self-Host GPU Break-Even (2026)"
        desc = f"Calculate exact {m['name']} token costs ({in_fmt}/1M input, {out_fmt}/1M output), local VRAM requirements ({m['vram_int4']}GB Q4), and Cloud GPU self-hosting break-even."
        faq_q2 = f"When is it cheaper to self-host {m['name']} on a Cloud GPU?"
        faq_a2 = f"Renting a dedicated {gpu['name']} costs roughly ${gpu['hourly_usd']}/hour (${daily_gpu_cost}/day). If your workload exceeds {breakeven_m_tokens_day}M blended tokens per day, self-hosting {m['name']} on {aff['name']} can significantly reduce your monthly inference bill."
        weights_badge = '<span class="badge badge-green">Open-Weight / Self-Hostable</span>'
        vram_label = "VRAM Needed (4-Bit Q4 / FP16)"
        vram_link_text = f"{m['vram_int4']} GB / {m['vram_fp16']} GB →"
        gpu_row_label = f"24/7 Dedicated {html.escape(gpu['name'])} Self-Host Cost"
        verdict_win_prefix = f"Self-Hosting {html.escape(m['name'])} Wins"
    else:
        title = f"{m['name']} API Cost Calculator & Open-Weight GPU Break-Even (2026)"
        desc = f"Calculate exact {m['name']} API token costs ({in_fmt}/1M input, {out_fmt}/1M output) and compare against self-hosting an open-weight GPU alternative."
        faq_q2 = f"Can you self-host {m['name']} or switch to a Cloud GPU?"
        faq_a2 = f"{m['name']} is a proprietary API model, so its weights are not downloadable. However, if your API spend exceeds ${round(daily_gpu_cost * 30):,}/month (~{breakeven_m_tokens_day}M tokens/day), running an equivalent open-weight model (Llama, Qwen, or DeepSeek) on a dedicated {gpu['name']} (${gpu['hourly_usd']}/hr) on {aff['name']} can cut costs substantially."
        weights_badge = '<span class="badge badge-warn">Proprietary Cloud API</span>'
        vram_label = f"Open-Weight Eq. VRAM (~{m['params_b']:g}B Class)"
        vram_link_text = f"~{m['vram_int4']} GB Q4 Equivalent Guide →"
        gpu_row_label = f"24/7 {html.escape(gpu['name'])} (Open-Weight Alternative)"
        verdict_win_prefix = "Switching to Open-Weight GPU Wins"

    json_ld = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": f"How much does the {m['name']} API cost per 1M tokens?",
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": f"{m['name']} costs {in_fmt} per 1M input tokens and {out_fmt} per 1M output tokens, with a context window of {m['context_length']:,} tokens."
                }
            },
            {
                "@type": "Question",
                "name": faq_q2,
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": faq_a2
                }
            }
        ]
    }

    related = [x for x in all_models if x["slug"] != m["slug"] and not x["is_secondary"]][:8]
    related_links = "".join(
        f'<li><a href="{x["slug"]}.html">{html.escape(x["name"])} Cost Calculator ({format_price(x["input_1m"])}/1M in)</a></li>'
        for x in related
    )

    page = shared_head(title, desc, canonical, config, json_ld)
    page += shared_nav(prefix="../", config=config)
    page += f"""
<main class="container">
  <section class="hero">
    <div style="display:flex;gap:8px;flex-wrap:wrap;">
      <span class="badge">{html.escape(m['provider'])} Model Pricing &amp; Calculator</span>
      {weights_badge}
    </div>
    <h1>{html.escape(m['name'])} API Cost Calculator</h1>
    <p>{html.escape(m['description'])}</p>
  </section>

  <div class="grid-2">
    <div class="card">
      <h2>Interactive Monthly Spend Estimator</h2>
      <p style="color:var(--muted);font-size:0.9rem;margin-bottom:16px;">Adjust sliders to calculate your daily and monthly bill for <strong>{html.escape(m['name'])}</strong>.</p>

      <div class="slider-group">
        <label><span>Input Tokens per Day</span> <span id="inVal">2,000,000</span></label>
        <input type="range" id="inSlider" min="100000" max="{in_max}" step="100000" value="2000000" oninput="recalc()" />
      </div>

      <div class="slider-group">
        <label><span>Output Tokens per Day</span> <span id="outVal">500,000</span></label>
        <input type="range" id="outSlider" min="50000" max="{out_max}" step="50000" value="500000" oninput="recalc()" />
      </div>

      <div id="mixInfo" style="color:var(--muted);font-size:0.84rem;margin:-8px 0 14px;">Mix: 80% input / 20% output · Blended API rate: —</div>

      <div class="slider-group">
        <label><span>GPU Utilization (Active / Peak Headroom)</span> <span id="utilVal">100% (steady 24/7)</span></label>
        <input type="range" id="utilSlider" min="20" max="100" step="5" value="100" oninput="recalc()" />
      </div>

      <div class="slider-group">
        <label><span>Monthly Ops &amp; Infra Overhead</span> <span id="opsVal">$0/mo</span></label>
        <input type="range" id="opsSlider" min="0" max="1500" step="50" value="0" oninput="recalc()" />
      </div>

      <div class="stat-row">
        <span>Estimated Daily API Cost</span>
        <strong id="dailyCost">$0.00</strong>
      </div>
      <div class="stat-row">
        <span>Estimated Monthly API Cost (30 days)</span>
        <strong id="monthlyCost" class="price-big">$0.00</strong>
      </div>
      <div class="stat-row">
        <span>{gpu_row_label}<small id="gpuBreakdown" style="display:block;color:var(--muted);font-size:0.8rem;font-weight:400;"></small></span>
        <strong id="gpuMonthlyCost">${round(daily_gpu_cost * 30, 2):,.2f}/mo</strong>
      </div>
      <div class="stat-row">
        <span>Live Break-Even at Your Mix &amp; Util</span>
        <strong id="liveBreakeven">—</strong>
      </div>
      <div id="verdictBox" style="margin-top:16px;padding:12px;border-radius:8px;background:rgba(16,185,129,0.12);border:1px solid var(--accent-green);font-size:0.92rem;"></div>
      <div class="share-row">
        <button type="button" class="btn-ghost" onclick="copyCalcLink(this)">🔗 Copy Shareable Link</button>
        <button type="button" class="btn-ghost" onclick="copyMarkdown(this)">📋 Copy Markdown (Reddit / GitHub)</button>
      </div>
    </div>

    <div class="card">
      <h2>Official Token Pricing &amp; Specs</h2>
      <div class="stat-row">
        <span>Model ID</span>
        <code>{html.escape(m['id'])}</code>
      </div>
      <div class="stat-row">
        <span>Provider</span>
        <strong>{html.escape(m['provider'])}</strong>
      </div>
      <div class="stat-row">
        <span>Weight Availability</span>
        <strong>{"Open-Weight (Downloadable GGUF / Safetensors)" if m["is_open_weight"] else "Proprietary Cloud API (Closed Weights)"}</strong>
      </div>
      <div class="stat-row">
        <span>Input Price (per 1M tokens)</span>
        <strong>{in_fmt}</strong>
      </div>
      <div class="stat-row">
        <span>Output Price (per 1M tokens)</span>
        <strong>{out_fmt}</strong>
      </div>
      <div class="stat-row">
        <span>Max Context Window</span>
        <strong>{m['context_length']:,} tokens</strong>
      </div>
      <div class="stat-row">
        <span>{vram_label}</span>
        <strong><a href="../vram/{m['slug']}-vram-requirements.html">{vram_link_text}</a></strong>
      </div>
      <div class="stat-row">
        <span>API vs. GPU Daily Break-Even</span>
        <strong>~{breakeven_m_tokens_day}M tokens/day</strong>
      </div>

      <div style="margin-top:20px;display:flex;gap:12px;flex-wrap:wrap;">
        <a href="{html.escape(aff['url'])}" target="_blank" rel="nofollow sponsored" data-track-cta="model_{gpu['provider_key']}" class="btn-cta">{html.escape(aff['cta_text'])}</a>
        <a href="{html.escape(alt_aff['url'])}" target="_blank" rel="nofollow sponsored" data-track-cta="model_{alt_key}" class="btn-cta btn-green">{html.escape(alt_aff['cta_text'])}</a>
      </div>
    </div>
  </div>

  <div class="affiliate-box">
    <div>
      <span class="badge badge-green">{html.escape(aff['badge'])}</span>
      <h3>Stop Overpaying for High-Volume AI Inference</h3>
      <p style="color:var(--muted);max-width:700px;">{"Running " + html.escape(m['name']) + " at scale? Deploy pre-configured vLLM or Ollama pods on dedicated H100, A100, or RTX 4090 GPUs by the second." if m["is_open_weight"] else "Spending heavily on " + html.escape(m['name']) + " API calls? Benchmark open-weight alternatives (Llama 4, Qwen 3, DeepSeek V3/R1) on dedicated H100, A100, or RTX 4090 GPUs from $0.20/hr."}</p>
    </div>
    <a href="{html.escape(aff['url'])}" target="_blank" rel="nofollow sponsored" data-track-cta="model_banner_{gpu['provider_key']}" class="btn-cta">{html.escape(aff['cta_text'])} →</a>
  </div>
  {render_deploy_card(m, config)}
  <div class="card" style="margin-top:24px;">
    <h2>Compare {html.escape(m['name'])} With Other Popular Models</h2>
    <ul style="margin-top:12px;padding-left:20px;display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%, 260px),1fr));gap:8px;">
      {related_links}
    </ul>
  </div>
</main>
<script>
  const MODEL_NAME = {json.dumps(m['name'])};
  const CANONICAL_URL = {json.dumps(canonical)};
  const IN_RATE = {m['input_1m']};
  const OUT_RATE = {m['output_1m']};
  const GPU_HOURLY = {gpu['hourly_usd']};
  const GPU_MONTHLY = {round(daily_gpu_cost * 30, 2)};
  const IS_OPEN = {"true" if m["is_open_weight"] else "false"};
  function initFromParams() {{
    const p = new URLSearchParams(window.location.search);
    if (p.get('in')) document.getElementById('inSlider').value = p.get('in');
    if (p.get('out')) document.getElementById('outSlider').value = p.get('out');
    if (p.get('util')) document.getElementById('utilSlider').value = p.get('util');
    if (p.get('ops')) document.getElementById('opsSlider').value = p.get('ops');
    recalc();
  }}
  function recalc() {{
    const inTok = parseFloat(document.getElementById('inSlider').value);
    const outTok = parseFloat(document.getElementById('outSlider').value);
    const utilPct = parseFloat(document.getElementById('utilSlider').value);
    const opsMo = parseFloat(document.getElementById('opsSlider').value);
    document.getElementById('inVal').textContent = inTok.toLocaleString();
    document.getElementById('outVal').textContent = outTok.toLocaleString();

    const totalTok = Math.max(1, inTok + outTok);
    const inPct = Math.round((inTok / totalTok) * 100);
    const outPct = 100 - inPct;
    const blendedRate = ((inTok * IN_RATE) + (outTok * OUT_RATE)) / totalTok;
    document.getElementById('mixInfo').textContent = `Mix: ${{inPct}}% input / ${{outPct}}% output · Blended API rate: $${{blendedRate.toFixed(2)}}/1M tokens`;

    const effActiveHr = GPU_HOURLY / (utilPct / 100);
    const adjGpuMonthly = (GPU_MONTHLY / (utilPct / 100)) + opsMo;
    document.getElementById('utilVal').textContent = utilPct === 100
      ? `100% (steady 24/7 · $${{GPU_HOURLY.toFixed(2)}}/hr)`
      : `${{utilPct}}% (${{(100 / utilPct).toFixed(1)}}× peak headroom · $${{effActiveHr.toFixed(2)}}/active hr)`;
    document.getElementById('opsVal').textContent = opsMo > 0
      ? `$${{opsMo.toLocaleString()}}/mo (drivers, scaling, storage)`
      : '$0/mo';

    const daily = ((inTok / 1e6) * IN_RATE) + ((outTok / 1e6) * OUT_RATE);
    const monthly = daily * 30;
    document.getElementById('dailyCost').textContent = '$' + daily.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
    document.getElementById('monthlyCost').textContent = '$' + monthly.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
    document.getElementById('gpuMonthlyCost').textContent = '$' + adjGpuMonthly.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}}) + '/mo';
    document.getElementById('gpuBreakdown').textContent = (utilPct === 100 && opsMo === 0)
      ? `1× dedicated GPU 24/7 @ $${{GPU_HOURLY.toFixed(2)}}/hr`
      : `$${{Math.round(GPU_MONTHLY).toLocaleString()}} base ÷ ${{utilPct}}% util + $${{opsMo.toLocaleString()}} ops`;

    const liveBeM = blendedRate > 0 ? ((adjGpuMonthly / 30) / blendedRate) : 0;
    document.getElementById('liveBreakeven').textContent = blendedRate > 0
      ? `~${{liveBeM.toFixed(1)}}M tokens/day`
      : 'Free / $0 API';

    const verdict = document.getElementById('verdictBox');
    if (monthly > adjGpuMonthly) {{
      const save = Math.round(monthly - adjGpuMonthly).toLocaleString();
      if (IS_OPEN) {{
        verdict.innerHTML = '💡 <strong>{verdict_win_prefix}:</strong> Even after ' + utilPct + '% utilization &amp; $' + opsMo.toLocaleString() + '/mo ops, renting a dedicated Cloud GPU saves roughly <strong>$' + save + '/month</strong> vs. API billing.';
      }} else {{
        verdict.innerHTML = '💡 <strong>{verdict_win_prefix}:</strong> Even after ' + utilPct + '% utilization &amp; $' + opsMo.toLocaleString() + '/mo ops, migrating high-volume tasks to a self-hosted open-weight equivalent (Llama / Qwen / DeepSeek) can save roughly <strong>$' + save + '/month</strong>.';
      }}
    }} else {{
      verdict.innerHTML = '✅ <strong>Pay-Per-Token API Wins:</strong> At this volume (' + utilPct + '% GPU utilization, $' + opsMo.toLocaleString() + '/mo ops), pay-per-token API or scale-to-zero Serverless GPUs are cheaper than a dedicated GPU instance.';
    }}
  }}
  function copyCalcLink(btn) {{
    const inTok = document.getElementById('inSlider').value;
    const outTok = document.getElementById('outSlider').value;
    const util = document.getElementById('utilSlider').value;
    const ops = document.getElementById('opsSlider').value;
    const url = CANONICAL_URL + '?in=' + inTok + '&out=' + outTok + '&util=' + util + '&ops=' + ops;
    navigator.clipboard.writeText(url);
    const orig = btn.textContent;
    btn.textContent = '✅ Link Copied!';
    setTimeout(() => btn.textContent = orig, 2000);
  }}
  function copyMarkdown(btn) {{
    const inRaw = document.getElementById('inSlider').value;
    const outRaw = document.getElementById('outSlider').value;
    const util = document.getElementById('utilSlider').value;
    const ops = document.getElementById('opsSlider').value;
    const inTok = parseFloat(inRaw).toLocaleString();
    const outTok = parseFloat(outRaw).toLocaleString();
    const monthly = document.getElementById('monthlyCost').textContent;
    const gpuMo = document.getElementById('gpuMonthlyCost').textContent;
    const md = `**${{MODEL_NAME}} Cost Estimate** (${{inTok}} in / ${{outTok}} out per day @ ${{util}}% util, $${{ops}}/mo ops): **${{monthly}}/mo** API vs **${{gpuMo}}** on Cloud GPU — [Calculate on TokenBench](${{CANONICAL_URL}}?in=${{inRaw}}&out=${{outRaw}}&util=${{util}}&ops=${{ops}})`;
    navigator.clipboard.writeText(md);
    const orig = btn.textContent;
    btn.textContent = '✅ Markdown Copied!';
    setTimeout(() => btn.textContent = orig, 2000);
  }}
  initFromParams();
</script>
"""
    page += shared_footer(config, updated_str, prefix="../")
    return page

def build_vram_page(m, config, updated_str):
    base_url = config["base_url"].rstrip("/")
    vram_slug = f"{m['slug']}-vram-requirements"
    canonical = f"{base_url}/vram/{vram_slug}.html"
    gpu = config["gpu_tiers"][m["gpu_tier_idx"]]
    aff_runpod = config["affiliates"]["runpod"]
    aff_vast = config["affiliates"]["vastai"]

    in_fmt = format_price(m["input_1m"])
    out_fmt = format_price(m["output_1m"])
    slider_max = max(192, int((m["vram_fp16"] // 32 + 2) * 32))

    if m["is_open_weight"]:
        title = f"{m['name']} VRAM Requirements & GPU Hardware Calculator (FP16, Q8, Q4)"
        desc = f"How much VRAM do you need to run {m['name']} locally? See exact memory requirements for FP16 ({m['vram_fp16']} GB), 8-bit ({m['vram_int8']} GB), and 4-bit GGUF/AWQ ({m['vram_int4']} GB) plus Cloud GPU rental costs."
        hero_h1 = f"{html.escape(m['name'])} VRAM Requirements &amp; GPU Guide"
        hero_p = f"Find out if your local GPU can run <strong>{html.escape(m['name'])}</strong> (~{m['params_b']:g}B parameters) in FP16, 8-bit, or 4-bit GGUF/EXL2/AWQ quantization—or rent the exact Cloud GPU needed from ${gpu['hourly_usd']}/hr."
        notice_html = ""
        card1_title = f"VRAM Required by Quantization Level (~{m['params_b']:g}B Params)"
        card2_title = f"Can My Local GPU Run {html.escape(m['name'])}?"
        faq_a = f"Running {m['name']} (~{m['params_b']:g}B parameters) requires approximately {m['vram_fp16']} GB VRAM in 16-bit (FP16/BF16), {m['vram_int8']} GB VRAM in 8-bit (Q8_0) quantization, and {m['vram_int4']} GB VRAM in 4-bit (Q4_K_M / AWQ) quantization including KV cache overhead."
    else:
        title = f"Can You Run {m['name']} Locally? VRAM Equivalent & Open-Weight Alternatives"
        desc = f"{m['name']} is a proprietary Cloud API. See the exact GPU VRAM required ({m['vram_int4']} GB Q4 / {m['vram_fp16']} GB FP16) to run an equivalent ~{m['params_b']:g}B open-weight alternative locally or on RunPod/Vast.ai."
        hero_h1 = f"{html.escape(m['name'])}: Local VRAM Equivalent &amp; Self-Hosting Guide"
        hero_p = f"<strong>{html.escape(m['name'])}</strong> is a closed-weight commercial API hosted by {html.escape(m['provider'])}. Below are the exact VRAM specifications needed to self-host a comparable <strong>~{m['params_b']:g}B-class open-weight LLM</strong> (such as Meta Llama, Qwen, Mistral, or DeepSeek) in FP16, Q8, or Q4 quantization."
        notice_html = f"""
  <div class="notice-box">
    <strong>ℹ️ Closed-Weight API Notice:</strong> {html.escape(m['name'])} weights are proprietary to {html.escape(m['provider'])} and cannot be downloaded as local GGUF/Safetensors files. The calculator below shows the hardware and VRAM required to run an <strong>equivalent ~{m['params_b']:g}B parameter open-weight alternative</strong> locally or on a dedicated Cloud GPU.
  </div>"""
        card1_title = f"Equivalent Open-Weight VRAM (~{m['params_b']:g}B Class Model)"
        card2_title = f"Can My GPU Run a ~{m['params_b']:g}B Open-Weight Alternative?"
        faq_a = f"{m['name']} is a closed-source API and cannot be downloaded directly. To run a comparable ~{m['params_b']:g}B parameter open-weight model locally, you need roughly {m['vram_fp16']} GB VRAM in FP16, {m['vram_int8']} GB VRAM in 8-bit (Q8_0), or {m['vram_int4']} GB VRAM in 4-bit (Q4_K_M / AWQ) quantization."

    json_ld = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": f"How much VRAM is needed to run {m['name']} or an open-weight equivalent?",
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": faq_a
                }
            }
        ]
    }

    page = shared_head(title, desc, canonical, config, json_ld)
    page += shared_nav(prefix="../", config=config)
    page += f"""
<main class="container">
  <section class="hero">
    <div style="display:flex;gap:8px;flex-wrap:wrap;">
      <span class="badge">LLM VRAM &amp; Hardware Calculator</span>
      {'<span class="badge badge-green">Open-Weight Model</span>' if m["is_open_weight"] else '<span class="badge badge-warn">Proprietary API · Open-Weight Equivalent</span>'}
    </div>
    <h1>{hero_h1}</h1>
    <p>{hero_p}</p>
  </section>
  {notice_html}
  <div class="grid-2">
    <div class="card">
      <h2>{card1_title}</h2>
      <div class="stat-row">
        <span>Estimated Parameter Class</span>
        <strong>~{m['params_b']:g}B Parameters</strong>
      </div>
      <div class="stat-row">
        <span>16-Bit Full Precision (FP16 / BF16)</span>
        <strong>{m['vram_fp16']} GB VRAM</strong>
      </div>
      <div class="stat-row">
        <span>8-Bit Quantization (Q8_0 / FP8)</span>
        <strong>{m['vram_int8']} GB VRAM</strong>
      </div>
      <div class="stat-row">
        <span>4-Bit Quantization (Q4_K_M / AWQ / EXL2)</span>
        <strong style="color:var(--accent-green)">{m['vram_int4']} GB VRAM</strong>
      </div>
      <div class="stat-row">
        <span>Recommended Cloud GPU</span>
        <strong>{html.escape(gpu['name'])} (${gpu['hourly_usd']}/hr)</strong>
      </div>
      <div class="stat-row">
        <span>Cloud API Token Pricing</span>
        <strong><a href="../models/{m['slug']}.html">{in_fmt}/1M In · {out_fmt}/1M Out →</a></strong>
      </div>
      <div style="margin-top:20px;display:flex;gap:12px;flex-wrap:wrap;">
        <a href="{html.escape(aff_runpod['url'])}" target="_blank" rel="nofollow sponsored" data-track-cta="vram_runpod" class="btn-cta">{html.escape(aff_runpod['cta_text'])}</a>
        <a href="{html.escape(aff_vast['url'])}" target="_blank" rel="nofollow sponsored" data-track-cta="vram_vastai" class="btn-cta btn-green">{html.escape(aff_vast['cta_text'])}</a>
      </div>
    </div>

    <div class="card">
      <h2>{card2_title}</h2>
      <p style="color:var(--muted);font-size:0.86rem;margin-bottom:10px;">Quick-select your hardware preset or drag the slider:</p>
      <div class="filter-bar" style="margin-bottom:12px;">
        <button type="button" class="filter-pill" onclick="setGpuVram(16)">RTX 4060 Ti (16GB)</button>
        <button type="button" class="filter-pill" onclick="setGpuVram(24)">RTX 3090/4090 (24GB)</button>
        <button type="button" class="filter-pill" onclick="setGpuVram(32)">RTX 5090 (32GB)</button>
        <button type="button" class="filter-pill" onclick="setGpuVram(48)">2x 3090 / A6000 (48GB)</button>
        <button type="button" class="filter-pill" onclick="setGpuVram(64)">Mac Studio (64GB)</button>
        <button type="button" class="filter-pill" onclick="setGpuVram(80)">A100 / H100 (80GB)</button>
        <button type="button" class="filter-pill" onclick="setGpuVram(192)">Mac Ultra / 2x H100 (192GB)</button>
      </div>
      <div class="slider-group">
        <label><span>Select Your Local GPU VRAM</span> <span id="myVramLabel">24 GB VRAM</span></label>
        <input type="range" id="myVram" min="8" max="{slider_max}" step="8" value="24" oninput="checkVram()" />
      </div>
      <div id="vramVerdict" style="margin-top:16px;padding:14px;border-radius:8px;background:rgba(56,189,248,0.12);border:1px solid var(--accent);"></div>
      <p style="color:var(--muted);font-size:0.88rem;margin-top:14px;">Don't have enough local VRAM? Rent a pre-configured Ollama / vLLM PyTorch pod on RunPod or Vast.ai in 30 seconds with zero upfront hardware cost.</p>
      <div class="share-row">
        <button type="button" class="btn-ghost" onclick="copyVramLink(this)">🔗 Copy Hardware Check Link</button>
        <button type="button" class="btn-ghost" onclick="copyVramMd(this)">📋 Copy Markdown for Reddit / GitHub</button>
      </div>
    </div>
  </div>
  {render_deploy_card(m, config)}
  {render_hardware_buy_card(m, config)}
</main>
<script>
  const MODEL_NAME = {json.dumps(m['name'])};
  const CANONICAL_URL = {json.dumps(canonical)};
  const FP16 = {m['vram_fp16']};
  const INT8 = {m['vram_int8']};
  const INT4 = {m['vram_int4']};
  function setGpuVram(val) {{
    const slider = document.getElementById('myVram');
    if (val > parseFloat(slider.max)) slider.max = val;
    slider.value = val;
    checkVram();
  }}
  function checkVram() {{
    const v = parseFloat(document.getElementById('myVram').value);
    document.getElementById('myVramLabel').textContent = v + ' GB VRAM';
    const box = document.getElementById('vramVerdict');
    if (v >= FP16) {{
      box.innerHTML = '✅ <strong>Full Speed (FP16 / BF16):</strong> ' + v + ' GB VRAM can run this model class at full 16-bit precision (' + FP16 + ' GB needed) with generous KV-cache context headroom.';
    }} else if (v >= INT8) {{
      box.innerHTML = '🟢 <strong>Near-Lossless 8-Bit (Q8_0 / FP8):</strong> ' + v + ' GB VRAM comfortably runs 8-bit quantization (' + INT8 + ' GB needed) or 4-bit quantization (' + INT4 + ' GB needed) with large context windows. For full unquantized FP16 (' + FP16 + ' GB), rent a Cloud GPU below.';
    }} else if (v >= INT4) {{
      box.innerHTML = '⚡ <strong>Runs 4-Bit Quantized (Q4_K_M / AWQ):</strong> ' + v + ' GB VRAM can run this model class using 4-bit quantization (' + INT4 + ' GB needed). For 8-bit (' + INT8 + ' GB) or full FP16 (' + FP16 + ' GB), rent a Cloud GPU below.';
    }} else {{
      box.innerHTML = '❌ <strong>Not Enough Local VRAM:</strong> Even 4-bit quantization requires ~' + INT4 + ' GB VRAM (you selected ' + v + ' GB). Rent an RTX 4090, A100, or H100 on RunPod / Vast.ai for pennies per hour instead of buying expensive hardware.';
    }}
  }}
  function copyVramLink(btn) {{
    const v = document.getElementById('myVram').value;
    navigator.clipboard.writeText(CANONICAL_URL + '?vram=' + v);
    const orig = btn.textContent;
    btn.textContent = '✅ Link Copied!';
    setTimeout(() => btn.textContent = orig, 2000);
  }}
  function copyVramMd(btn) {{
    const v = document.getElementById('myVram').value;
    const md = `**${{MODEL_NAME}} VRAM Requirements**: Q4: **${{INT4}} GB** · Q8: **${{INT8}} GB** · FP16: **${{FP16}} GB** — [Check GPU compatibility on TokenBench](${{CANONICAL_URL}}?vram=${{v}})`;
    navigator.clipboard.writeText(md);
    const orig = btn.textContent;
    btn.textContent = '✅ Markdown Copied!';
    setTimeout(() => btn.textContent = orig, 2000);
  }}
  (function() {{
    const p = new URLSearchParams(window.location.search);
    if (p.get('vram')) setGpuVram(parseFloat(p.get('vram')));
    else checkVram();
  }})();
</script>
"""
    page += shared_footer(config, updated_str, prefix="../")
    return vram_slug, page

def build_vram_hub_page(models, config, updated_str):
    base_url = config["base_url"].rstrip("/")
    canonical = f"{base_url}/vram-electricity-calculator.html"
    title = "LLM VRAM Calculator & Local GPU Electricity vs Cloud Cost (2026)"
    desc = "Calculate exact VRAM requirements (FP16, Q8, Q4) for any LLM including context window KV cache, and compare your home GPU electricity bill ($/kWh) against Cloud GPUs."

    aff_runpod = config["affiliates"]["runpod"]
    aff_vast = config["affiliates"]["vastai"]

    json_ld = {
        "@context": "https://schema.org",
        "@type": "WebApplication",
        "name": "TokenBench LLM VRAM & GPU Electricity Calculator",
        "url": canonical,
        "applicationCategory": "DeveloperApplication",
        "operatingSystem": "All",
        "description": desc,
        "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"}
    }

    # Feature genuinely open-weight models first in the VRAM directory grid
    open_models = [m for m in models if m["is_open_weight"] and not m["is_secondary"]]
    other_models = [m for m in models if not m["is_open_weight"] and not m["is_secondary"]]
    featured_vram_models = (open_models + other_models)[:96]

    vram_links = "".join(
        f'<a href="vram/{m["slug"]}-vram-requirements.html" class="card" style="display:block;padding:14px 18px;text-decoration:none;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;gap:6px;"><strong style="color:#fff;">{html.escape(m["name"])}</strong>'
        f'<span class="badge {"badge-green" if m["is_open_weight"] else "badge-warn"}" style="margin:0;font-size:0.7rem;">{"Open-Weight" if m["is_open_weight"] else "API Eq."}</span></div>'
        f'<div style="color:var(--muted);font-size:0.84rem;margin-top:6px;">~{m["params_b"]:g}B · 4-bit: {m["vram_int4"]} GB · FP16: {m["vram_fp16"]} GB →</div></a>'
        for m in featured_vram_models
    )

    page = shared_head(title, desc, canonical, config, json_ld=json_ld)
    page += shared_nav(prefix="", config=config)
    page += f"""
<main class="container">
  <section class="hero">
    <span class="badge">VRAM &amp; Power Bill Calculator</span>
    <h1>LLM VRAM &amp; Home GPU Electricity vs. Cloud Calculator</h1>
    <p>Calculate how much GPU memory your LLM requires across FP16, 8-bit, and 4-bit quantization (including KV cache context overhead) and compare your local PC's power bill against on-demand Cloud GPUs.</p>
  </section>

  <div class="grid-2">
    <div class="card">
      <h2>Interactive LLM VRAM Calculator</h2>
      <div class="filter-bar" style="margin-bottom:12px;">
        <button type="button" class="filter-pill" onclick="setParamPreset(8)">8B (Llama 3.1 / Qwen)</button>
        <button type="button" class="filter-pill" onclick="setParamPreset(14)">14B (Phi-4 / Qwen)</button>
        <button type="button" class="filter-pill" onclick="setParamPreset(32)">32B (Qwen3 / DeepSeek Distill)</button>
        <button type="button" class="filter-pill" onclick="setParamPreset(70)">70B (Llama 3.3 / Qwen 72B)</button>
        <button type="button" class="filter-pill" onclick="setParamPreset(123)">123B (Mistral Large)</button>
        <button type="button" class="filter-pill" onclick="setParamPreset(405)">405B (Frontier MoE)</button>
      </div>
      <div class="slider-group">
        <label><span>Model Size (Billion Parameters)</span> <span id="paramVal">70B</span></label>
        <input type="range" id="paramSlider" min="1" max="405" step="1" value="70" oninput="calcHub()" />
      </div>
      <div class="slider-group">
        <label><span>Context Window (KV Cache Tokens)</span> <span id="ctxVal">8,192 tokens</span></label>
        <input type="range" id="ctxSlider" min="2048" max="131072" step="2048" value="8192" oninput="calcHub()" />
      </div>
      <div class="stat-row">
        <span>FP16 / BF16 Full Precision</span>
        <strong id="hubFp16">154.0 GB VRAM</strong>
      </div>
      <div class="stat-row">
        <span>8-Bit Quantized (Q8_0 / FP8)</span>
        <strong id="hubInt8">80.5 GB VRAM</strong>
      </div>
      <div class="stat-row">
        <span>4-Bit Quantized (Q4_K_M / AWQ)</span>
        <strong id="hubInt4" style="color:var(--accent-green)">45.5 GB VRAM</strong>
      </div>
      <div class="stat-row">
        <span>Recommended GPU Tier</span>
        <strong id="hubGpuRec" style="color:var(--accent)">NVIDIA A100 80GB</strong>
      </div>
      <div class="share-row">
        <button type="button" class="btn-ghost" onclick="copyHubLink(this)">🔗 Copy Shareable Link</button>
        <button type="button" class="btn-ghost" onclick="copyHubMd(this)">📋 Copy Markdown Summary</button>
      </div>
    </div>

    <div class="card">
      <h2>Home GPU Electricity Bill Calculator</h2>
      <div class="slider-group">
        <label><span>System Power Draw Under Load (Watts)</span> <span id="wattVal">550 W</span></label>
        <input type="range" id="wattSlider" min="150" max="1800" step="50" value="550" oninput="calcHub()" />
      </div>
      <div class="slider-group">
        <label><span>Electricity Rate ($ / kWh)</span> <span id="kwhVal">$0.18 / kWh</span></label>
        <input type="range" id="kwhSlider" min="0.05" max="0.60" step="0.01" value="0.18" oninput="calcHub()" />
      </div>
      <div class="slider-group">
        <label><span>Active Inference Hours per Day</span> <span id="hrsVal">24 hrs/day</span></label>
        <input type="range" id="hrsSlider" min="1" max="24" step="1" value="24" oninput="calcHub()" />
      </div>
      <div class="stat-row">
        <span>Monthly Energy Consumption</span>
        <strong id="kwhMonthly">396.0 kWh/mo</strong>
      </div>
      <div class="stat-row">
        <span>Monthly Home GPU Power Cost</span>
        <strong id="elecMonthly" class="price-big">$71.28/mo</strong>
      </div>
      <div style="margin-top:16px;display:flex;gap:12px;flex-wrap:wrap;">
        <a href="{html.escape(aff_runpod['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta">{html.escape(aff_runpod['cta_text'])}</a>
        <a href="{html.escape(aff_vast['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta btn-green">{html.escape(aff_vast['cta_text'])}</a>
      </div>
    </div>
  </div>
  {render_cli_card(config)}
  {render_hardware_hub_section(config)}
  <h2 style="color:#fff;margin-top:36px;margin-bottom:8px;">Open-Weight &amp; Popular Model VRAM Guides</h2>
  <p style="color:var(--muted);margin-bottom:16px;">Select any model below to view quantization memory tiers and check if your GPU can run it.</p>
  <div class="grid-3">
    {vram_links}
  </div>
</main>
<script>
  const CANONICAL_URL = {json.dumps(canonical)};
  function setParamPreset(val) {{
    document.getElementById('paramSlider').value = val;
    calcHub();
  }}
  function calcHub() {{
    const p = parseFloat(document.getElementById('paramSlider').value);
    const ctx = parseFloat(document.getElementById('ctxSlider').value);
    document.getElementById('paramVal').textContent = p + 'B';
    document.getElementById('ctxVal').textContent = ctx.toLocaleString() + ' tokens';

    const kvExtraGb = Math.max(0, (ctx - 8192) / 8192) * (p * 0.025);
    const fp16 = (p * 2.2) + (kvExtraGb * 2);
    const int8 = (p * 1.15) + (kvExtraGb * 1.2);
    const int4 = (p * 0.65) + kvExtraGb;

    document.getElementById('hubFp16').textContent = fp16.toFixed(1) + ' GB VRAM';
    document.getElementById('hubInt8').textContent = int8.toFixed(1) + ' GB VRAM';
    document.getElementById('hubInt4').textContent = int4.toFixed(1) + ' GB VRAM';

    let rec = '1x RTX 4090 (24GB)';
    if (int4 > 140) rec = '4x–8x H100 80GB Cluster';
    else if (int4 > 75) rec = '2x H100 / A100 (160GB VRAM)';
    else if (int4 > 22) rec = '1x A100 / H100 (80GB VRAM)';
    document.getElementById('hubGpuRec').textContent = rec;

    const w = parseFloat(document.getElementById('wattSlider').value);
    const kwh = parseFloat(document.getElementById('kwhSlider').value);
    const hrs = parseFloat(document.getElementById('hrsSlider').value);
    document.getElementById('wattVal').textContent = w + ' W';
    document.getElementById('kwhVal').textContent = '$' + kwh.toFixed(2) + ' / kWh';
    document.getElementById('hrsVal').textContent = hrs + ' hrs/day';
    const monthlyKwh = (w / 1000) * hrs * 30;
    const monthlyCost = monthlyKwh * kwh;
    document.getElementById('kwhMonthly').textContent = monthlyKwh.toFixed(1) + ' kWh/mo';
    document.getElementById('elecMonthly').textContent = '$' + monthlyCost.toFixed(2) + '/mo';
  }}
  function getHubUrl() {{
    const p = document.getElementById('paramSlider').value;
    const ctx = document.getElementById('ctxSlider').value;
    const w = document.getElementById('wattSlider').value;
    const kwh = document.getElementById('kwhSlider').value;
    const hrs = document.getElementById('hrsSlider').value;
    return `${{CANONICAL_URL}}?p=${{p}}&ctx=${{ctx}}&w=${{w}}&kwh=${{kwh}}&hrs=${{hrs}}`;
  }}
  function copyHubLink(btn) {{
    navigator.clipboard.writeText(getHubUrl());
    const orig = btn.textContent;
    btn.textContent = '✅ Link Copied!';
    setTimeout(() => btn.textContent = orig, 2000);
  }}
  function copyHubMd(btn) {{
    const p = document.getElementById('paramSlider').value;
    const ctx = parseFloat(document.getElementById('ctxSlider').value).toLocaleString();
    const q4 = document.getElementById('hubInt4').textContent;
    const q8 = document.getElementById('hubInt8').textContent;
    const fp16 = document.getElementById('hubFp16').textContent;
    const elec = document.getElementById('elecMonthly').textContent;
    const md = `**${{p}}B LLM VRAM & Power Estimate (@ ${{ctx}} context)**: Q4: **${{q4}}** · Q8: **${{q8}}** · FP16: **${{fp16}}** · Home Electricity: **${{elec}}** — [Interactive Calculator](${{getHubUrl()}})`;
    navigator.clipboard.writeText(md);
    const orig = btn.textContent;
    btn.textContent = '✅ Markdown Copied!';
    setTimeout(() => btn.textContent = orig, 2000);
  }}
  (function() {{
    const q = new URLSearchParams(window.location.search);
    if (q.get('p')) document.getElementById('paramSlider').value = q.get('p');
    if (q.get('ctx')) document.getElementById('ctxSlider').value = q.get('ctx');
    if (q.get('w')) document.getElementById('wattSlider').value = q.get('w');
    if (q.get('kwh')) document.getElementById('kwhSlider').value = q.get('kwh');
    if (q.get('hrs')) document.getElementById('hrsSlider').value = q.get('hrs');
    calcHub();
  }})();
</script>
"""
    page += shared_footer(config, updated_str, prefix="")
    return page

def build_comparison_page(m1, m2, config, updated_str):
    base_url = config["base_url"].rstrip("/")
    comp_slug = f"{m1['slug']}-vs-{m2['slug']}"
    canonical = f"{base_url}/compare/{comp_slug}.html"
    title = f"{m1['name']} vs {m2['name']}: API Pricing & Token Cost Comparison (2026)"
    desc = f"Interactive price comparison: {m1['name']} ({format_price(m1['input_1m'])}/1M in) vs {m2['name']} ({format_price(m2['input_1m'])}/1M in). Calculate which LLM is cheaper for your workload."

    cheaper = m1 if m1["blended_1m"] <= m2["blended_1m"] else m2
    aff_runpod = config["affiliates"]["runpod"]
    aff_do = config["affiliates"]["vastai"]

    json_ld = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": f"Is {m1['name']} cheaper than {m2['name']}?",
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": f"{cheaper['name']} has the lower blended token price. {m1['name']} costs {format_price(m1['input_1m'])}/1M input and {format_price(m1['output_1m'])}/1M output tokens, whereas {m2['name']} costs {format_price(m2['input_1m'])}/1M input and {format_price(m2['output_1m'])}/1M output tokens."
                }
            }
        ]
    }

    vram_cell_1 = f'<a href="../vram/{m1["slug"]}-vram-requirements.html">{m1["vram_int4"]} GB (Open-Weight)</a>' if m1["is_open_weight"] else f'<a href="../vram/{m1["slug"]}-vram-requirements.html">API Only (~{m1["vram_int4"]} GB eq.)</a>'
    vram_cell_2 = f'<a href="../vram/{m2["slug"]}-vram-requirements.html">{m2["vram_int4"]} GB (Open-Weight)</a>' if m2["is_open_weight"] else f'<a href="../vram/{m2["slug"]}-vram-requirements.html">API Only (~{m2["vram_int4"]} GB eq.)</a>'

    page = shared_head(title, desc, canonical, config, json_ld=json_ld)
    page += shared_nav(prefix="../", config=config)
    page += f"""
<main class="container">
  <section class="hero">
    <span class="badge">Head-to-Head Pricing Benchmark</span>
    <h1>{html.escape(m1['name'])} vs. {html.escape(m2['name'])}</h1>
    <p>Interactive cost comparison based on live API token rates and context window limits. Overall lower blended token cost: <strong>{html.escape(cheaper['name'])}</strong>.</p>
  </section>

  <div class="grid-2">
    <div class="card">
      <h2>Interactive Side-by-Side Cost Calculator</h2>
      <p style="color:var(--muted);font-size:0.9rem;margin-bottom:14px;">Slide your daily token volume to compare monthly bills between both models in real time.</p>
      <div class="slider-group">
        <label><span>Daily Input Tokens</span> <span id="compInVal">5,000,000</span></label>
        <input type="range" id="compInSlider" min="100000" max="50000000" step="100000" value="5000000" oninput="calcComp()" />
      </div>
      <div class="slider-group">
        <label><span>Daily Output Tokens</span> <span id="compOutVal">1,000,000</span></label>
        <input type="range" id="compOutSlider" min="50000" max="20000000" step="50000" value="1000000" oninput="calcComp()" />
      </div>
      <div class="stat-row">
        <span><strong>{html.escape(m1['name'])}</strong> Monthly Cost</span>
        <strong id="m1Monthly" style="font-size:1.25rem;">$0.00/mo</strong>
      </div>
      <div class="stat-row">
        <span><strong>{html.escape(m2['name'])}</strong> Monthly Cost</span>
        <strong id="m2Monthly" style="font-size:1.25rem;">$0.00/mo</strong>
      </div>
      <div id="compWinnerBox" style="margin-top:14px;padding:12px;border-radius:8px;background:rgba(16,185,129,0.12);border:1px solid var(--accent-green);font-size:0.92rem;"></div>
      <div class="share-row">
        <button type="button" class="btn-ghost" onclick="copyCompLink(this)">🔗 Copy Comparison Link</button>
        <button type="button" class="btn-ghost" onclick="copyCompMd(this)">📋 Copy Markdown Table</button>
      </div>
    </div>

    <div class="card">
      <h2>Side-by-Side Specs &amp; Token Rates</h2>
      <div class="table-wrap">
        <table style="min-width:0;table-layout:fixed;width:100%;">
          <thead>
            <tr>
              <th style="width:32%;white-space:normal;word-break:break-word;">Metric</th>
              <th style="width:34%;white-space:normal;word-break:break-word;"><a href="../models/{m1['slug']}.html">{html.escape(m1['name'])}</a></th>
              <th style="width:34%;white-space:normal;word-break:break-word;"><a href="../models/{m2['slug']}.html">{html.escape(m2['name'])}</a></th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Provider</td>
              <td>{html.escape(m1['provider'])}</td>
              <td>{html.escape(m2['provider'])}</td>
            </tr>
            <tr>
              <td>Model Type</td>
              <td>{"Open-Weight" if m1["is_open_weight"] else "Proprietary API"}</td>
              <td>{"Open-Weight" if m2["is_open_weight"] else "Proprietary API"}</td>
            </tr>
            <tr>
              <td>Input Cost / 1M</td>
              <td><strong>{format_price(m1['input_1m'])}</strong></td>
              <td><strong>{format_price(m2['input_1m'])}</strong></td>
            </tr>
            <tr>
              <td>Output Cost / 1M</td>
              <td><strong>{format_price(m1['output_1m'])}</strong></td>
              <td><strong>{format_price(m2['output_1m'])}</strong></td>
            </tr>
            <tr>
              <td>Context Window</td>
              <td>{m1['context_length']:,} tokens</td>
              <td>{m2['context_length']:,} tokens</td>
            </tr>
            <tr>
              <td>4-Bit Quantized VRAM</td>
              <td>{vram_cell_1}</td>
              <td>{vram_cell_2}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="affiliate-box">
    <div>
      <span class="badge badge-green">Cloud Infrastructure Deal</span>
      <h3>Deploy Your AI App or Open-Source LLM for Less</h3>
      <p style="color:var(--muted);">Rent serverless or dedicated GPUs on RunPod and Vast.ai to cut your monthly AI infrastructure bill by up to 70%.</p>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;">
      <a href="{html.escape(aff_runpod['url'])}" target="_blank" rel="nofollow sponsored" data-track-cta="compare_runpod" class="btn-cta">{html.escape(aff_runpod['cta_text'])}</a>
      <a href="{html.escape(aff_do['url'])}" target="_blank" rel="nofollow sponsored" data-track-cta="compare_vastai" class="btn-cta btn-green">{html.escape(aff_do['cta_text'])}</a>
    </div>
  </div>
</main>
<script>
  const CANONICAL_URL = {json.dumps(canonical)};
  const M1_NAME = {json.dumps(m1['name'])};
  const M2_NAME = {json.dumps(m2['name'])};
  const M1_IN = {m1['input_1m']};
  const M1_OUT = {m1['output_1m']};
  const M2_IN = {m2['input_1m']};
  const M2_OUT = {m2['output_1m']};
  function calcComp() {{
    const inTok = parseFloat(document.getElementById('compInSlider').value);
    const outTok = parseFloat(document.getElementById('compOutSlider').value);
    document.getElementById('compInVal').textContent = inTok.toLocaleString();
    document.getElementById('compOutVal').textContent = outTok.toLocaleString();
    const cost1 = (((inTok / 1e6) * M1_IN) + ((outTok / 1e6) * M1_OUT)) * 30;
    const cost2 = (((inTok / 1e6) * M2_IN) + ((outTok / 1e6) * M2_OUT)) * 30;
    document.getElementById('m1Monthly').textContent = '$' + cost1.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}}) + '/mo';
    document.getElementById('m2Monthly').textContent = '$' + cost2.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}}) + '/mo';
    const diff = Math.abs(cost1 - cost2).toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
    const winner = cost1 <= cost2 ? M1_NAME : M2_NAME;
    document.getElementById('compWinnerBox').innerHTML = '🏆 <strong>' + winner + ' is cheaper:</strong> Saves you <strong>$' + diff + '/month</strong> at this token volume.';
  }}
  function copyCompLink(btn) {{
    const inTok = document.getElementById('compInSlider').value;
    const outTok = document.getElementById('compOutSlider').value;
    navigator.clipboard.writeText(`${{CANONICAL_URL}}?in=${{inTok}}&out=${{outTok}}`);
    const orig = btn.textContent;
    btn.textContent = '✅ Link Copied!';
    setTimeout(() => btn.textContent = orig, 2000);
  }}
  function copyCompMd(btn) {{
    const m1c = document.getElementById('m1Monthly').textContent;
    const m2c = document.getElementById('m2Monthly').textContent;
    const md = `| Model | Input / 1M | Output / 1M | Est. Monthly |\n|---|---|---|---|\n| **${{M1_NAME}}** | $${{M1_IN}} | $${{M1_OUT}} | **${{m1c}}** |\n| **${{M2_NAME}}** | $${{M2_IN}} | $${{M2_OUT}} | **${{m2c}}** |\n\n*Compared via [TokenBench](${{CANONICAL_URL}})*`;
    navigator.clipboard.writeText(md);
    const orig = btn.textContent;
    btn.textContent = '✅ Markdown Copied!';
    setTimeout(() => btn.textContent = orig, 2000);
  }}
  (function() {{
    const p = new URLSearchParams(window.location.search);
    if (p.get('in')) document.getElementById('compInSlider').value = p.get('in');
    if (p.get('out')) document.getElementById('compOutSlider').value = p.get('out');
    calcComp();
  }})();
</script>
"""
    page += shared_footer(config, updated_str, prefix="../")
    return comp_slug, page

def build_gpu_page(config, updated_str):
    base_url = config["base_url"].rstrip("/")
    canonical = f"{base_url}/gpu-cloud-pricing.html"
    title = "Cloud GPU Price Comparison (2026): H100, A100, RTX 4090 Hourly Rates"
    desc = "Compare hourly Cloud GPU rental prices across RunPod, Vast.ai, Vultr, and DigitalOcean. Find the cheapest GPU for LLM inference and fine-tuning."

    json_ld = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": "How much does it cost to rent an NVIDIA H100, A100, or RTX 4090 per hour?",
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": "On-demand Cloud GPU rental rates start around $0.39/hr for an NVIDIA RTX 4090 (24GB VRAM) on Vast.ai, $1.39/hr for an NVIDIA A100 (80GB VRAM) on RunPod, and $2.69/hr for an NVIDIA H100 SXM (80GB VRAM)."
                }
            }
        ]
    }

    rows = ""
    for gpu in config["gpu_tiers"]:
        aff = config["affiliates"][gpu["provider_key"]]
        rows += f"""
        <tr>
          <td><strong>{html.escape(gpu['name'])}</strong></td>
          <td>{gpu['vram_gb']} GB</td>
          <td><strong style="color:var(--accent-green)">${gpu['hourly_usd']:.2f}/hr</strong> (${gpu['hourly_usd']*24*30:,.0f}/mo)</td>
          <td>~{gpu['throughput_tok_sec']} tok/s</td>
          <td>{html.escape(gpu['best_for'])}</td>
          <td><a href="{html.escape(aff['url'])}" target="_blank" rel="nofollow sponsored" data-track-cta="gpu_matrix_{gpu['id']}" class="btn-cta">{html.escape(aff['cta_text'])}</a></td>
        </tr>
        """

    cards = ""
    for key, aff in config["affiliates"].items():
        desc_text = aff.get(
            "description",
            "High-performance cloud compute and GPU instances ready for AI agents, vLLM, Ollama, and production APIs."
        )
        cards += f"""
        <div class="card">
          <span class="badge badge-green">{html.escape(aff['badge'])}</span>
          <h3>{html.escape(aff['name'])}</h3>
          <p style="color:var(--muted);font-size:0.92rem;margin:10px 0 18px;">{html.escape(desc_text)}</p>
          <a href="{html.escape(aff['url'])}" target="_blank" rel="nofollow sponsored" data-track-cta="gpu_partner_{key}" class="btn-cta">{html.escape(aff['cta_text'])} →</a>
        </div>
        """

    page = shared_head(title, desc, canonical, config, json_ld=json_ld)
    page += shared_nav(prefix="", config=config)
    page += f"""
<main class="container">
  <section class="hero">
    <span class="badge">Live Cloud GPU &amp; VPS Benchmark</span>
    <h1>Cloud GPU Hourly Price Comparison (H100, A100, RTX 4090)</h1>
    <p>Compare on-demand GPU instances and claim free developer cloud credits for self-hosting LLMs, Stable Diffusion, and AI agents.</p>
  </section>

  <div class="card">
    <h2>On-Demand Cloud GPU Pricing Matrix</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>GPU Instance</th>
            <th>VRAM</th>
            <th>On-Demand Rate</th>
            <th>Est. Speed</th>
            <th>Best Workload</th>
            <th>Deploy</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </div>
  </div>

  <h2 style="margin-top:36px;color:#fff;">Verified Cloud GPU &amp; VPS Partner Offers</h2>
  <div class="grid-3">
    {cards}
  </div>

  <div class="affiliate-box" style="margin-top:32px;border-color:var(--accent-purple);">
    <div>
      <span class="badge" style="background:rgba(129,140,248,0.18);color:var(--accent-purple);">B2B Sponsorship Slot</span>
      <h3>Run an AI Cloud, GPU Marketplace, or Inference API?</h3>
      <p style="color:var(--muted);max-width:720px;">Feature your H100 / A100 / serverless GPU pricing across 1,550+ AI cost &amp; VRAM calculator pages and our local <code>tokenbench</code> CLI.</p>
    </div>
    <a href="{html.escape(config.get('sponsor_inquiry_url', 'https://github.com/stigla610/ai-cost-engine/issues'))}" target="_blank" rel="noopener" data-track-cta="sponsor_inquiry" class="btn-cta">📣 Sponsor a Featured GPU Tier →</a>
  </div>
</main>
"""
    page += shared_footer(config, updated_str, prefix="")
    return page

def build_benchmarks_page(models, config, updated_str):
    base_url = config["base_url"].rstrip("/")
    canonical = f"{base_url}/benchmarks.html"
    title = "2026 LLM API Cost, Cloud GPU Break-Even & VRAM Scaling Benchmarks"
    desc = "Live 2026 benchmark reports: LLM API vs. Cloud GPU break-even volume, KV-cache VRAM scaling (8K–128K context), and cheapest models ranked by daily token volume."

    primary = [m for m in models if not m["is_secondary"]]
    a100_mo = round(config["gpu_tiers"][1]["hourly_usd"] * 24 * 30, 2)
    h100_mo = round(config["gpu_tiers"][2]["hourly_usd"] * 24 * 30, 2)

    # Guide 1: Break-even rows across top flagship & open-weight models
    breakeven_rows = ""
    for m in primary[:18]:
        mo_2_5m = ((2.0 * m["input_1m"]) + (0.5 * m["output_1m"])) * 30.0
        mo_12_5m = ((10.0 * m["input_1m"]) + (2.5 * m["output_1m"])) * 30.0
        mo_60m = ((50.0 * m["input_1m"]) + (10.0 * m["output_1m"])) * 30.0
        gpu = config["gpu_tiers"][m["gpu_tier_idx"]]
        daily_gpu = gpu["hourly_usd"] * 24.0
        be_tokens = round(daily_gpu / max(m["blended_1m"], 0.05), 1)
        breakeven_rows += f"""
        <tr>
          <td><a href="models/{m['slug']}.html"><strong>{html.escape(m['name'])}</strong></a></td>
          <td>{"Open-Weight" if m["is_open_weight"] else "Proprietary API"}</td>
          <td>{format_price(m['input_1m'])} / {format_price(m['output_1m'])}</td>
          <td>${mo_2_5m:,.2f}/mo</td>
          <td>${mo_12_5m:,.2f}/mo</td>
          <td>${mo_60m:,.2f}/mo</td>
          <td><strong>~{be_tokens}M/day</strong> ({html.escape(gpu['name'].split('(')[0].strip())})</td>
        </tr>"""

    # Sensitivity breakdown using #1 flagship frontier model vs A100 80GB
    sens_model = next((m for m in primary if m["input_1m"] >= 1.0), primary[0])
    sens_scenarios = [
        ("100% Util · $0 Ops · 80/20 In/Out", "Steady 24/7 batch, input-heavy", 1.0, 0.0, 0.80),
        ("100% Util · $0 Ops · 50/50 In/Out", "Steady 24/7, output-heavy", 1.0, 0.0, 0.50),
        ("50% Util · $300 Ops · 80/20 In/Out", "Daytime traffic / idle overnight + ops", 0.50, 300.0, 0.80),
        ("30% Util · $500 Ops · 80/20 In/Out", "Bursty SaaS + scaling/ops overhead", 0.30, 500.0, 0.80),
    ]
    sens_rows = ""
    for sc_label, sc_note, util_frac, ops_mo, in_frac in sens_scenarios:
        out_frac = 1.0 - in_frac
        blended = (in_frac * sens_model["input_1m"]) + (out_frac * sens_model["output_1m"])
        eff_mo = (a100_mo / util_frac) + ops_mo
        be_m_day = (eff_mo / 30.0) / max(blended, 0.01)
        sens_rows += f"""
        <tr>
          <td><strong>{html.escape(sc_label)}</strong><br/><small style="color:var(--muted)">{html.escape(sc_note)}</small></td>
          <td>${blended:.2f} / 1M</td>
          <td>${eff_mo:,.0f}/mo</td>
          <td><strong style="color:var(--accent-green)">~{be_m_day:.1f}M tokens/day</strong></td>
        </tr>"""

    # Guide 2: KV Cache VRAM scaling for common open-weight sizes
    kv_sizes = [
        ("8B (Llama 3.1 8B / Qwen3 8B)", 8.0),
        ("14B (Phi-4 / Qwen3 14B)", 14.0),
        ("32B (Qwen3 32B / Qwen2.5 Coder)", 32.0),
        ("70B (Llama 3.3 70B / Qwen 72B)", 70.0),
        ("123B (Mistral Large)", 123.0),
        ("235B MoE (Qwen3 235B-A22B)", 235.0),
    ]
    kv_rows = ""
    for label, p in kv_sizes:
        def q4_at(ctx):
            return round((p * 0.65) + max(0.0, (ctx - 8192) / 8192.0) * (p * 0.025), 1)
        def q8_at(ctx):
            return round((p * 1.15) + (max(0.0, (ctx - 8192) / 8192.0) * (p * 0.025) * 1.2), 1)
        kv_rows += f"""
        <tr>
          <td><strong>{html.escape(label)}</strong></td>
          <td>{q4_at(8192)} GB</td>
          <td>{q4_at(32768)} GB</td>
          <td><strong style="color:var(--accent-green)">{q4_at(131072)} GB</strong></td>
          <td>{q8_at(8192)} GB</td>
          <td>{q8_at(32768)} GB</td>
        </tr>"""

    # Guide 3: Cheapest primary models sorted by 10M/day (8M in + 2M out) monthly bill
    cheapest_sorted = sorted(primary, key=lambda x: (8.0 * x["input_1m"]) + (2.0 * x["output_1m"]))[:18]
    cheapest_rows = ""
    for rank, m in enumerate(cheapest_sorted, 1):
        mo_10m = ((8.0 * m["input_1m"]) + (2.0 * m["output_1m"])) * 30.0
        cheapest_rows += f"""
        <tr>
          <td>#{rank}</td>
          <td><a href="models/{m['slug']}.html"><strong>{html.escape(m['name'])}</strong></a></td>
          <td>{html.escape(m['provider'])}</td>
          <td>{"Open-Weight" if m["is_open_weight"] else "Proprietary API"}</td>
          <td>{format_price(m['input_1m'])}</td>
          <td>{format_price(m['output_1m'])}</td>
          <td><strong style="color:var(--accent-green)">${mo_10m:,.2f}/mo</strong></td>
        </tr>"""

    json_ld = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": title,
        "description": desc,
        "url": canonical,
        "dateModified": updated_str,
        "author": {"@type": "Organization", "name": config["site_name"]},
    }

    page = shared_head(title, desc, canonical, config, json_ld=json_ld)
    page += shared_nav(prefix="", config=config)
    page += f"""
<main class="container">
  <section class="hero">
    <span class="badge">Live Data Benchmarks · Updated {updated_str}</span>
    <h1>2026 LLM API Cost, Cloud GPU Break-Even &amp; VRAM Benchmarks</h1>
    <p>Data-backed engineering guides generated directly from live API pricing across {len(models):,} models—showing exact GPU break-even thresholds, long-context KV-cache VRAM scaling, and lowest-cost inference tiers.</p>
  </section>

  <div class="card" id="breakeven" style="margin-bottom:24px;">
    <span class="badge badge-green">Report 1</span>
    <h2>LLM API vs. Dedicated Cloud GPU Break-Even (2026)</h2>
    <p style="color:var(--muted);font-size:0.92rem;margin-bottom:12px;">
      An on-demand <strong>NVIDIA A100 80GB</strong> runs roughly <strong>${a100_mo:,.0f}/mo</strong> (24/7 @ $1.39/hr) and an <strong>H100 SXM</strong> runs <strong>${h100_mo:,.0f}/mo</strong> (@ $2.69/hr). Here is how live API bills compare at Startup (2.5M tokens/day), Scale-Up (12.5M/day), and Agent Fleet (60M/day) workloads:
    </p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Type</th>
            <th>In / Out per 1M</th>
            <th>2.5M/day</th>
            <th>12.5M/day</th>
            <th>60M/day</th>
            <th>GPU Break-Even</th>
          </tr>
        </thead>
        <tbody>{breakeven_rows}</tbody>
      </table>
    </div>

    <h3 style="margin-top:24px;margin-bottom:8px;">Utilization, Ops Overhead &amp; Output-Mix Sensitivity</h3>
    <p style="color:var(--muted);font-size:0.9rem;margin-bottom:10px;">
      Comparing <strong><a href="models/{sens_model['slug']}.html">{html.escape(sens_model['name'])}</a></strong> ({format_price(sens_model['input_1m'])} in / {format_price(sens_model['output_1m'])} out per 1M) against an <strong>NVIDIA A100 80GB</strong> (${a100_mo:,.0f}/mo base). Lower utilization, driver/scaling ops, or output-heavy mixes shift the break-even significantly:
    </p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Operating Scenario</th>
            <th>Blended API Rate</th>
            <th>Effective GPU + Ops Cost</th>
            <th>Break-Even Volume</th>
          </tr>
        </thead>
        <tbody>{sens_rows}</tbody>
      </table>
    </div>
    <p style="color:var(--muted);font-size:0.86rem;margin-top:10px;">
      🎛️ <a href="models/{sens_model['slug']}.html">Click any model above to slide Utilization (20%–100%), Ops Overhead, and Input/Output Mix live →</a>
    </p>
  </div>

  <div class="card" id="vram-scaling" style="margin-bottom:24px;">
    <span class="badge">Report 2</span>
    <h2>Local LLM VRAM &amp; KV-Cache Scaling (8K vs. 32K vs. 128K Context)</h2>
    <p style="color:var(--muted);font-size:0.92rem;margin-bottom:12px;">
      Many local LLM guides only measure model weights at 2K–4K context. Once you push context to 32K or 128K tokens, KV-cache memory grows significantly—for example, a 70B Q4 model jumps from <strong>45.5 GB</strong> (8K) to <strong>71.8 GB</strong> (128K). Check your own hardware using <a href="vram-electricity-calculator.html">the interactive VRAM &amp; Electricity Calculator</a>.
    </p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Model Parameter Tier</th>
            <th>Q4 @ 8K Context</th>
            <th>Q4 @ 32K Context</th>
            <th>Q4 @ 128K Context</th>
            <th>Q8 @ 8K Context</th>
            <th>Q8 @ 32K Context</th>
          </tr>
        </thead>
        <tbody>{kv_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="card" id="cheapest">
    <span class="badge badge-green">Report 3</span>
    <h2>Cheapest Production LLMs Ranked (10M Tokens/Day Workload)</h2>
    <p style="color:var(--muted);font-size:0.92rem;margin-bottom:12px;">
      Lowest blended-cost primary models (8M input + 2M output tokens/day = 300M tokens/month), ranked live:
    </p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Model</th>
            <th>Provider</th>
            <th>Type</th>
            <th>Input / 1M</th>
            <th>Output / 1M</th>
            <th>10M/day Monthly Bill</th>
          </tr>
        </thead>
        <tbody>{cheapest_rows}</tbody>
      </table>
    </div>
  </div>

  {render_cli_card(config)}
</main>
"""
    page += shared_footer(config, updated_str, prefix="")
    return page

def build_external_articles(models, config, updated_str):
    """Generate ready-to-publish Markdown articles for Dev.to / Hashnode / Medium in ./articles/."""
    articles_dir = os.path.join(BASE_DIR, "articles")
    os.makedirs(articles_dir, exist_ok=True)
    base_url = config["base_url"].rstrip("/")
    primary = [m for m in models if not m["is_secondary"]]

    # Article 1: API vs Cloud GPU Break-Even
    a1_rows = []
    for m in primary[:15]:
        mo_25 = ((20.0 * m["input_1m"]) + (5.0 * m["output_1m"])) * 30.0
        mo_60 = ((50.0 * m["input_1m"]) + (10.0 * m["output_1m"])) * 30.0
        gpu = config["gpu_tiers"][m["gpu_tier_idx"]]
        be = round((gpu["hourly_usd"] * 24.0) / max(m["blended_1m"], 0.05), 1)
        a1_rows.append(
            f"| [{m['name']}]({base_url}/models/{m['slug']}.html) | {format_price(m['input_1m'])} / {format_price(m['output_1m'])} | ${mo_25:,.2f}/mo | ${mo_60:,.2f}/mo | ~{be}M/day |"
        )
    art1 = f"""---
title: "LLM API vs. Dedicated Cloud GPU Break-Even Math in 2026 (Live Data)"
published: true
tags: ai, llm, gpu, cloud
canonical_url: {base_url}/benchmarks.html#breakeven
---

At what point does paying per-million API tokens become more expensive than renting a dedicated NVIDIA A100 (`$1.39/hr`, `~$1,001/mo`) or H100 (`$2.69/hr`, `~$1,937/mo`)?

Using live pricing pulled across {len(models)} models on [TokenBench]({base_url}/), here are the exact monthly bills and daily token break-even points:

| Model | Input / Output (1M) | 25M Tokens/Day | 60M Tokens/Day | Dedicated GPU Break-Even |
| :--- | :--- | :--- | :--- | :--- |
{chr(10).join(a1_rows)}

### Try It With Your Own Token Volume
* **Simulate all 260+ models simultaneously:** [{base_url}/]({base_url}/)
* **Compare home GPU electricity (`$/kWh`) vs. cloud rentals:** [{base_url}/vram-electricity-calculator.html]({base_url}/vram-electricity-calculator.html)
* **Check what fits on your local GPU in terminal:**
```bash
curl -sL {base_url}/tokenbench.py | python3 -
```
"""
    with open(os.path.join(articles_dir, "01-llm-api-vs-cloud-gpu-breakeven-2026.md"), "w", encoding="utf-8") as f:
        f.write(art1)

    # Article 2: Local LLM VRAM & KV-Cache Guide
    open_models = [m for m in primary if m["is_open_weight"]][:16]
    a2_rows = []
    for m in open_models:
        p = m["params_b"]
        q4_32k = round((p * 0.65) + max(0.0, (32768 - 8192) / 8192.0) * (p * 0.025), 1)
        a2_rows.append(
            f"| [{m['name']}]({base_url}/vram/{m['slug']}-vram-requirements.html) | ~{p:g}B | **{m['vram_int4']} GB** | {q4_32k} GB | {m['vram_int8']} GB | {m['vram_fp16']} GB |"
        )
    art2 = f"""---
title: "How Much VRAM Do Open-Weight LLMs Actually Need at 8K vs. 32K Context?"
published: true
tags: localllama, ollama, vllm, machinelearning
canonical_url: {base_url}/vram-electricity-calculator.html
---

Most VRAM cheat sheets only count model weights at tiny context lengths. Once you crank your context window up to 32K–128K tokens for coding or RAG, the KV cache adds gigabytes of extra VRAM.

Here are the exact requirements (weights + KV-cache headroom) across top open-weight models from [TokenBench]({base_url}/vram-electricity-calculator.html):

| Open-Weight Model | Params | Q4 @ 8K Context | Q4 @ 32K Context | Q8 @ 8K Context | FP16 @ 8K |
| :--- | :--- | :--- | :--- | :--- | :--- |
{chr(10).join(a2_rows)}

### Auto-Check Your GPU in 1 Second
Run the zero-dependency Python CLI (`nvidia-smi` + Apple Silicon Unified Memory detection) to see which models fit on your hardware right now with copy-paste `ollama run` and `vllm serve` commands:

```bash
curl -sL {base_url}/tokenbench.py | python3 -
```

Interactive calculator + electricity (`$/kWh`) estimator: [{base_url}/vram-electricity-calculator.html]({base_url}/vram-electricity-calculator.html)
"""
    with open(os.path.join(articles_dir, "02-local-llm-vram-kv-cache-guide-2026.md"), "w", encoding="utf-8") as f:
        f.write(art2)

    # Article 3: Cheapest LLM APIs Ranked
    cheapest = sorted(primary, key=lambda x: (8.0 * x["input_1m"]) + (2.0 * x["output_1m"]))[:18]
    a3_rows = []
    for i, m in enumerate(cheapest, 1):
        mo_10 = ((8.0 * m["input_1m"]) + (2.0 * m["output_1m"])) * 30.0
        a3_rows.append(
            f"| #{i} | [{m['name']}]({base_url}/models/{m['slug']}.html) | {m['provider']} | {format_price(m['input_1m'])} | {format_price(m['output_1m'])} | **${mo_10:,.2f}/mo** |"
        )
    art3 = f"""---
title: "Cheapest Production LLM APIs in 2026 (Ranked for 10M Tokens/Day)"
published: true
tags: ai, llm, openai, deepseek
canonical_url: {base_url}/benchmarks.html#cheapest
---

If you are running agents, batch classification, or summarization pipelines at 10M tokens/day (8M input + 2M output), token rates vary by over **100x** between providers.

Below is the live ranking updated {updated_str} via [TokenBench]({base_url}/):

| Rank | Model | Provider | Input / 1M | Output / 1M | 10M/Day Monthly Bill |
| :--- | :--- | :--- | :--- | :--- | :--- |
{chr(10).join(a3_rows)}

Compare side-by-side costs or simulate custom token ratios on the live dashboard: [{base_url}/]({base_url}/)
"""
    with open(os.path.join(articles_dir, "03-cheapest-llm-apis-ranked-by-workload-2026.md"), "w", encoding="utf-8") as f:
        f.write(art3)

def build_index_page(models, comparisons, config, updated_str, total_pages):
    base_url = config["base_url"].rstrip("/")
    canonical = f"{base_url}/index.html"
    title = f"{config['site_name']} — {config['site_tagline']}"
    desc = "Compare live API token pricing across 300+ AI models (OpenAI, Anthropic, DeepSeek, Meta Llama, Qwen, Mistral, Google Gemini), simulate monthly workload bills, calculate LLM VRAM requirements, and compare Cloud GPU rental rates."

    json_ld = {
        "@context": "https://schema.org",
        "@type": "WebApplication",
        "name": config["site_name"],
        "url": canonical,
        "applicationCategory": "DeveloperApplication",
        "operatingSystem": "All",
        "description": desc,
        "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"}
    }

    display_models = [m for m in models if not m["is_secondary"]][:260]

    table_rows = ""
    for m in display_models:
        search_terms = f"{m['name'].lower()} {m['provider'].lower()} {m['id'].lower()} {'open-weight' if m['is_open_weight'] else 'proprietary api'}"
        gpu_monthly = round(config["gpu_tiers"][m["gpu_tier_idx"]]["hourly_usd"] * 24 * 30, 2)
        default_monthly = ((2.0 * m["input_1m"]) + (0.5 * m["output_1m"])) * 30.0
        if m["is_open_weight"]:
            vram_col = f'<a href="vram/{m["slug"]}-vram-requirements.html"><strong>{m["vram_int4"]} GB</strong></a> <small style="color:var(--accent-green)">(Open)</small>'
        else:
            vram_col = f'<a href="vram/{m["slug"]}-vram-requirements.html" style="color:var(--muted)" title="Proprietary API — click for open-weight equivalent VRAM">API (~{m["vram_int4"]} GB eq.)</a>'

        table_rows += f"""
        <tr class="model-row" data-name="{html.escape(search_terms)}" data-open="{'1' if m['is_open_weight'] else '0'}" data-in="{m['input_1m']}" data-out="{m['output_1m']}" data-ctx="{m['context_length']}" data-vram="{m['vram_int4']}" data-gpu="{gpu_monthly}" data-monthly="{default_monthly:.2f}">
          <td><a href="models/{m['slug']}.html"><strong>{html.escape(m['name'])}</strong></a><br/><small style="color:var(--muted)">{html.escape(m['id'])}</small></td>
          <td>{html.escape(m['provider'])}</td>
          <td>{format_price(m['input_1m'])}</td>
          <td>{format_price(m['output_1m'])}</td>
          <td class="sim-cell"><strong>${default_monthly:,.2f}/mo</strong></td>
          <td>{m['context_length']:,}</td>
          <td>{vram_col}</td>
          <td><a href="models/{m['slug']}.html">Calculator →</a></td>
        </tr>
        """

    comp_links = ""
    for c_slug, m1, m2 in comparisons[:90]:
        comp_links += f"""
        <a href="compare/{c_slug}.html" class="card" style="display:block;padding:14px 18px;text-decoration:none;">
          <strong style="color:#fff;">{html.escape(m1['name'])} vs {html.escape(m2['name'])}</strong>
          <div style="color:var(--muted);font-size:0.85rem;margin-top:4px;">{format_price(m1['input_1m'])}/1M vs {format_price(m2['input_1m'])}/1M input tokens →</div>
        </a>
        """

    aff_main = config["affiliates"]["runpod"]
    aff_sec = config["affiliates"]["vastai"]

    page = shared_head(title, desc, canonical, config, json_ld=json_ld)
    page += shared_nav(prefix="", config=config)
    page += f"""
<main class="container">
  <section class="hero">
    <span class="badge">Auto-Updated Daily · {total_pages:,}+ Live AI Pricing &amp; VRAM Calculators</span>
    <h1>AI API Token Cost, VRAM &amp; Cloud GPU Break-Even Engine</h1>
    <p>Instantly compare per-million token costs across OpenAI, Anthropic, Google, DeepSeek, Qwen, Mistral, and Meta Llama models—and simulate your exact monthly bill across all {len(display_models)} models at once.</p>
  </section>

  <div class="card" style="margin-bottom:24px;border-color:rgba(56,189,248,0.4);">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:12px;">
      <div>
        <span class="badge badge-green" style="margin-bottom:4px;">Live Batch Workload Simulator</span>
        <h2 style="margin-bottom:2px;">Simulate Your Monthly Bill Across All {len(display_models)} Models</h2>
      </div>
      <div class="filter-bar" style="margin-bottom:0;" id="workloadPresets">
        <button type="button" class="filter-pill" onclick="setWorkloadPreset(200000, 50000, this)">Indie App (250K/day)</button>
        <button type="button" class="filter-pill active" onclick="setWorkloadPreset(2000000, 500000, this)">Startup (2.5M/day)</button>
        <button type="button" class="filter-pill" onclick="setWorkloadPreset(10000000, 2500000, this)">Scale-Up (12.5M/day)</button>
        <button type="button" class="filter-pill" onclick="setWorkloadPreset(50000000, 10000000, this)">Agent Fleet (60M/day)</button>
        <button type="button" class="filter-pill" onclick="setWorkloadPreset(5000000, 5000000, this)">Output-Heavy 50/50 (10M/day)</button>
      </div>
    </div>
    <div class="grid-2" style="margin:12px 0 6px;">
      <div class="slider-group" style="margin-bottom:4px;">
        <label><span>Daily Input Tokens</span> <span id="simInLabel" style="color:var(--accent)">2,000,000 / day</span></label>
        <input type="range" id="simInSlider" min="100000" max="100000000" step="100000" value="2000000" oninput="updateWorkloadSim()" />
      </div>
      <div class="slider-group" style="margin-bottom:4px;">
        <label><span>Daily Output Tokens</span> <span id="simOutLabel" style="color:var(--accent)">500,000 / day</span></label>
        <input type="range" id="simOutSlider" min="50000" max="40000000" step="50000" value="500000" oninput="updateWorkloadSim()" />
      </div>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-top:8px;">
      <small style="color:var(--muted);">💡 <span id="simMixLabel" style="color:var(--text);font-weight:600;">Current Mix: 80% input / 20% output</span> · Click any column header below (e.g. <strong>Est. Monthly</strong> or <strong>Input / 1M</strong>) to sort all models from cheapest to most expensive.</small>
      <button type="button" class="btn-ghost" onclick="copyIndexLink(this)">🔗 Copy Shareable Benchmark Link</button>
    </div>
  </div>

  <div class="affiliate-box">
    <div>
      <span class="badge badge-green">Featured Infrastructure Deals</span>
      <h3>Building an AI Product or Running Open-Weight LLMs?</h3>
      <p style="color:var(--muted);">Deploy dedicated RTX 4090, A100, and H100 GPUs from $0.20/hr with instant PyTorch, vLLM, and Ollama templates.</p>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;">
      <a href="{html.escape(aff_main['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta">{html.escape(aff_main['cta_text'])}</a>
      <a href="{html.escape(aff_sec['url'])}" target="_blank" rel="nofollow sponsored" class="btn-cta btn-green">{html.escape(aff_sec['cta_text'])}</a>
    </div>
  </div>
  {render_cli_card(config)}
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:14px;">
      <h2>Live AI Model Pricing &amp; VRAM Directory (Per 1M Tokens)</h2>
      <input type="text" id="searchInput" placeholder="Filter models (e.g. gpt-5, claude, deepseek, llama, qwen, mistral)..." style="max-width:380px;" oninput="filterTable()" />
    </div>
    <div class="filter-bar">
      <button type="button" class="filter-pill active" onclick="setQuickFilter('', this)">All ({len(display_models)})</button>
      <button type="button" class="filter-pill" onclick="setQuickFilter('OPEN_ONLY', this)">🟢 Open-Weight Only</button>
      <button type="button" class="filter-pill" onclick="setQuickFilter('openai', this)">OpenAI</button>
      <button type="button" class="filter-pill" onclick="setQuickFilter('anthropic', this)">Anthropic</button>
      <button type="button" class="filter-pill" onclick="setQuickFilter('google', this)">Google</button>
      <button type="button" class="filter-pill" onclick="setQuickFilter('deepseek', this)">DeepSeek</button>
      <button type="button" class="filter-pill" onclick="setQuickFilter('llama', this)">Meta Llama</button>
      <button type="button" class="filter-pill" onclick="setQuickFilter('qwen', this)">Qwen</button>
      <button type="button" class="filter-pill" onclick="setQuickFilter('mistral', this)">Mistral</button>
      <button type="button" class="filter-pill" onclick="setQuickFilter('grok', this)">xAI Grok</button>
    </div>
    <div class="table-wrap">
      <table id="pricingTable">
        <thead>
          <tr>
            <th class="sortable" onclick="sortTable('name')">Model Name ↕</th>
            <th class="sortable" onclick="sortTable('provider')">Provider ↕</th>
            <th class="sortable" onclick="sortTable('in')">Input / 1M ↕</th>
            <th class="sortable" onclick="sortTable('out')">Output / 1M ↕</th>
            <th class="sortable" onclick="sortTable('monthly')" style="color:var(--accent)">Est. Monthly ↕</th>
            <th class="sortable" onclick="sortTable('ctx')">Context ↕</th>
            <th class="sortable" onclick="sortTable('vram')">Q4 VRAM / Type ↕</th>
            <th>Calculator</th>
          </tr>
        </thead>
        <tbody id="modelTableBody">
          {table_rows}
        </tbody>
      </table>
    </div>
  </div>

  <section id="comparisons" style="margin-top:42px;">
    <h2 style="color:#fff;margin-bottom:10px;">Popular Head-to-Head Model Cost Comparisons</h2>
    <p style="color:var(--muted);margin-bottom:18px;">Click any pair below to view interactive side-by-side token calculators and monthly cost breakdowns.</p>
    <div class="grid-3">
      {comp_links}
    </div>
  </section>
</main>
<script>
  const CANONICAL_URL = {json.dumps(canonical)};
  let currentMode = '';
  let sortCol = '';
  let sortAsc = true;

  function setWorkloadPreset(inVal, outVal, btn) {{
    document.getElementById('simInSlider').value = inVal;
    document.getElementById('simOutSlider').value = outVal;
    if (btn) {{
      document.querySelectorAll('#workloadPresets .filter-pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    }}
    updateWorkloadSim();
  }}

  function updateWorkloadSim() {{
    const inTok = parseFloat(document.getElementById('simInSlider').value);
    const outTok = parseFloat(document.getElementById('simOutSlider').value);
    document.getElementById('simInLabel').textContent = inTok.toLocaleString() + ' / day';
    document.getElementById('simOutLabel').textContent = outTok.toLocaleString() + ' / day';
    const totalTok = Math.max(1, inTok + outTok);
    const inPct = Math.round((inTok / totalTok) * 100);
    const outPct = 100 - inPct;
    const mixEl = document.getElementById('simMixLabel');
    if (mixEl) mixEl.textContent = `Current Mix: ${{inPct}}% input / ${{outPct}}% output`;
    const rows = document.querySelectorAll('.model-row');
    rows.forEach(r => {{
      const inRate = parseFloat(r.getAttribute('data-in') || '0');
      const outRate = parseFloat(r.getAttribute('data-out') || '0');
      const gpuRate = parseFloat(r.getAttribute('data-gpu') || '999999');
      const monthly = (((inTok / 1e6) * inRate) + ((outTok / 1e6) * outRate)) * 30;
      r.setAttribute('data-monthly', monthly.toFixed(4));
      const cell = r.querySelector('.sim-cell');
      if (cell) {{
        const fmt = '$' + monthly.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}}) + '/mo';
        if (monthly > gpuRate) {{
          cell.innerHTML = '<strong>' + fmt + '</strong><br/><small style="color:var(--accent-green)">💡 GPU Cheaper</small>';
        }} else {{
          cell.innerHTML = '<strong>' + fmt + '</strong>';
        }}
      }}
    }});
    if (sortCol === 'monthly') sortTable('monthly', true);
  }}

  function sortTable(col, keepDir) {{
    if (!keepDir) {{
      if (sortCol === col) sortAsc = !sortAsc;
      else {{ sortCol = col; sortAsc = true; }}
    }}
    const tbody = document.getElementById('modelTableBody');
    const rows = Array.from(tbody.querySelectorAll('.model-row'));
    rows.sort((a, b) => {{
      let va, vb;
      if (col === 'name' || col === 'provider') {{
        va = (a.children[col === 'name' ? 0 : 1].textContent || '').toLowerCase();
        vb = (b.children[col === 'name' ? 0 : 1].textContent || '').toLowerCase();
        return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
      }} else {{
        va = parseFloat(a.getAttribute('data-' + col) || '0');
        vb = parseFloat(b.getAttribute('data-' + col) || '0');
        return sortAsc ? (va - vb) : (vb - va);
      }}
    }});
    rows.forEach(r => tbody.appendChild(r));
  }}

  function setQuickFilter(q, btn) {{
    document.querySelectorAll('.card .filter-bar .filter-pill').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const input = document.getElementById('searchInput');
    if (q === 'OPEN_ONLY') {{
      currentMode = 'OPEN_ONLY';
      input.value = '';
    }} else {{
      currentMode = '';
      input.value = q;
    }}
    filterTable();
  }}

  function filterTable() {{
    const q = document.getElementById('searchInput').value.toLowerCase().trim();
    const rows = document.querySelectorAll('.model-row');
    rows.forEach(r => {{
      const name = r.getAttribute('data-name') || '';
      const isOpen = r.getAttribute('data-open') === '1';
      const matchesOpen = (currentMode !== 'OPEN_ONLY') || isOpen;
      const matchesText = !q || name.includes(q);
      r.style.display = (matchesOpen && matchesText) ? '' : 'none';
    }});
  }}

  function copyIndexLink(btn) {{
    const inTok = document.getElementById('simInSlider').value;
    const outTok = document.getElementById('simOutSlider').value;
    const q = encodeURIComponent(document.getElementById('searchInput').value.trim());
    const url = `${{CANONICAL_URL}}?in=${{inTok}}&out=${{outTok}}${{q ? '&q=' + q : ''}}`;
    navigator.clipboard.writeText(url);
    const orig = btn.textContent;
    btn.textContent = '✅ Link Copied!';
    setTimeout(() => btn.textContent = orig, 2000);
  }}

  (function() {{
    const p = new URLSearchParams(window.location.search);
    if (p.get('in')) document.getElementById('simInSlider').value = p.get('in');
    if (p.get('out')) document.getElementById('simOutSlider').value = p.get('out');
    if (p.get('q')) document.getElementById('searchInput').value = p.get('q');
    updateWorkloadSim();
    filterTable();
  }})();
</script>
"""
    page += shared_footer(config, updated_str, prefix="")
    return page

def build_brand_assets(dist_dir):
    favicon_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="14" fill="#0b0f19"/>
  <path d="M36 6L14 36h16l-4 22 24-30H34z" fill="#38bdf8"/>
</svg>"""
    with open(os.path.join(dist_dir, "favicon.svg"), "w", encoding="utf-8") as f:
        f.write(favicon_svg)

    og_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0b0f19"/>
      <stop offset="100%" stop-color="#1e1b4b"/>
    </linearGradient>
  </defs>
  <rect width="1200" height="630" fill="url(#bg)"/>
  <rect x="60" y="60" width="1080" height="510" rx="24" fill="#131b2e" stroke="#38bdf8" stroke-width="2"/>
  <text x="110" y="165" font-family="Inter,Segoe UI,sans-serif" font-size="32" font-weight="700" fill="#38bdf8">⚡ TOKENBENCH &amp; GPU COST ENGINE</text>
  <text x="110" y="255" font-family="Inter,Segoe UI,sans-serif" font-size="56" font-weight="800" fill="#ffffff">AI API Token Pricing, VRAM</text>
  <text x="110" y="325" font-family="Inter,Segoe UI,sans-serif" font-size="56" font-weight="800" fill="#ffffff">&amp; Cloud GPU Break-Even Calculator</text>
  <text x="110" y="405" font-family="Inter,Segoe UI,sans-serif" font-size="28" fill="#94a3b8">Compare 440+ LLMs (GPT-5, Claude, Gemini, DeepSeek, Llama, Qwen, Mistral)</text>
  <rect x="110" y="455" width="360" height="64" rx="12" fill="#10b981"/>
  <text x="145" y="497" font-family="Inter,Segoe UI,sans-serif" font-size="26" font-weight="700" fill="#ffffff">1,500+ Live Calculators →</text>
</svg>"""
    with open(os.path.join(dist_dir, "og-card.svg"), "w", encoding="utf-8") as f:
        f.write(og_svg)

def submit_indexnow(config, urls):
    key = config.get("indexnow_key", "").strip()
    if not key:
        return
    base_url = config["base_url"].rstrip("/")
    host = base_url.split("://")[-1].split("/")[0]
    for i in range(0, len(urls), 1000):
        batch = urls[i:i+1000]
        payload = {
            "host": host,
            "key": key,
            "keyLocation": f"{base_url}/{key}.txt",
            "urlList": batch
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.indexnow.org/indexnow",
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                print(f"IndexNow ping batch {i//1000 + 1} succeeded with HTTP {resp.status} for {len(batch)} URLs!")
        except Exception as e:
            print(f"IndexNow ping note: {e}")

def main():
    config = load_config()
    updated_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print("Fetching live AI model pricing data...")
    all_models = fetch_models()
    max_models = int(config.get("max_individual_model_pages", 450))
    models = all_models[:max_models]
    print(f"Selected {len(models)} models ({sum(1 for m in models if m['is_open_weight'])} open-weight) for programmatic calculator pages.")

    # Clean old comparison files so only balanced cross-provider comparisons exist
    comp_dir = os.path.join(DIST_DIR, "compare")
    if os.path.exists(comp_dir):
        shutil.rmtree(comp_dir)

    os.makedirs(os.path.join(DIST_DIR, "models"), exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(os.path.join(DIST_DIR, "vram"), exist_ok=True)
    build_brand_assets(DIST_DIR)
    cli_src = os.path.join(BASE_DIR, "tokenbench.py")
    if os.path.exists(cli_src):
        shutil.copyfile(cli_src, os.path.join(DIST_DIR, "tokenbench.py"))

    urls = []
    base_url = config["base_url"].rstrip("/")

    # 1. Build individual model calculator pages
    for m in models:
        html_out = build_model_page(m, models, config, updated_str)
        out_path = os.path.join(DIST_DIR, "models", f"{m['slug']}.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_out)
        urls.append(f"{base_url}/models/{m['slug']}.html")

    # 2. Build VRAM calculator pages for all models so every model's VRAM link works
    max_vram = max(len(models), int(config.get("max_vram_pages", 450)))
    for m in models[:max_vram]:
        v_slug, v_html = build_vram_page(m, config, updated_str)
        v_path = os.path.join(DIST_DIR, "vram", f"{v_slug}.html")
        with open(v_path, "w", encoding="utf-8") as f:
            f.write(v_html)
        urls.append(f"{base_url}/vram/{v_slug}.html")

    # 3. Build comparison pages for top primary models (diverse across OpenAI, Anthropic, Google, DeepSeek, Llama, Qwen, Mistral, xAI)
    max_comp = int(config.get("max_comparison_pages", 650))
    primary_for_comp = [m for m in models if not m["is_secondary"]][:38]
    comparisons = []
    for i in range(len(primary_for_comp)):
        for j in range(i + 1, len(primary_for_comp)):
            if len(comparisons) >= max_comp:
                break
            m1, m2 = primary_for_comp[i], primary_for_comp[j]
            c_slug, c_html = build_comparison_page(m1, m2, config, updated_str)
            c_path = os.path.join(comp_dir, f"{c_slug}.html")
            with open(c_path, "w", encoding="utf-8") as f:
                f.write(c_html)
            comparisons.append((c_slug, m1, m2))
            urls.append(f"{base_url}/compare/{c_slug}.html")
        if len(comparisons) >= max_comp:
            break

    # 4. Build VRAM & Electricity Hub Page
    vram_hub_html = build_vram_hub_page(models, config, updated_str)
    with open(os.path.join(DIST_DIR, "vram-electricity-calculator.html"), "w", encoding="utf-8") as f:
        f.write(vram_hub_html)
    urls.insert(0, f"{base_url}/vram-electricity-calculator.html")

    # 5. Build Cloud GPU comparison page
    gpu_html = build_gpu_page(config, updated_str)
    with open(os.path.join(DIST_DIR, "gpu-cloud-pricing.html"), "w", encoding="utf-8") as f:
        f.write(gpu_html)
    urls.insert(0, f"{base_url}/gpu-cloud-pricing.html")

    # 5.5. Build Benchmarks & Editorial Guides Page + External Markdown Articles
    benchmarks_html = build_benchmarks_page(models, config, updated_str)
    with open(os.path.join(DIST_DIR, "benchmarks.html"), "w", encoding="utf-8") as f:
        f.write(benchmarks_html)
    urls.insert(0, f"{base_url}/benchmarks.html")
    build_external_articles(models, config, updated_str)

    # 6. Build Main Index Hub
    index_html = build_index_page(models, comparisons, config, updated_str, len(urls) + 1)
    with open(os.path.join(DIST_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)
    urls.insert(0, f"{base_url}/index.html")

    # 7. Build sitemap.xml & robots.txt & .nojekyll & IndexNow key file
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sitemap_entries = "\n".join(
        f"  <url><loc>{html.escape(u)}</loc><lastmod>{today}</lastmod><changefreq>daily</changefreq></url>"
        for u in urls
    )
    sitemap_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{sitemap_entries}
</urlset>
"""
    with open(os.path.join(DIST_DIR, "sitemap.xml"), "w", encoding="utf-8") as f:
        f.write(sitemap_xml)

    robots_txt = f"User-agent: *\nAllow: /\nSitemap: {base_url}/sitemap.xml\n"
    with open(os.path.join(DIST_DIR, "robots.txt"), "w", encoding="utf-8") as f:
        f.write(robots_txt)

    with open(os.path.join(DIST_DIR, ".nojekyll"), "w", encoding="utf-8") as f:
        f.write("")

    indexnow_key = config.get("indexnow_key", "").strip()
    if indexnow_key:
        with open(os.path.join(DIST_DIR, f"{indexnow_key}.txt"), "w", encoding="utf-8") as f:
            f.write(indexnow_key)

    print(f"Successfully generated {len(urls)} total SEO pages in {DIST_DIR}")
    if "--ping-indexnow" in sys.argv:
        submit_indexnow(config, urls)

if __name__ == "__main__":
    main()
