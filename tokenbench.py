#!/usr/bin/env python3
"""
TokenBench CLI — Local GPU Hardware Checker & LLM VRAM / Deploy Command Advisor.

Zero-dependency Python 3 utility that detects your local GPU VRAM (NVIDIA via nvidia-smi,
Apple Silicon Unified Memory, or Windows/Linux GPU fallback) and shows which open-weight LLMs
fit in FP16, 8-bit (Q8_0), or 4-bit (Q4_K_M / AWQ) quantization at your target context length,
along with ready-to-run Ollama and vLLM commands.

Usage:
  python tokenbench.py
  python tokenbench.py --vram 24 --ctx 32768 --search qwen
  python tokenbench.py --live --search llama
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import urllib.request

SITE_BASE_URL = "https://stigla610.github.io/ai-cost-engine"

CURATED_OPEN_MODELS = [
    {"name": "Llama 3.2 1B", "params_b": 1.2, "ollama": "llama3.2:1b", "hf": "meta-llama/Llama-3.2-1B-Instruct", "ctx": 131072},
    {"name": "Llama 3.2 3B", "params_b": 3.2, "ollama": "llama3.2:3b", "hf": "meta-llama/Llama-3.2-3B-Instruct", "ctx": 131072},
    {"name": "Gemma 3 4B", "params_b": 4.0, "ollama": "gemma3:4b", "hf": "google/gemma-3-4b-it", "ctx": 131072},
    {"name": "Llama 3.1 8B Instruct", "params_b": 8.0, "ollama": "llama3.1:8b", "hf": "meta-llama/Llama-3.1-8B-Instruct", "ctx": 131072},
    {"name": "Qwen3 8B", "params_b": 8.0, "ollama": "qwen3:8b", "hf": "Qwen/Qwen3-8B", "ctx": 131072},
    {"name": "Mistral Nemo 12B", "params_b": 12.2, "ollama": "mistral-nemo:12b", "hf": "mistralai/Mistral-Nemo-Instruct-2407", "ctx": 131072},
    {"name": "Gemma 3 12B", "params_b": 12.0, "ollama": "gemma3:12b", "hf": "google/gemma-3-12b-it", "ctx": 131072},
    {"name": "Microsoft Phi-4 14B", "params_b": 14.0, "ollama": "phi4:14b", "hf": "microsoft/phi-4", "ctx": 16384},
    {"name": "Qwen3 14B", "params_b": 14.0, "ollama": "qwen3:14b", "hf": "Qwen/Qwen3-14B", "ctx": 131072},
    {"name": "OpenAI gpt-oss-20b", "params_b": 20.0, "ollama": "gpt-oss:20b", "hf": "openai/gpt-oss-20b", "ctx": 131072},
    {"name": "Codestral 22B", "params_b": 22.0, "ollama": "codestral:22b", "hf": "mistralai/Codestral-22B-v0.1", "ctx": 32768},
    {"name": "Mistral Small 3.2 24B", "params_b": 24.0, "ollama": "mistral-small3.2:24b", "hf": "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "ctx": 131072},
    {"name": "Gemma 4 26B A4B", "params_b": 26.0, "ollama": "gemma4:26b", "hf": "google/gemma-4-26b-a4b-it", "ctx": 131072},
    {"name": "Gemma 3 27B", "params_b": 27.0, "ollama": "gemma3:27b", "hf": "google/gemma-3-27b-it", "ctx": 131072},
    {"name": "Qwen3 30B A3B (MoE)", "params_b": 30.0, "ollama": "qwen3:30b-a3b", "hf": "Qwen/Qwen3-30B-A3B", "ctx": 131072},
    {"name": "Gemma 4 31B", "params_b": 31.0, "ollama": "gemma4:31b", "hf": "google/gemma-4-31b-it", "ctx": 131072},
    {"name": "Qwen3 32B", "params_b": 32.0, "ollama": "qwen3:32b", "hf": "Qwen/Qwen3-32B", "ctx": 131072},
    {"name": "Qwen2.5 Coder 32B", "params_b": 32.0, "ollama": "qwen2.5-coder:32b", "hf": "Qwen/Qwen2.5-Coder-32B-Instruct", "ctx": 32768},
    {"name": "Llama 3.3 70B Instruct", "params_b": 70.0, "ollama": "llama3.3:70b", "hf": "meta-llama/Llama-3.3-70B-Instruct", "ctx": 131072},
    {"name": "Qwen2.5 72B Instruct", "params_b": 72.0, "ollama": "qwen2.5:72b", "hf": "Qwen/Qwen2.5-72B-Instruct", "ctx": 32768},
    {"name": "Llama 4 Scout (109B MoE)", "params_b": 109.0, "ollama": "llama4:scout", "hf": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "ctx": 131072},
    {"name": "OpenAI gpt-oss-120b", "params_b": 120.0, "ollama": "gpt-oss:120b", "hf": "openai/gpt-oss-120b", "ctx": 131072},
    {"name": "Mistral Large 123B", "params_b": 123.0, "ollama": "mistral-large:123b", "hf": "mistralai/Mistral-Large-Instruct-2411", "ctx": 131072},
    {"name": "Qwen3 235B A22B (MoE)", "params_b": 235.0, "ollama": "qwen3:235b-a22b", "hf": "Qwen/Qwen3-235B-A22B", "ctx": 131072},
    {"name": "Llama 4 Maverick (400B MoE)", "params_b": 400.0, "ollama": "llama4:maverick", "hf": "meta-llama/Llama-4-Maverick-17B-128E-Instruct", "ctx": 131072},
    {"name": "DeepSeek V3 (671B MoE)", "params_b": 671.0, "ollama": "deepseek-v3:671b", "hf": "deepseek-ai/DeepSeek-V3", "ctx": 163840},
    {"name": "DeepSeek R1 (671B MoE)", "params_b": 671.0, "ollama": "deepseek-r1:671b", "hf": "deepseek-ai/DeepSeek-R1-0528", "ctx": 163840},
]


def _configure_stdio():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def detect_hardware():
    """Detect local GPU VRAM (NVIDIA, Apple Silicon Unified Memory, or Windows GPU fallback)."""
    # 1. Try nvidia-smi (Linux / Windows with NVIDIA driver)
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            gpus = []
            total_mb = 0.0
            for line in proc.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    try:
                        mb = float(parts[1])
                        gpus.append(f"{parts[0]} ({mb / 1024.0:.0f} GB)")
                        total_mb += mb
                    except ValueError:
                        continue
            if total_mb > 0:
                return round(total_mb / 1024.0, 1), f"NVIDIA GPU: {' + '.join(gpus)}", "nvidia-smi"
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    # 2. Apple Silicon / macOS unified memory
    if sys.platform == "darwin":
        try:
            mem_out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=3).strip()
            arch = platform.machine().lower()
            total_gb = int(mem_out) / (1024 ** 3)
            if "arm" in arch or "aarch64" in arch:
                usable = round(total_gb * 0.75, 1)
                return usable, f"Apple Silicon Unified Memory ({total_gb:.0f} GB RAM -> ~{usable} GB usable Metal VRAM)", "apple-sysctl"
            return round(total_gb * 0.5, 1), f"macOS System Memory ({total_gb:.0f} GB RAM)", "mac-sysctl"
        except Exception:
            pass

    # 3. Windows GPU fallback via PowerShell CIM + Registry qwMemorySize for >4GB GPUs
    if sys.platform == "win32":
        ps_cmd = (
            "$gpus = Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM; "
            "$reg = Get-ItemProperty 'HKLM:\\SYSTEM\\ControlSet001\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' "
            "-ErrorAction SilentlyContinue | Select-Object DriverDesc,'HardwareInformation.qwMemorySize'; "
            "@{gpus=@($gpus); reg=@($reg)} | ConvertTo-Json -Depth 3"
        )
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=6,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                data = json.loads(proc.stdout)
                best_bytes = 0
                best_name = None
                for r in data.get("reg") or []:
                    qw = r.get("HardwareInformation.qwMemorySize")
                    desc = r.get("DriverDesc")
                    if isinstance(qw, (int, float)) and qw > best_bytes:
                        best_bytes = qw
                        best_name = desc
                if best_bytes == 0:
                    for g in data.get("gpus") or []:
                        ram = g.get("AdapterRAM")
                        name = g.get("Name")
                        if isinstance(ram, (int, float)) and ram > best_bytes:
                            best_bytes = ram
                            best_name = name
                if best_bytes > 0 and best_name:
                    gb = round(best_bytes / (1024 ** 3), 1)
                    return gb, f"Windows GPU: {best_name} ({gb} GB detected)", "win32-cim"
        except Exception:
            pass

    return 16.0, "Default fallback (16.0 GB assumed — pass --vram <GB> to specify yours)", "fallback"


def calc_vram(params_b, ctx):
    kv_extra = max(0.0, (ctx - 8192) / 8192.0) * (params_b * 0.025)
    fp16 = round((params_b * 2.2) + (kv_extra * 2.0), 1)
    q8 = round((params_b * 1.15) + (kv_extra * 1.2), 1)
    q4 = round((params_b * 0.65) + kv_extra, 1)
    return fp16, q8, q4


def _estimate_params_b(model_id, name):
    combined = f"{model_id} {name}".lower()
    m_moe = re.search(r"\b(\d+)x(\d+(?:\.\d+)?)b\b", combined)
    if m_moe:
        return round(float(m_moe.group(1)) * float(m_moe.group(2)) * 0.8, 1)
    m_tril = re.search(r"\b(\d+(?:\.\d+)?)t\b", combined)
    if m_tril:
        val_t = float(m_tril.group(1))
        if 0.5 <= val_t <= 10:
            return round(val_t * 1000.0, 1)
    m_bil = re.search(r"(?<![a-z0-9])(\d+(?:\.\d+)?)b\b", combined)
    if m_bil:
        val = float(m_bil.group(1))
        if 0.5 <= val <= 2000:
            return val
    if "deepseek" in combined:
        if re.search(r"\b(flash|lite|mini|nano|small)\b", combined):
            return 14.0
        if any(k in combined for k in ["v3", "r1", "chat", "reasoner", "prover", "v4"]):
            return 671.0
    if "phi-4" in combined or "phi-3" in combined:
        return 3.8 if re.search(r"\b(mini|small|multimodal)\b", combined) else 14.0
    if "llama-4-maverick" in combined:
        return 400.0
    if "llama-4-scout" in combined:
        return 109.0
    if "mistral-large" in combined or "pixtral-large" in combined:
        return 123.0
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
    return 32.0


def _is_open_weight(model_id, name):
    mid = model_id.lower().lstrip("~")
    combined = f"{mid} {name.lower()}"
    if any(k in mid for k in [":batch", "-image", "-audio", "-realtime"]):
        return False
    if "gpt-oss" in mid:
        return True
    if mid.startswith(("openai/", "anthropic/", "x-ai/", "amazon/", "perplexity/", "ai21/", "inflection/", "openrouter/")):
        return False
    if mid.startswith("google/"):
        return "gemma" in mid
    if mid.startswith("meta/"):
        return "open" in mid or "30b" in mid or "llama" in mid
    if mid.startswith("meta-llama/"):
        return True
    if mid.startswith("qwen/"):
        if re.search(r"\b\d+(?:\.\d+)?[bt]\b", combined) or "qwq" in mid or "qwen-2.5" in mid or "qwen2.5" in mid:
            return True
        if any(k in mid for k in ["max", "plus", "turbo", "flash"]):
            return False
        return True
    if mid.startswith("mistralai/"):
        return "mistral-medium" not in mid
    if mid.startswith("deepseek/"):
        return not (model_id.startswith("~") or "flash" in mid or "v4-pro" in mid or "pro-latest" in mid)
    if mid.startswith((
        "microsoft/phi", "microsoft/wizard", "nvidia/", "nousresearch/",
        "cognitivecomputations/", "sao10k/", "thedrummer/", "gryphe/",
        "allenai/", "ibm-granite/", "liquid/", "01-ai/", "moonshotai/kimi",
        "z-ai/glm", "thudm/glm", "cohere/command-r", "cohere/command-a"
    )):
        return True
    return bool(re.search(r"\b\d+(?:\.\d+)?b\b", combined))


def _infer_commands(model_id, name, params_b):
    mid = model_id.lower()
    # Map common OpenRouter IDs cleanly to Ollama tags & HF repos
    if mid.startswith("meta-llama/"):
        if "llama-4-maverick" in mid:
            return "llama4:maverick", "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
        if "llama-4-scout" in mid:
            return "llama4:scout", "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        m = re.search(r"llama-(\d+(?:\.\d+)?)-(\d+b)", mid)
        if m:
            ver, size = m.group(1), m.group(2)
            return f"llama{ver}:{size}", f"meta-llama/Llama-{ver}-{size.upper()}-Instruct"
    if mid.startswith("qwen/"):
        if "coder" in mid and "32b" in mid:
            return "qwen2.5-coder:32b", "Qwen/Qwen2.5-Coder-32B-Instruct"
        if "2.5" in mid and "72b" in mid:
            return "qwen2.5:72b", "Qwen/Qwen2.5-72B-Instruct"
        m = re.search(r"qwen(\d+(?:\.\d+)?)-(\d+b)(?:-(a\d+b))?", mid)
        if m:
            ver, size, act = m.group(1), m.group(2), m.group(3)
            tag = f"qwen{ver}:{size}" + (f"-{act}" if act else "")
            hf = f"Qwen/Qwen{ver}-{size.upper()}" + (f"-{act.upper()}" if act else "")
            return tag, hf
    if mid.startswith("deepseek/"):
        if "r1" in mid:
            return f"deepseek-r1:{int(params_b)}b", "deepseek-ai/DeepSeek-R1-0528"
        if "chat" in mid or "v3" in mid:
            return "deepseek-v3:671b", "deepseek-ai/DeepSeek-V3"
    if mid.startswith("google/gemma"):
        m = re.search(r"gemma-(\d+)-(\d+b)", mid)
        if m:
            return f"gemma{m.group(1)}:{m.group(2)}", f"google/gemma-{m.group(1)}-{m.group(2)}-it"
    if mid.startswith("microsoft/phi-4"):
        return "phi4:14b", "microsoft/phi-4"
    if mid.startswith("openai/gpt-oss-"):
        m = re.search(r"gpt-oss-(\d+b)", mid)
        if m:
            return f"gpt-oss:{m.group(1)}", f"openai/gpt-oss-{m.group(1)}"
    if mid.startswith("mistralai/"):
        if "codestral" in mid:
            return "codestral:22b", "mistralai/Codestral-22B-v0.1"
        if "mistral-small" in mid:
            return "mistral-small3.2:24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
        if "mistral-large" in mid:
            return "mistral-large:123b", "mistralai/Mistral-Large-Instruct-2411"
        if "mixtral-8x22b" in mid:
            return "mixtral:8x22b", "mistralai/Mixtral-8x22B-Instruct-v0.1"
    slug_tail = model_id.split("/")[-1]
    return slug_tail, model_id


def fetch_live_open_models():
    url = "https://openrouter.ai/api/v1/models"
    req = urllib.request.Request(url, headers={"User-Agent": "TokenBench-CLI/2.1"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    results = []
    seen = set()
    for item in raw.get("data", []):
        mid = item.get("id", "")
        name = item.get("name", mid).lstrip("~").strip()
        if not mid or mid in seen or mid.startswith("~"):
            continue
        if not _is_open_weight(mid, name):
            continue
        pricing = item.get("pricing") or {}
        try:
            in_1m = round(float(pricing.get("prompt", 0) or 0) * 1_000_000, 4)
            out_1m = round(float(pricing.get("completion", 0) or 0) * 1_000_000, 4)
        except (ValueError, TypeError):
            in_1m, out_1m = 0.0, 0.0
        if in_1m == 0 and out_1m == 0 and ":free" not in mid:
            continue
        seen.add(mid)
        params_b = _estimate_params_b(mid, name)
        ollama_tag, hf_repo = _infer_commands(mid, name, params_b)
        results.append({
            "id": mid,
            "name": name,
            "params_b": params_b,
            "ollama": ollama_tag,
            "hf": hf_repo,
            "ctx": int(item.get("context_length") or 32768),
            "in_1m": in_1m,
            "out_1m": out_1m,
        })
    results.sort(key=lambda x: (x["params_b"], x["name"]))
    return results


def evaluate_models(models, vram_gb, ctx, search_term=""):
    q = search_term.lower().strip()
    evaluated = []
    for m in models:
        if q and q not in m["name"].lower() and q not in m.get("id", "").lower() and q not in m["ollama"].lower():
            continue
        fp16, q8, q4 = calc_vram(m["params_b"], ctx)
        if vram_gb >= fp16:
            status = "FP16"
            badge = "✅ FP16"
        elif vram_gb >= q8:
            status = "Q8_0"
            badge = "🟢 Q8_0"
        elif vram_gb >= q4:
            status = "Q4_K_M"
            badge = "⚡ Q4_K_M"
        else:
            status = "CLOUD"
            badge = "☁️ Cloud GPU"

        tp = 1 if q4 <= 75 else (2 if q4 <= 155 else (4 if q4 <= 310 else 8))
        ollama_cmd = f"ollama run {m['ollama']}"
        vllm_cmd = f"vllm serve {m['hf']} --max-model-len {min(ctx, m.get('ctx', ctx))} --tensor-parallel-size {tp}"
        evaluated.append({
            **m,
            "vram_fp16": fp16,
            "vram_q8": q8,
            "vram_q4": q4,
            "status": status,
            "badge": badge,
            "ollama_cmd": ollama_cmd,
            "vllm_cmd": vllm_cmd,
        })
    return evaluated


def main():
    _configure_stdio()
    parser = argparse.ArgumentParser(
        prog="tokenbench",
        description="Detect local GPU VRAM and check which open-weight LLMs fit in FP16 / Q8 / Q4.",
    )
    parser.add_argument("--vram", type=float, default=None, help="Override detected VRAM in GB (e.g. --vram 24)")
    parser.add_argument("--ctx", type=int, default=8192, help="Context window tokens for KV cache sizing (default: 8192)")
    parser.add_argument("--search", type=str, default="", help="Filter models by name/family (e.g. qwen, llama, deepseek)")
    parser.add_argument("--live", action="store_true", help="Fetch live 130+ open-weight models from OpenRouter API")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of formatted table")
    args = parser.parse_args()

    detected_vram, hw_desc, source = detect_hardware()
    vram_gb = float(args.vram) if args.vram is not None else detected_vram
    if args.vram is not None:
        hw_desc = f"Manual Override ({vram_gb:g} GB VRAM) [detected: {hw_desc}]"

    models = CURATED_OPEN_MODELS
    data_source = "curated"
    if args.live:
        try:
            models = fetch_live_open_models()
            data_source = "openrouter-live"
        except Exception as e:
            if not args.json:
                print(f"[note] Live fetch failed ({e}); using built-in model catalog.", file=sys.stderr)

    results = evaluate_models(models, vram_gb, args.ctx, args.search)

    if args.json:
        print(json.dumps({
            "vram_gb": vram_gb,
            "hardware": hw_desc,
            "detection_source": source,
            "context_tokens": args.ctx,
            "catalog_source": data_source,
            "models": results,
        }, indent=2))
        return 0

    print("")
    print("⚡ TokenBench Local GPU & LLM Compatibility Checker")
    print("=" * 78)
    print(f"  Hardware : {hw_desc}")
    print(f"  Usable   : {vram_gb:g} GB VRAM   |   Context: {args.ctx:,} tokens   |   Catalog: {data_source}")
    print("=" * 78)
    print("")
    print(f"{'Model':<29} {'Params':>7} {'Q4':>8} {'Q8':>8} {'FP16':>8}  {'Fit':<12} {'Run Command'}")
    print("-" * 78)

    fit_count = 0
    for r in results:
        if r["status"] != "CLOUD":
            fit_count += 1
            cmd = r["ollama_cmd"]
        else:
            cmd = f"Need >= {r['vram_q4']} GB (Cloud GPU)"
        print(
            f"{r['name'][:28]:<29} "
            f"{r['params_b']:>6g}B "
            f"{r['vram_q4']:>6.1f}G "
            f"{r['vram_q8']:>6.1f}G "
            f"{r['vram_fp16']:>6.1f}G  "
            f"{r['badge']:<12} {cmd}"
        )

    runnable = [r for r in results if r["status"] != "CLOUD"]
    if runnable:
        preferred = [r for r in runnable if not any(k in r["name"].lower() for k in ("guard", "safeguard"))]
        top = (preferred or runnable)[-1]
        print("")
        print(f"💡 Best local pick for {vram_gb:g} GB VRAM: {top['name']} ({top['badge']})")
        print(f"   Ollama : {top['ollama_cmd']}")
        print(f"   vLLM   : {top['vllm_cmd']}")

    print("")
    print(f"Summary: {fit_count}/{len(results)} models fit locally on {vram_gb:g} GB VRAM @ {args.ctx:,} context.")
    print(f"📊 Full API Cost & Electricity Break-Even: {SITE_BASE_URL}/vram-electricity-calculator.html")
    print(f"🚀 Batch Simulator (440+ Models):         {SITE_BASE_URL}/")
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
