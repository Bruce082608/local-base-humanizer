#!/usr/bin/env python3
"""
Pluggable Inference Backends for Local Base-Model Humanizer Studio.

Provides two interoperable backends:
  1. OllamaBackend (default, ZERO external dependencies):
     - Thread-local persistent HTTP/HTTPS keep-alive client against Ollama
       `/api/generate` & `/api/tags`, eliminating per-token TCP reconnect overhead.
     - Automatically infers model family (qwen3, llama3, mistral, gemma, etc.)
       and approximate parameter scale (in billions) to detect cross-family
       surrogates and Base/Instruct scale mismatches.
  2. LlamaCppBackend (optional, requires `llama-cpp-python` + `numpy`):
     - Direct in-process GGUF loading with incremental KV-cache decoding and
       EXACT full-vocabulary logits (`log P_base(t) - alpha * log P_instruct(t)`)
       — zero Top-20 truncation cliff, 5-15x higher throughput for Best-of-N!
     - Single-pass EXACT unconditional Binoculars scoring
       (Hans et al., ICML 2024: mean_NLL_observer / mean_XENT(performer, observer)),
       exact unconditional PPL, exact GLTR ranks, and exact per-sentence Burstiness.
     - Can also discover GGUF blobs already downloaded by Ollama (read-only) so
       users do not have to download weights twice.
"""

import atexit
import ctypes
import http.client
import inspect
import json
import math
import os
import re
import socket
import threading
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_NUM_CTX = 2048

# Lazy-imported optional modules
_LLAMA_CPP_CHECKED = False
_LLAMA_CPP_AVAILABLE = False
_LLAMA_CPP_VERSION = ""
_NUMPY_AVAILABLE = False


def _probe_llama_cpp() -> Tuple[bool, str]:
    global _LLAMA_CPP_CHECKED, _LLAMA_CPP_AVAILABLE, _LLAMA_CPP_VERSION, _NUMPY_AVAILABLE
    if _LLAMA_CPP_CHECKED:
        return _LLAMA_CPP_AVAILABLE, _LLAMA_CPP_VERSION
    _LLAMA_CPP_CHECKED = True
    try:
        import numpy  # noqa: F401
        _NUMPY_AVAILABLE = True
    except Exception:
        _NUMPY_AVAILABLE = False
    try:
        import llama_cpp
        _LLAMA_CPP_AVAILABLE = _NUMPY_AVAILABLE
        _LLAMA_CPP_VERSION = getattr(llama_cpp, "__version__", "installed")
    except Exception:
        _LLAMA_CPP_AVAILABLE = False
        _LLAMA_CPP_VERSION = ""
    return _LLAMA_CPP_AVAILABLE, _LLAMA_CPP_VERSION


def get_backend_capabilities() -> Dict[str, Any]:
    """Report which optional local backends/detectors are available in this Python env."""
    lc_ok, lc_ver = _probe_llama_cpp()
    neural_ok = False
    neural_device = "cpu"
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        neural_ok = True
        if torch.cuda.is_available():
            neural_device = "cuda"
    except Exception:
        neural_ok = False

    return {
        "ollama_available": True,
        "llama_cpp_available": lc_ok,
        "llama_cpp_version": lc_ver,
        "neural_detector_available": neural_ok,
        "neural_default_device": "cpu",  # Default CPU to preserve GPU VRAM for 8B/4B GGUFs
        "cuda_available": neural_device == "cuda",
    }


def normalize_ollama_base(url: str) -> str:
    u = (url or DEFAULT_OLLAMA_BASE).strip().rstrip("/")
    if u.endswith("/api/generate"):
        u = u[: -len("/api/generate")]
    elif u.endswith("/api/tags"):
        u = u[: -len("/api/tags")]
    return u


def infer_model_family(model_name: str) -> str:
    """Infer coarse model architecture/tokenizer family from an Ollama or GGUF name."""
    m = (model_name or "").lower()
    if "qwen3" in m or "qwen-3" in m:
        return "qwen3"
    if "qwen2.5" in m or "qwen2" in m or "qwen" in m:
        return "qwen2"
    if "llama-3" in m or "llama3" in m:
        return "llama3"
    if "llama-2" in m or "llama2" in m:
        return "llama2"
    if "llama" in m:
        return "llama"
    if "mistral" in m or "mixtral" in m:
        return "mistral"
    if "gemma-3" in m or "gemma3" in m:
        return "gemma3"
    if "gemma-2" in m or "gemma2" in m:
        return "gemma2"
    if "gemma" in m:
        return "gemma"
    if "deepseek" in m:
        return "deepseek"
    if "falcon" in m:
        return "falcon"
    if "phi-4" in m or "phi4" in m or "phi-3" in m or "phi3" in m or "phi" in m:
        return "phi"
    if "yi-" in m or "yi:" in m:
        return "yi"
    if "olmo" in m:
        return "olmo"
    return "other"


def infer_param_billions(model_name: str, size_bytes: int = 0) -> Optional[float]:
    """Estimate parameter scale (in billions) from model name or quantized byte size."""
    m = re.search(r"(?:^|[^a-z0-9])(\d+(?:\.\d+)?)\s*b(?:[^a-z0-9]|$)", (model_name or "").lower())
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    if size_bytes and size_bytes > 0:
        # Rough Q4_K_M heuristic (~0.62 GB per billion params)
        return round((size_bytes / 1e9) / 0.62, 1)
    return None


# ---------------------------------------------------------------------------
# Zero-Dependency Persistent Keep-Alive Ollama HTTP Client
# ---------------------------------------------------------------------------

class OllamaClient:
    """Thread-local HTTP/1.1 keep-alive client (with TCP_NODELAY) for Ollama REST endpoints."""

    def __init__(self):
        self._local = threading.local()
        self._all_lock = threading.Lock()
        self._all_conns: List[http.client.HTTPConnection] = []

    def _get_conn(self, scheme: str, host: str, port: int, timeout: float) -> http.client.HTTPConnection:
        conns: Dict[Tuple[str, str, int], http.client.HTTPConnection] = getattr(self._local, "conns", None)
        if conns is None:
            conns = {}
            self._local.conns = conns
        key = (scheme, host, port)
        conn = conns.get(key)
        if conn is None:
            if scheme == "https":
                conn = http.client.HTTPSConnection(host, port, timeout=timeout)
            else:
                conn = http.client.HTTPConnection(host, port, timeout=timeout)
            conns[key] = conn
            with self._all_lock:
                self._all_conns.append(conn)
        else:
            conn.timeout = timeout
            if conn.sock is not None:
                try:
                    conn.sock.settimeout(timeout)
                except OSError:
                    pass
        if conn.sock is None:
            conn.connect()
            try:
                conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
        return conn

    def _drop_conn(self, scheme: str, host: str, port: int) -> None:
        conns = getattr(self._local, "conns", None)
        if not conns:
            return
        conn = conns.pop((scheme, host, port), None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            with self._all_lock:
                try:
                    self._all_conns.remove(conn)
                except ValueError:
                    pass

    def close_all(self) -> None:
        """Close all open persistent connections cleanly."""
        with self._all_lock:
            conns = list(self._all_conns)
            self._all_conns.clear()
        for c in conns:
            try:
                c.close()
            except Exception:
                pass
        local_conns = getattr(self._local, "conns", None)
        if isinstance(local_conns, dict):
            local_conns.clear()

    def request_json(
        self,
        ollama_base: str,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        base = normalize_ollama_base(ollama_base)
        parsed = urllib.parse.urlparse(base)
        scheme = parsed.scheme or "http"
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if scheme == "https" else 11434)
        full_path = (parsed.path.rstrip("/") + path) if parsed.path else path

        body_bytes = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Connection": "keep-alive"}
        if body_bytes is not None:
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body_bytes))

        last_exc: Optional[Exception] = None
        for attempt in range(2):
            conn = self._get_conn(scheme, host, port, timeout)
            try:
                conn.request(method, full_path, body=body_bytes, headers=headers)
                resp = conn.getresponse()
                raw = resp.read()
                if resp.status >= 400:
                    raise RuntimeError(f"Ollama HTTP {resp.status}: {raw.decode('utf-8', errors='replace')[:300]}")
                return json.loads(raw.decode("utf-8"))
            except (
                http.client.RemoteDisconnected,
                http.client.BadStatusLine,
                http.client.CannotSendRequest,
                http.client.ResponseNotReady,
                BrokenPipeError,
                ConnectionResetError,
                ConnectionAbortedError,
            ) as exc:
                self._drop_conn(scheme, host, port)
                last_exc = exc
                if attempt == 0:
                    continue
                raise
            except Exception:
                self._drop_conn(scheme, host, port)
                raise
        raise last_exc or RuntimeError("Ollama request failed")

    def list_models(self, ollama_base: str = DEFAULT_OLLAMA_BASE, timeout: float = 5.0) -> Tuple[bool, List[Dict], str]:
        try:
            data = self.request_json(ollama_base, "GET", "/api/tags", timeout=timeout)
            models = data.get("models", [])
            result = []
            for m in models:
                name = m.get("name", "")
                if not name:
                    continue
                size = m.get("size", 0) or 0
                result.append({
                    "name": name,
                    "size": size,
                    "is_base": "base" in name.lower(),
                    "family": infer_model_family(name),
                    "param_b": infer_param_billions(name, size),
                })
            result.sort(key=lambda x: (not x["is_base"], -x["size"] if x["is_base"] else 0, x["name"]))
            return True, result, ""
        except Exception as exc:
            return False, [], str(exc)

    def generate_raw(
        self,
        ollama_base: str,
        model: str,
        raw_prompt: str,
        *,
        num_predict: int = 240,
        temperature: float = 0.0,
        num_ctx: int = DEFAULT_NUM_CTX,
        timeout: float = 90.0,
        keep_alive: str = "10m",
    ) -> str:
        payload = {
            "model": model,
            "prompt": raw_prompt,
            "raw": True,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {
                "num_predict": num_predict,
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        }
        d = self.request_json(ollama_base, "POST", "/api/generate", payload=payload, timeout=timeout)
        return (d.get("response") or "").strip()

    def next_token_logprobs(
        self,
        ollama_base: str,
        model: str,
        raw_prompt: str,
        top_k: int = 20,
        timeout: float = 60.0,
        keep_alive: str = "10m",
        num_ctx: int = DEFAULT_NUM_CTX,
    ) -> Tuple[Optional[str], List[Tuple[str, float]]]:
        payload = {
            "model": model,
            "prompt": raw_prompt,
            "raw": True,
            "stream": False,
            "keep_alive": keep_alive,
            "logprobs": True,
            "top_logprobs": top_k,
            "options": {"num_predict": 1, "temperature": 0.0, "num_ctx": num_ctx},
        }
        d = self.request_json(ollama_base, "POST", "/api/generate", payload=payload, timeout=timeout)
        lps = d.get("logprobs", [])
        if not lps:
            return None, []
        top = lps[0].get("top_logprobs", [])
        return lps[0].get("token", ""), [(x["token"], x["logprob"]) for x in top]


OLLAMA_CLIENT = OllamaClient()
atexit.register(OLLAMA_CLIENT.close_all)


# ---------------------------------------------------------------------------
# Local GGUF Discovery (Standalone files + Ollama downloaded blobs, read-only)
# ---------------------------------------------------------------------------

def discover_local_ggufs(extra_dirs: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """
    Discover local GGUF files from:
      1. Optional user directories / project `./models` folder (*.gguf)
      2. Standard user Ollama manifests directory (read-only), mapping model names
         to their underlying local GGUF blob paths so users need not re-download.
    """
    found: List[Dict[str, Any]] = []
    seen_paths: set = set()

    # 1. Explicit *.gguf files in local directories
    search_dirs: List[Path] = []
    if extra_dirs:
        search_dirs.extend(Path(d) for d in extra_dirs if d)
    search_dirs.append(Path(__file__).resolve().parent / "models")

    for d in search_dirs:
        if not d.is_dir():
            continue
        try:
            for fp in sorted(d.rglob("*.gguf")):
                ap = str(fp.resolve())
                if ap in seen_paths:
                    continue
                seen_paths.add(ap)
                size = fp.stat().st_size
                name = fp.name
                found.append({
                    "label": name,
                    "path": ap,
                    "size": size,
                    "is_base": "base" in name.lower(),
                    "family": infer_model_family(name),
                    "param_b": infer_param_billions(name, size),
                    "source": "file",
                })
        except OSError:
            continue

    # 2. Ollama local manifests -> blob GGUF files (read-only)
    ollama_roots: List[Path] = []
    env_models = os.environ.get("OLLAMA_MODELS")
    if env_models:
        ollama_roots.append(Path(env_models))
    ollama_roots.append(Path.home() / ".ollama" / "models")

    for root in ollama_roots:
        manifests_dir = root / "manifests"
        blobs_dir = root / "blobs"
        if not manifests_dir.is_dir() or not blobs_dir.is_dir():
            continue
        try:
            for mf in manifests_dir.rglob("*"):
                if not mf.is_file():
                    continue
                try:
                    data = json.loads(mf.read_text(encoding="utf-8"))
                except Exception:
                    continue
                layers = data.get("layers") or []
                model_digest = None
                for layer in layers:
                    if layer.get("mediaType") == "application/vnd.ollama.image.model":
                        model_digest = layer.get("digest")
                        break
                if not model_digest or not model_digest.startswith("sha256:"):
                    continue
                blob_path = blobs_dir / model_digest.replace(":", "-")
                if not blob_path.is_file():
                    continue
                ap = str(blob_path.resolve())
                if ap in seen_paths:
                    continue
                seen_paths.add(ap)
                rel_parts = mf.relative_to(manifests_dir).parts
                if len(rel_parts) >= 3:
                    # e.g. registry.ollama.ai / library / qwen3 / 4b -> qwen3:4b
                    if rel_parts[0] == "registry.ollama.ai" and rel_parts[1] == "library":
                        label = f"{rel_parts[2]}:{'/'.join(rel_parts[3:])}" if len(rel_parts) > 3 else rel_parts[2]
                    else:
                        label = f"{'/'.join(rel_parts[:-1])}:{rel_parts[-1]}"
                else:
                    label = mf.name
                size = blob_path.stat().st_size
                found.append({
                    "label": f"[Ollama GGUF] {label}",
                    "path": ap,
                    "size": size,
                    "is_base": "base" in label.lower(),
                    "family": infer_model_family(label),
                    "param_b": infer_param_billions(label, size),
                    "source": "ollama_blob",
                })
        except OSError:
            continue

    found.sort(key=lambda x: (not x["is_base"], -x["size"] if x["is_base"] else 0, x["label"]))
    return found


# ---------------------------------------------------------------------------
# Optional Full-Vocab Exact Logits Backend (llama-cpp-python + numpy)
# ---------------------------------------------------------------------------

def _log_softmax_1d(logits):
    """Numerically stable 1D log-softmax in float32/float64 numpy."""
    import numpy as np
    arr = np.asarray(logits, dtype=np.float64)
    mx = float(np.max(arr))
    shifted = arr - mx
    lse = mx + math.log(float(np.sum(np.exp(shifted))))
    return (arr - lse).astype(np.float32)


def _log_softmax_2d(logits_2d):
    """Numerically stable row-wise log-softmax over shape (L, V)."""
    import numpy as np
    arr = np.asarray(logits_2d, dtype=np.float64)
    mx = np.max(arr, axis=-1, keepdims=True)
    shifted = arr - mx
    lse = mx + np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return arr - lse


def compute_binoculars_from_logits(
    obs_logits_2d,
    perf_logits_2d,
    target_ids: Sequence[int],
    token_strings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Exact Binoculars + unconditional PPL + GLTR + sentence-PPL Burstiness
    from full-vocabulary next-token logit matrices of shape (L, V) aligned with
    `target_ids` of length L (where row i predicts target_ids[i]).

    Uses vectorized numpy when installed (standard with llama-cpp-python),
    with a pure-Python fallback for zero-dependency verification.

    Reference: Hans et al., "Spotting LLMs With Binoculars: Zero-Shot Detection
    of Machine-Generated Text" (ICML 2024):
      B(s) = mean_i(-log P_obs(x_i | x_<i)) / mean_i(H(P_perf(.|x_<i), P_obs(.|x_<i)))
    Higher B(s) -> more human-like (typical human >= 0.90-0.95).
    """
    L = len(target_ids)
    if L == 0:
        return {
            "binoculars": None,
            "uncond_base_ppl": None,
            "uncond_inst_ppl": None,
            "gltr_top1": None,
            "gltr_top5": None,
            "gltr_top10": None,
            "gltr_oob": None,
            "uncond_sent_ppl_cv": None,
        }

    try:
        import numpy as np
        use_np = True
    except ImportError:
        use_np = False

    if use_np:
        obs_lp = _log_softmax_2d(obs_logits_2d[:L])
        perf_lp = _log_softmax_2d(perf_logits_2d[:L])
        perf_p = np.exp(perf_lp)

        idx = np.arange(L)
        tgt = np.asarray(target_ids, dtype=np.int64)

        obs_nll = -obs_lp[idx, tgt]
        perf_nll_arr = -perf_lp[idx, tgt]
        x_ent = -np.sum(perf_p * obs_lp, axis=-1)

        mean_obs_nll = float(np.mean(obs_nll))
        mean_perf_nll = float(np.mean(perf_nll_arr))
        mean_xent = float(np.mean(x_ent))

        target_perf_lp = perf_lp[idx, tgt][:, None]
        ranks = np.sum(perf_lp > target_perf_lp, axis=-1)
        gltr_top1 = float(np.mean(ranks == 0))
        gltr_top5 = float(np.mean(ranks < 5))
        gltr_top10 = float(np.mean(ranks < 10))
        gltr_oob = float(np.mean(ranks >= 20))
        perf_nll_list = [float(x) for x in perf_nll_arr]
    else:
        obs_nll_list: List[float] = []
        perf_nll_list = []
        x_ent_list: List[float] = []
        ranks_list: List[int] = []

        for i in range(L):
            o_row = [float(v) for v in obs_logits_2d[i]]
            p_row = [float(v) for v in perf_logits_2d[i]]
            o_mx = max(o_row)
            o_lse = o_mx + math.log(sum(math.exp(v - o_mx) for v in o_row))
            o_lp = [v - o_lse for v in o_row]

            p_mx = max(p_row)
            p_lse = p_mx + math.log(sum(math.exp(v - p_mx) for v in p_row))
            p_lp = [v - p_lse for v in p_row]
            p_pr = [math.exp(v) for v in p_lp]

            t = int(target_ids[i])
            obs_nll_list.append(-o_lp[t])
            perf_nll_list.append(-p_lp[t])
            x_ent_list.append(-sum(pp * ol for pp, ol in zip(p_pr, o_lp)))
            ranks_list.append(sum(1 for lp in p_lp if lp > p_lp[t]))

        mean_obs_nll = sum(obs_nll_list) / L
        mean_perf_nll = sum(perf_nll_list) / L
        mean_xent = sum(x_ent_list) / L
        gltr_top1 = sum(1 for r in ranks_list if r == 0) / L
        gltr_top5 = sum(1 for r in ranks_list if r < 5) / L
        gltr_top10 = sum(1 for r in ranks_list if r < 10) / L
        gltr_oob = sum(1 for r in ranks_list if r >= 20) / L

    binoculars = mean_obs_nll / max(mean_xent, 1e-6)

    # Per-sentence unconditional PPL Burstiness (CV of sentence-level performer NLL)
    sent_ppl_cv = None
    if token_strings and len(token_strings) >= L:
        sent_ppls: List[float] = []
        cur_nlls: List[float] = []
        for i in range(L):
            cur_nlls.append(perf_nll_list[i])
            ts = token_strings[i]
            if any(p in ts for p in (".", "!", "?")) and len(cur_nlls) >= 5:
                sent_ppls.append(math.exp(min(12.0, sum(cur_nlls) / len(cur_nlls))))
                cur_nlls = []
        if len(cur_nlls) >= 5:
            sent_ppls.append(math.exp(min(12.0, sum(cur_nlls) / len(cur_nlls))))
        if len(sent_ppls) >= 2:
            m_val = sum(sent_ppls) / len(sent_ppls)
            var = sum((x - m_val) ** 2 for x in sent_ppls) / len(sent_ppls)
            sent_ppl_cv = round(math.sqrt(var) / max(m_val, 1e-6), 3)

    return {
        "binoculars": round(binoculars, 4),
        "uncond_base_ppl": round(math.exp(min(15.0, mean_obs_nll)), 2),
        "uncond_inst_ppl": round(math.exp(min(15.0, mean_perf_nll)), 2),
        "gltr_top1": round(gltr_top1, 3),
        "gltr_top5": round(gltr_top5, 3),
        "gltr_top10": round(gltr_top10, 3),
        "gltr_oob": round(gltr_oob, 3),
        "uncond_sent_ppl_cv": sent_ppl_cv,
    }


class LlamaCppManager:
    """
    Manages cached `llama_cpp.Llama` instances for exact full-vocabulary
    contrastive decoding and single-pass Binoculars evaluation.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._models: Dict[Tuple[str, int, int], Any] = {}
        self._tok_str_cache: Dict[int, Dict[int, str]] = {}
        self._supports_prev_tokens: Dict[int, bool] = {}

    def get_model(
        self,
        gguf_path: str,
        *,
        n_ctx: int = DEFAULT_NUM_CTX,
        n_gpu_layers: int = -1,
    ) -> Any:
        ok, _ = _probe_llama_cpp()
        if not ok:
            raise RuntimeError(
                "llama-cpp-python and numpy are required for the `llama_cpp` backend. "
                "Install with: pip install llama-cpp-python numpy"
            )
        if not gguf_path or not os.path.isfile(gguf_path):
            raise FileNotFoundError(f"GGUF model file not found: {gguf_path}")

        abs_path = os.path.abspath(gguf_path)
        key = (abs_path, int(n_ctx), int(n_gpu_layers))
        with self._lock:
            llm = self._models.get(key)
            if llm is not None:
                return llm
            from llama_cpp import Llama
            llm = Llama(
                model_path=abs_path,
                n_ctx=int(n_ctx),
                n_gpu_layers=int(n_gpu_layers),
                logits_all=True,
                verbose=False,
            )
            self._models[key] = llm
            self._tok_str_cache[id(llm)] = {}
            try:
                sig = inspect.signature(llm.detokenize)
                self._supports_prev_tokens[id(llm)] = "prev_tokens" in sig.parameters
            except Exception:
                self._supports_prev_tokens[id(llm)] = False
            return llm

    def detokenize_piece(self, llm: Any, tid: int, prev_tid: Optional[int] = None) -> str:
        """
        Convert a single token ID to its exact text piece while preserving
        leading whitespace (handles both BPE and SentencePiece tokenizers).
        """
        cache = self._tok_str_cache.setdefault(id(llm), {})
        if tid in cache:
            return cache[tid]

        piece = ""
        try:
            if self._supports_prev_tokens.get(id(llm), False):
                ref_prev = [prev_tid] if prev_tid is not None else [max(0, llm.token_bos())]
                raw = llm.detokenize([tid], prev_tokens=ref_prev, special=True)
            else:
                raw = llm.detokenize([tid], special=True)
            piece = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        except Exception:
            try:
                raw = llm.detokenize([tid])
                piece = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
            except Exception:
                piece = ""
        cache[tid] = piece
        return piece

    def get_last_logits(self, llm: Any):
        """
        Fast zero-overhead extraction of the most recently evaluated token's
        full-vocab logits (avoids slow `llm.eval_logits` `.tolist()` conversion).
        Works across llama-cpp-python 0.2.x and 0.3.x.
        """
        import numpy as np
        n_vocab = int(llm.n_vocab())
        # Prefer direct C pointer when available (always valid even with logits_all=False)
        try:
            import llama_cpp
            ctx_ptr = getattr(getattr(llm, "_ctx", None), "ctx", None)
            if ctx_ptr is not None:
                ptr = llama_cpp.llama_get_logits(ctx_ptr)
                if ptr:
                    return np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()
        except Exception:
            pass
        # Fallback to llm.scores numpy buffer (valid with logits_all=True)
        idx = max(0, int(getattr(llm, "n_tokens", 1)) - 1)
        return np.asarray(llm.scores[idx, :n_vocab], dtype=np.float32).copy()

    def get_sequence_logits(self, llm: Any, tokens: Sequence[int]):
        """
        Run a single batch pass on `tokens` and return logits of shape (len(tokens), n_vocab).
        """
        import numpy as np
        n_vocab = int(llm.n_vocab())
        llm.reset()
        llm.eval(list(tokens))
        L = len(tokens)
        scores = getattr(llm, "scores", None)
        if scores is not None and len(scores) >= L:
            arr = np.asarray(scores[:L, :n_vocab], dtype=np.float32)
            # Sanity check that scores buffer was actually populated (non-zero)
            if float(np.max(np.abs(arr))) > 0.0:
                return arr.copy()
        # Safe fallback if backend ignored logits_all=True: incremental eval
        out = np.zeros((L, n_vocab), dtype=np.float32)
        llm.reset()
        for i, tid in enumerate(tokens):
            llm.eval([tid])
            out[i] = self.get_last_logits(llm)
        return out

    def verify_vocab_alignment(self, base_llm: Any, inst_llm: Any) -> int:
        """
        Verify two GGUF models share a compatible vocabulary for exact logit contrast.
        Returns shared vocab size; raises ValueError if mismatched families.
        """
        vb = int(base_llm.n_vocab())
        vi = int(inst_llm.n_vocab())
        shared_v = min(vb, vi)
        if abs(vb - vi) > 256:
            raise ValueError(
                f"Vocabulary size mismatch between Base ({vb}) and Instruct ({vi}) GGUF models. "
                "Contrastive decoding requires same-family/same-tokenizer models."
            )
        # Spot-check common vocabulary indices
        probe_ids = [100, 250, 500, 1000, 2000, 3500, 5000, 8000]
        mismatches = 0
        for tid in probe_ids:
            if tid >= shared_v:
                continue
            sb = self.detokenize_piece(base_llm, tid)
            si = self.detokenize_piece(inst_llm, tid)
            if sb != si:
                mismatches += 1
        if mismatches >= 3:
            raise ValueError(
                f"Tokenizer vocabulary mismatch detected ({mismatches}/{len(probe_ids)} probe tokens differ). "
                "Ensure Base and Instruct GGUF models belong to the same model family."
            )
        return shared_v

    def generate_raw(
        self,
        llm: Any,
        raw_prompt: str,
        *,
        num_predict: int = 240,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Direct raw-prompt completion via llama-cpp-python (for Stage-0 notes & Judge)."""
        res = llm(
            raw_prompt,
            max_tokens=num_predict,
            temperature=temperature,
            stop=stop or ["<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<end_of_turn>"],
            echo=False,
        )
        choices = res.get("choices") or []
        return (choices[0].get("text") or "").strip() if choices else ""

    def score_unconditional_binoculars(
        self,
        base_llm: Any,
        inst_llm: Any,
        text: str,
    ) -> Dict[str, Any]:
        """
        Single-pass exact unconditional Binoculars + PPL + GLTR + Sentence Burstiness
        on standalone `text` (no prompt/notes conditioning!).
        """
        cleaned = text.strip()
        if not cleaned:
            return {}
        tokens = base_llm.tokenize(cleaned.encode("utf-8"), add_bos=True, special=False)
        if len(tokens) < 4:
            return {}
        n_ctx = min(int(base_llm.n_ctx()), int(inst_llm.n_ctx()))
        if len(tokens) >= n_ctx:
            tokens = tokens[: n_ctx - 1]

        shared_v = self.verify_vocab_alignment(base_llm, inst_llm)
        obs_logits = self.get_sequence_logits(base_llm, tokens)[:-1, :shared_v]
        perf_logits = self.get_sequence_logits(inst_llm, tokens)[:-1, :shared_v]
        target_ids = tokens[1:]
        tok_strs = [self.detokenize_piece(base_llm, tid, tokens[i]) for i, tid in enumerate(target_ids)]
        return compute_binoculars_from_logits(obs_logits, perf_logits, target_ids, token_strings=tok_strs)


class LlamaCppContrastiveSession:
    """
    Incremental KV-cache decoding session for EXACT full-vocabulary contrastive decoding:
        score(t) = log P_base(t) - alpha(t) * max(log P_instruct(t), i_floor)
    Eliminates Ollama's Top-20 truncation cliff and HTTP round-trip overhead.
    """

    def __init__(
        self,
        manager: LlamaCppManager,
        base_llm: Any,
        inst_llm: Any,
        base_prompt: str,
        inst_prompt: str,
    ):
        self.manager = manager
        self.base_llm = base_llm
        self.inst_llm = inst_llm
        self.shared_v = manager.verify_vocab_alignment(base_llm, inst_llm)
        self.eos_ids = {int(base_llm.token_eos()), int(inst_llm.token_eos())}

        b_toks = base_llm.tokenize(base_prompt.encode("utf-8"), add_bos=True, special=True)
        i_toks = inst_llm.tokenize(inst_prompt.encode("utf-8"), add_bos=True, special=True)
        n_ctx_b = int(base_llm.n_ctx())
        n_ctx_i = int(inst_llm.n_ctx())
        if len(b_toks) >= n_ctx_b - 16:
            b_toks = b_toks[-(n_ctx_b - 64):]
        if len(i_toks) >= n_ctx_i - 16:
            i_toks = i_toks[-(n_ctx_i - 64):]

        self.last_tid = b_toks[-1] if b_toks else None
        base_llm.reset()
        base_llm.eval(b_toks)
        inst_llm.reset()
        inst_llm.eval(i_toks)

    def next_exact_candidates(
        self,
        top_base_k: int = 48,
    ) -> List[Tuple[int, str, float, float, bool]]:
        """
        Return top `top_base_k` candidates from Base with their EXACT full-vocab
        `b_lp` and EXACT full-vocab `i_lp`:
            [(token_id, token_str, exact_b_lp, exact_i_lp, is_eos), ...]
        sorted descending by `exact_b_lp`.
        """
        import numpy as np
        b_logits = self.manager.get_last_logits(self.base_llm)[: self.shared_v]
        i_logits = self.manager.get_last_logits(self.inst_llm)[: self.shared_v]

        b_lp_all = _log_softmax_1d(b_logits)
        i_lp_all = _log_softmax_1d(i_logits)

        k = min(max(8, int(top_base_k)), self.shared_v)
        top_idx = np.argpartition(b_lp_all, -k)[-k:]
        top_idx = top_idx[np.argsort(b_lp_all[top_idx])[::-1]]

        out: List[Tuple[int, str, float, float, bool]] = []
        for tid_np in top_idx:
            tid = int(tid_np)
            t_str = self.manager.detokenize_piece(self.base_llm, tid, self.last_tid)
            out.append((tid, t_str, float(b_lp_all[tid]), float(i_lp_all[tid]), tid in self.eos_ids))
        return out

    def accept_token(self, tid: int) -> None:
        self.last_tid = tid
        self.base_llm.eval([tid])
        self.inst_llm.eval([tid])


class LlamaCppSingleSession:
    """Incremental KV-cache decoding session for fast Anti-Greedy decoding on GGUF."""

    def __init__(self, manager: LlamaCppManager, llm: Any, prompt: str):
        self.manager = manager
        self.llm = llm
        self.n_vocab = int(llm.n_vocab())
        self.eos_id = int(llm.token_eos())

        toks = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
        n_ctx = int(llm.n_ctx())
        if len(toks) >= n_ctx - 16:
            toks = toks[-(n_ctx - 64):]
        self.last_tid = toks[-1] if toks else None
        llm.reset()
        llm.eval(toks)

    def next_top_tokens(self, top_k: int = 32) -> Tuple[Optional[str], List[Tuple[int, str, float, bool]]]:
        import numpy as np
        logits = self.manager.get_last_logits(self.llm)[: self.n_vocab]
        lp_all = _log_softmax_1d(logits)
        k = min(max(8, int(top_k)), self.n_vocab)
        top_idx = np.argpartition(lp_all, -k)[-k:]
        top_idx = top_idx[np.argsort(lp_all[top_idx])[::-1]]

        out: List[Tuple[int, str, float, bool]] = []
        for tid_np in top_idx:
            tid = int(tid_np)
            t_str = self.manager.detokenize_piece(self.llm, tid, self.last_tid)
            out.append((tid, t_str, float(lp_all[tid]), tid == self.eos_id))
        tok0 = out[0][1] if out else None
        return tok0, out

    def accept_token(self, tid: int) -> None:
        self.last_tid = tid
        self.llm.eval([tid])


LLAMA_CPP_MANAGER = LlamaCppManager()
