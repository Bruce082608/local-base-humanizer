#!/usr/bin/env python3
"""
Local Base-Model Humanizer Studio — Zero-Dependency Local Web GUI Server.

Supports:
  - ZeroGPT Mode (Anti-Greedy + AI-Buzzword Blacklist + Self-Bootstrap)
  - GPTZero Mode (Contrastive Anti-RLHF + Human-Style Few-Shot + Dynamic Burstiness)
  - Pluggable Backends: Ollama HTTP (zero-dep) & optional exact full-vocab llama.cpp (GGUF)
  - Multi-Signal Local Surrogate Reranking: Burstiness + Cross-Family Unconditional GLTR
    + Exact Binoculars + Optional Local DeBERTa/RoBERTa Neural AI Detector
"""

import argparse
import json
import mimetypes
import random
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import backends
import core
import human_demos
import surrogates

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

ACTIVE_JOBS = {}
JOBS_LOCK = threading.Lock()


class HumanizerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def _send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _serve_static(self, rel_path: str):
        safe_path = (STATIC_DIR / rel_path.lstrip("/")).resolve()
        if not str(safe_path).startswith(str(STATIC_DIR.resolve())) or not safe_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        ctype, _ = mimetypes.guess_type(str(safe_path))
        if not ctype:
            ctype = "application/octet-stream"
        data = safe_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            f"{ctype}; charset=utf-8" if ctype.startswith("text/") or ctype.endswith("javascript") else ctype,
        )
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            return self._serve_static("index.html")
        if path.startswith("/static/"):
            return self._serve_static(path[len("/static/"):])
        if path == "/api/defaults":
            return self._send_json({
                "mode": "contrastive",
                "backend": "ollama",
                "ollama_url": core.DEFAULT_OLLAMA_BASE,
                "model": core.DEFAULT_MODEL,
                "instruct_model": core.DEFAULT_INSTRUCT_MODEL,
                "base_gguf": "",
                "instruct_gguf": "",
                "n_gpu_layers": -1,
                "length_ratio": core.DEFAULT_LENGTH_RATIO,
                "p_skip": 0.72,
                "min_ppl": 3.8,
                "alpha": 0.56,
                "min_inst_ppl": 350,
                "contrastive_temp": 0.62,
                "attempts": 4,
                "pool_size": 6,
                "seed": 401,
                "num_ctx": core.DEFAULT_NUM_CTX,
                "use_telegraphic": True,
                "constrain_facts": True,
                "use_judge": True,
                "dynamic_alpha": True,
                "use_human_demos": True,
                "cad_blind": False,
                "surrogate_ollama_model": "",
                "surrogate_probes": 16,
                "neural_detector_model": "",
                "neural_detector_device": "cpu",
                "binoculars_threshold": 0.90,
                "banned_substrs": core.DEFAULT_BANNED_SUBSTRS,
                "contrastive_banned_substrs": core.CONTRASTIVE_BANNED_SUBSTRS,
            })
        if path == "/api/capabilities":
            caps = backends.get_backend_capabilities()
            ggufs = backends.discover_local_ggufs()
            return self._send_json({
                **caps,
                "local_ggufs": ggufs,
                "recommended_neural_detectors": surrogates.RECOMMENDED_NEURAL_DETECTORS,
                "builtin_human_demos_count": len(human_demos.BUILTIN_HUMAN_DEMOS),
            })
        if path == "/api/models":
            qs = parse_qs(parsed.query)
            ollama_url = qs.get("ollama_url", [core.DEFAULT_OLLAMA_BASE])[0]
            ok, models, err = core.list_ollama_models(ollama_url)
            return self._send_json({
                "ok": ok,
                "models": models,
                "error": err,
                "ollama_url": core.normalize_ollama_base(ollama_url),
            })

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/cancel":
            try:
                body = self._read_json()
                job_id = body.get("job_id")
                with JOBS_LOCK:
                    ev = ACTIVE_JOBS.get(job_id)
                    if ev:
                        ev.set()
                return self._send_json({"ok": True})
            except Exception as exc:
                return self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

        if path == "/api/rewrite/stream":
            return self._handle_rewrite_stream()

        if path == "/api/rewrite/one":
            return self._handle_rewrite_one()

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _begin_sse(self):
        self.close_connection = True
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def _send_sse(self, event_name: str, data: dict) -> bool:
        try:
            msg = f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            self.wfile.write(msg.encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return False

    @staticmethod
    def _parse_common_params(body: dict) -> dict:
        backend = body.get("backend", "ollama")
        if backend not in ("ollama", "llama_cpp"):
            backend = "ollama"
        return {
            "mode": body.get("mode", "contrastive"),
            "backend": backend,
            "ollama_url": body.get("ollama_url", core.DEFAULT_OLLAMA_BASE),
            "model": body.get("model", core.DEFAULT_MODEL),
            "instruct_model": body.get("instruct_model", core.DEFAULT_INSTRUCT_MODEL),
            "base_gguf": str(body.get("base_gguf", "") or "").strip(),
            "instruct_gguf": str(body.get("instruct_gguf", "") or "").strip(),
            "n_gpu_layers": int(body.get("n_gpu_layers", -1)),
            "length_ratio": max(0.6, min(1.6, float(body.get("length_ratio", core.DEFAULT_LENGTH_RATIO)))),
            "p_skip": float(body.get("p_skip", 0.72)),
            "min_ppl": float(body.get("min_ppl", 3.8)),
            "alpha": float(body.get("alpha", 0.56)),
            "min_inst_ppl": float(body.get("min_inst_ppl", 350.0)),
            "contrastive_temp": float(body.get("contrastive_temp", 0.62)),
            "attempts": max(1, min(32, int(body.get("attempts", 4)))),
            "pool_size": max(2, min(15, int(body.get("pool_size", 6)))),
            "seed": int(body.get("seed", 401)),
            "num_ctx": max(512, min(8192, int(body.get("num_ctx", core.DEFAULT_NUM_CTX)))),
            "use_telegraphic": bool(body.get("use_telegraphic", True)),
            "constrain_facts": bool(body.get("constrain_facts", True)),
            "use_judge": bool(body.get("use_judge", True)),
            "dynamic_alpha": bool(body.get("dynamic_alpha", True)),
            "use_human_demos": bool(body.get("use_human_demos", True)),
            "cad_blind": bool(body.get("cad_blind", False)),
            "custom_human_demos_text": str(body.get("custom_human_demos_text", "") or ""),
            "surrogate_ollama_model": str(body.get("surrogate_ollama_model", "") or "").strip(),
            "surrogate_probes": max(6, min(32, int(body.get("surrogate_probes", 16)))),
            "neural_detector_model": str(body.get("neural_detector_model", "") or "").strip(),
            "neural_detector_device": str(body.get("neural_detector_device", "cpu") or "cpu").strip(),
            "binoculars_threshold": max(0.60, min(1.20, float(body.get("binoculars_threshold", 0.90)))),
            "banned": body.get("banned_substrs", core.DEFAULT_BANNED_SUBSTRS),
            "contrastive_banned": body.get("contrastive_banned_substrs", core.CONTRASTIVE_BANNED_SUBSTRS),
            "use_bootstrap": bool(body.get("use_bootstrap", True)),
        }

    def _handle_rewrite_stream(self):
        try:
            body = self._read_json()
        except Exception as exc:
            return self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        p = self._parse_common_params(body)
        mode = p["mode"]
        raw_text = body.get("text", "")
        job_id = body.get("job_id") or f"job-{random.randint(100000, 999999)}"
        locked_blocks = body.get("locked_blocks", {})

        blocks = core.parse_blocks(raw_text)
        cancel_ev = threading.Event()
        with JOBS_LOCK:
            ACTIVE_JOBS[job_id] = cancel_ev

        self._begin_sse()
        self._send_sse("init", {"job_id": job_id, "mode": mode, "backend": p["backend"], "blocks": blocks})

        bootstrap_demos = []
        if mode == "anti_greedy" and p["use_bootstrap"] and locked_blocks:
            for idx_str, locked_info in locked_blocks.items():
                try:
                    idx_i = int(idx_str)
                    if 0 <= idx_i < len(blocks) and blocks[idx_i]["kind"] == "para":
                        if locked_info.get("text"):
                            bootstrap_demos.append(
                                core.build_bootstrap_demo(blocks[idx_i]["text"], locked_info["text"])
                            )
                except ValueError:
                    pass

        demos = [core.DEMO_A, core.DEMO_B]
        if bootstrap_demos:
            demos = bootstrap_demos + demos

        para_idx = 0
        try:
            for b_idx, blk in enumerate(blocks):
                if cancel_ev.is_set():
                    self._send_sse("cancelled", {"job_id": job_id})
                    break

                if blk["kind"] == "heading":
                    self._send_sse("block_done", {
                        "index": b_idx,
                        "kind": "heading",
                        "source": blk["text"],
                        "text": blk["text"],
                        "ppl": None,
                        "inst_ppl": None,
                        "locked": False,
                    })
                    continue

                if str(b_idx) in locked_blocks:
                    info = locked_blocks[str(b_idx)]
                    locked_txt = info.get("text", blk["text"])
                    s_w = max(1, len(blk["text"].split()))
                    o_w = len(locked_txt.split())
                    self._send_sse("block_done", {
                        "index": b_idx,
                        "kind": "para",
                        "mode": info.get("mode", mode),
                        "backend": info.get("backend", p["backend"]),
                        "source": blk["text"],
                        "text": locked_txt,
                        "ppl": info.get("ppl"),
                        "inst_ppl": info.get("inst_ppl"),
                        "skip_ratio": info.get("skip_ratio"),
                        "fidelity_pct": info.get("fidelity_pct"),
                        "judge_score": info.get("judge_score"),
                        "human_score_pct": info.get("human_score_pct"),
                        "burstiness_score": info.get("burstiness_score"),
                        "binoculars": info.get("binoculars"),
                        "neural_ai_prob": info.get("neural_ai_prob"),
                        "gltr_top1": info.get("gltr_top1"),
                        "overlap_3gram": info.get("overlap_3gram"),
                        "source_words": s_w,
                        "output_words": o_w,
                        "length_pct": round(100.0 * o_w / s_w),
                        "elapsed": 0.0,
                        "attempt": 0,
                        "locked": True,
                    })
                    para_idx += 1
                    continue

                def on_prog(ev, _b_idx=b_idx):
                    if not self._send_sse("progress", {"index": _b_idx, **ev}):
                        cancel_ev.set()

                if mode == "contrastive":
                    anchor = core.make_contrastive_anchor(blk["text"], para_idx)
                    self._send_sse("block_start", {
                        "index": b_idx,
                        "para_idx": para_idx,
                        "mode": "contrastive",
                        "backend": p["backend"],
                        "seed_start": anchor,
                    })
                    res = core.rewrite_paragraph_contrastive(
                        blk["text"],
                        anchor=anchor,
                        ollama_base=p["ollama_url"],
                        base_model=p["model"],
                        instruct_model=p["instruct_model"],
                        backend=p["backend"],
                        base_gguf=p["base_gguf"],
                        instruct_gguf=p["instruct_gguf"],
                        n_gpu_layers=p["n_gpu_layers"],
                        length_ratio=p["length_ratio"],
                        alpha=p["alpha"],
                        temp=p["contrastive_temp"],
                        pool_size=min(p["pool_size"], 8),
                        min_inst_ppl=p["min_inst_ppl"],
                        attempts=p["attempts"],
                        base_seed=p["seed"] + para_idx * 101,
                        use_telegraphic=p["use_telegraphic"],
                        constrain_facts=p["constrain_facts"],
                        use_judge=p["use_judge"],
                        cad_blind=p["cad_blind"],
                        dynamic_alpha=p["dynamic_alpha"],
                        use_human_demos=p["use_human_demos"],
                        custom_human_demos_text=p["custom_human_demos_text"],
                        banned_substrs=p["contrastive_banned"],
                        num_ctx=p["num_ctx"],
                        surrogate_ollama_model=p["surrogate_ollama_model"],
                        surrogate_probes=p["surrogate_probes"],
                        neural_detector_model=p["neural_detector_model"],
                        neural_detector_device=p["neural_detector_device"],
                        binoculars_threshold=p["binoculars_threshold"],
                        on_progress=on_prog,
                        cancel_event=cancel_ev,
                    )
                else:
                    demo = demos[para_idx % len(demos)]
                    seed_start = core.make_seed_start(blk["text"], para_idx)
                    self._send_sse("block_start", {
                        "index": b_idx,
                        "para_idx": para_idx,
                        "mode": "anti_greedy",
                        "backend": p["backend"],
                        "seed_start": seed_start,
                    })
                    res = core.rewrite_paragraph(
                        blk["text"],
                        demo=demo,
                        seed_start=seed_start,
                        ollama_base=p["ollama_url"],
                        model=p["model"],
                        backend=p["backend"],
                        base_gguf=p["base_gguf"],
                        n_gpu_layers=p["n_gpu_layers"],
                        length_ratio=p["length_ratio"],
                        p_skip_top1=p["p_skip"],
                        pool_size=p["pool_size"],
                        min_ppl=p["min_ppl"],
                        attempts=p["attempts"],
                        base_seed=p["seed"] + para_idx * 101,
                        banned_substrs=p["banned"],
                        constrain_facts=p["constrain_facts"],
                        num_ctx=p["num_ctx"],
                        surrogate_ollama_model=p["surrogate_ollama_model"],
                        surrogate_probes=p["surrogate_probes"],
                        neural_detector_model=p["neural_detector_model"],
                        neural_detector_device=p["neural_detector_device"],
                        on_progress=on_prog,
                        cancel_event=cancel_ev,
                    )

                if res.get("cancelled"):
                    self._send_sse("cancelled", {"job_id": job_id})
                    break

                self._send_sse("block_done", {
                    "index": b_idx,
                    "kind": "para",
                    "source": blk["text"],
                    "locked": False,
                    **res,
                })
                para_idx += 1

            if not cancel_ev.is_set():
                self._send_sse("done", {"job_id": job_id})
        except Exception as exc:
            self._send_sse("error", {"message": str(exc)})
        finally:
            with JOBS_LOCK:
                ACTIVE_JOBS.pop(job_id, None)

    def _handle_rewrite_one(self):
        try:
            body = self._read_json()
        except Exception as exc:
            return self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        p = self._parse_common_params(body)
        mode = p["mode"]
        job_id = body.get("job_id") or f"one-{random.randint(100000, 999999)}"
        b_idx = int(body.get("index", 0))
        para_idx = int(body.get("para_idx", b_idx))
        source_para = body.get("source", "")
        seed = int(body.get("seed", random.randint(1000, 99999)))
        custom_demo = body.get("bootstrap_demo")

        cancel_ev = threading.Event()
        with JOBS_LOCK:
            ACTIVE_JOBS[job_id] = cancel_ev

        self._begin_sse()

        def on_prog(ev):
            if not self._send_sse("progress", {"index": b_idx, **ev}):
                cancel_ev.set()

        try:
            if mode == "contrastive":
                anchor = core.make_contrastive_anchor(source_para, para_idx + random.randint(0, 5))
                self._send_sse("block_start", {
                    "index": b_idx,
                    "job_id": job_id,
                    "mode": "contrastive",
                    "backend": p["backend"],
                    "seed_start": anchor,
                })
                res = core.rewrite_paragraph_contrastive(
                    source_para,
                    anchor=anchor,
                    ollama_base=p["ollama_url"],
                    base_model=p["model"],
                    instruct_model=p["instruct_model"],
                    backend=p["backend"],
                    base_gguf=p["base_gguf"],
                    instruct_gguf=p["instruct_gguf"],
                    n_gpu_layers=p["n_gpu_layers"],
                    length_ratio=p["length_ratio"],
                    alpha=p["alpha"],
                    temp=p["contrastive_temp"],
                    pool_size=min(p["pool_size"], 8),
                    min_inst_ppl=p["min_inst_ppl"],
                    attempts=p["attempts"],
                    base_seed=seed,
                    use_telegraphic=p["use_telegraphic"],
                    constrain_facts=p["constrain_facts"],
                    use_judge=p["use_judge"],
                    cad_blind=p["cad_blind"],
                    dynamic_alpha=p["dynamic_alpha"],
                    use_human_demos=p["use_human_demos"],
                    custom_human_demos_text=p["custom_human_demos_text"],
                    banned_substrs=p["contrastive_banned"],
                    num_ctx=p["num_ctx"],
                    surrogate_ollama_model=p["surrogate_ollama_model"],
                    surrogate_probes=p["surrogate_probes"],
                    neural_detector_model=p["neural_detector_model"],
                    neural_detector_device=p["neural_detector_device"],
                    binoculars_threshold=p["binoculars_threshold"],
                    on_progress=on_prog,
                    cancel_event=cancel_ev,
                )
            else:
                if custom_demo and custom_demo.get("source") and custom_demo.get("text"):
                    demo = core.build_bootstrap_demo(custom_demo["source"], custom_demo["text"])
                else:
                    demo = [core.DEMO_A, core.DEMO_B][para_idx % 2]
                seed_start = core.make_seed_start(source_para, para_idx + random.randint(0, 3))
                self._send_sse("block_start", {
                    "index": b_idx,
                    "job_id": job_id,
                    "mode": "anti_greedy",
                    "backend": p["backend"],
                    "seed_start": seed_start,
                })
                res = core.rewrite_paragraph(
                    source_para,
                    demo=demo,
                    seed_start=seed_start,
                    ollama_base=p["ollama_url"],
                    model=p["model"],
                    backend=p["backend"],
                    base_gguf=p["base_gguf"],
                    n_gpu_layers=p["n_gpu_layers"],
                    length_ratio=p["length_ratio"],
                    p_skip_top1=p["p_skip"],
                    pool_size=p["pool_size"],
                    min_ppl=p["min_ppl"],
                    attempts=p["attempts"],
                    base_seed=seed,
                    banned_substrs=p["banned"],
                    constrain_facts=p["constrain_facts"],
                    num_ctx=p["num_ctx"],
                    surrogate_ollama_model=p["surrogate_ollama_model"],
                    surrogate_probes=p["surrogate_probes"],
                    neural_detector_model=p["neural_detector_model"],
                    neural_detector_device=p["neural_detector_device"],
                    on_progress=on_prog,
                    cancel_event=cancel_ev,
                )

            if res.get("cancelled"):
                self._send_sse("cancelled", {"job_id": job_id})
            else:
                self._send_sse("block_done", {
                    "index": b_idx,
                    "kind": "para",
                    "source": source_para,
                    "locked": False,
                    **res,
                })
                self._send_sse("done", {"job_id": job_id})
        except Exception as exc:
            self._send_sse("error", {"message": str(exc)})
        finally:
            with JOBS_LOCK:
                ACTIVE_JOBS.pop(job_id, None)


def main():
    parser = argparse.ArgumentParser(description="Local Base-Model Humanizer Studio GUI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open system browser")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), HumanizerHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Local Base-Model Humanizer Studio running at {url}")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


if __name__ == "__main__":
    main()
