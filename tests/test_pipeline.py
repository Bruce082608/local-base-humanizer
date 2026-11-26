#!/usr/bin/env python3
"""
Comprehensive zero-dependency test suite verifying all Step 1 & Step 2 upgrades:
  - human_demos.py: built-in pool, custom JSONL / Notes-Paragraph / raw-text parsing,
    TF-IDF retrieval + custom boost + retry rotation, dynamic entity leak extraction
  - core.py (Step 1): CONTRASTIVE_BANNED_SUBSTRS enforcement, proper-noun/author/acronym
    protection in _is_active_fact_token, multi-perspective Stage-0 note extraction,
    free-start & non-formulaic anchors, opening-cliché blocker, BurstinessScheduler
  - surrogates.py (Step 2): Burstiness & 3-gram overlap scoring, unconditional cross-family
    Ollama GLTR probing, exact Binoculars math, neural detector label-index inference,
    multi-signal surrogate aggregation & hard vetoes
  - backends.py: persistent keep-alive OllamaClient, model family & scale inference,
    capabilities probing, Binoculars exact logit equations
  - app.py: live HTTP server endpoints (/api/defaults, /api/capabilities, /api/models,
    POST /api/rewrite/stream SSE end-to-end against a mock Ollama server)
"""

import json
import math
import sys
import threading
import unittest
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app
import backends
import core
import human_demos
import surrogates


class _MockOllamaState:
    def __init__(self):
        self.calls = 0
        self.prompts_seen = []
        self.models_seen = []
        self.step_counter = 0

    def reset(self):
        self.calls = 0
        self.prompts_seen.clear()
        self.models_seen.clear()
        self.step_counter = 0


MOCK_STATE = _MockOllamaState()

# Synthetic token sequence used by mock Base/Instruct to emit realistic multi-sentence prose
# INCLUDING deliberate banned/archaic traps (" whilst", " utilize") that Step 1 MUST filter out!
_MOCK_SCRIPT_TOKENS = [
    "In", " a", " five", "-node", " test", " bed", " reported", " by", " Weber", " and",
    " Klein", " [4],", " throughput", " reached", " 91.2%", " across", " 24", " runs.",
    " Latency", " stayed", " flat.", " Under", " heavier", " load", " across", " edge",
    " gateways,", " the", " BraTS", " benchmark", " dropped", " error", " rates", " by",
    " 14.8%", " while", " keeping", " peak", " memory", " below", " 9.5", " GB.",
    " Short", " runs", " finished", " early.", " Even", " when", " cluster", " bandwidth",
    " fluctuated", " sharply,", " Weber", " and", " Klein", " [4]", " kept", " variance",
    " modest", " across", " every", " trial", " window.",
]


class _MockOllamaHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def _send_json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            return self._send_json({
                "models": [
                    {"name": "qwen3-8b-base:latest", "size": 5_000_000_000},
                    {"name": "qwen3:4b", "size": 2_500_000_000},
                    {"name": "llama3.2:3b", "size": 2_000_000_000},
                ]
            })
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path != "/api/generate":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        MOCK_STATE.calls += 1
        model = body.get("model", "")
        prompt = body.get("prompt", "")
        MOCK_STATE.models_seen.append(model)
        MOCK_STATE.prompts_seen.append(prompt)

        # Non-logprobs calls: Stage-0 telegraphic notes OR Judge
        if not body.get("logprobs"):
            if "integer scale from 0 to 10" in prompt:
                return self._send_json({"response": "9"})
            return self._send_json({
                "response": (
                    "Study: Weber and Klein [4] tested 24 runs on a 5-node cluster.\n"
                    "Result: throughput reached 91.2% and BraTS error dropped 14.8%.\n"
                    "Detail: peak memory stayed under 9.5 GB across edge gateways."
                )
            })

        # Unconditional cross-family GLTR probe (llama3.2:3b)
        if "llama3" in model:
            # Mix rank-0, rank-3, and out-of-top-20 so GLTR produces realistic human stats
            words = prompt.strip().split()
            idx = len(words)
            if idx % 3 == 0:
                top = [(" expected", -0.4), (" latency", -1.2), (" memory", -1.8), (" throughput", -2.1)]
            elif idx % 3 == 1:
                top = [(" alpha", -0.9), (" beta", -1.5), (" peak", -1.9), (" BraTS", -2.2)]
            else:
                top = [(" unrelated", -0.8), (" token", -1.6), (" filler", -2.3), (" noise", -3.1)]
            return self._send_json({
                "response": top[0][0],
                "logprobs": [{"token": top[0][0], "top_logprobs": [{"token": t, "logprob": lp} for t, lp in top]}],
            })

        # Token-by-token generation (Base vs Instruct)
        # Estimate progress from generated suffix length after last "\n"
        after_hdr = prompt.rsplit("\n", 1)[-1]
        gen_word_idx = len(after_hdr.split())
        tok = _MOCK_SCRIPT_TOKENS[gen_word_idx % len(_MOCK_SCRIPT_TOKENS)]

        is_base = "base" in model.lower()
        if is_base:
            # Deliberately put banned tokens (" whilst", " utilize") at high base prob to prove Step 1 filters them!
            top = [
                (" whilst", -0.35),
                (" utilize", -0.45),
                (tok, -0.75),
                (" however", -1.40),
                (" tested", -1.85),
                (" across", -2.10),
                (" rates", -2.30),
                (".", -2.50),
            ]
        else:
            # Instruct puts low probability on `tok` and high probability on AI clichés
            top = [
                (" utilize", -0.20),
                (" furthermore", -0.70),
                (" however", -1.10),
                (" tested", -2.40),
                (" across", -2.80),
                (tok, -6.20),
                (".", -3.00),
            ]
        return self._send_json({
            "response": top[0][0],
            "logprobs": [{"token": top[0][0], "top_logprobs": [{"token": t, "logprob": lp} for t, lp in top]}],
        })


class TestHumanizerUpgrades(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_ollama = ThreadingHTTPServer(("127.0.0.1", 0), _MockOllamaHandler)
        cls.ollama_port = cls.mock_ollama.server_address[1]
        cls.ollama_url = f"http://127.0.0.1:{cls.ollama_port}"
        cls.ollama_thread = threading.Thread(target=cls.mock_ollama.serve_forever, daemon=True)
        cls.ollama_thread.start()

        cls.app_server = ThreadingHTTPServer(("127.0.0.1", 0), app.HumanizerHandler)
        cls.app_port = cls.app_server.server_address[1]
        cls.app_url = f"http://127.0.0.1:{cls.app_port}"
        cls.app_thread = threading.Thread(target=cls.app_server.serve_forever, daemon=True)
        cls.app_thread.start()

    @classmethod
    def tearDownClass(cls):
        backends.OLLAMA_CLIENT.close_all()
        cls.app_server.shutdown()
        cls.app_server.server_close()
        cls.mock_ollama.shutdown()
        cls.mock_ollama.server_close()

    def setUp(self):
        MOCK_STATE.reset()

    # ------------------------------------------------------------------
    # Step 1 Tests
    # ------------------------------------------------------------------
    def test_human_demos_pool_and_custom_parsers(self):
        self.assertGreaterEqual(len(human_demos.BUILTIN_HUMAN_DEMOS), 8)
        # Ensure built-in human demos contain zero high-risk AI clichés
        banned = ("utilize", "leverage", "delve", "tapestry", "furthermore", "moreover", "underscore", "pivotal", "—")
        for d in human_demos.BUILTIN_HUMAN_DEMOS:
            low = d["paragraph"].lower()
            for b in banned:
                self.assertNotIn(b, low, f"Built-in demo {d['id']} should not contain AI cliché '{b}'")

        # JSONL parser
        jsonl = json.dumps({
            "text": (
                "Across 18 alpine weather stations measured between 2016 and 2019, snowpack runoff "
                "shifted earlier by eleven days on south-facing slopes. North-facing basins changed less. "
                "Gauge records also picked up heavier winter rain events below 1,800 meters."
            )
        })
        parsed_jsonl = human_demos.parse_custom_human_demos(jsonl)
        self.assertEqual(len(parsed_jsonl), 1)
        self.assertTrue(parsed_jsonl[0]["notes"])

        # Notes/Paragraph pair parser
        pair_txt = (
            "Notes:\n"
            "River gauge study [11]; 18 alpine stations; runoff earlier by 11 days.\n\n"
            "Paragraph:\n"
            "Across 18 alpine weather stations in [11], snowpack runoff shifted earlier by eleven days "
            "on south-facing slopes during 2016 to 2019. North-facing basins changed much less overall."
        )
        parsed_pair = human_demos.parse_custom_human_demos(pair_txt)
        self.assertEqual(len(parsed_pair), 1)
        self.assertIn("[11]", parsed_pair[0]["notes"])

        # Retrieval prioritizes domain match & custom demos + rotates across attempts
        pool = human_demos.get_demo_pool(custom_demos=parsed_jsonl)
        r1 = human_demos.retrieve_human_demos("alpine snowpack runoff weather stations meters", pool, k=1, attempt=1, seed=7)
        self.assertEqual(r1[0]["domain"], "custom")
        r_med = human_demos.retrieve_human_demos("BraTS MRI tumor segmentation Dice 3D U-Net", human_demos.BUILTIN_HUMAN_DEMOS, k=1, attempt=1, seed=3)
        self.assertEqual(r_med[0]["id"], "medical_segmentation")

        # Dynamic leak extraction catches demo entities absent from source
        leaks = human_demos.extract_demo_leak_words(r_med, "Wind power forecasting on a 5-node cluster.")
        self.assertIn("brats", leaks)
        self.assertIn("dice", leaks)

    def test_fact_guard_protects_proper_nouns_authors_and_acronyms(self):
        src = "Weber and Klein [4] evaluated BraTS MRI segmentation using HDFS with 91.2% Dice across 24 runs."
        guard = core.build_fact_guard(src)
        self.assertIn("weber", guard["protected_terms"])
        self.assertIn("klein", guard["protected_terms"])
        self.assertIn("brats", guard["protected_terms"])
        self.assertIn("hdfs", guard["protected_terms"])

        # Numbers & citations protected
        self.assertTrue(core._is_active_fact_token("reached 91", ".2", guard))
        self.assertTrue(core._is_active_fact_token("in [", "4]", guard))
        # Authors, acronyms, and BPE prefixes protected!
        self.assertTrue(core._is_active_fact_token("results from", " Weber", guard))
        self.assertTrue(core._is_active_fact_token("using the", " Bra", guard))
        self.assertTrue(core._is_active_fact_token("using the Bra", "TS", guard))
        self.assertTrue(core._is_active_fact_token("stored in", " HDFS", guard))
        # Ordinary non-fact content words NOT protected
        self.assertFalse(core._is_active_fact_token("the system", " improves", guard))

        # Fabricated numbers & citations still blocked
        self.assertTrue(core._violates_fact_guard("accuracy reached", " 99", guard))
        self.assertTrue(core._violates_fact_guard("as shown in [", "7]", guard))
        self.assertFalse(core._violates_fact_guard("as shown in [", "4]", guard))

    def test_dynamic_burstiness_scheduler_modulates_alpha(self):
        sched = core.BurstinessScheduler(seed=0, enabled=True)
        # Plan 0: sent 0 is short(12), sent 1 is long(30)
        a_short, nudge_short = sched.modulate(0.60, " throughput", sent_idx=0, words_in_sent=5, in_fact=False)
        a_long_pivot, _ = sched.modulate(0.60, " throughput", sent_idx=1, words_in_sent=12, in_fact=False)
        a_glue, _ = sched.modulate(0.60, " the", sent_idx=1, words_in_sent=12, in_fact=False)
        a_fact, _ = sched.modulate(0.60, " BraTS", sent_idx=1, words_in_sent=12, in_fact=True)
        _, nudge_period_short = sched.modulate(0.60, ".", sent_idx=0, words_in_sent=12, in_fact=False)
        _, nudge_period_long_early = sched.modulate(0.60, ".", sent_idx=1, words_in_sent=5, in_fact=False)

        self.assertLess(a_short, a_long_pivot)
        self.assertLess(a_glue, a_long_pivot)
        self.assertAlmostEqual(a_fact, 0.12, places=3)
        self.assertGreater(nudge_period_short, 0.0)
        self.assertLess(nudge_period_long_early, 0.0)
        self.assertEqual(nudge_short, 0.0)

    def test_anchors_free_start_and_opening_cliche_blocker(self):
        para = "Weber and Klein [4] measured throughput on a 5-node cluster across 24 runs."
        anchors = [core.make_contrastive_anchor(para, i) for i in range(8)]
        # Half of buckets are free-start ""
        self.assertGreaterEqual(anchors.count(""), 4)
        # Explicit lead-in heading preserved
        lead_para = "RQ1. Federated edge forecasting. Operators tested 24 gateways in [6]."
        self.assertTrue(core.make_contrastive_anchor(lead_para, 0).startswith("RQ1. Federated edge forecasting."))

        # Opening cliché blocker catches formulaic AI openers
        self.assertTrue(core._causes_opening_cliche("Recent", " studies"))
        self.assertTrue(core._causes_opening_cliche("In recent", " years"))
        self.assertFalse(core._causes_opening_cliche("Across 24", " runs"))

    def test_stage0_telegraphic_strips_labels_and_preserves_facts(self):
        src = "Weber and Klein [4] achieved 91.2% accuracy across 24 runs with 9.5 GB memory and 14.8% error drop."
        notes_v0 = core.extract_telegraphic_notes(
            src, ollama_base=self.ollama_url, instruct_model="qwen3:4b", variant_idx=0
        )
        # Study:/Result:/Detail: labels stripped
        self.assertNotIn("Study:", notes_v0)
        self.assertNotIn("Result:", notes_v0)
        self.assertIn("[4]", notes_v0)
        self.assertIn("91.2", notes_v0)

        # Shuffled variant keeps same facts
        notes_v1 = core.extract_telegraphic_notes(
            src, ollama_base=self.ollama_url, instruct_model="qwen3:4b", variant_idx=2, shuffle_seed=123
        )
        self.assertIn("91.2", notes_v1)
        self.assertIn("[4]", notes_v1)

    # ------------------------------------------------------------------
    # Step 2 Tests: Burstiness, Binoculars, Neural Detector Labels, Surrogates
    # ------------------------------------------------------------------
    def test_burstiness_and_overlap_distinguish_uniform_ai_vs_bursty_human(self):
        src = (
            "Recent studies demonstrate that federated learning effectively optimizes edge scheduling "
            "across distributed clusters. Furthermore, the framework highlights significant throughput "
            "improvements under heterogeneous workloads. Moreover, empirical results underscore robust "
            "convergence across varied network delays."
        )
        uniform_ai = (
            "Recent studies show that federated learning effectively optimizes edge scheduling "
            "across distributed clusters. In addition, the system demonstrates notable throughput "
            "improvements under heterogeneous workloads. Finally, empirical findings confirm stable "
            "convergence across varied network delays."
        )
        bursty_human = (
            "Edge clusters stall when slow links drag down every synchronous round. "
            "Switching to asynchronous updates changes the picture completely: throughput climbs even "
            "under uneven traffic, while convergence holds steady once stale gradients are capped at four rounds. "
            "Short runs settle fast."
        )
        m_ai = surrogates.compute_burstiness_and_overlap(uniform_ai, src, sent_nlls=[[1.5]*8, [1.55]*8, [1.52]*8])
        m_hu = surrogates.compute_burstiness_and_overlap(bursty_human, src, sent_nlls=[[1.1]*6, [2.7]*16, [1.2]*5])

        self.assertGreater(m_hu["sent_len_cv"], m_ai["sent_len_cv"])
        self.assertGreater(m_hu["burstiness_score"], m_ai["burstiness_score"])
        self.assertLess(m_hu["overlap_3gram"], m_ai["overlap_3gram"])

    def test_exact_binoculars_math_separates_ai_vs_human_logits(self):
        # Vocab size 30; 12 tokens across 2 sentences
        V = 30
        target_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        tok_strs = [" a", " b", " c", " d", " e", ".", " f", " g", " h", " i", " j", "."]

        # AI-like sequence: observer & performer both assign high logit to target_ids (low surprise)
        obs_ai = []
        perf_ai = []
        for t in target_ids:
            o = [0.0] * V
            p = [0.0] * V
            o[t] = 4.5
            p[t] = 5.0
            # Secondary mass elsewhere
            o[(t + 1) % V] = 2.0
            p[(t + 1) % V] = 1.5
            obs_ai.append(o)
            perf_ai.append(p)

        # Human-like sequence: observer & performer disagree more; target_ids are surprising to performer
        obs_hu = []
        perf_hu = []
        for idx, t in enumerate(target_ids):
            o = [0.0] * V
            p = [0.0] * V
            # Performer expects (t+1), not t -> human chooses unexpected token t!
            p[(t + 1) % V] = 5.0
            o[(t + 1) % V] = 3.5
            # Vary surprise between sentence 1 and sentence 2 for real PPL burstiness
            o[t] = 2.2 if idx < 6 else 0.8
            p[t] = 1.8 if idx < 6 else -0.5
            obs_hu.append(o)
            perf_hu.append(p)

        b_ai = backends.compute_binoculars_from_logits(obs_ai, perf_ai, target_ids, token_strings=tok_strs)
        b_hu = backends.compute_binoculars_from_logits(obs_hu, perf_hu, target_ids, token_strings=tok_strs)

        self.assertIsNotNone(b_ai["binoculars"])
        self.assertIsNotNone(b_hu["binoculars"])
        self.assertGreater(b_hu["binoculars"], b_ai["binoculars"])
        self.assertGreater(b_ai["gltr_top1"], b_hu["gltr_top1"])
        self.assertIsNotNone(b_hu["uncond_sent_ppl_cv"])
        self.assertGreater(b_hu["uncond_sent_ppl_cv"], 0.0)

    def test_neural_detector_label_index_inference(self):
        infer = surrogates.LocalNeuralDetector._infer_ai_class_index
        self.assertEqual(infer("TrustSafeAI/RADAR-Vicuna-7B", {0: "LABEL_0", 1: "LABEL_1"}), 0)
        self.assertEqual(infer("openai-community/roberta-large-openai-detector", {0: "Fake", 1: "Real"}), 0)
        self.assertEqual(infer("Hello-SimpleAI/chatgpt-detector-roberta", {0: "Human", 1: "ChatGPT"}), 1)
        self.assertEqual(infer("custom/detector", {0: "human_written", 1: "ai_generated"}), 1)
        self.assertEqual(infer("custom/detector", {0: "AI", 1: "Human"}), 0)

    def test_surrogate_aggregator_hard_vetoes(self):
        burst = {"burstiness_score": 0.85, "sent_len_cv": 0.48, "sent_ppl_cv": 0.35, "overlap_3gram": 0.04}
        # High neural AI probability MUST veto passing even if burstiness is great
        res_flagged = surrogates.aggregate_surrogate_score(burstiness=burst, neural_ai_prob=0.82)
        self.assertFalse(res_flagged["passes_surrogate"])
        # Low neural AI probability + high Binoculars clears cleanly
        res_pass = surrogates.aggregate_surrogate_score(
            burstiness=burst,
            binoculars_metrics={"binoculars": 0.96, "gltr_top1": 0.30, "gltr_oob": 0.22, "uncond_inst_ppl": 28.0},
            neural_ai_prob=0.08,
        )
        self.assertTrue(res_pass["passes_surrogate"])
        self.assertGreaterEqual(res_pass["human_score_pct"], 80)

    # ------------------------------------------------------------------
    # End-to-End Mock Ollama Contrastive + Surrogate + HTTP/SSE Tests
    # ------------------------------------------------------------------
    def test_contrastive_e2e_blocks_banned_tokens_and_runs_cross_family_gltr(self):
        src = (
            "Weber and Klein [4] evaluated a five-node cluster across 24 runs and achieved 91.2% throughput. "
            "On the BraTS benchmark, error rates dropped by 14.8% while peak memory stayed under 9.5 GB."
        )
        res = core.rewrite_paragraph_contrastive(
            src,
            anchor="",
            ollama_base=self.ollama_url,
            base_model="qwen3-8b-base:latest",
            instruct_model="qwen3:4b",
            backend="ollama",
            alpha=0.56,
            attempts=2,
            use_telegraphic=True,
            constrain_facts=True,
            use_judge=True,
            dynamic_alpha=True,
            use_human_demos=True,
            surrogate_ollama_model="llama3.2:3b",
            surrogate_probes=8,
        )
        # Step 1: Banned contrastive drift ("whilst") & AI cliché ("utilize") MUST NOT appear in output!
        self.assertNotIn("whilst", res["text"].lower())
        self.assertNotIn("utilize", res["text"].lower())
        # Step 1: Human demo block was injected into Base prompt ONLY, NOT Instruct prompt
        base_prompts = [p for m, p in zip(MOCK_STATE.models_seen, MOCK_STATE.prompts_seen) if "base" in m]
        inst_prompts = [p for m, p in zip(MOCK_STATE.models_seen, MOCK_STATE.prompts_seen) if m == "qwen3:4b" and "Write a formal" in p]
        self.assertTrue(any("---" in p for p in base_prompts), "Base prompt should contain retrieved human demo")
        self.assertTrue(all("---" not in p for p in inst_prompts), "Instruct prompt must stay demo-free")
        # Step 2: Cross-family Ollama GLTR surrogate actually ran and populated diagnostics!
        self.assertIn("llama3.2:3b", MOCK_STATE.models_seen)
        self.assertIsNotNone(res["gltr_top1"])
        self.assertIsNotNone(res["uncond_ppl"])
        self.assertIsNotNone(res["human_score_pct"])
        self.assertEqual(res["judge_score"], 9.0)

    def test_app_endpoints_and_sse_stream_e2e(self):
        # 1. GET /api/defaults
        with urllib.request.urlopen(f"{self.app_url}/api/defaults", timeout=5) as r:
            defaults = json.loads(r.read().decode("utf-8"))
        self.assertEqual(defaults["backend"], "ollama")
        self.assertTrue(defaults["dynamic_alpha"])
        self.assertTrue(defaults["use_human_demos"])
        self.assertIn("whilst", defaults["contrastive_banned_substrs"])

        # 2. GET /api/capabilities
        with urllib.request.urlopen(f"{self.app_url}/api/capabilities", timeout=5) as r:
            caps = json.loads(r.read().decode("utf-8"))
        self.assertTrue(caps["ollama_available"])
        self.assertIn("llama_cpp_available", caps)
        self.assertIn("neural_detector_available", caps)
        self.assertGreaterEqual(caps["builtin_human_demos_count"], 8)

        # 3. GET /api/models (against mock Ollama)
        with urllib.request.urlopen(f"{self.app_url}/api/models?ollama_url={self.ollama_url}", timeout=5) as r:
            models_resp = json.loads(r.read().decode("utf-8"))
        self.assertTrue(models_resp["ok"])
        by_name = {m["name"]: m for m in models_resp["models"]}
        self.assertEqual(by_name["qwen3-8b-base:latest"]["family"], "qwen3")
        self.assertEqual(by_name["qwen3-8b-base:latest"]["param_b"], 8.0)
        self.assertEqual(by_name["llama3.2:3b"]["family"], "llama3")

        # 4. POST /api/rewrite/stream (full SSE lifecycle)
        payload = json.dumps({
            "mode": "contrastive",
            "backend": "ollama",
            "ollama_url": self.ollama_url,
            "model": "qwen3-8b-base:latest",
            "instruct_model": "qwen3:4b",
            "surrogate_ollama_model": "llama3.2:3b",
            "surrogate_probes": 6,
            "attempts": 1,
            "dynamic_alpha": True,
            "use_human_demos": True,
            "text": (
                "1.1 Benchmark Overview\n\n"
                "Weber and Klein [4] tested 24 runs on a five-node cluster and reached 91.2% throughput, "
                "cutting BraTS error by 14.8% below 9.5 GB memory."
            ),
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.app_url}/api/rewrite/stream",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw_sse = resp.read().decode("utf-8")

        events = []
        for chunk in raw_sse.strip().split("\n\n"):
            ev, data = "", ""
            for ln in chunk.splitlines():
                if ln.startswith("event:"):
                    ev = ln[6:].strip()
                elif ln.startswith("data:"):
                    data = ln[5:].strip()
            if ev and data:
                events.append((ev, json.loads(data)))

        ev_names = [e[0] for e in events]
        self.assertIn("init", ev_names)
        self.assertIn("block_start", ev_names)
        self.assertIn("block_done", ev_names)
        self.assertIn("done", ev_names)
        para_done = [d for ev, d in events if ev == "block_done" and d.get("kind") == "para"]
        self.assertEqual(len(para_done), 1)
        self.assertNotIn("whilst", para_done[0]["text"].lower())
        self.assertIsNotNone(para_done[0].get("human_score_pct"))
        self.assertIsNotNone(para_done[0].get("gltr_top1"))

    def test_llama_cpp_vocab_alignment_validation(self):
        class _FakeLlama:
            def __init__(self, vocab_size, prefix):
                self._v = vocab_size
                self._p = prefix
            def n_vocab(self):
                return self._v
            def token_bos(self):
                return 1
            def detokenize(self, tids, special=True):
                return f"{self._p}_{tids[0]}".encode("utf-8")

        mgr = backends.LlamaCppManager()
        # Same family -> passes
        self.assertEqual(mgr.verify_vocab_alignment(_FakeLlama(32000, "tok"), _FakeLlama(32000, "tok")), 32000)
        # Different vocab sizes -> raises ValueError
        with self.assertRaises(ValueError):
            mgr.verify_vocab_alignment(_FakeLlama(151936, "qwen"), _FakeLlama(32000, "llama"))
        # Same size, different token mappings -> raises ValueError
        with self.assertRaises(ValueError):
            mgr.verify_vocab_alignment(_FakeLlama(32000, "qwen"), _FakeLlama(32000, "mistral"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
