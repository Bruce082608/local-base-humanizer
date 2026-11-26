#!/usr/bin/env python3
"""
Local Multi-Signal Surrogate Evaluator for Cross-Family Reranking & Diagnostics.

Breaks single-model Goodhart overfitting (where Qwen3 grades its own conditional
InstPPL) by combining four independent, local, read-only evaluation signals:

  1. Linguistic Burstiness & Anti-Paraphrase Distance (always on, 0ms, stdlib):
     - Sentence-length coefficient of variation (`sent_len_cv`)
     - Sentence-level perplexity burstiness (`sent_ppl_cv`, from decoding or exact pass)
     - Non-factual 3-gram overlap against source (`overlap_3gram`)
     - Sentence opener diversity (`opener_diversity`)
  2. Unconditional Cross-Family Ollama GLTR Prober (zero pip dependencies):
     - Strided content-word probes on STANDALONE candidate text (no notes prompt!)
       using an independent Ollama model (e.g. llama3.2 / mistral / gemma) or
       unconditional fallback, measuring truncation-free GLTR Top-1 / Top-5 / OOB
       rates + conservative unconditional PPL + per-sentence NLL variance.
  3. Exact Unconditional Binoculars (when `llama_cpp` backend is active):
     - Single-pass full-vocab `mean_NLL_obs / mean_XENT(perf, obs)` (Hans et al., 2024)
       + exact GLTR + exact sentence-level unconditional PPL Burstiness.
  4. Optional Local Neural AI Detector (`torch` + `transformers`, runs on CPU by default):
     - Supports both Desklib custom mean-pooling DeBERTa-v3-large academic models
       (`desklib/ai-text-detector-academic-v1.01`, `desklib/ai-text-detector-v1.01`)
       and standard `AutoModelForSequenceClassification` detectors
       (`TrustSafeAI/RADAR-Vicuna-7B`, `Hello-SimpleAI/chatgpt-detector-roberta`,
       `fakespot-ai/roberta-base-ai-text-detection-v1`, or local HF paths).
"""

import math
import re
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import backends

RECOMMENDED_NEURAL_DETECTORS: List[Dict[str, str]] = [
    {
        "id": "desklib/ai-text-detector-academic-v1.01",
        "label": "Desklib Academic DeBERTa-v3-Large (学术专用 · 推荐)",
        "arch": "desklib",
    },
    {
        "id": "desklib/ai-text-detector-v1.01",
        "label": "Desklib General DeBERTa-v3-Large (RAID 榜单强基线)",
        "arch": "desklib",
    },
    {
        "id": "TrustSafeAI/RADAR-Vicuna-7B",
        "label": "IBM RADAR RoBERTa-Large (对抗改写/Paraphrase 检测)",
        "arch": "seq_cls",
    },
    {
        "id": "Hello-SimpleAI/chatgpt-detector-roberta",
        "label": "HC3 ChatGPT RoBERTa-Base (轻量高速)",
        "arch": "seq_cls",
    },
]

_SURROGATE_SKIP_WORDS = {
    "a", "an", "the", "of", "in", "on", "to", "for", "and", "or", "by", "at", "as", "is", "are",
}


def _split_sentences_clean(text: str) -> List[str]:
    s = re.sub(r"\s+", " ", (text or "").strip())
    if not s:
        return []
    protected = re.sub(r"\b(et al|e\.g|i\.e|vs|cf|Fig|Tab|Eq|No|Vol|Dr|Mr|Ms)\.", r"\1<DOT>", s)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[\(\"'])", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def _extract_non_fact_words(text: str) -> List[str]:
    no_cites = re.sub(r"\[\d+(?:\s*[,;\-–]\s*\d+)*\]", " ", text or "")
    raw = re.findall(r"[A-Za-z][a-z'\-]{1,}", no_cites)
    return [w.lower() for w in raw if not any(ch.isdigit() for ch in w)]


def compute_burstiness_and_overlap(
    candidate_text: str,
    source_text: str = "",
    *,
    sent_nlls: Optional[Sequence[Sequence[float]]] = None,
    override_sent_ppl_cv: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Fast, zero-dependency structural burstiness & anti-paraphrase overlap analyzer.

    Returns:
      - sent_count, sent_len_mean, sent_len_std, sent_len_cv
      - sent_ppl_cv (from per-sentence token NLLs or exact unconditional pass)
      - overlap_3gram (0.0-1.0 fraction of candidate 3-grams verbatim in source)
      - opener_diversity (0.0-1.0 uniqueness of 2-word sentence starts)
      - burstiness_score (0.0-1.0 calibrated human-rhythm score)
    """
    sents = _split_sentences_clean(candidate_text)
    sent_count = len(sents)
    lens = [len(s.split()) for s in sents if s.split()]

    if lens:
        len_mean = sum(lens) / len(lens)
        len_var = sum((x - len_mean) ** 2 for x in lens) / len(lens)
        len_std = math.sqrt(len_var)
        sent_len_cv = len_std / max(len_mean, 1.0)
    else:
        len_mean, len_std, sent_len_cv = 0.0, 0.0, 0.0

    # Sentence opener diversity (first 2 normalized words of each sentence)
    if len(sents) >= 2:
        openers = []
        for s in sents:
            ws = [re.sub(r"[^a-z]", "", w.lower()) for w in s.split()[:2]]
            openers.append(" ".join(ws))
        opener_diversity = len(set(openers)) / len(openers)
    else:
        opener_diversity = 1.0

    # Per-sentence PPL coefficient of variation (real perplexity burstiness)
    sent_ppl_cv: Optional[float] = override_sent_ppl_cv
    if sent_ppl_cv is None and sent_nlls:
        valid_ppls: List[float] = []
        for sn in sent_nlls:
            if len(sn) >= 4:
                avg_nll = sum(sn) / len(sn)
                valid_ppls.append(math.exp(min(10.0, avg_nll)))
        if len(valid_ppls) >= 2:
            pm = sum(valid_ppls) / len(valid_ppls)
            pv = sum((x - pm) ** 2 for x in valid_ppls) / len(valid_ppls)
            sent_ppl_cv = math.sqrt(pv) / max(pm, 1e-6)

    # Anti-paraphrase lexical 3-gram overlap against source
    overlap_3gram = 0.0
    if source_text:
        s_ws = _extract_non_fact_words(source_text)
        c_ws = _extract_non_fact_words(candidate_text)
        if len(s_ws) >= 3 and len(c_ws) >= 3:
            s_3 = {tuple(s_ws[i : i + 3]) for i in range(len(s_ws) - 2)}
            c_3 = [tuple(c_ws[i : i + 3]) for i in range(len(c_ws) - 2)]
            hits = sum(1 for g in c_3 if g in s_3)
            overlap_3gram = hits / max(1, len(c_3))

    # Calibrated Burstiness score in [0, 1]
    # Human academic writing typically shows sent_len_cv ~ 0.30-0.65 (uniform AI often < 0.20)
    if sent_count <= 1:
        len_cv_score = 0.45
    elif sent_len_cv < 0.15:
        len_cv_score = 0.20 + (sent_len_cv / 0.15) * 0.25
    elif sent_len_cv <= 0.45:
        len_cv_score = 0.45 + ((sent_len_cv - 0.15) / 0.30) * 0.55
    elif sent_len_cv <= 0.78:
        len_cv_score = 1.0
    else:
        len_cv_score = max(0.55, 1.0 - (sent_len_cv - 0.78) * 0.6)

    if sent_ppl_cv is not None:
        if sent_ppl_cv < 0.10:
            ppl_cv_score = 0.25 + (sent_ppl_cv / 0.10) * 0.30
        elif sent_ppl_cv <= 0.40:
            ppl_cv_score = 0.55 + ((sent_ppl_cv - 0.10) / 0.30) * 0.45
        else:
            ppl_cv_score = 1.0
        burstiness_score = 0.50 * len_cv_score + 0.35 * ppl_cv_score + 0.15 * opener_diversity
    else:
        burstiness_score = 0.80 * len_cv_score + 0.20 * opener_diversity

    return {
        "sent_count": sent_count,
        "sent_len_mean": round(len_mean, 1),
        "sent_len_std": round(len_std, 1),
        "sent_len_cv": round(sent_len_cv, 3),
        "sent_ppl_cv": round(sent_ppl_cv, 3) if sent_ppl_cv is not None else None,
        "overlap_3gram": round(overlap_3gram, 3),
        "opener_diversity": round(opener_diversity, 2),
        "burstiness_score": round(max(0.0, min(1.0, burstiness_score)), 3),
    }


# ---------------------------------------------------------------------------
# Zero-Dependency Unconditional Cross-Family Ollama GLTR Prober
# ---------------------------------------------------------------------------

def _match_word_in_top_logprobs(
    target_word: str,
    top_logprobs: Sequence[Tuple[str, float]],
) -> Tuple[Optional[int], Optional[float]]:
    """
    Find the best BPE-aligned match for `target_word` (preceded by a space) inside
    Ollama's `top_logprobs` list. Returns `(rank_0_based, logprob)` or `(None, None)`.
    """
    bare = target_word.strip()
    if not bare:
        return None, None
    bare_low = bare.lower()

    best_exact: Optional[Tuple[int, float]] = None
    best_prefix: Optional[Tuple[int, int, float]] = None  # (match_len, -rank, lp)
    best_ci: Optional[Tuple[int, int, float]] = None

    for rank, (tok, lp) in enumerate(top_logprobs):
        if not tok:
            continue
        # Expect a word-starting token (leading space) when prefix ends without space
        if not tok[0].isspace():
            continue
        piece = tok.lstrip()
        if not piece:
            continue
        if piece == bare:
            best_exact = (rank, lp)
            break
        if bare.startswith(piece) and len(piece) >= 2:
            cand = (len(piece), -rank, lp)
            if best_prefix is None or cand > best_prefix:
                best_prefix = cand
        elif bare_low.startswith(piece.lower()) and len(piece) >= 3:
            cand = (len(piece), -rank, lp)
            if best_ci is None or cand > best_ci:
                best_ci = cand

    if best_exact is not None:
        return best_exact
    if best_prefix is not None:
        return -best_prefix[1], best_prefix[2]
    if best_ci is not None:
        return -best_ci[1], best_ci[2]
    return None, None


def score_ollama_unconditional_gltr(
    candidate_text: str,
    *,
    ollama_base: str = backends.DEFAULT_OLLAMA_BASE,
    model: str,
    num_probes: int = 16,
    num_ctx: int = backends.DEFAULT_NUM_CTX,
    tail_logprob_clamp: float = -8.5,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """
    Strided Unconditional GLTR Probe via Ollama (zero pip dependencies).

    Evaluates STANDALONE `candidate_text` (NO `notes_text` prompt!) at `num_probes`
    evenly distributed content-word positions across all sentences. Because the
    prefix grows monotonically, Ollama reuses KV cache across probes (~0.4-0.9s).

    GLTR top-1 / top-5 / out-of-top-20 rates are 100% EXACT under top_k=20
    (completely immune to the conditional-distribution + top-20 tail inflation bug!).
    """
    cleaned = re.sub(r"\s+", " ", (candidate_text or "").strip())
    words = cleaned.split()
    if len(words) < 12 or not model:
        return {}

    # Map each word index to its sentence index so we probe across every sentence
    sent_ids: List[int] = []
    cur_s = 0
    for w in words:
        sent_ids.append(cur_s)
        if w.endswith((".", "!", "?")) and not w.lower().endswith(("al.", "e.g.", "i.e.", "vs.", "fig.", "eq.")):
            cur_s += 1

    # Eligible probe indices: skip first 3 words, digits, citations, ultra-short stopwords
    eligible: List[int] = []
    for idx in range(3, len(words)):
        w = words[idx]
        bare = w.strip(".,;:()[]\"'")
        if len(bare) < 3:
            continue
        if any(ch.isdigit() for ch in bare) or "[" in w or "]" in w:
            continue
        if bare.lower() in _SURROGATE_SKIP_WORDS:
            continue
        eligible.append(idx)

    if len(eligible) < 4:
        eligible = list(range(3, len(words)))
    if not eligible:
        return {}

    n = min(max(4, int(num_probes)), len(eligible))
    if n >= len(eligible):
        chosen_indices = eligible
    else:
        step = (len(eligible) - 1) / max(1, n - 1)
        chosen_indices = sorted({eligible[int(round(i * step))] for i in range(n)})

    top1_hits = 0
    top5_hits = 0
    oob_misses = 0
    nlls: List[float] = []
    per_sent_nlls: Dict[int, List[float]] = {}

    for wi in chosen_indices:
        if cancel_event and cancel_event.is_set():
            return {}
        prefix = " ".join(words[:wi])
        target_word = words[wi].strip(".,;:()[]\"'")
        try:
            _, top = backends.OLLAMA_CLIENT.next_token_logprobs(
                ollama_base,
                model,
                prefix,
                top_k=20,
                timeout=30.0,
                num_ctx=num_ctx,
            )
        except Exception:
            return {}
        if not top:
            continue
        rank, lp = _match_word_in_top_logprobs(target_word, top)
        if rank is None or lp is None:
            oob_misses += 1
            tail_lp = max(tail_logprob_clamp, min(x[1] for x in top) - 0.75)
            nll = -tail_lp
        else:
            if rank == 0:
                top1_hits += 1
            if rank < 5:
                top5_hits += 1
            nll = -max(lp, tail_logprob_clamp)
        nlls.append(nll)
        per_sent_nlls.setdefault(sent_ids[wi], []).append(nll)

    total = len(nlls)
    if total == 0:
        return {}

    mean_nll = sum(nlls) / total
    uncond_ppl = math.exp(min(12.0, mean_nll))

    # Sentence-level unconditional NLL dispersion
    sent_ppls = [
        math.exp(min(10.0, sum(vals) / len(vals)))
        for vals in per_sent_nlls.values()
        if vals
    ]
    uncond_sent_ppl_cv = None
    if len(sent_ppls) >= 2:
        m_val = sum(sent_ppls) / len(sent_ppls)
        v_val = sum((x - m_val) ** 2 for x in sent_ppls) / len(sent_ppls)
        uncond_sent_ppl_cv = round(math.sqrt(v_val) / max(m_val, 1e-6), 3)

    return {
        "surrogate_model": model,
        "probes": total,
        "gltr_top1": round(top1_hits / total, 3),
        "gltr_top5": round(top5_hits / total, 3),
        "gltr_oob": round(oob_misses / total, 3),
        "uncond_ppl": round(uncond_ppl, 2),
        "uncond_sent_ppl_cv": uncond_sent_ppl_cv,
    }


# ---------------------------------------------------------------------------
# Optional Local Neural AI Detector (DeBERTa / RoBERTa via torch + transformers)
# ---------------------------------------------------------------------------

class LocalNeuralDetector:
    """
    Lazy-loaded, cached local encoder-only neural AI detector (RoBERTa / DeBERTa).
    Runs on CPU by default so it does NOT compete with GPU VRAM used by 8B/4B LLMs.

    Supports both:
      - Desklib's custom mean-pooling DeBERTa-v3-large head (`desklib/ai-text-detector-*`)
      - Standard `AutoModelForSequenceClassification` detectors (`TrustSafeAI/RADAR-Vicuna-7B`,
        `Hello-SimpleAI/chatgpt-detector-roberta`, `fakespot-ai/roberta-base-ai-text-detection-v1`, etc.)
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    @staticmethod
    def _infer_ai_class_index(model_id: str, id2label: Dict[Any, str]) -> int:
        mid = (model_id or "").lower()
        # Verified IBM RADAR convention: class 0 = AI-generated, class 1 = Human
        if "radar" in mid:
            return 0
        # OpenAI RoBERTa detector convention: Fake = 0, Real = 1
        if "roberta-base-openai-detector" in mid or "roberta-large-openai-detector" in mid:
            return 0

        if isinstance(id2label, dict):
            for k, v in id2label.items():
                lbl = str(v).lower()
                if any(tag in lbl for tag in ("ai", "fake", "chatgpt", "gpt", "machine", "generated", "llm", "bot")):
                    try:
                        return int(k)
                    except (TypeError, ValueError):
                        pass
            for k, v in id2label.items():
                lbl = str(v).lower()
                if any(tag in lbl for tag in ("human", "real", "written")):
                    try:
                        return 1 - int(k) if int(k) in (0, 1) else 1
                    except (TypeError, ValueError):
                        pass
        return 1

    def _load(self, model_id: str, device: str = "cpu") -> Dict[str, Any]:
        key = (model_id.strip(), device.strip().lower())
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                return entry

            try:
                import torch
                import torch.nn as nn
                from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
            except ImportError as exc:
                raise RuntimeError(
                    "Local neural AI detector requires `torch` and `transformers`. "
                    "Install with: pip install torch transformers sentencepiece"
                ) from exc

            resolved_dev = device.strip().lower()
            if resolved_dev == "auto":
                resolved_dev = "cuda" if torch.cuda.is_available() else "cpu"
            elif resolved_dev == "cuda" and not torch.cuda.is_available():
                resolved_dev = "cpu"
            torch_dev = torch.device(resolved_dev)

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            is_desklib = "desklib" in model_id.lower()

            if is_desklib:
                class _DesklibAIDetectionModel(PreTrainedModel):
                    config_class = AutoConfig

                    def __init__(self, config):
                        super().__init__(config)
                        self.model = AutoModel.from_config(config)
                        self.classifier = nn.Linear(config.hidden_size, 1)
                        self.init_weights()

                    def forward(self, input_ids, attention_mask=None):
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        last_hidden = outputs[0]
                        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                        pooled = torch.sum(last_hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                        logits = self.classifier(pooled)
                        return {"logits": logits}

                model = _DesklibAIDetectionModel.from_pretrained(model_id)
                model.to(torch_dev)
                model.eval()
                entry = {
                    "kind": "desklib",
                    "model": model,
                    "tokenizer": tokenizer,
                    "device": torch_dev,
                    "max_len": 768,
                    "ai_idx": 0,
                }
            else:
                model = AutoModelForSequenceClassification.from_pretrained(model_id)
                model.to(torch_dev)
                model.eval()
                id2label = getattr(getattr(model, "config", None), "id2label", {}) or {}
                ai_idx = self._infer_ai_class_index(model_id, id2label)
                max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", 512) or 512
                entry = {
                    "kind": "seq_cls",
                    "model": model,
                    "tokenizer": tokenizer,
                    "device": torch_dev,
                    "max_len": min(512, max(256, int(max_pos) - 2)),
                    "ai_idx": ai_idx,
                }

            self._cache[key] = entry
            return entry

    def predict_ai_prob(self, text: str, model_id: str, device: str = "cpu") -> Optional[float]:
        """Return estimated AI probability in [0.0, 1.0] (lower = more human)."""
        if not text or not text.strip() or not model_id or not model_id.strip():
            return None
        import torch
        entry = self._load(model_id.strip(), device=device)
        tokenizer = entry["tokenizer"]
        model = entry["model"]
        dev = entry["device"]

        encoded = tokenizer(
            text.strip(),
            padding=True,
            truncation=True,
            max_length=entry["max_len"],
            return_tensors="pt",
        )
        encoded = {k: v.to(dev) for k, v in encoded.items()}
        with torch.no_grad():
            if entry["kind"] == "desklib":
                out = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
                ai_prob = float(torch.sigmoid(out["logits"].view(-1))[0].item())
            else:
                out = model(**encoded)
                logits = out.logits
                if logits.shape[-1] == 1:
                    ai_prob = float(torch.sigmoid(logits.view(-1))[0].item())
                else:
                    probs = torch.softmax(logits, dim=-1)[0]
                    ai_idx = min(entry["ai_idx"], logits.shape[-1] - 1)
                    ai_prob = float(probs[ai_idx].item())
        return round(max(0.0, min(1.0, ai_prob)), 4)


NEURAL_DETECTOR = LocalNeuralDetector()


# ---------------------------------------------------------------------------
# Unified Multi-Signal Composite Surrogate Aggregator
# ---------------------------------------------------------------------------

def aggregate_surrogate_score(
    *,
    burstiness: Dict[str, Any],
    gltr: Optional[Dict[str, Any]] = None,
    binoculars_metrics: Optional[Dict[str, Any]] = None,
    neural_ai_prob: Optional[float] = None,
    binoculars_threshold: float = 0.90,
) -> Dict[str, Any]:
    """
    Combine all available independent local surrogate signals into:
      - `human_score`: calibrated 0.0-1.0 likelihood of clearing external detectors
      - `human_score_pct`: 0-100 integer badge for UI
      - `passes_surrogate`: boolean pass gate used alongside fidelity & judge
      - flat diagnostic fields (`burstiness_score`, `sent_len_cv`, `sent_ppl_cv`,
        `overlap_3gram`, `gltr_top1`, `gltr_oob`, `uncond_ppl`, `binoculars`, `neural_ai_prob`)
    """
    gltr = gltr or {}
    bino = binoculars_metrics or {}

    b_score = float(burstiness.get("burstiness_score", 0.5))
    overlap_3 = float(burstiness.get("overlap_3gram", 0.0))
    # Anti-paraphrase lexical distance component (overlap <= 0.08 -> 1.0, overlap >= 0.35 -> 0.15)
    anti_para_score = max(0.15, min(1.0, 1.0 - max(0.0, overlap_3 - 0.06) / 0.32))

    # Merge GLTR & unconditional metrics (exact Binoculars pass overrides strided Ollama probe)
    gltr_top1 = bino.get("gltr_top1") if bino.get("gltr_top1") is not None else gltr.get("gltr_top1")
    gltr_top5 = bino.get("gltr_top5") if bino.get("gltr_top5") is not None else gltr.get("gltr_top5")
    gltr_oob = bino.get("gltr_oob") if bino.get("gltr_oob") is not None else gltr.get("gltr_oob")
    uncond_ppl = bino.get("uncond_inst_ppl") if bino.get("uncond_inst_ppl") is not None else gltr.get("uncond_ppl")
    binoculars_val = bino.get("binoculars")
    sent_ppl_cv = (
        bino.get("uncond_sent_ppl_cv")
        if bino.get("uncond_sent_ppl_cv") is not None
        else (
            gltr.get("uncond_sent_ppl_cv")
            if gltr.get("uncond_sent_ppl_cv") is not None
            else burstiness.get("sent_ppl_cv")
        )
    )

    # Weighted blend across whichever orthogonal signals are active
    weighted_sum = 0.0
    total_w = 0.0

    # 1. Rhythm & anti-paraphrase structure (always active)
    weighted_sum += 0.22 * b_score + 0.16 * anti_para_score
    total_w += 0.38

    # 2. Unconditional GLTR signal (truncation-free! Human top1 ~ 0.22-0.44, AI ~ 0.58-0.78)
    gltr_human_sub = None
    if gltr_top1 is not None:
        t1_human = max(0.0, min(1.0, (0.68 - float(gltr_top1)) / 0.36))
        oob_human = max(0.0, min(1.0, (float(gltr_oob or 0.0) - 0.04) / 0.22))
        gltr_human_sub = 0.68 * t1_human + 0.32 * oob_human
        weighted_sum += 0.32 * gltr_human_sub
        total_w += 0.32

    # 3. Exact Binoculars score (Hans et al., 2024: human >= ~0.90-0.96, AI <= ~0.82)
    bino_human_sub = None
    if binoculars_val is not None:
        # Smooth ramp from (threshold - 0.14) -> 0.0 to (threshold + 0.08) -> 1.0
        bino_human_sub = max(0.0, min(1.0, (float(binoculars_val) - (binoculars_threshold - 0.14)) / 0.22))
        weighted_sum += 0.38 * bino_human_sub
        total_w += 0.38

    # 4. Local cross-architecture Neural AI Detector (DeBERTa/RoBERTa — strongest external proxy!)
    neural_human_sub = None
    if neural_ai_prob is not None:
        neural_human_sub = max(0.0, min(1.0, 1.0 - float(neural_ai_prob)))
        weighted_sum += 0.55 * neural_human_sub
        total_w += 0.55

    human_score = weighted_sum / max(total_w, 1e-6)

    # Hard veto if an active high-confidence external detector flags the candidate strongly as AI
    passes = True
    if neural_ai_prob is not None and float(neural_ai_prob) > 0.38:
        passes = False
    if binoculars_val is not None and float(binoculars_val) < binoculars_threshold:
        passes = False
    if gltr_top1 is not None and float(gltr_top1) > 0.56:
        passes = False
    if overlap_3 > 0.26:
        passes = False
    if human_score < 0.58:
        passes = False

    return {
        "human_score": round(max(0.0, min(1.0, human_score)), 3),
        "human_score_pct": int(round(100.0 * max(0.0, min(1.0, human_score)))),
        "passes_surrogate": passes,
        "burstiness_score": round(b_score, 3),
        "sent_len_cv": burstiness.get("sent_len_cv"),
        "sent_ppl_cv": round(sent_ppl_cv, 3) if sent_ppl_cv is not None else None,
        "overlap_3gram": round(overlap_3, 3),
        "gltr_top1": round(float(gltr_top1), 3) if gltr_top1 is not None else None,
        "gltr_top5": round(float(gltr_top5), 3) if gltr_top5 is not None else None,
        "gltr_oob": round(float(gltr_oob), 3) if gltr_oob is not None else None,
        "uncond_ppl": round(float(uncond_ppl), 2) if uncond_ppl is not None else None,
        "binoculars": round(float(binoculars_val), 4) if binoculars_val is not None else None,
        "neural_ai_prob": round(float(neural_ai_prob), 4) if neural_ai_prob is not None else None,
        "surrogate_model": gltr.get("surrogate_model"),
    }


def evaluate_candidate_surrogates(
    candidate_text: str,
    source_text: str = "",
    *,
    sent_nlls: Optional[Sequence[Sequence[float]]] = None,
    # Ollama unconditional cross-family GLTR probe
    ollama_base: str = backends.DEFAULT_OLLAMA_BASE,
    surrogate_ollama_model: str = "",
    surrogate_probes: int = 16,
    num_ctx: int = backends.DEFAULT_NUM_CTX,
    # Exact llama.cpp Binoculars (pass already-loaded Llama handles if active)
    llama_base_llm: Any = None,
    llama_inst_llm: Any = None,
    binoculars_threshold: float = 0.90,
    # Optional local HF neural AI detector
    neural_detector_model: str = "",
    neural_detector_device: str = "cpu",
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """Run all configured local surrogate evaluators on a finished candidate paragraph."""
    bino_metrics: Dict[str, Any] = {}
    if llama_base_llm is not None and llama_inst_llm is not None:
        try:
            bino_metrics = backends.LLAMA_CPP_MANAGER.score_unconditional_binoculars(
                llama_base_llm, llama_inst_llm, candidate_text
            )
        except Exception:
            bino_metrics = {}

    gltr_metrics: Dict[str, Any] = {}
    if not bino_metrics and surrogate_ollama_model and surrogate_ollama_model.strip():
        gltr_metrics = score_ollama_unconditional_gltr(
            candidate_text,
            ollama_base=ollama_base,
            model=surrogate_ollama_model.strip(),
            num_probes=surrogate_probes,
            num_ctx=num_ctx,
            cancel_event=cancel_event,
        )

    override_cv = bino_metrics.get("uncond_sent_ppl_cv")
    if override_cv is None:
        override_cv = gltr_metrics.get("uncond_sent_ppl_cv")

    burst = compute_burstiness_and_overlap(
        candidate_text,
        source_text,
        sent_nlls=sent_nlls,
        override_sent_ppl_cv=override_cv,
    )

    neural_prob: Optional[float] = None
    if neural_detector_model and neural_detector_model.strip():
        try:
            neural_prob = NEURAL_DETECTOR.predict_ai_prob(
                candidate_text,
                model_id=neural_detector_model.strip(),
                device=neural_detector_device,
            )
        except Exception:
            neural_prob = None

    return aggregate_surrogate_score(
        burstiness=burst,
        gltr=gltr_metrics,
        binoculars_metrics=bino_metrics,
        neural_ai_prob=neural_prob,
        binoculars_threshold=binoculars_threshold,
    )
