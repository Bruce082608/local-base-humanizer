#!/usr/bin/env python3
"""
Core engine for Local Base-Model Humanizer Studio (Ollama).

Supports two complementary local decoding engines:
  1. anti_greedy (ZeroGPT Mode):
     Unaligned local base model with token-level logprobs rank-1 skipping,
     hard AI-buzzword logit blacklist filtering, alternating verified 0%-AI
     in-context demos, and automatic per-paragraph PPL retry.
  2. contrastive (GPTZero Mode — Contrastive Anti-RLHF Decoding):
     Same-vocabulary local pair (Base model minus alpha * Instruct model):
       score(t) = log P_base(t) - alpha * log P_instruct(t)
     Mathematically cancels RLHF/instruction-tuned token shifts while staying
     within the base model's plausible pre-training distribution, tracking
     both BasePPL and InstPPL to bypass both AI-Generated and AI-Paraphrased
     classifiers.

Length preservation:
  - Automatically splits long paragraphs (> ~105 words) along sentence boundaries
    into balanced chunks so small 4B models do not compress 300-600 word blocks
    into short summaries, then stitches chunks back seamlessly.
  - Dynamically scales min_tokens / max_tokens from source word count and
    length_ratio (default 1.0x), suppressing premature EOS/newline stops until
    the target word budget is reached, and finishing cleanly on a sentence period.
"""

import argparse
import json
import math
import random
import re
import sys
import threading
import time
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_MODEL = "lab-qwen3-4b-base:latest"
DEFAULT_INSTRUCT_MODEL = "lab-qwen3-4b:latest"
DEFAULT_LENGTH_RATIO = 1.0

DEFAULT_BANNED_SUBSTRS = [
    "utiliz", "employ", "leverag", "demonstrat", "illustrat", "highlight",
    "underscor", "notabl", "furthermore", "moreover", "meanwhile", "similarly",
    "additionally", "consequently", "subsequently", "specifically", "significantly",
    "effectively", "overall", "allowing", "enabling", "ensuring", "resulting",
    "providing", "supporting", "offering", "incorporating", "integrating",
    "enhancing", "facilitat", "crucial", "pivotal", "paramount", "landscape",
    "paradigm", "framework", "methodology", "scalability", "robust", "seamless",
    "comprehensive", "numerous", "delve", "tapestry", "testament", "multifaceted",
    "intricate", "evolving", "dynamic", "innovative", "cutting-edge", "—", "–",
    "suggest", "together", "summary", "thus", "hence", "therefore", "complementary",
    "provisional", "feasibility", "adjacent", "conclusion", "various", "focus",
    "briefly", "potential", "indicating", "improving", "describing", "transform",
    "instance", "fact", "note", "collectively", "example", "recommend",
    "important", "topic", "variety", "discuss", "solve", "issues", "further",
]

DEMO_A = """Notes:
Chang et al. [6] report a campus electricity-data platform where Hadoop handles collection and storage, Hive handles warehousing, Spark does ETL, and Impala handles queries. Preview reports query and ETL tests across different processing-node counts. Neither [5] nor [6] is direct PV or wind forecasting evidence.

Paragraph:
For Chang et al. [6], the authors report a campus electricity-data platform where the Hadoop cluster is used to manage the data collection and storage, whereas Hive helps with warehousing. For the rest of the data management, Spark is used to run data integration jobs, and for the query, they use the Impala. They report their ETL performance tests with different data sizes across different numbers of nodes. Although these papers are not directly related to PV and wind forecasting, Chang et al. [6] help the readers choose a suitable storage platform for the future PV or wind data."""

DEMO_B = """Notes:
Liu et al. [3] built ultra-short-term wind-power prediction on a 5-node Hadoop cluster with Intel Core i5 CPUs and Ubuntu. The model mixes multivariate phase-space reconstruction with multivariate linear regression using past power and chosen NWP variables. Five MapReduce jobs run in parallel and HDFS stores data. Tested on Hunan wind-farm data in China. Faster calculation reported without numerical speedup ratio.

Paragraph:
In the wind-power system that was built by Liu et al. [3], ultra-short-term prediction was run on a five-node Hadoop cluster with Intel Core i5 processors and Ubuntu. A combination of multivariate phase-space reconstruction and multivariate linear regression is used to combine past power measurements and selected weather variables. Five MapReduce jobs execute in parallel, with HDFS handling data distributed between the nodes. The system is tested on wind-farm data in the Chinese province of Hunan. Although there was a faster calculation time, there was no significant numerical speedup, and the target is to improve predictions of wind power generation instead of wind speed."""

HEADING_RE = re.compile(r"^(?:#{1,6}\s+.{1,100}|\d+(?:\.\d+)*\.?(?:\s+[A-Z].{0,100})?)$")
WORD_START_RE = re.compile(r"^ [A-Za-z][A-Za-z'\-]*$")
STOP_TOKENS = ("<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<think>", "</think>", "")


def normalize_ollama_base(url: str) -> str:
    u = (url or DEFAULT_OLLAMA_BASE).strip().rstrip("/")
    if u.endswith("/api/generate"):
        u = u[: -len("/api/generate")]
    elif u.endswith("/api/tags"):
        u = u[: -len("/api/tags")]
    return u


def list_ollama_models(ollama_base: str = DEFAULT_OLLAMA_BASE, timeout: float = 5.0) -> Tuple[bool, List[Dict], str]:
    base = normalize_ollama_base(ollama_base)
    try:
        req = urllib.request.Request(f"{base}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("models", [])
            result = [
                {
                    "name": m.get("name", ""),
                    "size": m.get("size", 0),
                    "is_base": "base" in m.get("name", "").lower(),
                }
                for m in models
                if m.get("name")
            ]
            result.sort(key=lambda x: (not x["is_base"], x["name"]))
            return True, result, ""
    except Exception as exc:
        return False, [], str(exc)


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 110 or s.endswith("."):
        return False
    return bool(HEADING_RE.match(s))


def parse_blocks(raw_text: str) -> List[Dict[str, str]]:
    """Split raw text into heading and paragraph blocks."""
    blocks: List[Dict[str, str]] = []
    buf: List[str] = []
    for line in raw_text.splitlines():
        s = line.strip()
        if not s:
            if buf:
                blocks.append({"kind": "para", "text": " ".join(buf)})
                buf = []
            continue
        if is_heading(s):
            if buf:
                blocks.append({"kind": "para", "text": " ".join(buf)})
                buf = []
            blocks.append({"kind": "heading", "text": s})
        else:
            # Preserve Word/web single-newline paragraph breaks when a previous line is already a full paragraph
            if buf and len(buf[-1]) >= 120 and buf[-1][-1] in ".!?)]\"'":
                blocks.append({"kind": "para", "text": " ".join(buf)})
                buf = [s]
            else:
                buf.append(s)
    if buf:
        blocks.append({"kind": "para", "text": " ".join(buf)})
    return blocks


def split_sentences(text: str) -> List[str]:
    """Split English academic prose into sentences without breaking on 'et al.' or decimals."""
    s = re.sub(r"\s+", " ", text.strip())
    if not s:
        return []
    # Protect common abbreviations from sentence splitting
    protected = re.sub(r"\b(et al|e\.g|i\.e|vs|cf|Fig|Tab|Eq|No|Vol|Dr|Mr|Ms)\.", r"\1<DOT>", s)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[\(\"'])", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def chunk_paragraph(para: str, max_chunk_words: int = 160) -> List[str]:
    """
    Split only truly long paragraphs (> ~175 words) into balanced sentence-aligned chunks
    (~110-160 words each). Normal 70-170 word paragraphs stay unchunked to avoid artificial
    mid-paragraph transition seams that trigger AI detectors.
    """
    words = para.split()
    if len(words) <= max_chunk_words + 15:
        return [para.strip()]

    sents = split_sentences(para)
    if len(sents) <= 1:
        return [para.strip()]

    # Aim for balanced chunks rather than leaving a tiny tail
    n_chunks = max(2, math.ceil(len(words) / max_chunk_words))
    target_per_chunk = len(words) / n_chunks

    chunks: List[str] = []
    cur: List[str] = []
    cur_words = 0

    for idx, sent in enumerate(sents):
        sw = len(sent.split())
        remaining_chunks = n_chunks - len(chunks)
        if cur and remaining_chunks > 1 and (cur_words + sw > target_per_chunk * 1.15) and cur_words >= 55:
            chunks.append(" ".join(cur))
            cur = [sent]
            cur_words = sw
        else:
            cur.append(sent)
            cur_words += sw
    if cur:
        chunks.append(" ".join(cur))
    return chunks


DEMO_LEAK_WORDS = {
    "chang", "impala", "campus", "hunan", "ubuntu", "intel", "i5",
    "phase-space", "warehousing", "province", "etl", "china", "chinese",
}


def _chunk_demo_blacklist(chunk_text: str) -> List[str]:
    """Return demo-specific entities not present in chunk_text so they can never leak."""
    low = chunk_text.lower()
    return [w for w in DEMO_LEAK_WORDS if w not in low]


def _causes_demo_leak_or_repeat(cur_text: str, tok: str, demo_ban: List[str]) -> bool:
    """Catch BPE-split demo leaks (e.g. ' Hu'+'nan') and 3-gram repetition loops."""
    tail = (cur_text[-60:] + tok).lower()
    if any(b in tail for b in demo_ban):
        return True
    # Block exact 4-word repetition loops inside the same paragraph
    cand = re.sub(r"\s+", " ", (cur_text + tok)).strip().lower()
    words = cand.split()
    if len(words) >= 8:
        last4 = " ".join(words[-4:])
        prev = " ".join(words[:-4])
        if last4 in prev:
            return True
    return False


def is_banned(tok: str, banned_substrs: List[str]) -> bool:
    t = tok.lower().strip()
    if not t:
        return False
    return any(b in t for b in banned_substrs if b)


def is_clean_ascii(tok: str) -> bool:
    return all(32 <= ord(c) < 127 or c in "\n\t" for c in tok)


def next_token_logprobs(
    ollama_base: str,
    model: str,
    raw_prompt: str,
    top_k: int = 20,
    timeout: float = 60.0,
    keep_alive: str = "10m",
) -> Tuple[Optional[str], List[Tuple[str, float]]]:
    url = f"{normalize_ollama_base(ollama_base)}/api/generate"
    payload = {
        "model": model,
        "prompt": raw_prompt,
        "raw": True,
        "stream": False,
        "keep_alive": keep_alive,
        "logprobs": True,
        "top_logprobs": top_k,
        "options": {"num_predict": 1, "temperature": 0.0},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        d = json.loads(resp.read().decode("utf-8"))
        lps = d.get("logprobs", [])
        if not lps:
            return None, []
        top = lps[0].get("top_logprobs", [])
        return lps[0].get("token", ""), [(x["token"], x["logprob"]) for x in top]


def make_seed_start(para: str, idx: int) -> str:
    """Create a concrete, non-copying seed opening for Anti-Greedy mode."""
    stripped = para.strip()
    lead_m = re.match(r"^((?:[A-Z]{1,4}\d+\.\s+)?[A-Z][A-Za-z0-9\-\s]{3,55}\.)\s+(.+)$", stripped)
    if lead_m and not lead_m.group(1).lower().endswith("et al."):
        lead_title = lead_m.group(1)
        starters = [
            f"{lead_title} In future work here,",
            f"{lead_title} Under this kind of setup,",
            f"{lead_title} For this part of the study,",
            f"{lead_title} In practical prototypes here,",
        ]
        return starters[idx % len(starters)]

    # Match an author name in the first three sentences
    sents = split_sentences(stripped)
    head_text = " ".join(sents[:3]) if sents else stripped
    m = re.search(r"\b([A-Z][A-Za-z\-]+(?:\s+et\s+al\.|\s+and\s+[A-Z][A-Za-z\-]+))\b", head_text)
    if m:
        author = m.group(1)
        author_starters = [
            f"In the study that was built by {author},",
            f"For {author}, the authors",
            f"In the setup studied by {author},",
            f"For the work of {author},",
        ]
        return author_starters[idx % len(author_starters)]

    # Never copy source words[:4] verbatim (causes ZeroGPT to flag the whole paragraph).
    # Rotate across diverse, natural plain-English openings so consecutive paragraphs never repeat a template.
    cites = re.findall(r"\[\d+\](?:[–\-]\[\d+\])?", stripped)
    unique_cites: List[str] = []
    for c in cites:
        if c not in unique_cites:
            unique_cites.append(c)

    diverse_openers = [
        f"In the setup covered by {unique_cites[0]}," if unique_cites else "When we look at this setup,",
        "For this stage of the deployment,",
        f"Based on the results from {unique_cites[0]}," if unique_cites else "From the reviewed cases here,",
        "In a practical utility setting,",
        f"When looking closely at {unique_cites[0]} and {unique_cites[1]}," if len(unique_cites) >= 2 else "To make this work on real hardware,",
        "On the operational side here,",
        f"For the cases reported in {unique_cites[0]}," if unique_cites else "Before scaling up to full operation,",
        "With these kinds of field constraints,",
    ]
    return diverse_openers[idx % len(diverse_openers)]


def _ends_clean_sentence(text: str) -> bool:
    """True when text ends on a real sentence period (not 'et al.' or decimal '2.')."""
    s = text.rstrip()
    if not s.endswith("."):
        return False
    low = s.lower()
    if low.endswith(("et al.", "e.g.", "i.e.", "vs.", "fig.", "eq.", "no.")):
        return False
    if re.search(r"\b\d{1,3}\.$", s):
        return False
    return s.count("(") == s.count(")")


CONTRASTIVE_REGISTERS = [
    (
        "Notes from reading technical papers:\n{notes}\n\n"
        "Rough plain prose paragraph (complete sentences, direct, concrete):\n{anchor}"
    ),
    (
        "Engineering literature notes:\n{notes}\n\n"
        "Direct plain-English technical paragraph (factual, concrete numbers, complete sentences):\n{anchor}"
    ),
    (
        "Summary notes from technical studies:\n{notes}\n\n"
        "Plain academic draft paragraph (straightforward verbs, varied sentence lengths):\n{anchor}"
    ),
    (
        "Technical review notes:\n{notes}\n\n"
        "Concrete plain-prose technical paragraph (active voice, clear facts):\n{anchor}"
    ),
    (
        "Workshop paper scratch notes:\n{notes}\n\n"
        "Unpolished academic prose paragraph (direct statements, factual, complete sentences):\n{anchor}"
    ),
    (
        "Reading notes on engineering systems:\n{notes}\n\n"
        "Plain factual paragraph (direct comparisons, concrete system details):\n{anchor}"
    ),
]


def _extract_topic_phrase(para: str) -> str:
    """Extract a concrete 2-3 word domain phrase from para to anchor Contrastive mode factually."""
    # Prefer concrete technical systems / capitalized terms or key nouns from the first 2 sentences
    sents = split_sentences(para.strip())
    head = " ".join(sents[:2]) if sents else para
    for term in [
        "Spark Streaming", "Hadoop MapReduce", "Spark", "Flink", "Kafka", "HDFS", "HBase",
        "FedNorm", "SMASH", "BBData", "LoRaWAN", "MQTT", "SCADA", "AMI", "iPACS-5612C1",
        "wind-farm", "wind farm", "smart-meter", "smart meter", "microgrid", "photovoltaic",
        "solar-farm", "edge terminal", "edge servers", "federated", "cluster tuning",
        "centralized service", "checkpointing", "pilot project",
    ]:
        if term.lower() in head.lower():
            return term
    # Generic noun-phrase fallback: first 2 content words (>4 chars)
    stop = {"along", "these", "methods", "which", "their", "there", "where", "while", "about", "order", "could", "would", "should", "using", "based"}
    cands = [w.strip(".,;:()[]") for w in head.split() if len(w.strip(".,;:()[]")) >= 5 and w.strip(".,;:()[]").lower() not in stop]
    return " ".join(cands[:2]) if cands else "this setup"


def make_contrastive_anchor(para: str, idx: int) -> str:
    """Create a varied, topic-grounded 3-6 word factual anchor (zero hallucination, zero repetition)."""
    stripped = para.strip()

    sents = split_sentences(stripped)
    head = " ".join(sents[:2]) if sents else stripped
    auth_m = re.search(r"\b([A-Z][A-Za-z\-]+(?:\s+et\s+al\.|\s+and\s+[A-Z][A-Za-z\-]+))\b", head)

    cites = re.findall(r"\[\d+\](?:[–\-]\[\d+\])?", stripped)
    unique_cites: List[str] = []
    for c in cites:
        if c not in unique_cites:
            unique_cites.append(c)

    lead_m = re.match(r"^((?:[A-Z]{1,4}\d+\.\s+)?[A-Z][A-Za-z0-9\-\s]{2,40})\.", stripped)
    has_lead = bool(lead_m and not lead_m.group(1).lower().endswith("et al"))
    topic = _extract_topic_phrase(stripped)

    bucket = idx % 6
    if bucket == 0 and auth_m:
        return f"{auth_m.group(1)} looked at {topic}"
    if bucket == 1 and unique_cites:
        return f"On {topic} in {unique_cites[0]},"
    if bucket == 2 and has_lead:
        return f"For {lead_m.group(1).lower()},"
    if bucket == 3 and auth_m:
        return f"The {topic} setup by {auth_m.group(1)}"
    if bucket == 4 and unique_cites:
        return f"Looking at {topic} ({unique_cites[0]}),"
    if bucket == 5 and unique_cites:
        return f"For {topic} in {unique_cites[0]},"

    if auth_m:
        return f"{auth_m.group(1)} tested {topic}"
    if has_lead:
        return f"About {lead_m.group(1).lower()},"
    if unique_cites:
        return f"With {topic} in {unique_cites[0]},"
    return f"For {topic},"


def trim_to_last_sentence(text: str) -> str:
    # Strip any accidental few-shot label echo (e.g. "Source:", "Rewrite:", "Notes:")
    text = re.split(r"\b(?:Source|Rewrite|Notes|Paragraph)\s*:", text, maxsplit=1)[0]
    text = re.sub(r"\s+", " ", text).strip()
    matches = [
        m for m in re.finditer(r"(?<!\bal)\.(?:\s|$)", text)
        if not re.search(r"\b\d{1,3}\.$", text[: m.start() + 1])
    ]
    if matches:
        return text[: matches[-1].start() + 1].strip()
    # Ensure chunk ends with sentence punctuation when stitched with following chunks
    cleaned = text.rstrip(" ,;:-")
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


AI_LEX_STEMS = (
    "ensur", "requir", "variou", "provid", "allow", "enabl", "support", "involv",
    "includ", "approach", "method", "manag", "integrat", "evaluat", "perform",
    "consider", "indicat", "however", "moreover", "further", "particular",
    "specific", "overall", "similar", "complex", "effectiv", "efficien",
    "comprehen", "robust", "crucial", "vital", "essential", "significan",
    "utiliz", "employ", "leverag", "demonstrat", "facilitat", "validat",
)


def _human_burstiness_and_lex(text: str) -> Tuple[float, float]:
    """Return (sentence_length_std, ai_lexicon_pct) for ZeroGPT-aware local candidate ranking."""
    sents = split_sentences(text)
    slens = [len(s.split()) for s in sents if s.strip()]
    if len(slens) >= 2:
        mean_s = sum(slens) / len(slens)
        sent_std = (sum((x - mean_s) ** 2 for x in slens) / len(slens)) ** 0.5
    else:
        sent_std = 0.0
    words = [w.strip(".,;:()[]\"'").lower() for w in text.split() if w.strip(".,;:()[]\"'")]
    if not words:
        return sent_std, 100.0
    hits = sum(1 for w in words if any(st in w for st in AI_LEX_STEMS))
    return sent_std, 100.0 * hits / len(words)


def _length_penalty_factor(actual_words: int, target_words: int) -> float:
    """Score multiplier in (0, 1] favoring outputs close to target_words."""
    if target_words <= 0:
        return 1.0
    ratio = actual_words / target_words
    if 0.85 <= ratio <= 1.25:
        return 1.0
    if ratio < 0.85:
        return max(0.15, (ratio / 0.85) ** 2)
    return max(0.4, 1.25 / ratio)


def _rewrite_chunk_anti_greedy(
    chunk_text: str,
    demo: str,
    seed_start: str,
    *,
    ollama_base: str,
    model: str,
    length_ratio: float,
    p_skip_top1: float,
    max_p1_for_skip: float,
    pool_size: int,
    min_alt_prob: float,
    min_ppl: float,
    attempts: int,
    base_seed: int,
    banned_clean: List[str],
    on_progress: Optional[Callable[[Dict], None]],
    cancel_event: Optional[threading.Event],
    prefix_so_far: str = "",
) -> Dict:
    demo_ban = _chunk_demo_blacklist(chunk_text)
    chunk_banned = banned_clean + demo_ban
    src_words = max(1, len(chunk_text.split()))
    target_words = max(20, int(round(src_words * length_ratio)))
    seed_words = len(seed_start.split())
    gen_target_words = max(15, target_words - seed_words)
    # Gate EOS/newlines directly on word progress so output matches target_words reliably
    soft_stop_words = max(target_words - 4, int(round(target_words * 0.94)))
    max_tokens = max(95, int(gen_target_words * 1.70) + 40)

    best_result = {
        "text": chunk_text.strip(),
        "ppl": 0.0,
        "tokens": 0,
        "skipped": 0,
        "word_starts": 0,
        "attempt": 0,
        "seed": base_seed,
        "elapsed": 0.0,
        "cancelled": False,
    }
    best_rank = -1.0

    for attempt in range(1, max(1, attempts) + 1):
        if cancel_event and cancel_event.is_set():
            best_result["cancelled"] = True
            return best_result

        seed_val = base_seed + (attempt - 1) * 37
        ctx = (
            f"{demo.strip()}\n\n"
            f"Notes:\n"
            f"{chunk_text.strip()}\n\n"
            f"Paragraph:\n{seed_start}"
        )
        rng = random.Random(seed_val)
        gen = ""
        nlls: List[float] = []
        skipped = 0
        word_starts = 0
        t0 = time.time()

        if on_progress:
            live = f"{prefix_so_far} {seed_start}".strip() if prefix_so_far else seed_start
            on_progress({
                "event": "attempt_start",
                "mode": "anti_greedy",
                "attempt": attempt,
                "attempts": attempts,
                "seed": seed_val,
                "text": live,
            })

        cur_words = seed_words
        for step in range(max_tokens):
            if cancel_event and cancel_event.is_set():
                best_result["cancelled"] = True
                return best_result

            tok0, top = next_token_logprobs(ollama_base, model, ctx + gen, top_k=20)
            if not top:
                break

            cur_full_pre = re.sub(r"\s+", " ", (seed_start + gen)).strip()
            at_sent_end = _ends_clean_sentence(cur_full_pre)
            reached_len = (cur_words >= soft_stop_words and at_sent_end) or (cur_words >= int(target_words * 1.22))

            # Suppress premature EOS / newline until word budget reached
            def _tok_allowed(t: str) -> bool:
                if t in STOP_TOKENS:
                    return False
                if not is_clean_ascii(t):
                    return False
                if not reached_len and "\n" in t:
                    return False
                return True

            if reached_len and tok0 in STOP_TOKENS:
                break

            clean = [
                (t, lp)
                for (t, lp) in top
                if _tok_allowed(t)
                and not is_banned(t, chunk_banned)
                and not _causes_demo_leak_or_repeat(seed_start + gen, t, demo_ban)
            ]
            if not clean:
                clean = [
                    (t, lp)
                    for (t, lp) in top
                    if _tok_allowed(t) and not _causes_demo_leak_or_repeat(seed_start + gen, t, demo_ban)
                ]
            if not clean:
                clean = [(t, lp) for (t, lp) in top if _tok_allowed(t)]
            if not clean:
                break

            chosen_tok, chosen_lp = clean[0]
            p0 = math.exp(chosen_lp)
            if WORD_START_RE.match(chosen_tok):
                word_starts += 1
                if p0 <= max_p1_for_skip and rng.random() < p_skip_top1:
                    alts = [
                        (t, lp)
                        for (t, lp) in clean[1 : max(2, pool_size + 1)]
                        if WORD_START_RE.match(t) and math.exp(lp) >= min_alt_prob and "\n" not in t
                    ]
                    if alts:
                        weights = [math.exp(lp) ** 0.45 for _, lp in alts]
                        chosen_tok, chosen_lp = rng.choices(alts, weights=weights, k=1)[0]
                        skipped += 1

            if reached_len and "\n" in chosen_tok:
                break

            gen += chosen_tok
            nlls.append(-chosen_lp)

            cur_full = re.sub(r"\s+", " ", (seed_start + gen)).strip()
            cur_words = len(cur_full.split())

            if on_progress and (step % 4 == 0 or chosen_tok.endswith(".")):
                running_ppl = math.exp(sum(nlls) / max(len(nlls), 1))
                live = f"{prefix_so_far} {cur_full}".strip() if prefix_so_far else cur_full
                on_progress({
                    "event": "token",
                    "mode": "anti_greedy",
                    "attempt": attempt,
                    "attempts": attempts,
                    "text": live,
                    "ppl": round(running_ppl, 2),
                    "tokens": len(nlls),
                })

            # Stop gracefully at the first clean sentence end after reaching target word budget
            if cur_words >= soft_stop_words and _ends_clean_sentence(cur_full):
                break

        elapsed = round(time.time() - t0, 2)
        ppl = math.exp(sum(nlls) / max(len(nlls), 1)) if nlls else 0.0
        full = trim_to_last_sentence(seed_start + gen)
        actual_words = len(full.split())
        len_factor = _length_penalty_factor(actual_words, target_words)
        ok_len = actual_words >= int(target_words * 0.85)

        cand = {
            "text": full,
            "ppl": round(ppl, 2),
            "tokens": len(nlls),
            "skipped": skipped,
            "word_starts": word_starts,
            "attempt": attempt,
            "seed": seed_val,
            "elapsed": elapsed,
            "cancelled": False,
        }

        rank = min(ppl, 12.0) * len_factor
        if rank > best_rank and actual_words >= 10:
            best_rank = rank
            best_result = cand

        if ppl >= min_ppl and ok_len:
            break

    return best_result


def rewrite_paragraph(
    source_para: str,
    demo: str,
    seed_start: str,
    *,
    ollama_base: str = DEFAULT_OLLAMA_BASE,
    model: str = DEFAULT_MODEL,
    length_ratio: float = DEFAULT_LENGTH_RATIO,
    max_tokens: Optional[int] = None,
    p_skip_top1: float = 0.72,
    max_p1_for_skip: float = 0.88,
    pool_size: int = 6,
    min_alt_prob: float = 0.01,
    min_ppl: float = 4.2,
    attempts: int = 3,
    base_seed: int = 401,
    banned_substrs: Optional[List[str]] = None,
    on_progress: Optional[Callable[[Dict], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict:
    """ZeroGPT Mode: Length-preserving paragraph rewrite (auto-chunks long paragraphs)."""
    if banned_substrs is None:
        banned_substrs = DEFAULT_BANNED_SUBSTRS
    banned_clean = [b.lower().strip() for b in banned_substrs if b.strip()]

    chunks = chunk_paragraph(source_para, max_chunk_words=160)
    demos = [demo, DEMO_B if demo == DEMO_A else DEMO_A]

    parts: List[str] = []
    total_tokens = 0
    total_skipped = 0
    total_ws = 0
    total_elapsed = 0.0
    weighted_log_ppl = 0.0
    last_attempt = 1

    for c_idx, chunk in enumerate(chunks):
        c_seed_start = seed_start if c_idx == 0 else make_seed_start(chunk, c_idx)
        c_demo = demos[c_idx % 2]
        prefix_so_far = " ".join(parts)

        res = _rewrite_chunk_anti_greedy(
            chunk,
            demo=c_demo,
            seed_start=c_seed_start,
            ollama_base=ollama_base,
            model=model,
            length_ratio=length_ratio,
            p_skip_top1=p_skip_top1,
            max_p1_for_skip=max_p1_for_skip,
            pool_size=pool_size,
            min_alt_prob=min_alt_prob,
            min_ppl=min_ppl,
            attempts=attempts,
            base_seed=base_seed + c_idx * 131,
            banned_clean=banned_clean,
            on_progress=on_progress,
            cancel_event=cancel_event,
            prefix_so_far=prefix_so_far,
        )
        if res.get("cancelled"):
            return {
                "mode": "anti_greedy",
                "text": " ".join(parts) if parts else source_para.strip(),
                "ppl": 0.0,
                "inst_ppl": None,
                "tokens": total_tokens,
                "skipped": total_skipped,
                "word_starts": total_ws,
                "skip_ratio": 0.0,
                "attempt": res.get("attempt", 0),
                "seed": base_seed,
                "elapsed": round(total_elapsed, 2),
                "cancelled": True,
            }

        parts.append(res["text"])
        toks = max(1, res["tokens"])
        total_tokens += res["tokens"]
        total_skipped += res["skipped"]
        total_ws += res["word_starts"]
        total_elapsed += res["elapsed"]
        weighted_log_ppl += math.log(max(1.0, res["ppl"])) * toks
        last_attempt = res["attempt"]

    combined_text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    agg_ppl = math.exp(weighted_log_ppl / max(1, total_tokens)) if total_tokens else 0.0
    skip_ratio = round(100.0 * total_skipped / max(1, total_ws), 1)
    src_words = max(1, len(source_para.split()))
    out_words = len(combined_text.split())

    final = {
        "mode": "anti_greedy",
        "text": combined_text,
        "ppl": round(agg_ppl, 2),
        "inst_ppl": None,
        "tokens": total_tokens,
        "skipped": total_skipped,
        "word_starts": total_ws,
        "skip_ratio": skip_ratio,
        "source_words": src_words,
        "output_words": out_words,
        "length_pct": round(100.0 * out_words / src_words),
        "chunks": len(chunks),
        "attempt": last_attempt,
        "seed": base_seed,
        "elapsed": round(total_elapsed, 2),
        "cancelled": False,
    }
    if on_progress:
        on_progress({"event": "attempt_done", **final})
    return final


def _rewrite_chunk_contrastive(
    chunk_text: str,
    anchor: str,
    *,
    ollama_base: str,
    base_model: str,
    instruct_model: str,
    length_ratio: float,
    alpha: float,
    min_base_prob: float,
    temp: float,
    pool_size: int,
    min_inst_ppl: float,
    attempts: int,
    base_seed: int,
    on_progress: Optional[Callable[[Dict], None]],
    cancel_event: Optional[threading.Event],
    prefix_so_far: str = "",
) -> Dict:
    demo_leak_ban = _chunk_demo_blacklist(chunk_text)
    src_words = max(1, len(chunk_text.split()))
    target_words = max(22, int(round(src_words * length_ratio)))
    anchor_words = len(anchor.split())
    gen_target_words = max(18, target_words - anchor_words)

    # Gate EOS/newlines directly on word progress without runaway 240-word rambles
    soft_stop_words = max(target_words - 6, int(round(target_words * 0.90)))
    max_tokens = max(95, int(gen_target_words * 1.55) + 30)

    best_result = {
        "text": chunk_text.strip(),
        "ppl": 0.0,
        "inst_ppl": 0.0,
        "tokens": 0,
        "attempt": 0,
        "seed": base_seed,
        "elapsed": 0.0,
        "ok_complete": False,
        "cancelled": False,
    }
    best_rank = -1.0

    for attempt in range(1, max(1, attempts) + 1):
        if cancel_event and cancel_event.is_set():
            best_result["cancelled"] = True
            return best_result

        seed_val = base_seed + (attempt - 1) * 53
        # Rotate natural human drafting registers + slight alpha/temp burstiness jitter
        reg_idx = (seed_val // 17 + attempt) % len(CONTRASTIVE_REGISTERS)
        base_prompt = CONTRASTIVE_REGISTERS[reg_idx].format(
            notes=chunk_text.strip(),
            anchor=anchor,
        )
        inst_prompt = (
            f"<|im_start|>user\nWrite a formal academic paragraph from these notes:\n"
            f"{chunk_text.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n{anchor}"
        )
        # Per-paragraph/attempt burstiness jitter breaks document-level uniformity while staying academic & faithful
        alpha_jit = max(0.45, min(0.52, alpha + [0.0, 0.02, -0.02, 0.03, 0.01, -0.01][reg_idx]))
        temp_jit = max(0.44, min(0.54, temp + [0.0, 0.03, -0.03, 0.04, 0.02, -0.02][reg_idx]))

        rng = random.Random(seed_val)
        gen = ""
        base_nlls: List[float] = []
        inst_nlls: List[float] = []
        tokens_since_bracket = 20
        cur_words = anchor_words
        t0 = time.time()

        if on_progress:
            live = f"{prefix_so_far} {anchor}".strip() if prefix_so_far else anchor
            on_progress({
                "event": "attempt_start",
                "mode": "contrastive",
                "attempt": attempt,
                "attempts": attempts,
                "seed": seed_val,
                "text": live,
            })

        for step in range(max_tokens):
            if cancel_event and cancel_event.is_set():
                best_result["cancelled"] = True
                return best_result

            _, b_top = next_token_logprobs(ollama_base, base_model, base_prompt + gen, top_k=20)
            if not b_top:
                break
            _, i_top = next_token_logprobs(ollama_base, instruct_model, inst_prompt + gen, top_k=20)
            i_map = {t: lp for t, lp in i_top}
            min_i_lp = (min(i_map.values()) - 1.5) if i_map else -10.0

            cur_full_pre = re.sub(r"\s+", " ", (anchor + gen)).strip()
            at_sent_end = _ends_clean_sentence(cur_full_pre)
            reached_len = (cur_words >= soft_stop_words and at_sent_end) or (cur_words >= int(target_words * 1.22))
            top_b_lp = max(lp for _, lp in b_top)
            scored = []
            for tok, b_lp in b_top:
                if tok in STOP_TOKENS:
                    continue
                if not is_clean_ascii(tok) or "—" in tok or "–" in tok:
                    continue
                if "\n" in tok and not reached_len:
                    continue
                if "[" in tok and tokens_since_bracket < 12:
                    continue
                if math.exp(b_lp) < min_base_prob and b_lp < top_b_lp - 3.0:
                    continue
                i_lp = i_map.get(tok, min_i_lp)
                c_score = b_lp - alpha_jit * i_lp
                scored.append((tok, c_score, b_lp, i_lp))

            if not scored:
                # Fallback to any clean non-stop ASCII token so we don't abort mid-paragraph
                fallback = [(t, b_lp) for (t, b_lp) in b_top if t not in STOP_TOKENS and is_clean_ascii(t) and ("\n" not in t or reached_len)]
                if not fallback:
                    fallback = [(t, b_lp) for (t, b_lp) in b_top if t not in STOP_TOKENS and is_clean_ascii(t)]
                if not fallback:
                    break
                tok, b_lp = fallback[0]
                i_lp = i_map.get(tok, min_i_lp)
                scored = [(tok, b_lp - alpha_jit * i_lp, b_lp, i_lp)]

            scored.sort(key=lambda x: x[1], reverse=True)
            pool = scored[: max(2, pool_size)]
            mx = pool[0][1]
            eff_temp = max(0.05, temp_jit)
            weights = [math.exp((x[1] - mx) / eff_temp) for x in pool]
            chosen_tok, _, b_lp, i_lp = rng.choices(pool, weights=weights, k=1)[0]

            if "\n" in chosen_tok and reached_len:
                break
            gen += chosen_tok
            tokens_since_bracket = 0 if "[" in chosen_tok else (tokens_since_bracket + 1)
            base_nlls.append(-b_lp)
            inst_nlls.append(-i_lp)

            cur_full = re.sub(r"\s+", " ", (anchor + gen)).strip()
            if cur_words >= int(target_words * 0.75) and re.search(r"\b(?:Source|Rewrite|Notes|Paragraph)\s*:", gen):
                break
            cur_words = len(cur_full.split())

            if on_progress and (step % 4 == 0 or chosen_tok.endswith(".")):
                r_bppl = math.exp(sum(base_nlls) / max(len(base_nlls), 1))
                r_ippl = math.exp(sum(inst_nlls) / max(len(inst_nlls), 1))
                live = f"{prefix_so_far} {cur_full}".strip() if prefix_so_far else cur_full
                on_progress({
                    "event": "token",
                    "mode": "contrastive",
                    "attempt": attempt,
                    "attempts": attempts,
                    "text": live,
                    "ppl": round(r_bppl, 2),
                    "inst_ppl": round(r_ippl, 1),
                    "tokens": len(base_nlls),
                })

            if cur_words >= soft_stop_words and _ends_clean_sentence(cur_full):
                break

        elapsed = round(time.time() - t0, 2)
        base_ppl = math.exp(sum(base_nlls) / max(len(base_nlls), 1)) if base_nlls else 0.0
        inst_ppl = math.exp(sum(inst_nlls) / max(len(inst_nlls), 1)) if inst_nlls else 0.0
        full = trim_to_last_sentence(anchor + gen)
        actual_words = len(full.split())
        ok_complete = (
            full.endswith(".")
            and actual_words >= int(target_words * 0.86)
            and full.count("(") == full.count(")")
        )
        len_factor = _length_penalty_factor(actual_words, target_words)

        cand = {
            "text": full,
            "ppl": round(base_ppl, 2),
            "inst_ppl": round(inst_ppl, 1),
            "tokens": len(base_nlls),
            "attempt": attempt,
            "seed": seed_val,
            "elapsed": elapsed,
            "ok_complete": ok_complete,
            "cancelled": False,
        }

        # Cap inst_ppl influence so extreme outliers don't beat a properly-sized complete chunk
        eff_inst = min(inst_ppl, 1200.0)
        punct_mult = 1.0 if (full.endswith(".") and full.count("(") == full.count(")")) else 0.4
        # Strongly penalize severely truncated candidates (<65% target words)
        short_mult = 1.0 if actual_words >= int(target_words * 0.65) else 0.05
        rank = eff_inst * len_factor * punct_mult * short_mult
        if rank > best_rank and actual_words >= 10:
            best_rank = rank
            best_result = cand

        if ok_complete and inst_ppl >= min_inst_ppl and base_ppl >= 6.0:
            break

    return best_result


def rewrite_paragraph_contrastive(
    source_para: str,
    anchor: str,
    *,
    ollama_base: str = DEFAULT_OLLAMA_BASE,
    base_model: str = DEFAULT_MODEL,
    instruct_model: str = DEFAULT_INSTRUCT_MODEL,
    length_ratio: float = DEFAULT_LENGTH_RATIO,
    alpha: float = 0.48,
    min_base_prob: float = 0.030,
    temp: float = 0.50,
    pool_size: int = 5,
    min_inst_ppl: float = 350.0,
    min_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    attempts: int = 3,
    base_seed: int = 401,
    on_progress: Optional[Callable[[Dict], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict:
    """
    GPTZero Mode: Length-preserving Contrastive Anti-RLHF Decoding (auto-chunks
    long paragraphs into ~80-word segments and stitches them cleanly).
    """
    chunks = chunk_paragraph(source_para, max_chunk_words=160)

    parts: List[str] = []
    total_tokens = 0
    total_elapsed = 0.0
    weighted_log_bppl = 0.0
    weighted_log_ippl = 0.0
    last_attempt = 1

    for c_idx, chunk in enumerate(chunks):
        c_anchor = anchor if c_idx == 0 else make_contrastive_anchor(chunk, c_idx)
        prefix_so_far = " ".join(parts)

        res = _rewrite_chunk_contrastive(
            chunk,
            anchor=c_anchor,
            ollama_base=ollama_base,
            base_model=base_model,
            instruct_model=instruct_model,
            length_ratio=length_ratio,
            alpha=alpha,
            min_base_prob=min_base_prob,
            temp=temp,
            pool_size=pool_size,
            min_inst_ppl=min_inst_ppl,
            attempts=attempts,
            base_seed=base_seed + c_idx * 137,
            on_progress=on_progress,
            cancel_event=cancel_event,
            prefix_so_far=prefix_so_far,
        )
        if res.get("cancelled"):
            return {
                "mode": "contrastive",
                "text": " ".join(parts) if parts else source_para.strip(),
                "ppl": 0.0,
                "inst_ppl": 0.0,
                "alpha": alpha,
                "tokens": total_tokens,
                "skipped": 0,
                "word_starts": 0,
                "skip_ratio": None,
                "attempt": res.get("attempt", 0),
                "seed": base_seed,
                "elapsed": round(total_elapsed, 2),
                "cancelled": True,
            }

        parts.append(res["text"])
        toks = max(1, res["tokens"])
        total_tokens += res["tokens"]
        total_elapsed += res["elapsed"]
        weighted_log_bppl += math.log(max(1.0, res["ppl"])) * toks
        weighted_log_ippl += math.log(max(1.0, res["inst_ppl"])) * toks
        last_attempt = res["attempt"]

    combined_text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    agg_bppl = math.exp(weighted_log_bppl / max(1, total_tokens)) if total_tokens else 0.0
    agg_ippl = math.exp(weighted_log_ippl / max(1, total_tokens)) if total_tokens else 0.0
    src_words = max(1, len(source_para.split()))
    out_words = len(combined_text.split())

    final = {
        "mode": "contrastive",
        "text": combined_text,
        "ppl": round(agg_bppl, 2),
        "inst_ppl": round(agg_ippl, 1),
        "alpha": alpha,
        "tokens": total_tokens,
        "skipped": 0,
        "word_starts": 0,
        "skip_ratio": None,
        "source_words": src_words,
        "output_words": out_words,
        "length_pct": round(100.0 * out_words / src_words),
        "chunks": len(chunks),
        "attempt": last_attempt,
        "seed": base_seed,
        "elapsed": round(total_elapsed, 2),
        "cancelled": False,
    }
    if on_progress:
        on_progress({"event": "attempt_done", **final})
    return final


def build_bootstrap_demo(source_para: str, rewritten_para: str) -> str:
    """Create a self-bootstrapping in-context demo from an already-verified paragraph."""
    return f"Notes:\n{source_para.strip()}\n\nParagraph:\n{rewritten_para.strip()}"


def main():
    parser = argparse.ArgumentParser(description="Local Base-Model Humanizer CLI (ZeroGPT & GPTZero Modes)")
    parser.add_argument("--input", "-i", help="Input text file (reads stdin if omitted)")
    parser.add_argument("--output", "-o", help="Output text file (prints stdout if omitted)")
    parser.add_argument(
        "--mode",
        choices=["anti_greedy", "contrastive"],
        default="anti_greedy",
        help="Decoding mode: anti_greedy (ZeroGPT) or contrastive (GPTZero Anti-RLHF)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama base model (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--instruct-model",
        default=DEFAULT_INSTRUCT_MODEL,
        help=f"Same-vocab Ollama instruct model for contrastive mode (default: {DEFAULT_INSTRUCT_MODEL})",
    )
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_BASE, help="Ollama base URL")
    parser.add_argument("--length-ratio", type=float, default=DEFAULT_LENGTH_RATIO, help="Target output/input word ratio (default: 1.0)")
    parser.add_argument("--p-skip", type=float, default=0.72, help="Anti-greedy Rank-1 skip prob (default: 0.72)")
    parser.add_argument("--min-ppl", type=float, default=4.2, help="Anti-greedy min target PPL (default: 4.2)")
    parser.add_argument("--alpha", type=float, default=0.48, help="Contrastive anti-RLHF weight alpha (default: 0.48)")
    parser.add_argument("--min-inst-ppl", type=float, default=350.0, help="Contrastive min target InstPPL (default: 350)")
    parser.add_argument("--temp", type=float, default=0.50, help="Contrastive sampling temperature (default: 0.50)")
    parser.add_argument("--attempts", type=int, default=3, help="Max resampling attempts per chunk (default: 3)")
    parser.add_argument("--seed", type=int, default=401, help="Base random seed (default: 401)")
    args = parser.parse_args()

    raw_text = open(args.input, "r", encoding="utf-8").read() if args.input else sys.stdin.read()
    blocks = parse_blocks(raw_text)
    demos = [DEMO_A, DEMO_B]
    out_lines: List[str] = []
    para_idx = 0

    for blk in blocks:
        if blk["kind"] == "heading":
            if out_lines:
                out_lines.append("")
            out_lines.append(blk["text"])
            continue

        sys.stderr.write(f"\n=== Paragraph {para_idx + 1} ({args.mode}, {len(blk['text'].split())} words) ===\n")
        if args.mode == "contrastive":
            res = rewrite_paragraph_contrastive(
                blk["text"],
                anchor=make_contrastive_anchor(blk["text"], para_idx),
                ollama_base=args.ollama_url,
                base_model=args.model,
                instruct_model=args.instruct_model,
                length_ratio=args.length_ratio,
                alpha=args.alpha,
                temp=args.temp,
                min_inst_ppl=args.min_inst_ppl,
                attempts=args.attempts,
                base_seed=args.seed + para_idx * 101,
            )
            sys.stderr.write(
                f"  -> {res['source_words']} -> {res['output_words']} words ({res['length_pct']}%) | chunks={res['chunks']} | BasePPL={res['ppl']} | InstPPL={res['inst_ppl']} | {res['elapsed']}s\n"
            )
        else:
            res = rewrite_paragraph(
                blk["text"],
                demo=demos[para_idx % len(demos)],
                seed_start=make_seed_start(blk["text"], para_idx),
                ollama_base=args.ollama_url,
                model=args.model,
                length_ratio=args.length_ratio,
                p_skip_top1=args.p_skip,
                min_ppl=args.min_ppl,
                attempts=args.attempts,
                base_seed=args.seed + para_idx * 101,
            )
            sys.stderr.write(
                f"  -> {res['source_words']} -> {res['output_words']} words ({res['length_pct']}%) | chunks={res['chunks']} | PPL={res['ppl']} | {res['elapsed']}s\n"
            )
        if out_lines and not is_heading(out_lines[-1]):
            out_lines.append("")
        out_lines.append(res["text"])
        para_idx += 1

    final_output = "\n".join(out_lines).strip() + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(final_output)
    else:
        sys.stdout.write(final_output)


if __name__ == "__main__":
    main()
