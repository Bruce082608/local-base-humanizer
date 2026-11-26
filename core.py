#!/usr/bin/env python3
"""
Core engine for Local Base-Model Humanizer Studio (Ollama & llama.cpp).

Supports two complementary local token-level `logprobs` / full-vocab logits
decoding engines:
  1. anti_greedy (ZeroGPT Mode):
     Unaligned local base model with token-level rank-1 skipping, adaptive
     plausibility filtering, hard AI-buzzword BPE blacklist, prefix-constrained
     numeric/citation/entity fact guard, alternating verified 0%-AI in-context
     demos, and fidelity + burstiness + surrogate-aware Best-of-N retry.
  2. contrastive (GPTZero Mode — Contrastive Anti-RLHF + Human-Style Decoding):
     Same-vocabulary local pair (Base model minus alpha(t) * Instruct model):
       score(t) = log P_base(t | human_demos, notes) - alpha(t) * max(log P_instruct(t), i_floor)
     Featuring:
       - Stage-0 Multi-Perspective Telegraphic Decomposition (rotates schemas &
         shuffles bullets across retries so paragraphs never share one fixed outline).
       - Human-Style Few-Shot Contrastive Conditioning (Base sees retrieved
         real-human examples while Instruct stays generic -> pulls toward human
         style rather than flat high-entropy noise).
       - Dynamic Sentence-Aware Burstiness Scheduling (`alpha(t)` alternates
         crisp low-alpha short sentences with richer higher-alpha complex sentences,
         relaxing penalty on grammatical glue and protected factual terms).
       - Strict Prefix-Constrained Fact Guard protecting numbers, citations,
         authors, acronyms, and proper nouns, plus active `CONTRASTIVE_BANNED_SUBSTRS`
         filtering (blocking both AI clichés and archaic contrastive drift).
       - Dual Inference Backends: zero-dependency persistent keep-alive Ollama HTTP
         OR exact full-vocabulary `llama-cpp-python` (GGUF) with 0 Top-20 cliff.
       - Multi-Signal Local Surrogate Reranking: Linguistic Burstiness + 3-gram
         anti-paraphrase overlap + unconditional cross-family Ollama GLTR probe
         + exact `llama.cpp` Binoculars + optional local DeBERTa/RoBERTa neural
         AI detector + read-only Instruct faithfulness judge.
"""

import argparse
import math
import random
import re
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import backends
import human_demos
import surrogates

DEFAULT_OLLAMA_BASE = backends.DEFAULT_OLLAMA_BASE
DEFAULT_MODEL = "lab-qwen3-4b-base:latest"
DEFAULT_INSTRUCT_MODEL = "lab-qwen3-4b:latest"
DEFAULT_LENGTH_RATIO = 1.0
DEFAULT_NUM_CTX = backends.DEFAULT_NUM_CTX
DEFAULT_I_FLOOR = -11.5
DEFAULT_BETA = 0.020

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

# Precision-focused ban list enforced in Contrastive mode: blocks extreme AI clichés
# AND archaic/legalese contrastive drift tokens that Base scores higher than Instruct.
CONTRASTIVE_BANNED_SUBSTRS = [
    "utiliz", "leverag", "delve", "tapestry", "testament", "multifaceted",
    "pivotal", "paramount", "cutting-edge", "underscor", "highlight",
    "furthermore", "moreover", "additionally", "thereby", "in-depth",
    "summary", "herein", "therein", "thereof", "wherein", "whilst",
    "heretofore", "aforementioned", "notwithstanding", "—", "–",
]

# Cliché paragraph openings blocked during the first few words of free-start generation
OPENING_CLICHE_PREFIXES = (
    "recent studies", "in recent years", "this study", "the study",
    "it is worth noting", "in summary", "overall,", "notably,",
    "furthermore,", "moreover,", "additionally,", "in conclusion",
)

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
STOP_TOKENS = ("<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<think>", "</think>", "<end_of_turn>", "")

DEMO_LEAK_WORDS = {
    "chang", "impala", "campus", "hunan", "ubuntu", "intel", "i5",
    "phase-space", "warehousing", "province", "etl", "china", "chinese",
}

STRUCTURAL_HALLUC_WORDS = (
    "figure", "fig", "table", "appendix", "chapter", "equation", "column", "page",
)

# Common grammatical glue words relaxed under Dynamic Burstiness scheduling
_GLUE_WORDS: Set[str] = {
    "the", "a", "an", "of", "in", "on", "to", "for", "with", "by", "at", "from",
    "into", "as", "that", "which", "who", "and", "or", "but", "is", "are", "was",
    "were", "be", "been", "being", "has", "have", "had", "this", "these", "those",
    "it", "its", "their", "they", "we", "our", "than", "not", "no", "per",
}

_PROTECTED_STOPWORDS: Set[str] = {
    "the", "this", "these", "those", "both", "each", "some", "many", "most",
    "data", "model", "models", "study", "studies", "paper", "method", "methods",
    "result", "results", "system", "table", "figure", "section", "however",
    "although", "because", "while", "where", "when", "after", "before", "across",
    "between", "within", "under", "over", "during", "through", "recent", "future",
}

# Diverse Stage-0 telegraphic extraction perspectives (breaks fixed Study/Method/Result template!)
NOTE_EXTRACTION_SCHEMAS: List[str] = [
    (
        "Break the paragraph below into 4-5 ultra-terse, unlabelled factual bullet fragments (NO headers like 'Study:' or 'Method:').\n"
        "Order: concrete quantitative outcomes/comparisons FIRST, technical setup/sample in the middle, caveats or broader context LAST.\n"
        "Strict rules:\n"
        "1. Preserve EVERY exact number, percentage, year, unit, comparison direction (higher/lower/fewer/faster), author name, technical term, and citation [k].\n"
        "2. Use clipped fragments rather than polished prose. Do NOT add new numbers or AI buzzwords (demonstrated, highlighted, crucial, significantly, furthermore, however, thereby, underscoring, robust, comprehensive).\n"
        "3. Output ONLY the short fragment lines."
    ),
    (
        "Extract 4-5 terse scratchpad fragments from the paragraph below (plain lines, no labels).\n"
        "Order: start with the main contrast, trade-off, or limitation, then state the system/dataset/method and exact reported figures.\n"
        "Strict rules:\n"
        "1. Keep EVERY exact number, percentage, year, unit, comparison direction, author name, acronym, and citation [k].\n"
        "2. Write clipped telegraphic phrases, NOT full sentences. Never invent numbers or use AI transition words (furthermore, moreover, notably, thereby, crucial, pivotal, underscoring, demonstrating).\n"
        "3. Output ONLY the fragment lines."
    ),
    (
        "Distill the paragraph below into 4-5 compact factual notes (no field labels, no prose transitions).\n"
        "Order: concrete experimental setup / cohort / dataset / authors first, observed measurements second, boundary condition or gap last.\n"
        "Strict rules:\n"
        "1. Keep EVERY exact number, unit, year, comparison direction, author name, technical term, and citation [k].\n"
        "2. Avoid complete academic sentences and ban AI clichés (utilize, leverage, demonstrate, highlight, significantly, effectively, framework, landscape, multifaceted).\n"
        "3. Output ONLY the short note lines."
    ),
    (
        "Compress the paragraph below into 4-5 raw research scratch notes (semicolon-separated phrases, no labels).\n"
        "Order: open with the specific question or technical challenge addressed, followed immediately by cited evidence and exact numerical values.\n"
        "Strict rules:\n"
        "1. Retain EVERY exact number, percentage, year, unit, comparison direction, author name, domain term, and citation [k].\n"
        "2. Keep fragments terse and unpolished; zero invented numbers and zero AI filler verbs (demonstrated, underscored, highlighted, facilitated, enabling, ensuring).\n"
        "3. Output ONLY the note lines."
    ),
    (
        "Convert the paragraph below into 4-5 concise, non-chronological technical note lines (no labels like 'Result:' or 'Context:').\n"
        "Order: lead with the strongest empirical finding or cited benchmark, then attach methodology details and scope limits.\n"
        "Strict rules:\n"
        "1. Preserve EVERY exact number, percentage, year, unit, direction of effect, author name, acronym, and citation [k].\n"
        "2. Do NOT write polished sentences, do NOT introduce numbers absent from the text, and avoid AI buzzwords (crucial, pivotal, paramount, furthermore, moreover, thereby, robust, comprehensive).\n"
        "3. Output ONLY the note lines."
    ),
]


def normalize_ollama_base(url: str) -> str:
    return backends.normalize_ollama_base(url)


def list_ollama_models(ollama_base: str = DEFAULT_OLLAMA_BASE, timeout: float = 5.0) -> Tuple[bool, List[Dict], str]:
    return backends.OLLAMA_CLIENT.list_models(ollama_base, timeout=timeout)


def format_instruct_prompt(instruct_model: str, user_msg: str, assistant_prefix: str = "") -> str:
    """Format a single-turn chat prompt matching the target Instruct model family."""
    m = (instruct_model or "").lower()
    if "qwen3" in m or "qwen-3" in m:
        return (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n{assistant_prefix}"
        )
    if "qwen" in m or "deepseek" in m:
        return (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_prefix}"
        )
    if "llama-3" in m or "llama3" in m or "llama" in m:
        return (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{assistant_prefix}"
        )
    if "mistral" in m or "mixtral" in m:
        return f"[INST] {user_msg} [/INST]{assistant_prefix}"
    if "gemma" in m:
        return (
            f"<start_of_turn>user\n{user_msg}<end_of_turn>\n"
            f"<start_of_turn>model\n{assistant_prefix}"
        )
    return f"User:\n{user_msg}\n\nAssistant:\n{assistant_prefix}"


def ollama_generate_raw(
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
    return backends.OLLAMA_CLIENT.generate_raw(
        ollama_base,
        model,
        raw_prompt,
        num_predict=num_predict,
        temperature=temperature,
        num_ctx=num_ctx,
        timeout=timeout,
        keep_alive=keep_alive,
    )


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

    n_chunks = max(2, math.ceil(len(words) / max_chunk_words))
    target_per_chunk = len(words) / n_chunks

    chunks: List[str] = []
    cur: List[str] = []
    cur_words = 0

    for sent in sents:
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


def _chunk_demo_blacklist(chunk_text: str, extra_leak_words: Optional[Sequence[str]] = None) -> List[str]:
    """Return demo-specific entities not present in chunk_text so they can never leak."""
    low = chunk_text.lower()
    pool = set(DEMO_LEAK_WORDS)
    if extra_leak_words:
        pool.update(w.lower() for w in extra_leak_words if w)
    return [w for w in sorted(pool) if w not in low]


def _causes_demo_leak_or_repeat(cur_text: str, tok: str, demo_ban: List[str]) -> bool:
    """Catch BPE-split demo leaks (e.g. ' Hu'+'nan') and 4-gram repetition loops."""
    tail = (cur_text[-60:] + tok).lower()
    if demo_ban and any(b in tail for b in demo_ban):
        return True
    cand = re.sub(r"\s+", " ", (cur_text + tok)).strip().lower()
    words = cand.split()
    if len(words) >= 8:
        last4 = " ".join(words[-4:])
        prev = " ".join(words[:-4])
        if last4 in prev:
            return True
    return False


def _causes_banned_bpe(cur_text: str, tok: str, banned_substrs: List[str]) -> bool:
    """Check both the token itself and the trailing BPE-assembled word against banned roots."""
    if not banned_substrs:
        return False
    t_low = tok.lower().strip()
    if t_low and any(b in t_low for b in banned_substrs if b):
        return True
    tail = (cur_text[-24:] + tok).lower()
    m = re.search(r"[a-z\-]+$", tail)
    if m:
        w = m.group(0)
        if any(b in w for b in banned_substrs if b):
            return True
    return False


def _causes_opening_cliche(cur_text: str, tok: str) -> bool:
    """Block formulaic AI paragraph openings during the first ~4 words of generation."""
    combined = re.sub(r"\s+", " ", (cur_text + tok)).strip().lower()
    if len(combined) > 36:
        return False
    return any(combined.startswith(prefix) for prefix in OPENING_CLICHE_PREFIXES)


def is_banned(tok: str, banned_substrs: List[str]) -> bool:
    t = tok.lower().strip()
    if not t:
        return False
    return any(b in t for b in banned_substrs if b)


def is_clean_ascii(tok: str) -> bool:
    return all(32 <= ord(c) < 127 or c in "\n\t" for c in tok)


# ---------------------------------------------------------------------------
# Strict Numeric / Citation / Entity Fact Guard & Fidelity Metrics
# ---------------------------------------------------------------------------

def _expand_citation_groups(text: str) -> Set[str]:
    """Extract all numeric citation IDs from patterns like [12], [2, 9], [4]-[6], [4–6]."""
    ids: Set[str] = set()
    for m in re.finditer(r"\[(\d+(?:\s*[,;\-–]\s*\d+)*)\]", text):
        inner = m.group(1)
        for num in re.findall(r"\d+", inner):
            ids.add(num)
    return ids


def extract_facts_signature(text: str) -> Dict[str, Set[str]]:
    """Extract verifiable factual anchors: citations, numbers, authors, acronyms, proper nouns."""
    cite_nums = _expand_citation_groups(text)
    cites = {f"[{n}]" for n in cite_nums}

    no_cites = re.sub(r"\[\d+(?:\s*[,;\-–]\s*\d+)*\]", " ", text)
    raw_nums = re.findall(r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", no_cites)
    nums: Set[str] = set()
    for rn in raw_nums:
        nums.add(rn)
        if "," in rn:
            nums.add(rn.replace(",", ""))

    authors = set(re.findall(r"\b([A-Z][a-zA-Z\-]{2,})\s+(?:et\s+al\.|and\s+[A-Z][a-zA-Z\-]{2,})", text))
    for m in re.finditer(r"\b([A-Z][a-zA-Z\-]{2,})\s+and\s+([A-Z][a-zA-Z\-]{2,})\b", text):
        a1, a2 = m.group(1), m.group(2)
        if a1.lower() not in {"the", "this", "these", "both", "data", "wind", "solar", "cement", "steel"}:
            authors.add(a1)
            authors.add(a2)

    acronyms = set(re.findall(r"\b(?:[A-Z]{2,}[a-z0-9]*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*|[A-Z]-[A-Z][a-z]+)\b", text))
    proper: Set[str] = set()
    for s in split_sentences(text):
        ws = s.split()
        for w in ws[1:]:
            cw = w.strip(".,;:()[]\"'")
            if len(cw) >= 2 and cw[0].isupper():
                proper.add(cw.lower())
    for a in acronyms | authors:
        proper.add(a.lower())

    return {
        "cites": cites,
        "cite_nums": cite_nums,
        "nums": nums,
        "authors": authors,
        "acronyms": acronyms,
        "proper": proper,
    }


def build_fact_guard(source_text: str) -> Dict:
    """
    Precompute prefix tries for strict prefix-constrained numeric & citation decoding,
    plus protected entity/acronym/author prefix sets so Contrastive Decoding does not
    penalize accurate domain terminology.
    """
    sig = extract_facts_signature(source_text)
    allowed_nums = set(sig["nums"])
    num_prefixes: Set[str] = set()
    for n in allowed_nums:
        for i in range(1, len(n) + 1):
            num_prefixes.add(n[:i])

    allowed_cite_nums = set(sig["cite_nums"])
    cite_prefixes: Set[str] = set()
    for cn in allowed_cite_nums:
        for i in range(1, len(cn) + 1):
            cite_prefixes.add(cn[:i])

    # Protected authors, acronyms, and non-generic proper nouns (and their 3+ char BPE prefixes)
    protected_terms: Set[str] = set()
    for term in sig["proper"]:
        clean_t = re.sub(r"[^a-z0-9\-]", "", term.lower())
        if len(clean_t) >= 2 and clean_t not in _PROTECTED_STOPWORDS:
            protected_terms.add(clean_t)
    protected_prefixes: Set[str] = set()
    for term in protected_terms:
        min_p = 2 if len(term) <= 3 else 3
        for i in range(min_p, len(term) + 1):
            protected_prefixes.add(term[:i])

    low = source_text.lower()
    block_dollar = ("$" not in source_text) and ("dollar" not in low) and ("usd" not in low)
    block_euro_sym = ("€" not in source_text) and ("eur" not in low)
    block_gdp = "gdp" not in low
    block_plus = "+" not in source_text
    blocked_struct = {w for w in STRUCTURAL_HALLUC_WORDS if w not in low}

    return {
        "sig": sig,
        "source_low": low,
        "allowed_nums": allowed_nums,
        "num_prefixes": num_prefixes,
        "allowed_cite_nums": allowed_cite_nums,
        "cite_prefixes": cite_prefixes,
        "protected_terms": protected_terms,
        "protected_prefixes": protected_prefixes,
        "block_dollar": block_dollar,
        "block_euro_sym": block_euro_sym,
        "block_gdp": block_gdp,
        "block_plus": block_plus,
        "blocked_struct": blocked_struct,
    }


def _violates_fact_guard(cur_text: str, tok: str, guard: Dict) -> bool:
    """
    True when appending `tok` to `cur_text` would fabricate a number, citation,
    fake figure/table reference, or disallowed currency not present in source_text.
    """
    tl = tok.lower().strip()

    # 1. Currency / fabricated metric / structural reference blockers (including BPE-assembled tail)
    tail_word_m = re.search(r"[a-z]+$", (cur_text[-12:] + tok).lower())
    tail_word = tail_word_m.group(0) if tail_word_m else tl
    if guard["block_dollar"] and ("$" in tok or tl == "usd" or tail_word == "usd"):
        return True
    if guard["block_euro_sym"] and "€" in tok:
        return True
    if guard["block_gdp"] and (tl == "gdp" or tail_word == "gdp"):
        return True
    if guard["block_plus"] and "+" in tok:
        return True
    if guard["blocked_struct"] and (tl in guard["blocked_struct"] or tail_word in guard["blocked_struct"]):
        return True
    if re.search(r"[\(\s]p\.$", (cur_text[-6:] + tok).lower()):
        return True
    if "/" in tok and re.search(r"/[a-z]{2,}", tl):
        if tl not in guard["sig"]["proper"]:
            return True
    if re.search(r"\d,$", cur_text) and tok and tok[0].isdigit():
        m_prev = re.search(r"(\d+(?:,\d+)*,)$", cur_text)
        if m_prev:
            cand_c = (m_prev.group(1) + re.match(r"^\d+", tok).group(0)).replace(",", "")
            if cand_c not in guard["num_prefixes"]:
                return True

    # 2. Check open citation state: e.g. "... [1" or "... [2, "
    m_open_cite = re.search(r"\[(\d*(?:\s*[,;\-–]\s*\d*)*)$", cur_text)
    if m_open_cite:
        inner = m_open_cite.group(1)
        m_last = re.search(r"(\d*)$", inner)
        cur_cid = m_last.group(1) if m_last else ""
        if re.search(r"[A-Za-z]", tok):
            return True
        if not guard["allowed_cite_nums"]:
            return True
        if "]" in tok:
            before_close = tok.split("]", 1)[0]
            cand_id = cur_cid + re.sub(r"[^\d]", "", before_close)
            if not cand_id or cand_id not in guard["allowed_cite_nums"]:
                return True
            return False
        if any(ch.isdigit() for ch in tok):
            segs = re.split(r"[,;\-–\s]+", (cur_cid + tok).strip())
            for idx_s, seg in enumerate(segs):
                if not seg:
                    continue
                if idx_s < len(segs) - 1:
                    if seg not in guard["allowed_cite_nums"]:
                        return True
                else:
                    if seg not in guard["cite_prefixes"]:
                        return True
            return False
        if cur_cid and cur_cid not in guard["allowed_cite_nums"]:
            return True
        return False

    # 3. Starting a new '[' citation when source has no citations
    if "[" in tok and not guard["allowed_cite_nums"]:
        return True
    if "[" in tok:
        after_br = tok.split("[", 1)[1]
        if after_br:
            if re.search(r"[A-Za-z]", after_br):
                return True
            if "]" in after_br:
                cid = after_br.split("]", 1)[0].strip()
                if cid and cid not in guard["allowed_cite_nums"]:
                    return True
            else:
                cid = re.sub(r"[^\d]", "", after_br)
                if cid and cid not in guard["cite_prefixes"]:
                    return True
        return False

    # 4. Active numeric span outside brackets
    m_cur_num = re.search(r"((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?)$", cur_text)
    if m_cur_num:
        cur_num = m_cur_num.group(1)
        cur_num_clean = cur_num.replace(",", "")
        cur_ends_dot = cur_num.endswith(".")
        cur_base = cur_num_clean[:-1] if cur_ends_dot else cur_num_clean

        if tok and (tok[0].isdigit() or (tok[0] in ".," and not cur_ends_dot)):
            if tok[0] == "." and cur_num_clean in guard["allowed_nums"]:
                rest = tok[1:]
                if not rest or not rest[0].isdigit():
                    return False
            if tok[0] == "," and cur_num_clean in guard["allowed_nums"]:
                rest = tok[1:]
                if not rest or not rest[0].isdigit():
                    return False

            combined = cur_num + tok
            m_ext = re.match(r"^((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?)(.*)$", combined)
            if not m_ext:
                return True
            ext_num = m_ext.group(1)
            ext_clean = ext_num.replace(",", "")
            remainder = m_ext.group(2)
            if remainder:
                check_val = ext_clean[:-1] if ext_clean.endswith(".") else ext_clean
                if check_val not in guard["allowed_nums"]:
                    return True
            else:
                if ext_clean not in guard["num_prefixes"]:
                    return True
            return False
        else:
            if cur_ends_dot:
                if cur_base not in guard["allowed_nums"]:
                    return True
            else:
                if cur_num_clean not in guard["allowed_nums"]:
                    return True

    # 4b. Block fabricated '28x28' / '28GB' dimensions or excessive reuse of an allowed number
    if re.search(r"\d$", cur_text) and tok and tok[0].lower() in ("x", "g", "m", "k") and not re.search(r"\b\d+[xgmk]", guard.get("source_low", "")):
        if not (tok.lower().startswith("x") and ("times" in guard.get("source_low", "") or "x" in guard.get("source_low", ""))):
            return True
    if any(ch.isdigit() for ch in tok):
        m_prev_d = re.search(r"((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?)$", cur_text)
        if not m_prev_d:
            first_d = re.search(r"(\d+(?:\.\d*)?)", tok)
            if first_d:
                pref = first_d.group(1)
                cands = [n for n in guard["allowed_nums"] if n.startswith(pref)]
                if cands:
                    used_counts = {n: len(re.findall(rf"(?<!\d){re.escape(n)}(?!\d)", cur_text)) for n in cands}
                    if all(cnt >= 2 for cnt in used_counts.values()):
                        return True

    # 5. Starting a new number inside `tok` (when tok contains any digit)
    if any(ch.isdigit() for ch in tok):
        if not guard["allowed_nums"]:
            return True
        for m_d in re.finditer(r"(\d+(?:,\d{3})*(?:\.\d*)?)", tok):
            seg = m_d.group(1).replace(",", "")
            ends_at_tok_end = (m_d.end() == len(tok))
            if ends_at_tok_end:
                if seg not in guard["num_prefixes"]:
                    return True
            else:
                check_val = seg[:-1] if seg.endswith(".") else seg
                if check_val not in guard["allowed_nums"]:
                    return True

    return False


def _is_active_fact_token(cur_text: str, tok: str, guard: Dict) -> bool:
    """
    True when `tok` is continuing/writing a whitelisted digit, citation,
    or protected source author / acronym / proper noun (including BPE prefixes).
    """
    if re.search(r"\[\d*$", cur_text) or re.search(r"\d\.?$", cur_text):
        return True
    if any(ch.isdigit() for ch in tok) or tok.strip() in ("[", "]", "%"):
        return True

    protected_terms = guard.get("protected_terms")
    protected_prefixes = guard.get("protected_prefixes")
    if not protected_terms:
        return False

    # Check assembled trailing word (handles BPE-split proper nouns like " Bra" + "TS", " Map" + "Reduce")
    m_tail = re.search(r"([A-Za-z][A-Za-z0-9\-]*)$", (cur_text[-20:] + tok))
    if m_tail:
        word_raw = m_tail.group(1)
        word_low = word_raw.lower()
        if word_low in protected_terms:
            return True
        if len(word_low) >= 3 and word_low in protected_prefixes:
            # Avoid false triggers on lowercase everyday words unless they match an exact capitalized/acronym prefix
            if word_raw[0].isupper() or word_low in protected_terms:
                return True
    return False


def compute_fidelity_metrics(source_text: str, candidate_text: str) -> Dict[str, float]:
    """Deterministic factual preservation & anti-hallucination scorecard (0.0 to 1.0)."""
    src_sig = extract_facts_signature(source_text)
    cand_sig = extract_facts_signature(candidate_text)

    if src_sig["cite_nums"]:
        cite_rec = len(src_sig["cite_nums"] & cand_sig["cite_nums"]) / len(src_sig["cite_nums"])
    else:
        cite_rec = 1.0
    halluc_cites = len(cand_sig["cite_nums"] - src_sig["cite_nums"])

    src_nums = {n.replace(",", "") for n in src_sig["nums"]}
    cand_nums = {n.replace(",", "") for n in cand_sig["nums"]}
    if src_nums:
        num_rec = len(src_nums & cand_nums) / len(src_nums)
    else:
        num_rec = 1.0
    halluc_nums = len(cand_nums - src_nums)

    if src_sig["authors"]:
        auth_rec = len({a.lower() for a in src_sig["authors"]} & {a.lower() for a in cand_sig["authors"]}) / len(src_sig["authors"])
    else:
        auth_rec = 1.0
    halluc_authors = len({a.lower() for a in cand_sig["authors"]} - {a.lower() for a in src_sig["authors"]})

    if src_sig["acronyms"]:
        acr_rec = len({a.lower() for a in src_sig["acronyms"]} & {a.lower() for a in cand_sig["acronyms"]}) / len(src_sig["acronyms"])
    else:
        acr_rec = 1.0

    stop = {
        "these", "those", "their", "there", "where", "while", "which", "about",
        "under", "using", "based", "other", "could", "would", "should", "between",
        "across", "within", "through", "however", "furthermore", "moreover",
        "significantly", "effectively", "study", "studies", "paper", "model",
    }
    s_words = {re.sub(r"[^a-z]", "", w.lower()) for w in source_text.split()}
    c_words = {re.sub(r"[^a-z]", "", w.lower()) for w in candidate_text.split()}
    s_content = {w for w in s_words if len(w) >= 5 and w not in stop}
    content_rec = len(s_content & c_words) / max(1, len(s_content))

    fidelity = (
        0.25 * cite_rec
        + 0.32 * num_rec
        + 0.13 * auth_rec
        + 0.10 * acr_rec
        + 0.20 * min(1.0, content_rec / 0.50)
        - 0.25 * halluc_cites
        - 0.18 * halluc_nums
        - 0.25 * halluc_authors
    )
    return {
        "fidelity": round(max(0.0, min(1.0, fidelity)), 3),
        "fidelity_pct": int(round(100.0 * max(0.0, min(1.0, fidelity)))),
        "cite_rec": round(cite_rec, 2),
        "num_rec": round(num_rec, 2),
        "halluc_cites": halluc_cites,
        "halluc_nums": halluc_nums,
        "halluc_authors": halluc_authors,
    }


def extract_telegraphic_notes(
    source_para: str,
    *,
    ollama_base: str = DEFAULT_OLLAMA_BASE,
    instruct_model: str = DEFAULT_INSTRUCT_MODEL,
    num_ctx: int = DEFAULT_NUM_CTX,
    variant_idx: int = 0,
    shuffle_seed: Optional[int] = None,
    backend: str = "ollama",
    llama_llm: Any = None,
) -> str:
    """
    Stage-0 Multi-Perspective Anti-Paraphrase Decomposition:
    Converts polished AI prose into terse, unlabelled telegraphic fragments
    using rotating organizational schemas (preventing fixed Study->Method->Result
    fingerprints across paragraphs) while deterministically guaranteeing zero
    dropped numbers or citations.
    """
    schema = NOTE_EXTRACTION_SCHEMAS[variant_idx % len(NOTE_EXTRACTION_SCHEMAS)]
    user_msg = f"{schema}\n\n{source_para.strip()}"
    prompt = format_instruct_prompt(instruct_model, user_msg)
    temp = 0.0 if variant_idx == 0 else 0.25

    try:
        if backend == "llama_cpp" and llama_llm is not None:
            raw = backends.LLAMA_CPP_MANAGER.generate_raw(
                llama_llm,
                prompt,
                num_predict=240,
                temperature=temp,
            )
        else:
            raw = ollama_generate_raw(
                ollama_base,
                instruct_model,
                prompt,
                num_predict=240,
                temperature=temp,
                num_ctx=num_ctx,
                timeout=60.0,
            )
    except Exception:
        return source_para.strip()

    # Strip bullets AND accidental 'Study:'/'Method:' labels so Base doesn't copy slot order
    lines = []
    for ln in raw.splitlines():
        s = ln.strip().lstrip("-*•0123456789.) ").strip()
        s = re.sub(r"^(?:Study|Method|Result|Detail|Context|Finding|Setup|Limitation|Note)\s*:\s*", "", s, flags=re.IGNORECASE)
        if s and len(s) >= 8:
            lines.append(s)
    if len(lines) < 2:
        return source_para.strip()

    if shuffle_seed is not None and len(lines) >= 3:
        rng = random.Random(shuffle_seed)
        # Rotate/shuffle note order across retries to break linear bullet-stitching
        perm = list(lines)
        rng.shuffle(perm)
        lines = perm

    notes = "\n".join(lines)

    # Lossless safety net: verify every source number and citation survived Stage-0
    src_sig = extract_facts_signature(source_para)
    note_sig = extract_facts_signature(notes)
    missing_cites = sorted(src_sig["cites"] - note_sig["cites"])
    missing_nums = sorted({n.replace(",", "") for n in src_sig["nums"]} - {n.replace(",", "") for n in note_sig["nums"]})
    if missing_cites or missing_nums:
        extras = []
        if missing_cites:
            extras.append("citations " + ", ".join(missing_cites))
        if missing_nums:
            extras.append("exact figures " + ", ".join(missing_nums))
        notes = notes + f"\nMust include exact items: {'; '.join(extras)}."

    return notes


def judge_faithfulness(
    source_para: str,
    candidate: str,
    *,
    ollama_base: str = DEFAULT_OLLAMA_BASE,
    instruct_model: str = DEFAULT_INSTRUCT_MODEL,
    num_ctx: int = DEFAULT_NUM_CTX,
    backend: str = "ollama",
    llama_llm: Any = None,
) -> Optional[float]:
    """
    Read-only Local Instruct Faithfulness & Coherence Judge (0.0 to 10.0).
    Evaluates a finished draft WITHOUT editing any token -> 0% AI-detection contamination!
    """
    user_msg = (
        "Score how accurately and coherently the DRAFT reports the facts, numbers, comparison directions, "
        "and citations of the SOURCE on an integer scale from 0 to 10.\n"
        "Penalize heavily for: inverted comparisons (e.g. saying baseline is higher when it was lower), "
        "fabricated facts/numbers/citations, broken nonsense grammar, or omitted main findings.\n"
        "Reply with ONLY a single integer from 0 to 10.\n\n"
        f"SOURCE:\n{source_para.strip()}\n\n"
        f"DRAFT:\n{candidate.strip()}"
    )
    prompt = format_instruct_prompt(instruct_model, user_msg)
    try:
        if backend == "llama_cpp" and llama_llm is not None:
            out = backends.LLAMA_CPP_MANAGER.generate_raw(
                llama_llm,
                prompt,
                num_predict=8,
                temperature=0.0,
            )
        else:
            out = ollama_generate_raw(
                ollama_base,
                instruct_model,
                prompt,
                num_predict=8,
                temperature=0.0,
                num_ctx=num_ctx,
                timeout=30.0,
            )
        m = re.search(r"\b(10|[0-9])\b", out)
        if m:
            return float(int(m.group(1)))
    except Exception:
        pass
    return None


def next_token_logprobs(
    ollama_base: str,
    model: str,
    raw_prompt: str,
    top_k: int = 20,
    timeout: float = 60.0,
    keep_alive: str = "10m",
    num_ctx: int = DEFAULT_NUM_CTX,
) -> Tuple[Optional[str], List[Tuple[str, float]]]:
    return backends.OLLAMA_CLIENT.next_token_logprobs(
        ollama_base,
        model,
        raw_prompt,
        top_k=top_k,
        timeout=timeout,
        keep_alive=keep_alive,
        num_ctx=num_ctx,
    )


# ---------------------------------------------------------------------------
# Dynamic Sentence-Aware Burstiness Scheduler (Step 1)
# ---------------------------------------------------------------------------

class BurstinessScheduler:
    """
    Replaces flat constant `alpha` with sentence-rhythm-aware dynamic modulation:
      - Alternates crisp short sentences (low alpha -> idiomatic low-PPL prose)
        with multi-clause long sentences (higher alpha on verbs/pivots -> high-PPL
        human turns), generating authentic human sentence-length & PPL burstiness.
      - Relaxes `alpha` on grammatical glue words so contrastive decoding shifts
        lexical/rhetorical choices instead of breaking syntax.
    """

    PLANS: List[List[Tuple[str, int]]] = [
        [("short", 12), ("long", 30), ("medium", 18), ("short", 11), ("long", 28), ("medium", 17)],
        [("medium", 18), ("short", 11), ("long", 32), ("short", 12), ("medium", 19), ("long", 27)],
        [("long", 29), ("short", 10), ("medium", 19), ("long", 28), ("short", 12), ("medium", 18)],
        [("short", 11), ("medium", 18), ("short", 12), ("long", 33), ("medium", 17), ("short", 11)],
    ]

    def __init__(self, seed: int = 0, enabled: bool = True):
        self.enabled = enabled
        self.plan = self.PLANS[abs(int(seed)) % len(self.PLANS)]

    def slot(self, sent_idx: int) -> Tuple[str, int]:
        return self.plan[sent_idx % len(self.plan)]

    def modulate(
        self,
        base_alpha: float,
        tok: str,
        *,
        sent_idx: int,
        words_in_sent: int,
        in_fact: bool,
    ) -> Tuple[float, float]:
        """
        Return `(effective_alpha, logit_nudge)` for candidate `tok`.
        """
        if in_fact:
            return base_alpha * 0.20, 0.0
        if not self.enabled:
            return base_alpha, 0.0

        kind, target_w = self.slot(sent_idx)
        stripped = tok.strip()
        low = stripped.lower()

        # 1. Sentence-rhythm alpha envelope
        if kind == "short":
            mult = 0.68 if words_in_sent >= 2 else 0.82
        elif kind == "long":
            # Higher contrastive push at sentence pivots & mid-clause transitions
            if words_in_sent <= 2 or (9 <= words_in_sent <= 18):
                mult = 1.22
            else:
                mult = 1.06
        else:
            mult = 1.00

        # 2. Relax contrastive penalty on grammatical glue / punctuation
        if low in _GLUE_WORDS or (stripped and all(not ch.isalnum() for ch in stripped)):
            mult *= 0.60

        eff_alpha = max(0.08, min(1.75, base_alpha * mult))

        # 3. Gentle sentence-length cadence nudge on period token '.'
        nudge = 0.0
        if stripped == "." or tok == ".":
            if kind == "short" and words_in_sent >= max(8, target_w - 2):
                nudge = 0.55
            elif kind == "long" and words_in_sent < 12:
                nudge = -0.45

        return eff_alpha, nudge


def make_seed_start(para: str, idx: int) -> str:
    """Create a domain-neutral, non-copying seed opening for Anti-Greedy mode."""
    stripped = para.strip()
    lead_m = re.match(r"^((?:[A-Z]{1,4}\d+\.\s+)?[A-Z][A-Za-z0-9\-\s]{3,55}\.)\s+(.+)$", stripped)
    if lead_m and not lead_m.group(1).lower().endswith("et al."):
        lead_title = lead_m.group(1)
        starters = [
            f"{lead_title} In this part of the work,",
            f"{lead_title} Under this kind of setup,",
            f"{lead_title} For this line of study,",
            f"{lead_title} In practical settings here,",
        ]
        return starters[idx % len(starters)]

    sents = split_sentences(stripped)
    head_text = " ".join(sents[:3]) if sents else stripped
    m = re.search(r"\b([A-Z][A-Za-z\-]+(?:\s+et\s+al\.|\s+and\s+[A-Z][A-Za-z\-]+))\b", head_text)
    if m:
        author = m.group(1)
        author_starters = [
            f"In the study reported by {author},",
            f"For {author}, the authors",
            f"In the work carried out by {author},",
            f"Looking at the findings from {author},",
        ]
        return author_starters[idx % len(author_starters)]

    cites = re.findall(r"\[\d+\](?:[–\-]\[\d+\])?", stripped)
    unique_cites: List[str] = []
    for c in cites:
        if c not in unique_cites:
            unique_cites.append(c)

    diverse_openers = [
        f"In the study covered by {unique_cites[0]}," if unique_cites else "When we look at these findings,",
        "In this part of the literature,",
        f"Based on the results from {unique_cites[0]}," if unique_cites else "From the reviewed cases here,",
        "In a practical research setting,",
        f"When looking closely at {unique_cites[0]} and {unique_cites[1]}," if len(unique_cites) >= 2 else "Under these conditions,",
        "From an empirical point of view,",
        f"For the cases reported in {unique_cites[0]}," if unique_cites else "Across the reviewed evidence,",
        "Given these experimental constraints,",
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
        "Notes from reading research papers:\n{notes}\n\n"
        "Plain factual paragraph (varied sentence lengths, direct statements, exact figures and citations):\n{anchor}"
    ),
    (
        "Literature notes:\n{notes}\n\n"
        "Direct plain-English technical paragraph (concrete phrasing, exact numbers, natural rhythm):\n{anchor}"
    ),
    (
        "Summary notes from empirical studies:\n{notes}\n\n"
        "Unpolished academic draft paragraph (mix of short and complex sentences, exact facts):\n{anchor}"
    ),
    (
        "Review notes:\n{notes}\n\n"
        "Concrete plain-prose paragraph (active voice where natural, accurate reported values):\n{anchor}"
    ),
    (
        "Paper scratch notes:\n{notes}\n\n"
        "Direct research prose paragraph (plain verbs, varied cadence, complete sentences):\n{anchor}"
    ),
    (
        "Reading notes on published studies:\n{notes}\n\n"
        "Plain factual paragraph (exact comparisons, no filler transitions, varied sentence lengths):\n{anchor}"
    ),
]


def _extract_topic_phrase(para: str) -> str:
    """Extract a concise 2-word domain phrase from para to anchor Contrastive mode factually."""
    sents = split_sentences(para.strip())
    head = " ".join(sents[:2]) if sents else para
    stop = {
        "along", "these", "those", "that", "this", "methods", "which", "their", "there", "where",
        "while", "about", "order", "could", "would", "should", "using", "based",
        "recent", "empirical", "studies", "indicate", "significantly", "effectively",
        "particular", "however", "furthermore", "moreover", "between", "across",
    }
    cands = [
        w.strip(".,;:()[]\"'")
        for w in head.split()
        if len(w.strip(".,;:()[]\"'")) >= 4 and w.strip(".,;:()[]\"'").lower() not in stop and not w.strip(".,;:()[]\"'").isdigit()
    ]
    return " ".join(cands[:2]) if cands else "this setup"


def make_contrastive_anchor(para: str, idx: int) -> str:
    """
    Create a non-formulaic opening anchor for Contrastive mode.

    - Always preserves explicit inline lead-in headings (e.g. `RQ1. ...`, `Edge forecasting.`).
    - On half of paragraphs, returns an empty string `""` (free start) so human-demo
      conditioned Base models open organically without a detectable 6-template prefix pattern!
    - On the other half, uses natural varied factual openers (direct author subject, etc.).
    """
    stripped = para.strip()
    lead_m = re.match(r"^((?:[A-Z]{1,4}\d+\.\s+)?[A-Z][A-Za-z0-9\-\s]{2,45}\.)\s+(.+)$", stripped)
    if lead_m and not lead_m.group(1).lower().endswith("et al."):
        return f"{lead_m.group(1)} "

    sents = split_sentences(stripped)
    head = " ".join(sents[:2]) if sents else stripped
    auth_m = re.search(r"\b([A-Z][A-Za-z\-]+(?:\s+et\s+al\.|\s+and\s+[A-Z][A-Za-z\-]+))\b", head)

    cites = re.findall(r"\[\d+\](?:[–\-]\[\d+\])?", stripped)
    unique_cites: List[str] = []
    for c in cites:
        if c not in unique_cites:
            unique_cites.append(c)

    bucket = idx % 8
    # Free-start buckets (50% of paragraphs) eliminate cross-paragraph opener fingerprints
    if bucket in (0, 2, 4, 6):
        return ""
    if bucket == 1 and auth_m and unique_cites:
        return f"{auth_m.group(1)} {unique_cites[0]}"
    if bucket == 3 and auth_m:
        return f"{auth_m.group(1)}"
    if bucket == 5 and unique_cites:
        return f"Evidence in {unique_cites[0]}"
    if bucket == 7:
        topic = _extract_topic_phrase(stripped)
        if topic and topic != "this setup":
            return f"{topic[0].upper() + topic[1:]}"
    return ""


def trim_to_last_sentence(text: str) -> str:
    text = re.split(r"\b(?:Source|Rewrite|Notes|Paragraph)\s*:", text, maxsplit=1)[0]
    text = re.sub(r"\s+", " ", text).strip()
    matches = [
        m for m in re.finditer(r"(?<!\bal)\.(?:\s|$)", text)
        if not re.search(r"\b\d{1,3}\.$", text[: m.start() + 1])
    ]
    if matches:
        return text[: matches[-1].start() + 1].strip()
    cleaned = text.rstrip(" ,;:-")
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


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
    backend: str,
    base_gguf: str,
    n_gpu_layers: int,
    length_ratio: float,
    p_skip_top1: float,
    max_p1_for_skip: float,
    pool_size: int,
    min_alt_prob: float,
    min_ppl: float,
    attempts: int,
    base_seed: int,
    banned_clean: List[str],
    constrain_facts: bool,
    num_ctx: int,
    surrogate_ollama_model: str,
    surrogate_probes: int,
    neural_detector_model: str,
    neural_detector_device: str,
    on_progress: Optional[Callable[[Dict], None]],
    cancel_event: Optional[threading.Event],
    prefix_so_far: str = "",
) -> Dict:
    demo_dyn_leaks = human_demos.extract_demo_leak_words([{"notes": demo, "paragraph": ""}], chunk_text)
    demo_ban = _chunk_demo_blacklist(chunk_text, extra_leak_words=demo_dyn_leaks)
    chunk_banned = banned_clean + demo_ban
    fact_guard = build_fact_guard(chunk_text) if constrain_facts else None

    llama_base_llm = None
    if backend == "llama_cpp":
        llama_base_llm = backends.LLAMA_CPP_MANAGER.get_model(
            base_gguf, n_ctx=num_ctx, n_gpu_layers=n_gpu_layers
        )

    src_words = max(1, len(chunk_text.split()))
    target_words = max(20, int(round(src_words * length_ratio)))
    seed_words = len(seed_start.split())
    gen_target_words = max(15, target_words - seed_words)
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
        "fidelity": 0.0,
        "fidelity_pct": 0,
        "num_rec": 0.0,
        "cite_rec": 0.0,
        "halluc_nums": 0,
        "halluc_cites": 0,
        "human_score": 0.0,
        "human_score_pct": 0,
        "burstiness_score": 0.0,
        "sent_len_cv": 0.0,
        "sent_ppl_cv": None,
        "overlap_3gram": 0.0,
        "gltr_top1": None,
        "gltr_oob": None,
        "uncond_ppl": None,
        "binoculars": None,
        "neural_ai_prob": None,
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
        sent_nlls: List[List[float]] = []
        cur_sent_nlls: List[float] = []
        skipped = 0
        word_starts = 0
        t0 = time.time()

        single_session = None
        if backend == "llama_cpp" and llama_base_llm is not None:
            single_session = backends.LlamaCppSingleSession(backends.LLAMA_CPP_MANAGER, llama_base_llm, ctx)

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

            if single_session is not None:
                tok0, raw_top = single_session.next_top_tokens(top_k=32)
                if not raw_top:
                    break
                top_items: List[Tuple[Optional[int], str, float, bool]] = raw_top
            else:
                tok0, o_top = next_token_logprobs(ollama_base, model, ctx + gen, top_k=20, num_ctx=num_ctx)
                if not o_top:
                    break
                top_items = [(None, t, lp, t in STOP_TOKENS) for (t, lp) in o_top]

            cur_full_pre = re.sub(r"\s+", " ", (seed_start + gen)).strip()
            at_sent_end = _ends_clean_sentence(cur_full_pre)
            reached_len = (cur_words >= soft_stop_words and at_sent_end) or (cur_words >= int(target_words * 1.22))

            def _tok_allowed(t: str, is_eos: bool) -> bool:
                if is_eos or t in STOP_TOKENS:
                    return False
                if not is_clean_ascii(t):
                    return False
                if not reached_len and "\n" in t:
                    return False
                return True

            if reached_len and (tok0 in STOP_TOKENS or (top_items and top_items[0][3])):
                break

            clean = [
                item
                for item in top_items
                if _tok_allowed(item[1], item[3])
                and not _causes_banned_bpe(seed_start + gen, item[1], chunk_banned)
                and not _causes_demo_leak_or_repeat(seed_start + gen, item[1], demo_ban)
                and (not fact_guard or not _violates_fact_guard(seed_start + gen, item[1], fact_guard))
            ]
            if not clean:
                clean = [
                    item
                    for item in top_items
                    if _tok_allowed(item[1], item[3])
                    and not _causes_demo_leak_or_repeat(seed_start + gen, item[1], demo_ban)
                    and (not fact_guard or not _violates_fact_guard(seed_start + gen, item[1], fact_guard))
                ]
            if not clean:
                clean = [item for item in top_items if _tok_allowed(item[1], item[3]) and not any(ch.isdigit() for ch in item[1]) and "[" not in item[1]]
            if not clean:
                clean = [item for item in top_items if _tok_allowed(item[1], item[3])]
            if not clean:
                break

            chosen_tid, chosen_tok, chosen_lp, _ = clean[0]
            p0 = math.exp(chosen_lp)
            in_fact = bool(fact_guard and _is_active_fact_token(seed_start + gen, chosen_tok, fact_guard))
            if WORD_START_RE.match(chosen_tok) and not in_fact:
                word_starts += 1
                if p0 <= max_p1_for_skip and rng.random() < p_skip_top1:
                    alt_floor = max(min_alt_prob, 0.05 * p0)
                    alts = [
                        item
                        for item in clean[1 : max(2, pool_size + 1)]
                        if WORD_START_RE.match(item[1]) and math.exp(item[2]) >= alt_floor and "\n" not in item[1]
                    ]
                    if alts:
                        weights = [math.exp(item[2]) ** 0.45 for item in alts]
                        chosen_tid, chosen_tok, chosen_lp, _ = rng.choices(alts, weights=weights, k=1)[0]
                        skipped += 1

            if reached_len and "\n" in chosen_tok:
                break

            if single_session is not None and chosen_tid is not None:
                single_session.accept_token(chosen_tid)

            gen += chosen_tok
            nll_val = -chosen_lp
            nlls.append(nll_val)
            cur_sent_nlls.append(nll_val)

            cur_full = re.sub(r"\s+", " ", (seed_start + gen)).strip()
            cur_words = len(cur_full.split())
            if _ends_clean_sentence(cur_full) and len(cur_sent_nlls) >= 4:
                sent_nlls.append(cur_sent_nlls)
                cur_sent_nlls = []

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

            if cur_words >= soft_stop_words and _ends_clean_sentence(cur_full):
                break

        if cur_sent_nlls:
            sent_nlls.append(cur_sent_nlls)

        elapsed = round(time.time() - t0, 2)
        ppl = math.exp(sum(nlls) / max(len(nlls), 1)) if nlls else 0.0
        full = trim_to_last_sentence(seed_start + gen)
        actual_words = len(full.split())
        len_factor = _length_penalty_factor(actual_words, target_words)
        ok_len = actual_words >= int(target_words * 0.85)
        fid = compute_fidelity_metrics(chunk_text, full)

        sur = surrogates.evaluate_candidate_surrogates(
            full,
            chunk_text,
            sent_nlls=sent_nlls,
            ollama_base=ollama_base,
            surrogate_ollama_model=surrogate_ollama_model,
            surrogate_probes=surrogate_probes,
            num_ctx=num_ctx,
            neural_detector_model=neural_detector_model,
            neural_detector_device=neural_detector_device,
            cancel_event=cancel_event,
        )

        cand = {
            "text": full,
            "ppl": round(ppl, 2),
            "tokens": len(nlls),
            "skipped": skipped,
            "word_starts": word_starts,
            "attempt": attempt,
            "seed": seed_val,
            "elapsed": elapsed,
            **fid,
            **sur,
            "cancelled": False,
        }

        ppl_sat = min(1.0, ppl / max(1.0, min_ppl))
        fluency = 1.0 if ppl <= 10.5 else max(0.35, 10.5 / ppl)
        has_ext_sur = (sur.get("neural_ai_prob") is not None) or (sur.get("gltr_top1") is not None)
        if has_ext_sur:
            det_part = 0.55 * sur["human_score"] + 0.45 * ppl_sat
        else:
            det_part = 0.25 * sur["burstiness_score"] + 0.75 * ppl_sat
        rank = (0.50 * fid["fidelity"] + 0.50 * det_part) * fluency * len_factor
        if rank > best_rank and actual_words >= 10:
            best_rank = rank
            best_result = cand

        sur_ok = (not has_ext_sur) or bool(sur.get("passes_surrogate"))
        if ppl >= min_ppl and ok_len and fid["fidelity"] >= 0.80 and fid["halluc_nums"] == 0 and fid["halluc_cites"] == 0 and sur_ok:
            break

    return best_result


def rewrite_paragraph(
    source_para: str,
    demo: str,
    seed_start: str,
    *,
    ollama_base: str = DEFAULT_OLLAMA_BASE,
    model: str = DEFAULT_MODEL,
    backend: str = "ollama",
    base_gguf: str = "",
    n_gpu_layers: int = -1,
    length_ratio: float = DEFAULT_LENGTH_RATIO,
    max_tokens: Optional[int] = None,
    p_skip_top1: float = 0.72,
    max_p1_for_skip: float = 0.88,
    pool_size: int = 6,
    min_alt_prob: float = 0.01,
    min_ppl: float = 3.8,
    attempts: int = 3,
    base_seed: int = 401,
    banned_substrs: Optional[List[str]] = None,
    constrain_facts: bool = True,
    num_ctx: int = DEFAULT_NUM_CTX,
    surrogate_ollama_model: str = "",
    surrogate_probes: int = 16,
    neural_detector_model: str = "",
    neural_detector_device: str = "cpu",
    on_progress: Optional[Callable[[Dict], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict:
    """ZeroGPT Mode: Length-preserving, fact-guarded paragraph rewrite (auto-chunks long paragraphs)."""
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
    sur_human_scores: List[float] = []
    sur_bursts: List[float] = []
    sur_neural_probs: List[float] = []
    sur_gltr_t1: List[float] = []

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
            backend=backend,
            base_gguf=base_gguf,
            n_gpu_layers=n_gpu_layers,
            length_ratio=length_ratio,
            p_skip_top1=p_skip_top1,
            max_p1_for_skip=max_p1_for_skip,
            pool_size=pool_size,
            min_alt_prob=min_alt_prob,
            min_ppl=min_ppl,
            attempts=attempts,
            base_seed=base_seed + c_idx * 131,
            banned_clean=banned_clean,
            constrain_facts=constrain_facts,
            num_ctx=num_ctx,
            surrogate_ollama_model=surrogate_ollama_model,
            surrogate_probes=surrogate_probes,
            neural_detector_model=neural_detector_model,
            neural_detector_device=neural_detector_device,
            on_progress=on_progress,
            cancel_event=cancel_event,
            prefix_so_far=prefix_so_far,
        )
        if res.get("cancelled"):
            return {
                "mode": "anti_greedy",
                "backend": backend,
                "text": " ".join(parts) if parts else source_para.strip(),
                "ppl": 0.0,
                "inst_ppl": None,
                "tokens": total_tokens,
                "skipped": total_skipped,
                "word_starts": total_ws,
                "skip_ratio": 0.0,
                "fidelity": 0.0,
                "fidelity_pct": 0,
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
        if res.get("human_score") is not None:
            sur_human_scores.append(float(res["human_score"]))
        if res.get("burstiness_score") is not None:
            sur_bursts.append(float(res["burstiness_score"]))
        if res.get("neural_ai_prob") is not None:
            sur_neural_probs.append(float(res["neural_ai_prob"]))
        if res.get("gltr_top1") is not None:
            sur_gltr_t1.append(float(res["gltr_top1"]))

    combined_text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    agg_ppl = math.exp(weighted_log_ppl / max(1, total_tokens)) if total_tokens else 0.0
    skip_ratio = round(100.0 * total_skipped / max(1, total_ws), 1)
    src_words = max(1, len(source_para.split()))
    out_words = len(combined_text.split())
    fid = compute_fidelity_metrics(source_para, combined_text)
    full_burst = surrogates.compute_burstiness_and_overlap(combined_text, source_para)
    avg_hs = round(sum(sur_human_scores) / len(sur_human_scores), 3) if sur_human_scores else full_burst["burstiness_score"]

    final = {
        "mode": "anti_greedy",
        "backend": backend,
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
        **fid,
        "human_score": avg_hs,
        "human_score_pct": int(round(100.0 * avg_hs)),
        "burstiness_score": round(sum(sur_bursts) / len(sur_bursts), 3) if sur_bursts else full_burst["burstiness_score"],
        "sent_len_cv": full_burst["sent_len_cv"],
        "overlap_3gram": full_burst["overlap_3gram"],
        "gltr_top1": round(sum(sur_gltr_t1) / len(sur_gltr_t1), 3) if sur_gltr_t1 else None,
        "neural_ai_prob": round(sum(sur_neural_probs) / len(sur_neural_probs), 4) if sur_neural_probs else None,
        "binoculars": None,
        "judge_score": None,
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
    backend: str,
    base_gguf: str,
    instruct_gguf: str,
    n_gpu_layers: int,
    length_ratio: float,
    alpha: float,
    min_base_prob: float,
    beta: float,
    i_floor: float,
    temp: float,
    pool_size: int,
    min_inst_ppl: float,
    attempts: int,
    base_seed: int,
    use_telegraphic: bool,
    constrain_facts: bool,
    use_judge: bool,
    cad_blind: bool,
    dynamic_alpha: bool,
    use_human_demos: bool,
    demo_pool: Sequence[Dict[str, str]],
    contrastive_banned: List[str],
    num_ctx: int,
    surrogate_ollama_model: str,
    surrogate_probes: int,
    neural_detector_model: str,
    neural_detector_device: str,
    binoculars_threshold: float,
    on_progress: Optional[Callable[[Dict], None]],
    cancel_event: Optional[threading.Event],
    prefix_so_far: str = "",
) -> Dict:
    fact_guard = build_fact_guard(chunk_text) if constrain_facts else None

    llama_base_llm = None
    llama_inst_llm = None
    inst_template_name = instruct_model
    if backend == "llama_cpp":
        llama_base_llm = backends.LLAMA_CPP_MANAGER.get_model(
            base_gguf, n_ctx=num_ctx, n_gpu_layers=n_gpu_layers
        )
        llama_inst_llm = backends.LLAMA_CPP_MANAGER.get_model(
            instruct_gguf, n_ctx=num_ctx, n_gpu_layers=n_gpu_layers
        )
        backends.LLAMA_CPP_MANAGER.verify_vocab_alignment(llama_base_llm, llama_inst_llm)
        inst_template_name = instruct_gguf or instruct_model

    # Cache Stage-0 telegraphic decompositions by variant bucket so retries explore
    # genuinely distinct discourse skeletons without regenerating on every single attempt.
    notes_cache: Dict[int, str] = {}

    def _get_notes_for_attempt(attempt_num: int, seed_val: int) -> str:
        if not use_telegraphic:
            return chunk_text.strip()
        # Refresh organizational perspective every 2 attempts (1-2 -> v0, 3-4 -> v1, 5-6 -> v2...)
        v_bucket = (attempt_num - 1) // 2
        if v_bucket not in notes_cache:
            schema_idx = (base_seed // 13 + v_bucket) % len(NOTE_EXTRACTION_SCHEMAS)
            shuffle_s = (seed_val + 97) if v_bucket > 0 else None
            notes_cache[v_bucket] = extract_telegraphic_notes(
                chunk_text,
                ollama_base=ollama_base,
                instruct_model=inst_template_name,
                num_ctx=num_ctx,
                variant_idx=schema_idx,
                shuffle_seed=shuffle_s,
                backend=backend,
                llama_llm=llama_inst_llm,
            )
        return notes_cache[v_bucket]

    src_words = max(1, len(chunk_text.split()))
    target_words = max(22, int(round(src_words * length_ratio)))
    anchor_words = len(anchor.split())
    gen_target_words = max(18, target_words - anchor_words)

    stop_ratio = 0.78 if use_telegraphic else 0.90
    soft_stop_words = max(24, int(round(target_words * stop_ratio)))
    max_tokens = max(95, int(gen_target_words * 1.50) + 30)

    # Number of human style demos that safely fit inside num_ctx alongside notes + output
    n_demos = 0
    if use_human_demos and demo_pool:
        n_demos = 2 if num_ctx >= 3072 else 1

    best_result = {
        "text": chunk_text.strip(),
        "ppl": 0.0,
        "inst_ppl": 0.0,
        "tokens": 0,
        "attempt": 0,
        "seed": base_seed,
        "elapsed": 0.0,
        "ok_complete": False,
        "fidelity": 0.0,
        "fidelity_pct": 0,
        "num_rec": 0.0,
        "cite_rec": 0.0,
        "halluc_nums": 0,
        "halluc_cites": 0,
        "judge_score": None,
        "human_score": 0.0,
        "human_score_pct": 0,
        "passes_surrogate": False,
        "burstiness_score": 0.0,
        "sent_len_cv": 0.0,
        "sent_ppl_cv": None,
        "overlap_3gram": 0.0,
        "gltr_top1": None,
        "gltr_oob": None,
        "uncond_ppl": None,
        "binoculars": None,
        "neural_ai_prob": None,
        "cancelled": False,
    }
    best_rank = -1.0
    alpha_boost = 0.0
    temp_boost = 0.0

    for attempt in range(1, max(1, attempts) + 1):
        if cancel_event and cancel_event.is_set():
            best_result["cancelled"] = True
            return best_result

        seed_val = base_seed + (attempt - 1) * 53
        reg_idx = (seed_val // 17 + attempt) % len(CONTRASTIVE_REGISTERS)
        notes_text = _get_notes_for_attempt(attempt, seed_val)

        # Retrieve real human few-shot demos for Base ONLY (Instruct sees generic prompt!)
        active_demos: List[Dict[str, str]] = []
        demo_ban: List[str] = []
        if n_demos > 0:
            active_demos = human_demos.retrieve_human_demos(
                chunk_text,
                demo_pool,
                k=n_demos,
                attempt=attempt,
                seed=base_seed,
            )
            demo_ban = human_demos.extract_demo_leak_words(active_demos, chunk_text)

        # Allow free start on later retries if anchor was non-empty and not an explicit heading
        cur_anchor = anchor
        if attempt >= 3 and anchor and not re.match(r"^(?:[A-Z]{1,4}\d+\.\s+|[A-Z][A-Za-z0-9\-\s]{2,45}\.\s)$", anchor):
            cur_anchor = "" if (attempt % 2 == 1) else anchor

        base_core = CONTRASTIVE_REGISTERS[reg_idx].format(
            notes=notes_text,
            anchor=cur_anchor,
        )
        if active_demos:
            demos_prefix = human_demos.format_human_demos_block(active_demos)
            base_prompt = f"{demos_prefix}\n\n---\n\n{base_core}"
        else:
            base_prompt = base_core

        if cad_blind:
            inst_prompt = format_instruct_prompt(
                inst_template_name,
                "Write a formal academic paragraph.",
                assistant_prefix=cur_anchor,
            )
        else:
            inst_prompt = format_instruct_prompt(
                inst_template_name,
                f"Write a formal academic paragraph from these notes:\n{notes_text}",
                assistant_prefix=cur_anchor,
            )

        alpha_jit = max(0.12, min(1.60, alpha + alpha_boost + [0.0, 0.03, -0.02, 0.04, 0.02, -0.02][reg_idx]))
        temp_jit = max(0.18, min(1.30, temp + temp_boost + [0.0, 0.03, -0.02, 0.04, 0.02, -0.02][reg_idx]))

        scheduler = BurstinessScheduler(seed=seed_val + attempt * 7, enabled=dynamic_alpha)
        rng = random.Random(seed_val)
        gen = ""
        base_nlls: List[float] = []
        inst_nlls: List[float] = []
        sent_inst_nlls: List[List[float]] = []
        cur_sent_inst_nlls: List[float] = []
        sent_idx = 0
        words_in_sent = len(cur_anchor.split())
        tokens_since_bracket = 20
        cur_words = len(cur_anchor.split())
        t0 = time.time()

        exact_session = None
        if backend == "llama_cpp" and llama_base_llm is not None and llama_inst_llm is not None:
            exact_session = backends.LlamaCppContrastiveSession(
                backends.LLAMA_CPP_MANAGER,
                llama_base_llm,
                llama_inst_llm,
                base_prompt,
                inst_prompt,
            )

        if on_progress:
            live = f"{prefix_so_far} {cur_anchor}".strip() if prefix_so_far else cur_anchor
            on_progress({
                "event": "attempt_start",
                "mode": "contrastive",
                "backend": backend,
                "attempt": attempt,
                "attempts": attempts,
                "seed": seed_val,
                "text": live,
            })

        for step in range(max_tokens):
            if cancel_event and cancel_event.is_set():
                best_result["cancelled"] = True
                return best_result

            # Fetch next-token candidates from either Exact Full-Vocab llama.cpp or Ollama
            if exact_session is not None:
                raw_cands = exact_session.next_exact_candidates(top_base_k=48)
                if not raw_cands:
                    break
                # (tid, tok, b_lp, raw_i_lp, is_eos)
                step_cands = raw_cands
            else:
                _, b_top = next_token_logprobs(ollama_base, base_model, base_prompt + gen, top_k=20, num_ctx=num_ctx)
                if not b_top:
                    break
                _, i_top = next_token_logprobs(ollama_base, instruct_model, inst_prompt + gen, top_k=20, num_ctx=num_ctx)
                i_map = {t: lp for t, lp in i_top}
                raw_min_i = (min(i_map.values()) - 1.5) if i_map else -10.0
                step_cands = [
                    (None, tok, b_lp, i_map.get(tok, raw_min_i), tok in STOP_TOKENS)
                    for tok, b_lp in b_top
                ]

            cur_full_pre = re.sub(r"\s+", " ", (cur_anchor + gen)).strip()
            at_sent_end = _ends_clean_sentence(cur_full_pre)
            reached_len = (cur_words >= soft_stop_words and at_sent_end) or (cur_words >= int(target_words * 1.20))

            top_b_lp = max(c[2] for c in step_cands)
            top_b_prob = math.exp(top_b_lp)
            plaus_thresh = max(min_base_prob, beta * top_b_prob)

            scored = []
            for tid, tok, b_lp, raw_i_lp, is_eos in step_cands:
                if is_eos or tok in STOP_TOKENS:
                    continue
                if not is_clean_ascii(tok) or "—" in tok or "–" in tok:
                    continue
                if "\n" in tok and not reached_len:
                    continue
                if "[" in tok and tokens_since_bracket < 12:
                    continue
                if re.search(r"\b(?:Notes|Source|Paragraph|Study|Method|Result|Detail|Context)\s*:", (cur_anchor + gen)[-15:] + tok):
                    continue
                # Enforce Contrastive BPE blacklist (AI clichés + archaic drift like whilst/thereof/herein)
                if contrastive_banned and _causes_banned_bpe(cur_anchor + gen, tok, contrastive_banned):
                    continue
                # Block AI cliché openings when starting freely
                if not cur_anchor and step < 6 and _causes_opening_cliche(gen, tok):
                    continue
                cand_ws = re.sub(r"[^a-z0-9\s\-]", " ", (cur_anchor + gen + tok).lower()).split()
                if len(cand_ws) >= 4 and cand_ws[-2:] == cand_ws[-4:-2]:
                    continue
                if WORD_START_RE.match(tok):
                    tw = tok.strip().lower()
                    if len(tw) >= 5 and len(cand_ws) >= 2 and tw in cand_ws[-4:-1]:
                        continue
                if _causes_demo_leak_or_repeat(cur_anchor + gen, tok, demo_ban):
                    continue
                if fact_guard and _violates_fact_guard(cur_anchor + gen, tok, fact_guard):
                    continue
                # Adaptive Plausibility Gate (Li et al. 2023)
                if math.exp(b_lp) < plaus_thresh and len(scored) >= 1:
                    continue

                eff_i_lp = max(raw_i_lp, i_floor)
                in_fact = bool(fact_guard and _is_active_fact_token(cur_anchor + gen, tok, fact_guard))
                eff_alpha, cadence_nudge = scheduler.modulate(
                    alpha_jit,
                    tok,
                    sent_idx=sent_idx,
                    words_in_sent=words_in_sent,
                    in_fact=in_fact,
                )
                c_score = b_lp - eff_alpha * eff_i_lp + cadence_nudge
                scored.append((tid, tok, c_score, b_lp, raw_i_lp))

            if not scored:
                fallback = [
                    c for c in step_cands
                    if not c[4]
                    and c[1] not in STOP_TOKENS
                    and is_clean_ascii(c[1])
                    and ("\n" not in c[1] or reached_len)
                    and not _causes_demo_leak_or_repeat(cur_anchor + gen, c[1], demo_ban)
                    and (not fact_guard or not _violates_fact_guard(cur_anchor + gen, c[1], fact_guard))
                ]
                if not fallback:
                    fallback = [
                        c for c in step_cands
                        if not c[4]
                        and c[1] not in STOP_TOKENS
                        and is_clean_ascii(c[1])
                        and ("\n" not in c[1] or reached_len)
                        and not any(ch.isdigit() for ch in c[1])
                        and "[" not in c[1]
                    ]
                if not fallback:
                    fallback = [c for c in step_cands if not c[4] and c[1] not in STOP_TOKENS and is_clean_ascii(c[1])]
                if not fallback:
                    break
                tid, tok, b_lp, raw_i_lp, _ = fallback[0]
                scored = [(tid, tok, b_lp - alpha_jit * max(raw_i_lp, i_floor), b_lp, raw_i_lp)]

            scored.sort(key=lambda x: x[2], reverse=True)
            pool = scored[: max(2, pool_size)]
            mx = pool[0][2]
            eff_temp = max(0.05, temp_jit)
            weights = [math.exp((x[2] - mx) / eff_temp) for x in pool]
            chosen_tid, chosen_tok, _, b_lp, raw_i_lp = rng.choices(pool, weights=weights, k=1)[0]

            if "\n" in chosen_tok and reached_len:
                break

            if exact_session is not None and chosen_tid is not None:
                exact_session.accept_token(chosen_tid)

            gen += chosen_tok
            tokens_since_bracket = 0 if "[" in chosen_tok else (tokens_since_bracket + 1)
            base_nlls.append(-b_lp)
            inst_nlls.append(-raw_i_lp)
            cur_sent_inst_nlls.append(-raw_i_lp)

            cur_full = re.sub(r"\s+", " ", (cur_anchor + gen)).strip()
            if cur_words >= int(target_words * 0.70) and re.search(r"\b(?:Source|Rewrite|Notes|Paragraph)\s*:", gen):
                break
            new_words = len(cur_full.split())
            if new_words > cur_words:
                words_in_sent += (new_words - cur_words)
            cur_words = new_words

            if _ends_clean_sentence(cur_full) and len(cur_sent_inst_nlls) >= 4:
                sent_inst_nlls.append(cur_sent_inst_nlls)
                cur_sent_inst_nlls = []
                sent_idx += 1
                words_in_sent = 0

            if on_progress and (step % 4 == 0 or chosen_tok.endswith(".")):
                r_bppl = math.exp(sum(base_nlls) / max(len(base_nlls), 1))
                r_ippl = math.exp(sum(inst_nlls) / max(len(inst_nlls), 1))
                live = f"{prefix_so_far} {cur_full}".strip() if prefix_so_far else cur_full
                on_progress({
                    "event": "token",
                    "mode": "contrastive",
                    "backend": backend,
                    "attempt": attempt,
                    "attempts": attempts,
                    "text": live,
                    "ppl": round(r_bppl, 2),
                    "inst_ppl": round(r_ippl, 1),
                    "tokens": len(base_nlls),
                })

            if cur_words >= soft_stop_words and _ends_clean_sentence(cur_full):
                break

        if cur_sent_inst_nlls:
            sent_inst_nlls.append(cur_sent_inst_nlls)

        elapsed = round(time.time() - t0, 2)
        base_ppl = math.exp(sum(base_nlls) / max(len(base_nlls), 1)) if base_nlls else 0.0
        inst_ppl = math.exp(sum(inst_nlls) / max(len(inst_nlls), 1)) if inst_nlls else 0.0
        full = trim_to_last_sentence(cur_anchor + gen)
        actual_words = len(full.split())
        min_len_ratio = 0.70 if use_telegraphic else 0.85
        ok_complete = (
            full.endswith(".")
            and actual_words >= int(target_words * min_len_ratio)
            and full.count("(") == full.count(")")
        )
        ratio = actual_words / max(1, target_words)
        if min_len_ratio <= ratio <= 1.25:
            len_factor = 1.0
        elif ratio < min_len_ratio:
            len_factor = max(0.15, (ratio / min_len_ratio) ** 2)
        else:
            len_factor = max(0.45, 1.25 / ratio)

        fid = compute_fidelity_metrics(chunk_text, full)

        # Multi-signal independent surrogate evaluation (Step 2)
        sur = surrogates.evaluate_candidate_surrogates(
            full,
            chunk_text,
            sent_nlls=sent_inst_nlls,
            ollama_base=ollama_base,
            surrogate_ollama_model=surrogate_ollama_model,
            surrogate_probes=surrogate_probes,
            num_ctx=num_ctx,
            llama_base_llm=llama_base_llm if backend == "llama_cpp" else None,
            llama_inst_llm=llama_inst_llm if backend == "llama_cpp" else None,
            binoculars_threshold=binoculars_threshold,
            neural_detector_model=neural_detector_model,
            neural_detector_device=neural_detector_device,
            cancel_event=cancel_event,
        )

        j_score: Optional[float] = None
        if use_judge and actual_words >= 15 and not (cancel_event and cancel_event.is_set()):
            j_score = judge_faithfulness(
                chunk_text,
                full,
                ollama_base=ollama_base,
                instruct_model=inst_template_name,
                num_ctx=num_ctx,
                backend=backend,
                llama_llm=llama_inst_llm,
            )

        cand = {
            "text": full,
            "ppl": round(base_ppl, 2),
            "inst_ppl": round(inst_ppl, 1),
            "tokens": len(base_nlls),
            "attempt": attempt,
            "seed": seed_val,
            "elapsed": elapsed,
            "ok_complete": ok_complete,
            **fid,
            **sur,
            "judge_score": j_score,
            "cancelled": False,
        }

        # Multi-signal pass-tier ranking:
        # When independent external surrogates (Neural Detector / Exact Binoculars / Unconditional GLTR)
        # are active, they govern detector clearance instead of letting conditional InstPPL create false greens!
        has_strong_surrogate = (
            (sur.get("neural_ai_prob") is not None)
            or (sur.get("binoculars") is not None)
            or (sur.get("gltr_top1") is not None)
        )
        clears_internal = (inst_ppl >= min_inst_ppl) and (base_ppl >= 5.5 if backend == "llama_cpp" else base_ppl >= 6.5)

        if has_strong_surrogate:
            clears_detector = bool(sur.get("passes_surrogate")) and (sur["human_score"] >= 0.62)
            if clears_detector:
                detector_mult = 1.65 + 0.55 * sur["human_score"]
            else:
                detector_mult = 0.20 + 0.90 * sur["human_score"]
        else:
            # Pure Ollama base+instruct mode: combine internal PPL with structural burstiness & anti-paraphrase distance
            clears_detector = clears_internal and (sur["overlap_3gram"] <= 0.24) and (sur["burstiness_score"] >= 0.48)
            if clears_detector:
                detector_mult = 1.75 + 0.35 * sur["burstiness_score"]
            else:
                ip_frac = min(1.0, inst_ppl / max(1.0, min_inst_ppl))
                bp_frac = min(1.0, base_ppl / 6.5)
                detector_mult = 0.25 + 0.75 * (0.50 * ip_frac + 0.25 * bp_frac + 0.25 * sur["human_score"])

        fluency_cap = 1.0 if base_ppl <= 13.0 else max(0.40, 13.0 / base_ppl)
        punct_mult = 1.0 if (full.endswith(".") and full.count("(") == full.count(")")) else 0.45
        short_mult = 1.0 if actual_words >= int(target_words * 0.60) else 0.08
        num_guard = 0.65 + 0.35 * fid["num_rec"]
        if j_score is not None:
            quality_part = 0.55 * fid["fidelity"] + 0.45 * (j_score / 10.0)
        else:
            quality_part = fid["fidelity"]
        rank = detector_mult * (0.25 + 0.75 * quality_part) * fluency_cap * len_factor * punct_mult * short_mult * num_guard

        if rank > best_rank and actual_words >= 10:
            best_rank = rank
            best_result = cand

        judge_ok = (j_score is None) or (j_score >= 7.0)
        if (
            ok_complete
            and clears_detector
            and fid["fidelity"] >= 0.78
            and fid["halluc_nums"] == 0
            and fid["halluc_cites"] == 0
            and judge_ok
        ):
            break

        # Bidirectional adaptive schedule:
        #   - If too AI-like (failed detector gate), step alpha & temp UP.
        #   - If detector cleared (or BasePPL too high) but Judge/Fidelity failed, step alpha DOWN!
        if not clears_detector:
            alpha_boost = min(0.32, alpha_boost + 0.07)
            temp_boost = min(0.18, temp_boost + 0.04)
        elif (not judge_ok) or (fid["fidelity"] < 0.78) or (base_ppl > 13.5):
            alpha_boost = max(-0.18, alpha_boost - 0.06)
            temp_boost = max(-0.12, temp_boost - 0.03)

    return best_result


def rewrite_paragraph_contrastive(
    source_para: str,
    anchor: str,
    *,
    ollama_base: str = DEFAULT_OLLAMA_BASE,
    base_model: str = DEFAULT_MODEL,
    instruct_model: str = DEFAULT_INSTRUCT_MODEL,
    backend: str = "ollama",
    base_gguf: str = "",
    instruct_gguf: str = "",
    n_gpu_layers: int = -1,
    length_ratio: float = DEFAULT_LENGTH_RATIO,
    alpha: float = 0.56,
    min_base_prob: float = 0.020,
    beta: float = DEFAULT_BETA,
    i_floor: float = DEFAULT_I_FLOOR,
    temp: float = 0.62,
    pool_size: int = 6,
    min_inst_ppl: float = 350.0,
    min_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    attempts: int = 4,
    base_seed: int = 401,
    use_telegraphic: bool = True,
    constrain_facts: bool = True,
    use_judge: bool = True,
    cad_blind: bool = False,
    dynamic_alpha: bool = True,
    use_human_demos: bool = True,
    custom_human_demos: Optional[Sequence[Dict[str, str]]] = None,
    custom_human_demos_text: str = "",
    banned_substrs: Optional[List[str]] = None,
    num_ctx: int = DEFAULT_NUM_CTX,
    surrogate_ollama_model: str = "",
    surrogate_probes: int = 16,
    neural_detector_model: str = "",
    neural_detector_device: str = "cpu",
    binoculars_threshold: float = 0.90,
    on_progress: Optional[Callable[[Dict], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict:
    """
    GPTZero Mode: Fact-guarded, length-preserving Contrastive Anti-RLHF + Human-Style
    Decoding with multi-perspective Stage-0, Dynamic Burstiness alpha(t), optional
    exact full-vocab llama.cpp backend, and multi-signal local surrogate reranking.
    """
    chunks = chunk_paragraph(source_para, max_chunk_words=160)
    demo_pool = human_demos.get_demo_pool(custom_human_demos, custom_human_demos_text) if use_human_demos else []
    c_banned = [
        b.lower().strip()
        for b in (banned_substrs if banned_substrs is not None else CONTRASTIVE_BANNED_SUBSTRS)
        if b and b.strip()
    ]

    parts: List[str] = []
    total_tokens = 0
    total_elapsed = 0.0
    weighted_log_bppl = 0.0
    weighted_log_ippl = 0.0
    judge_scores: List[float] = []
    human_scores: List[float] = []
    burst_scores: List[float] = []
    bino_vals: List[float] = []
    neural_vals: List[float] = []
    gltr_t1_vals: List[float] = []
    gltr_oob_vals: List[float] = []
    uncond_ppls: List[float] = []
    last_attempt = 1

    for c_idx, chunk in enumerate(chunks):
        c_anchor = anchor if c_idx == 0 else make_contrastive_anchor(chunk, c_idx + base_seed)
        prefix_so_far = " ".join(parts)

        res = _rewrite_chunk_contrastive(
            chunk,
            anchor=c_anchor,
            ollama_base=ollama_base,
            base_model=base_model,
            instruct_model=instruct_model,
            backend=backend,
            base_gguf=base_gguf,
            instruct_gguf=instruct_gguf,
            n_gpu_layers=n_gpu_layers,
            length_ratio=length_ratio,
            alpha=alpha,
            min_base_prob=min_base_prob,
            beta=beta,
            i_floor=i_floor,
            temp=temp,
            pool_size=pool_size,
            min_inst_ppl=min_inst_ppl,
            attempts=attempts,
            base_seed=base_seed + c_idx * 137,
            use_telegraphic=use_telegraphic,
            constrain_facts=constrain_facts,
            use_judge=use_judge,
            cad_blind=cad_blind,
            dynamic_alpha=dynamic_alpha,
            use_human_demos=use_human_demos,
            demo_pool=demo_pool,
            contrastive_banned=c_banned,
            num_ctx=num_ctx,
            surrogate_ollama_model=surrogate_ollama_model,
            surrogate_probes=surrogate_probes,
            neural_detector_model=neural_detector_model,
            neural_detector_device=neural_detector_device,
            binoculars_threshold=binoculars_threshold,
            on_progress=on_progress,
            cancel_event=cancel_event,
            prefix_so_far=prefix_so_far,
        )
        if res.get("cancelled"):
            return {
                "mode": "contrastive",
                "backend": backend,
                "text": " ".join(parts) if parts else source_para.strip(),
                "ppl": 0.0,
                "inst_ppl": 0.0,
                "alpha": alpha,
                "tokens": total_tokens,
                "skipped": 0,
                "word_starts": 0,
                "skip_ratio": None,
                "fidelity": 0.0,
                "fidelity_pct": 0,
                "judge_score": None,
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
        if res.get("judge_score") is not None:
            judge_scores.append(float(res["judge_score"]))
        if res.get("human_score") is not None:
            human_scores.append(float(res["human_score"]))
        if res.get("burstiness_score") is not None:
            burst_scores.append(float(res["burstiness_score"]))
        if res.get("binoculars") is not None:
            bino_vals.append(float(res["binoculars"]))
        if res.get("neural_ai_prob") is not None:
            neural_vals.append(float(res["neural_ai_prob"]))
        if res.get("gltr_top1") is not None:
            gltr_t1_vals.append(float(res["gltr_top1"]))
        if res.get("gltr_oob") is not None:
            gltr_oob_vals.append(float(res["gltr_oob"]))
        if res.get("uncond_ppl") is not None:
            uncond_ppls.append(float(res["uncond_ppl"]))
        last_attempt = res["attempt"]

    combined_text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    agg_bppl = math.exp(weighted_log_bppl / max(1, total_tokens)) if total_tokens else 0.0
    agg_ippl = math.exp(weighted_log_ippl / max(1, total_tokens)) if total_tokens else 0.0
    src_words = max(1, len(source_para.split()))
    out_words = len(combined_text.split())
    fid = compute_fidelity_metrics(source_para, combined_text)
    full_burst = surrogates.compute_burstiness_and_overlap(combined_text, source_para)

    avg_judge = round(sum(judge_scores) / len(judge_scores), 1) if judge_scores else None
    avg_human = round(sum(human_scores) / len(human_scores), 3) if human_scores else full_burst["burstiness_score"]
    avg_burst = round(sum(burst_scores) / len(burst_scores), 3) if burst_scores else full_burst["burstiness_score"]

    final = {
        "mode": "contrastive",
        "backend": backend,
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
        **fid,
        "human_score": avg_human,
        "human_score_pct": int(round(100.0 * avg_human)),
        "burstiness_score": avg_burst,
        "sent_len_cv": full_burst["sent_len_cv"],
        "overlap_3gram": full_burst["overlap_3gram"],
        "binoculars": round(sum(bino_vals) / len(bino_vals), 4) if bino_vals else None,
        "neural_ai_prob": round(sum(neural_vals) / len(neural_vals), 4) if neural_vals else None,
        "gltr_top1": round(sum(gltr_t1_vals) / len(gltr_t1_vals), 3) if gltr_t1_vals else None,
        "gltr_oob": round(sum(gltr_oob_vals) / len(gltr_oob_vals), 3) if gltr_oob_vals else None,
        "uncond_ppl": round(sum(uncond_ppls) / len(uncond_ppls), 2) if uncond_ppls else None,
        "judge_score": avg_judge,
        "cancelled": False,
    }
    if on_progress:
        on_progress({"event": "attempt_done", **final})
    return final


def build_bootstrap_demo(source_para: str, rewritten_para: str) -> str:
    """Create a self-bootstrapping in-context demo from an already-verified paragraph."""
    return f"Notes:\n{source_para.strip()}\n\nParagraph:\n{rewritten_para.strip()}"


def main():
    parser = argparse.ArgumentParser(description="Local Base-Model Humanizer CLI (Ollama & llama.cpp)")
    parser.add_argument("--input", "-i", help="Input text file (reads stdin if omitted)")
    parser.add_argument("--output", "-o", help="Output text file (prints stdout if omitted)")
    parser.add_argument(
        "--mode",
        choices=["anti_greedy", "contrastive"],
        default="contrastive",
        help="Decoding mode: anti_greedy (ZeroGPT) or contrastive (GPTZero Anti-RLHF + Human Demos)",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "llama_cpp"],
        default="ollama",
        help="Inference backend: ollama (default zero-dep HTTP) or llama_cpp (exact full-vocab GGUF)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama base model (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--instruct-model",
        default=DEFAULT_INSTRUCT_MODEL,
        help=f"Same-vocab Ollama instruct model for contrastive mode (default: {DEFAULT_INSTRUCT_MODEL})",
    )
    parser.add_argument("--base-gguf", default="", help="Local Base GGUF path (for --backend llama_cpp)")
    parser.add_argument("--instruct-gguf", default="", help="Local Instruct GGUF path (for --backend llama_cpp)")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="GPU layers to offload in llama_cpp (-1 = all)")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_BASE, help="Ollama base URL")
    parser.add_argument("--length-ratio", type=float, default=DEFAULT_LENGTH_RATIO, help="Target output/input word ratio (default: 1.0)")
    parser.add_argument("--p-skip", type=float, default=0.72, help="Anti-greedy Rank-1 skip prob (default: 0.72)")
    parser.add_argument("--min-ppl", type=float, default=3.8, help="Anti-greedy min target PPL (default: 3.8)")
    parser.add_argument("--alpha", type=float, default=0.56, help="Contrastive anti-RLHF weight alpha (default: 0.56)")
    parser.add_argument("--min-inst-ppl", type=float, default=350.0, help="Contrastive min target InstPPL (default: 350)")
    parser.add_argument("--temp", type=float, default=0.62, help="Contrastive sampling temperature (default: 0.62)")
    parser.add_argument("--i-floor", type=float, default=DEFAULT_I_FLOOR, help=f"Contrastive instruct logprob clamp floor (default: {DEFAULT_I_FLOOR})")
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help=f"Adaptive plausibility ratio beta (default: {DEFAULT_BETA})")
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help=f"KV cache context window (default: {DEFAULT_NUM_CTX})")
    parser.add_argument("--blind-cad", action="store_true", help="Use Blind-Instruct CAD (higher fidelity, softer anti-paraphrase)")
    parser.add_argument("--no-telegraphic", action="store_true", help="Disable Stage-0 telegraphic fact decomposition")
    parser.add_argument("--no-fact-guard", action="store_true", help="Disable strict numeric/citation/entity fact guard")
    parser.add_argument("--no-judge", action="store_true", help="Disable read-only local Instruct faithfulness judge")
    parser.add_argument("--no-dynamic-alpha", action="store_true", help="Disable dynamic sentence-aware Burstiness alpha(t) modulation")
    parser.add_argument("--no-human-demos", action="store_true", help="Disable real human-style few-shot contrastive conditioning")
    parser.add_argument("--human-demos", default="", help="Path to custom human corpus (.jsonl or .txt) for few-shot contrastive demos")
    parser.add_argument("--surrogate-model", default="", help="Optional cross-family Ollama model for unconditional GLTR reranking")
    parser.add_argument("--surrogate-probes", type=int, default=16, help="Number of strided unconditional GLTR probes per candidate (default: 16)")
    parser.add_argument("--neural-detector", default="", help="Optional local HF neural AI detector (e.g. desklib/ai-text-detector-academic-v1.01)")
    parser.add_argument("--neural-device", default="cpu", choices=["cpu", "cuda", "auto"], help="Device for local neural detector (default: cpu)")
    parser.add_argument("--binoculars-threshold", type=float, default=0.90, help="Min exact Binoculars human threshold in llama_cpp mode (default: 0.90)")
    parser.add_argument("--attempts", type=int, default=4, help="Max resampling attempts per chunk (default: 4; try 12-16 with llama_cpp)")
    parser.add_argument("--seed", type=int, default=401, help="Base random seed (default: 401)")
    args = parser.parse_args()

    custom_demos = human_demos.load_human_demos_file(args.human_demos) if args.human_demos else []
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

        sys.stderr.write(f"\n=== Paragraph {para_idx + 1} ({args.mode}/{args.backend}, {len(blk['text'].split())} words) ===\n")
        if args.mode == "contrastive":
            res = rewrite_paragraph_contrastive(
                blk["text"],
                anchor=make_contrastive_anchor(blk["text"], para_idx),
                ollama_base=args.ollama_url,
                base_model=args.model,
                instruct_model=args.instruct_model,
                backend=args.backend,
                base_gguf=args.base_gguf,
                instruct_gguf=args.instruct_gguf,
                n_gpu_layers=args.gpu_layers,
                length_ratio=args.length_ratio,
                alpha=args.alpha,
                beta=args.beta,
                i_floor=args.i_floor,
                temp=args.temp,
                min_inst_ppl=args.min_inst_ppl,
                attempts=args.attempts,
                base_seed=args.seed + para_idx * 101,
                use_telegraphic=not args.no_telegraphic,
                constrain_facts=not args.no_fact_guard,
                use_judge=not args.no_judge,
                cad_blind=args.blind_cad,
                dynamic_alpha=not args.no_dynamic_alpha,
                use_human_demos=not args.no_human_demos,
                custom_human_demos=custom_demos,
                num_ctx=args.num_ctx,
                surrogate_ollama_model=args.surrogate_model,
                surrogate_probes=args.surrogate_probes,
                neural_detector_model=args.neural_detector,
                neural_detector_device=args.neural_device,
                binoculars_threshold=args.binoculars_threshold,
            )
            extras = [f"Human={res['human_score_pct']}%", f"Burst={res['burstiness_score']}"]
            if res.get("binoculars") is not None:
                extras.append(f"Binoc={res['binoculars']}")
            if res.get("gltr_top1") is not None:
                extras.append(f"GLTR_T1={int(round(res['gltr_top1'] * 100))}%")
            if res.get("neural_ai_prob") is not None:
                extras.append(f"NeuralAI={int(round(res['neural_ai_prob'] * 100))}%")
            if res.get("judge_score") is not None:
                extras.append(f"Judge={res['judge_score']}/10")
            sys.stderr.write(
                f"  -> {res['source_words']} -> {res['output_words']} words ({res['length_pct']}%) | chunks={res['chunks']} "
                f"| BasePPL={res['ppl']} | InstPPL={res['inst_ppl']} | Fid={res['fidelity_pct']}% (hN={res['halluc_nums']}, hC={res['halluc_cites']}) "
                f"| {' | '.join(extras)} | {res['elapsed']}s\n"
            )
        else:
            res = rewrite_paragraph(
                blk["text"],
                demo=demos[para_idx % len(demos)],
                seed_start=make_seed_start(blk["text"], para_idx),
                ollama_base=args.ollama_url,
                model=args.model,
                backend=args.backend,
                base_gguf=args.base_gguf,
                n_gpu_layers=args.gpu_layers,
                length_ratio=args.length_ratio,
                p_skip_top1=args.p_skip,
                min_ppl=args.min_ppl,
                attempts=args.attempts,
                base_seed=args.seed + para_idx * 101,
                constrain_facts=not args.no_fact_guard,
                num_ctx=args.num_ctx,
                surrogate_ollama_model=args.surrogate_model,
                surrogate_probes=args.surrogate_probes,
                neural_detector_model=args.neural_detector,
                neural_detector_device=args.neural_device,
            )
            sys.stderr.write(
                f"  -> {res['source_words']} -> {res['output_words']} words ({res['length_pct']}%) | chunks={res['chunks']} "
                f"| PPL={res['ppl']} | Fid={res['fidelity_pct']}% (hN={res['halluc_nums']}, hC={res['halluc_cites']}) "
                f"| Burst={res['burstiness_score']} | {res['elapsed']}s\n"
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
