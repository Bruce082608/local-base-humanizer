#!/usr/bin/env python3
"""
Human-Style Few-Shot Demo Pool & Lightweight Zero-Dependency Retriever.

Supplies authentic, high-burstiness, cliché-free technical and academic writing
examples paired with terse telegraphic notes so Contrastive Decoding can pull
Base model generations toward real human prose while Subtracting generic
RLHF/Instruct style:

    log P_base(t | human_demos, notes) - alpha * log P_instruct(t | generic)

Also supports loading custom user-supplied human corpora (JSONL or plain text)
so users can inject their own pre-2022 papers before fine-tuning a LoRA.
"""

import json
import math
import os
import random
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set

# Curated built-in pool of varied human-style academic prose (no AI clichés,
# deliberate sentence-length burstiness, mixed active/passive voice, concrete phrasing).
BUILTIN_HUMAN_DEMOS: List[Dict[str, str]] = [
    {
        "id": "systems_hadoop",
        "domain": "systems",
        "notes": (
            "Chang et al. [6] campus electricity-data stack; Hadoop collects and stores raw streams, "
            "Hive runs warehousing, Spark handles ETL, Impala serves interactive queries. "
            "Benchmarks vary node count and table size. Indirect for PV/wind forecasting."
        ),
        "paragraph": (
            "Chang et al. [6] put together a campus electricity-data platform built around four "
            "separate tools. Hadoop takes care of raw ingestion and storage, while Hive sits on top "
            "as the warehouse layer. Spark carries out the ETL batch jobs, and ad-hoc queries go "
            "through Impala instead. In their scaling runs, the authors varied both dataset volume "
            "and cluster size to check query latency and ETL throughput. The setup itself does not "
            "forecast solar or wind output, yet those node-scaling figures still give useful guidance "
            "when picking a storage backend for metered generation series."
        ),
    },
    {
        "id": "wind_forecast_cluster",
        "domain": "energy",
        "notes": (
            "Liu et al. [3] ultra-short-term wind power on 5-node Hadoop cluster (Core i5, Ubuntu). "
            "Mixes multivariate phase-space reconstruction with linear regression on lagged power + NWP. "
            "Five MapReduce jobs in parallel; HDFS storage; Hunan wind farm data; faster run time, no exact speedup ratio."
        ),
        "paragraph": (
            "On a five-node Ubuntu cluster equipped with Core i5 machines, Liu et al. [3] ran "
            "ultra-short-term wind power forecasts using Hadoop. Their model pairs multivariate "
            "phase-space reconstruction with linear regression, feeding in both historical output "
            "and selected numerical weather predictors. Five MapReduce jobs execute side by side "
            "while HDFS spreads the input files across nodes. Tests on a wind farm in Hunan showed "
            "shorter execution times than a single-machine run, though the paper never states an "
            "exact speedup factor. The target variable throughout is power generation rather than raw wind speed."
        ),
    },
    {
        "id": "medical_segmentation",
        "domain": "biomed",
        "notes": (
            "BraTS multi-modal MRI tumor segmentation; 3D U-Net baseline vs patch-size 128^3 tuned variant. "
            "Whole-tumor Dice moves from 89.4% to 91.2%; enhancing core gains 1.6 points. "
            "Inference memory stays under 9.5 GB on one workstation GPU; small lesions still miss boundary voxels."
        ),
        "paragraph": (
            "Adjusting the training patch volume to 128^3 pushes the 3D U-Net baseline up to 91.2% "
            "Dice on whole-tumor regions in the BraTS MRI cohort, compared with 89.4% before tuning. "
            "Enhancing-core scores rise by 1.6 points as well. Peak memory during inference remains "
            "below 9.5 GB on a single workstation card, so clinics do not need multi-GPU nodes to run "
            "the network. Even so, tiny satellite lesions keep losing crisp edges along the tumor boundary, "
            "where partial-volume blur still confuses the decoder."
        ),
    },
    {
        "id": "carbon_policy_econ",
        "domain": "economics",
        "notes": (
            "EU ETS Phase III plant-level panel (2013-2020); steel and cement plants compared against free-allocation caps. "
            "Permit prices rose past 25 EUR/t after 2018; clinker-to-cement ratio fell 3.4% in exposed kilns. "
            "Pass-through to domestic buyers reached 68%, but export-heavy mills absorbed most cost."
        ),
        "paragraph": (
            "Plant-level records from Phase III of the EU ETS (2013 to 2020) show a split reaction "
            "once permit prices climbed past 25 EUR/t after 2018. Cement kilns with tighter free "
            "allocations trimmed their clinker-to-cement ratio by 3.4% over the next two years. "
            "Domestic suppliers passed roughly 68% of the compliance cost on to local buyers. "
            "Export-oriented steel mills had far less room to raise prices and ended up absorbing "
            "most of the carbon charge directly into operating margins."
        ),
    },
    {
        "id": "ml_ablation_limits",
        "domain": "ml",
        "notes": (
            "Sparse attention benchmark on LongBench; sliding window 512 tokens plus 32 sink tokens. "
            "Throughput jumps 2.1x at 16k context with 0.4-point drop on multi-doc QA. "
            "Synthetic needle-in-haystack retrieval breaks past 12k when key facts fall outside local window."
        ),
        "paragraph": (
            "Keeping 32 attention sinks alongside a 512-token sliding window cuts decoding cost sharply "
            "on LongBench. At a 16k context length, throughput runs 2.1x faster than dense self-attention "
            "and multi-document QA accuracy slips by only 0.4 points. The failure mode shows up in "
            "needle-in-a-haystack probes beyond 12k tokens. Once a relevant fact drops out of the active "
            "window, the model rarely recovers it from compressed sink states alone."
        ),
    },
    {
        "id": "clinical_cohort",
        "domain": "clinical",
        "notes": (
            "Retrospective cohort of 1,420 ICU patients across three hospitals (2017-2021); early lactate clearance at 6 h. "
            "Patients clearing >= 20% had 14.8% 28-day mortality vs 29.1% in non-clearers (adjusted OR 0.46). "
            "Missing serial arterial samples excluded 186 records; single-region sample limits generalization."
        ),
        "paragraph": (
            "Across 1,420 ICU admissions drawn from three hospitals between 2017 and 2021, six-hour "
            "lactate clearance tracked closely with short-term survival. Twenty-eight-day mortality "
            "stood at 14.8% among patients who cleared at least 20% of baseline lactate, against 29.1% "
            "in those who did not (adjusted OR 0.46). Because 186 charts lacked serial arterial draws and "
            "had to be dropped, some selection bias remains possible. All three centers also sit within "
            "one regional referral network, which narrows how broadly these cutoffs can be applied."
        ),
    },
    {
        "id": "materials_battery",
        "domain": "materials",
        "notes": (
            "NMC811 cathodes coated with 4 nm Al2O3 by atomic layer deposition; cycled 400 times at 1C and 45 C. "
            "Capacity retention reaches 86.5% vs 71.2% for bare electrodes; charge-transfer impedance grows half as fast. "
            "Thicker 12 nm coats raise initial overpotential and hurt rate capability above 3C."
        ),
        "paragraph": (
            "A 4 nm Al2O3 layer grown by atomic layer deposition slows high-temperature fade in NMC811 "
            "cathodes. After 400 cycles at 1C and 45 C, coated cells hold 86.5% of their starting capacity, "
            "whereas uncoated controls fall to 71.2%. Electrochemical impedance spectra show charge-transfer "
            "resistance rising at roughly half the baseline pace. Pushing the coating to 12 nm backfires, "
            "however: initial overpotential climbs immediately and discharge capacity drops off past 3C."
        ),
    },
    {
        "id": "federated_edge",
        "domain": "networking",
        "notes": (
            "Asynchronous vs synchronous federated averaging across 24 edge gateways with variable 4G links. "
            "Sync rounds stall behind the slowest 10% stragglers; async cuts wall-clock time to 80% accuracy by 38%. "
            "Stale gradients add 1.2% test-loss variance unless bounded at max lag 4 rounds."
        ),
        "paragraph": (
            "Over patchy 4G links across 24 edge gateways, synchronous federated averaging spends much "
            "of every round waiting on the slowest 10% of devices. Switching to asynchronous updates "
            "reaches 80% validation accuracy in 38% less wall-clock time. Unbounded staleness is noisy, "
            "though, adding about 1.2% variance to test loss during late epochs. Capping gradient lag at "
            "four rounds removes most of that jitter without giving back the speed gain."
        ),
    },
]

_STOPWORDS: Set[str] = {
    "the", "and", "for", "that", "with", "from", "this", "these", "those", "were",
    "was", "are", "have", "has", "had", "been", "into", "over", "under", "about",
    "between", "across", "within", "through", "while", "where", "which", "when",
    "than", "then", "also", "more", "most", "some", "such", "only", "both", "each",
    "their", "they", "them", "its", "not", "but", "can", "could", "would", "should",
    "used", "using", "based", "study", "studies", "paper", "results", "data", "model",
}

_COMMON_NON_ENTITY_CAPS: Set[str] = {
    "the", "this", "these", "those", "when", "while", "where", "after", "before",
    "across", "over", "under", "during", "between", "within", "because", "although",
    "however", "even", "keeping", "pushing", "adjusting", "switching", "capping",
    "tests", "plant", "domestic", "export", "patients", "twenty", "eight", "five",
    "four", "three", "two", "one", "peak", "initial", "missing", "small", "tiny",
    "enhancing", "whole", "unbounded", "synchronous", "asynchronous", "electrochemical",
    "all", "some", "most", "both", "each", "once", "since", "still", "yet", "only",
    "in", "on", "at", "by", "for", "to", "of", "as", "an", "a",
}


def _tokenize_for_retrieval(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) >= 3]


def _auto_telegraphic_from_paragraph(para: str) -> str:
    """
    Build concise pseudo-notes from a raw human paragraph when the user supplies
    plain human text without explicit Notes:/Paragraph: pairs.
    """
    cleaned = re.sub(r"\s+", " ", para.strip())
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[\(\"'])", cleaned)
    fragments: List[str] = []
    for sent in sents[:5]:
        words = sent.split()
        # Keep factual anchors (numbers, capitalized terms, content words >= 4 chars)
        kept = []
        for idx, w in enumerate(words):
            bare = w.strip(".,;:()[]\"'")
            if not bare:
                continue
            if any(ch.isdigit() for ch in bare) or "[" in w or (idx > 0 and bare[0].isupper()):
                kept.append(bare)
            elif len(bare) >= 5 and bare.lower() not in _STOPWORDS:
                kept.append(bare)
            if len(kept) >= 11:
                break
        if kept:
            fragments.append(" ".join(kept))
    return "; ".join(fragments) if fragments else cleaned[:220]


def parse_custom_human_demos(raw_text: str) -> List[Dict[str, str]]:
    """
    Parse user-supplied human examples from either:
      1. JSONL lines: {"notes": "...", "paragraph": "..."} or {"text": "..."}
      2. Explicit blocks: "Notes: ...\nParagraph: ..." separated by blank lines or '---'
      3. Plain human paragraphs separated by blank lines (auto-derives terse notes)
    """
    if not raw_text or not raw_text.strip():
        return []

    stripped = raw_text.strip()
    demos: List[Dict[str, str]] = []

    # 1. Try JSONL first if lines look like JSON objects
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if lines and lines[0].startswith("{"):
        parsed_any = False
        for i, ln in enumerate(lines):
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            para = (obj.get("paragraph") or obj.get("text") or obj.get("output") or "").strip()
            if len(para.split()) < 20:
                continue
            notes = (obj.get("notes") or obj.get("input") or "").strip()
            if not notes:
                notes = _auto_telegraphic_from_paragraph(para)
            demos.append({
                "id": str(obj.get("id") or f"custom_{i}"),
                "domain": str(obj.get("domain") or "custom"),
                "notes": notes,
                "paragraph": para,
            })
            parsed_any = True
        if parsed_any:
            return demos

    # 2. Explicit Notes: / Paragraph: sections
    if re.search(r"(?im)^Notes\s*:", stripped) and re.search(r"(?im)^Paragraph\s*:", stripped):
        chunks = re.split(r"(?im)(?=^Notes\s*:)", stripped)
        for i, ch in enumerate(chunks):
            ch = ch.strip()
            if not ch:
                continue
            m = re.search(r"(?is)^Notes\s*:\s*(.+?)\n+\s*Paragraph\s*:\s*(.+)$", ch)
            if m:
                notes = re.sub(r"\s+", " ", m.group(1)).strip()
                para = re.sub(r"\s+", " ", m.group(2)).strip()
                if len(para.split()) >= 20:
                    demos.append({
                        "id": f"custom_pair_{i}",
                        "domain": "custom",
                        "notes": notes,
                        "paragraph": para,
                    })
        if demos:
            return demos

    # 3. Plain paragraphs separated by blank lines or '---'
    blocks = re.split(r"\n\s*(?:---+\s*\n|\n)+", stripped)
    for i, blk in enumerate(blocks):
        para = re.sub(r"\s+", " ", blk.strip())
        if len(para.split()) >= 25:
            demos.append({
                "id": f"custom_para_{i}",
                "domain": "custom",
                "notes": _auto_telegraphic_from_paragraph(para),
                "paragraph": para,
            })
    return demos


def load_human_demos_file(path: str) -> List[Dict[str, str]]:
    """Load custom human demos from a .jsonl or .txt file path."""
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return parse_custom_human_demos(f.read())


def get_demo_pool(custom_demos: Optional[Sequence[Dict[str, str]]] = None, custom_text: str = "") -> List[Dict[str, str]]:
    """Combine user-supplied human demos (prioritized) with the curated built-in pool."""
    pool: List[Dict[str, str]] = []
    if custom_demos:
        for d in custom_demos:
            if isinstance(d, dict) and d.get("paragraph"):
                pool.append({
                    "id": str(d.get("id") or f"custom_{len(pool)}"),
                    "domain": str(d.get("domain") or "custom"),
                    "notes": str(d.get("notes") or _auto_telegraphic_from_paragraph(d["paragraph"])),
                    "paragraph": str(d["paragraph"]).strip(),
                })
    if custom_text and custom_text.strip():
        pool.extend(parse_custom_human_demos(custom_text))
    pool.extend(BUILTIN_HUMAN_DEMOS)
    return pool


def retrieve_human_demos(
    query_text: str,
    pool: Optional[Sequence[Dict[str, str]]] = None,
    *,
    k: int = 1,
    attempt: int = 1,
    seed: int = 0,
    prefer_custom: bool = True,
) -> List[Dict[str, str]]:
    """
    Lightweight TF-IDF cosine retrieval over the human-style demo pool.

    On attempt 1, picks the highest-relevance demo(s) (with a gentle boost for
    user-provided custom demos). On subsequent retries (attempt >= 2), rotates
    across top candidates so each attempt explores genuinely distinct human rhythm.
    """
    candidates = list(pool) if pool else list(BUILTIN_HUMAN_DEMOS)
    if not candidates or k <= 0:
        return []

    k = min(k, len(candidates))
    doc_tokens = [_tokenize_for_retrieval(d["notes"] + " " + d["paragraph"]) for d in candidates]
    q_tokens = _tokenize_for_retrieval(query_text)

    n_docs = len(candidates)
    df: Counter = Counter()
    for toks in doc_tokens:
        for t in set(toks):
            df[t] += 1

    def _tfidf_vec(toks: List[str]) -> Dict[str, float]:
        tf = Counter(toks)
        vec: Dict[str, float] = {}
        for t, cnt in tf.items():
            idf = math.log((n_docs + 1.0) / (df.get(t, 0) + 1.0)) + 1.0
            vec[t] = (1.0 + math.log(cnt)) * idf
        return vec

    q_vec = _tfidf_vec(q_tokens)
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

    scored = []
    rng = random.Random(seed * 31 + attempt * 17)
    for idx, (demo, toks) in enumerate(zip(candidates, doc_tokens)):
        d_vec = _tfidf_vec(toks)
        d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
        dot = sum(q_vec[t] * d_vec.get(t, 0.0) for t in q_vec)
        sim = dot / (q_norm * d_norm)
        # Prefer custom user-supplied human prose when available
        custom_bonus = 0.18 if (prefer_custom and demo.get("domain") == "custom") else 0.0
        # Slight tie-break jitter across attempts so retries vary human phrasing naturally
        jitter = rng.uniform(0.0, 0.04 if attempt == 1 else 0.14)
        scored.append((sim + custom_bonus + jitter, idx, demo))

    scored.sort(key=lambda x: x[0], reverse=True)

    if attempt > 1 and len(scored) > k:
        # Rotate starting offset through top pool so retries see fresh human demos
        top_window = min(len(scored), max(4, k + 3))
        offset = (attempt - 1 + seed) % top_window
        rotated = scored[offset:top_window] + scored[:offset]
        chosen = [item[2] for item in rotated[:k]]
    else:
        chosen = [item[2] for item in scored[:k]]
    return chosen


def format_human_demos_block(demos: Sequence[Dict[str, str]]) -> str:
    """Format retrieved human demos into a clean few-shot prefix for Base models."""
    if not demos:
        return ""
    blocks = []
    for d in demos:
        blocks.append(
            f"Notes:\n{d['notes'].strip()}\n\n"
            f"Paragraph:\n{d['paragraph'].strip()}"
        )
    return "\n\n---\n\n".join(blocks)


def extract_demo_leak_words(demos: Sequence[Dict[str, str]], source_text: str) -> List[str]:
    """
    Dynamically extract entity/topic tokens from the active human demos that do NOT
    appear in `source_text`, so contrastive/anti-greedy decoding never leaks entities
    from either built-in or user-supplied human examples.
    """
    if not demos:
        return []
    src_low = source_text.lower()
    leak_set: Set[str] = set()

    for d in demos:
        text = f"{d.get('notes', '')} {d.get('paragraph', '')}"
        # 1. Authors / proper nouns / acronyms (capitalized tokens)
        for m in re.finditer(r"\b([A-Z][A-Za-z0-9\-]{2,})\b", text):
            tok = m.group(1)
            low = tok.lower()
            if low in _COMMON_NON_ENTITY_CAPS or low in _STOPWORDS:
                continue
            if len(low) < 4 and not tok.isupper():
                continue
            if low not in src_low:
                leak_set.add(low)

        # 2. Distinctive domain-specific terms (e.g. 'hadoop', 'clinker', 'lactate', 'cathodes', 'al2o3')
        for m in re.finditer(r"\b([a-zA-Z]{2,}[0-9]+[a-zA-Z0-9]*)\b", text):
            low = m.group(1).lower()
            if low not in src_low and len(low) >= 3:
                leak_set.add(low)

    return sorted(leak_set)
