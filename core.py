#!/usr/bin/env python3
"""
Core engine for Local Base-Model Anti-Greedy Humanizer (Ollama).

Uses an unaligned local base model (e.g. lab-qwen3-4b-base:latest) with:
  1. Token-level logprobs anti-greedy decoding (skipping Rank-1 at content word starts)
  2. Hard AI-buzzword logit blacklist filtering
  3. Alternating verified 0%-AI in-context style demonstrations (plus optional self-bootstrapping)
  4. Automatic per-paragraph perplexity (PPL) verification & resampling
"""

import argparse
import json
import math
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_MODEL = "lab-qwen3-4b-base:latest"

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

HEADING_RE = re.compile(r"^(?:#{1,6}\s+|\d+(?:\.\d+)*\s+[A-Z]).{0,100}$")
WORD_START_RE = re.compile(r"^ [A-Za-z][A-Za-z'\-]*$")
STOP_TOKENS = ("<|endoftext|>", "<|im_end|>", "<|eot_id|>", "")


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
            # Sort base models first for convenience
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
            buf.append(s)
    if buf:
        blocks.append({"kind": "para", "text": " ".join(buf)})
    return blocks


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
) -> Tuple[Optional[str], List[Tuple[str, float]]]:
    url = f"{normalize_ollama_base(ollama_base)}/api/generate"
    payload = {
        "model": model,
        "prompt": raw_prompt,
        "raw": True,
        "stream": False,
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
    """Create a concrete, non-participial seed opening tailored to the paragraph."""
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

    m = re.search(r"\b([A-Z][A-Za-z\-]+(?:\s+et\s+al\.|\s+and\s+[A-Z][A-Za-z\-]+))\b", stripped)
    if m:
        author = m.group(1)
        author_starters = [
            f"In the study that was built by {author},",
            f"For {author}, the authors",
            f"In the setup studied by {author},",
            f"For the work of {author},",
        ]
        return author_starters[idx % len(author_starters)]

    words = stripped.split()
    return " ".join(words[:5])


def trim_to_last_sentence(text: str) -> str:
    text = re.sub(r" {2,}", " ", text)
    matches = list(re.finditer(r"(?<!\bal)\.(?:\s|$)", text))
    if matches:
        return text[: matches[-1].start() + 1].strip()
    return text.strip()


def rewrite_paragraph(
    source_para: str,
    demo: str,
    seed_start: str,
    *,
    ollama_base: str = DEFAULT_OLLAMA_BASE,
    model: str = DEFAULT_MODEL,
    max_tokens: Optional[int] = None,
    p_skip_top1: float = 0.72,
    max_p1_for_skip: float = 0.88,
    pool_size: int = 6,
    min_alt_prob: float = 0.01,
    min_ppl: float = 3.8,
    attempts: int = 3,
    base_seed: int = 401,
    banned_substrs: Optional[List[str]] = None,
    on_progress: Optional[Callable[[Dict], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict:
    """Rewrite a single paragraph with anti-greedy decoding and automatic PPL retries."""
    if banned_substrs is None:
        banned_substrs = DEFAULT_BANNED_SUBSTRS
    banned_clean = [b.lower().strip() for b in banned_substrs if b.strip()]

    word_count = max(1, len(source_para.split()))
    if max_tokens is None:
        max_tokens = max(95, min(220, int(word_count * 1.45)))

    best_result = {
        "text": source_para.strip(),
        "ppl": 0.0,
        "tokens": 0,
        "skipped": 0,
        "word_starts": 0,
        "skip_ratio": 0.0,
        "attempt": 0,
        "seed": base_seed,
        "elapsed": 0.0,
        "cancelled": False,
    }

    for attempt in range(1, max(1, attempts) + 1):
        if cancel_event and cancel_event.is_set():
            best_result["cancelled"] = True
            return best_result

        seed_val = base_seed + (attempt - 1) * 37
        ctx = f"{demo.strip()}\n\nNotes:\n{source_para.strip()}\n\nParagraph:\n{seed_start}"
        rng = random.Random(seed_val)
        gen = ""
        nlls: List[float] = []
        skipped = 0
        word_starts = 0
        t0 = time.time()

        if on_progress:
            on_progress({
                "event": "attempt_start",
                "attempt": attempt,
                "attempts": attempts,
                "seed": seed_val,
                "text": seed_start,
            })

        for step in range(max_tokens):
            if cancel_event and cancel_event.is_set():
                best_result["cancelled"] = True
                return best_result

            tok0, top = next_token_logprobs(ollama_base, model, ctx + gen, top_k=20)
            if not top or tok0 in STOP_TOKENS:
                break

            clean = [
                (t, lp)
                for (t, lp) in top
                if is_clean_ascii(t) and not is_banned(t, banned_clean) and t not in STOP_TOKENS
            ]
            if not clean:
                clean = [(t, lp) for (t, lp) in top if is_clean_ascii(t) and t not in STOP_TOKENS]
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
                        if WORD_START_RE.match(t) and math.exp(lp) >= min_alt_prob
                    ]
                    if alts:
                        weights = [math.exp(lp) ** 0.45 for _, lp in alts]
                        chosen_tok, chosen_lp = rng.choices(alts, weights=weights, k=1)[0]
                        skipped += 1

            gen += chosen_tok
            nlls.append(-chosen_lp)

            if on_progress and (step % 4 == 0 or chosen_tok.endswith(".")):
                running_ppl = math.exp(sum(nlls) / max(len(nlls), 1))
                on_progress({
                    "event": "token",
                    "attempt": attempt,
                    "attempts": attempts,
                    "text": seed_start + gen,
                    "ppl": round(running_ppl, 2),
                    "tokens": len(nlls),
                })

            if "\n\n" in gen and len(gen) > 150:
                gen = gen.split("\n\n")[0]
                break

        elapsed = round(time.time() - t0, 2)
        ppl = math.exp(sum(nlls) / max(len(nlls), 1)) if nlls else 0.0
        full = trim_to_last_sentence(seed_start + gen)
        skip_ratio = round(100.0 * skipped / max(1, word_starts), 1)

        cand = {
            "text": full,
            "ppl": round(ppl, 2),
            "tokens": len(nlls),
            "skipped": skipped,
            "word_starts": word_starts,
            "skip_ratio": skip_ratio,
            "attempt": attempt,
            "seed": seed_val,
            "elapsed": elapsed,
            "cancelled": False,
        }

        if on_progress:
            on_progress({"event": "attempt_done", **cand})

        if cand["ppl"] > best_result["ppl"] and len(cand["text"]) >= 60:
            best_result = cand

        min_len = max(100, int(len(source_para) * 0.42))
        if ppl >= min_ppl and len(full) >= min_len:
            break

    return best_result


def build_bootstrap_demo(source_para: str, rewritten_para: str) -> str:
    """Create a self-bootstrapping in-context demo from an already-verified paragraph."""
    return f"Notes:\n{source_para.strip()}\n\nParagraph:\n{rewritten_para.strip()}"


def main():
    parser = argparse.ArgumentParser(description="Local Base-Model Anti-Greedy Humanizer CLI")
    parser.add_argument("--input", "-i", help="Input text file (reads stdin if omitted)")
    parser.add_argument("--output", "-o", help="Output text file (prints stdout if omitted)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama base model (default: {DEFAULT_MODEL})")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_BASE, help="Ollama base URL")
    parser.add_argument("--p-skip", type=float, default=0.72, help="Rank-1 skip probability (default: 0.72)")
    parser.add_argument("--min-ppl", type=float, default=3.8, help="Minimum target PPL per paragraph (default: 3.8)")
    parser.add_argument("--attempts", type=int, default=3, help="Max resampling attempts per paragraph (default: 3)")
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

        sys.stderr.write(f"\n=== Paragraph {para_idx + 1} ===\n")
        res = rewrite_paragraph(
            blk["text"],
            demo=demos[para_idx % 2],
            seed_start=make_seed_start(blk["text"], para_idx),
            ollama_base=args.ollama_url,
            model=args.model,
            p_skip_top1=args.p_skip,
            min_ppl=args.min_ppl,
            attempts=args.attempts,
            base_seed=args.seed + para_idx * 101,
            on_progress=lambda ev: (
                sys.stderr.write(
                    f"  attempt {ev['attempt']} | seed={ev['seed']} | {ev['elapsed']}s | PPL={ev['ppl']}\n"
                )
                if ev.get("event") == "attempt_done"
                else None
            ),
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
