# Local Base-Model Humanizer Studio

A zero-dependency-by-default local GUI and CLI workbench for rewriting English academic and technical prose using **local unaligned base models** (`Qwen3-8B-Base`, `lab-qwen3-4b-base:latest`, `Mistral-7B-v0.1`, or any local Base/Instruct pair), featuring **dual decoding engines**, **human-style few-shot contrastive conditioning**, **dynamic Burstiness modulation**, **pluggable Ollama / exact full-vocab `llama.cpp` backends**, and **cross-family local surrogate reranking**.

一款默认零第三方依赖的本地可视化工作台（GUI）与命令行工具，支持 **Ollama HTTP（零依赖）** 与可选 **llama.cpp GGUF 全词表精确 Logits** 双后端，内置 **ZeroGPT 反贪心** 与 **GPTZero 人类示范 + 动态 Burstiness 对比解码** 双引擎，并集成**跨家族无条件代理裁判与本地神经 AI 检测器重排**。

![Local Base-Model Humanizer Studio — Dual-Engine GUI](docs/screenshot.png)

---

## Dual-Engine Architecture / 双解码引擎对比

| Mode / 模式 | Target / 适用场景 | Decoding Mechanism / 核心原理 | Key Metric / 关键指标 |
| :--- | :--- | :--- | :--- |
| **ZeroGPT 反贪心 (`anti_greedy`)** | ZeroGPT 等困惑度/词法分类器 | 单基础模型在内容词首以概率 `p_skip` 跳过 Top-1，结合自适应合理性门控、BPE 级 85+ AI 词根黑名单、**数值/引用/专名防幻觉白名单**、已锁定段落自举（Self-Bootstrap）与跨家族代理重排 | `PPL ≥ 3.8` & `Fid ≥ 85%` & `Burst` |
| **GPTZero 对比反 RLHF (`contrastive`)** | GPTZero Model 5.0b（同时对抗 `AI Generated` 与 `AI Paraphrased` 双检测头） | **多视角 Stage-0 去骨架电报体** + **真实人类示范对比**（仅喂给 Base，Instruct 保持通用模板）+ **动态句级 Burstiness `α(t)` 调制**：`score(t) = log P_base(t \| human_demos, notes) − α(t)·max(log P_instruct(t), i_floor)`，配合活跃对比黑名单（拦截 AI 高频词与 `whilst/thereof/herein` 古语漂移）、专名/数值/引用前缀白名单、双向自适应 `α` 重采样与**多信号跨家族本地代理重排** | `Human ≥ 65%` & `Binoc ≥ 0.90` / `Neural AI ≤ 30%` & `Fid ≥ 85%` |

---

## Key Features / 核心特性

### Core & Step-1 Decoding Upgrades / 核心与解码层升级
- **Zero External Dependencies by Default / 默认零依赖即开即用**：纯 Python 3.10+ 标准库驱动本地 Web GUI（`ThreadingHTTPServer` + SSE 实时流式推送 + 持久 Keep-Alive Ollama 客户端），无需 `pip install` 即可完整运行。
- **Human-Style Few-Shot Contrastive Conditioning / 真实人类示范对比 (`human_demos.py`)**：内置 8 段多学科高 Burstiness、零 AI 词根的真实风格范文，并支持直接在 GUI 或 CLI（`--human-demos`）导入**你自己的 2021 年前真实人类论文段落（纯文本 / `Notes:`+`Paragraph:` / JSONL）**。系统通过轻量 TF-IDF 按主题检索 1–2 段仅喂给 Base，而 Instruct 保持通用学术模板，同时动态隔离范文实体杜绝泄漏。
- **Dynamic Sentence-Aware Burstiness `α(t)` / 动态句级突发性调制**：不再全程使用恒定 `α` 制造“均匀噪声”，而是按句长节奏交替生成**短平快低 `α` 惯用句**与**长复合高 `α` 转折句**，并在语法虚词与受保护事实词上自动放松惩罚。
- **Multi-Perspective Stage-0 & Free-Start Anchors / 多视角 Stage-0 与去模板化段首**：5 套轮换的无标签要点拆解视角 + 重试乱序，配合 50% 自由起句（内置段首 AI 套话屏蔽）与显式小标题保留，消除跨段落的机械模板指纹。
- **Active Contrastive Blacklist & Entity-Aware Fact Guard / 对比黑名单与专名/数值/引用白名单**：在对比解码层硬屏蔽 AI 套话与 `whilst/thereof/herein` 等古语漂移；`Fact Guard` 同时保护原文数字、`[k]` 引用、作者名、术语缩写与专有名词 BPE 前缀。

### Step-2 Surrogate Evaluation & Exact Full-Vocab Backend / 跨家族代理裁判与精确后端
- **Cross-Family Multi-Signal Local Surrogate (`surrogates.py`)**：打破“Qwen3 自己给自己打分”的过拟合与条件 `InstPPL` 假绿灯，融合四路独立只读本地信号做 Best-of-N 择优：
  1. **结构 Burstiness + 反改写 3-gram 重叠率**（纯标准库，0 ms，常驻开启）：句长变异系数 `sent_len_cv` + 句级 PPL 变异系数 `sent_ppl_cv` + 原文 3-gram 重叠检测；
  2. **跨家族 Ollama 无条件 GLTR 探针**（零 pip 依赖）：选用任意异家族 Ollama 小模型（如 `llama3.2:3b`、`mistral:7b`、`gemma2:2b`），在**不带 notes 的独立文本**上做无截断误差的 GLTR Top-1 / Top-5 / OOB 率评估；
  3. **精确无条件 Binoculars 分数**（`llama.cpp` 后端自动开启）：单次前向计算 Hans et al. (ICML 2024) 精确全词表 Binoculars 比值 + 无条件句级 Burstiness；
  4. **可选本地神经 AI 检测器**（`torch` + `transformers`，默认跑在 CPU 不占 GPU 显存）：原生兼容 `desklib/ai-text-detector-academic-v1.01`（DeBERTa-v3-Large 学术专用）、`TrustSafeAI/RADAR-Vicuna-7B`（对抗改写检测）与 `Hello-SimpleAI/chatgpt-detector-roberta`。
- **Pluggable `llama-cpp-python` Full-Vocab Backend (`backends.py`)**：可选切换至原生 GGUF 推理，直接读取**全词表精确 logits**（彻底消除 Ollama `top_k=20` 截断悬崖），复用增量 KV Cache 提速 5–15×，可将 Best-of-N 搜索扩展到 `N = 12–32`；并能自动扫描 `./models/*.gguf` 与本机已拉取的 Ollama GGUF 缓存，无需重复下载权重。
- **Model Scale Mismatch Warning & Bidirectional Retry / 模型尺寸错配提醒与双向自适应重试**：GUI 自动识别 Base 明显大于 Instruct（如 `8B-Base` 配 `4B-Instruct`）带来的正向能力对比项并给出提醒；重试时根据失败原因是“偏 AI”还是“忠实度/流畅度不足”自动双向微调 `α`。

---

## Prerequisites / 环境准备

### 默认零依赖模式（Ollama）
1. **Python 3.10+**
2. **Ollama**（v0.4+，支持 `/api/generate` 返回 `logprobs`）运行在本地 `http://127.0.0.1:11434`
3. 准备一组**同家族、同词表**的 Base + Instruct 模型（显存允许时优先**同尺寸配对**以避免跨尺寸能力对比抵消），外加可选一个**异家族小模型**用于无条件 GLTR 代理评估：

```bash
# 同尺寸推荐（12 GB 显存）
ollama pull lab-qwen3-4b-base:latest
ollama pull qwen3:4b

# 或 8B 同尺寸（16–24 GB 显存）
ollama pull hf.co/mradermacher/Qwen3-8B-Base-GGUF:Q4_K_M
ollama pull qwen3:8b

# 可选：异家族小模型作为无条件 GLTR 跨家族代理裁判（仅占 ~2 GB）
ollama pull llama3.2:3b
```

### 可选 Step-2 加速器与本地神经检测器（`requirements-optional.txt`）

```bash
# 选项 A：启用 llama.cpp 全词表精确 Logits + 精确无条件 Binoculars（无 Top-20 截断，吞吐提速 5-15x）
pip install llama-cpp-python numpy

# 选项 B：启用本地 DeBERTa / RoBERTa 神经 AI 检测器做 Best-of-N 重排（默认跑 CPU，不占显存）
pip install torch transformers sentencepiece
```

---

## Quick Start / 快速上手

### 1. Launch / Stop GUI / 一键启动与关闭可视化界面

Windows 用户可直接双击项目根目录下的脚本：
- [`start.bat`](start.bat)（或 [`launch.bat`](launch.bat)）：一键启动并打开 `http://127.0.0.1:8765`
- [`stop.bat`](stop.bat)：一键关闭后台服务
- [`studio.bat`](studio.bat)：交互式管理控制台

或终端运行：

```bash
python app.py
# 指定其他端口：
python app.py --port 8766
```

### 2. CLI Mode / 命令行模式

**Ollama 对比模式 + 跨家族无条件 GLTR 代理 + 自定义人类范文库：**

```bash
python core.py --mode contrastive --input input.txt --output output.txt \
  --model hf.co/mradermacher/Qwen3-8B-Base-GGUF:Q4_K_M --instruct-model qwen3:8b \
  --surrogate-model llama3.2:3b --human-demos my_pre2022_papers.txt \
  --alpha 0.56 --temp 0.62 --attempts 6
```

**llama.cpp 全词表精确 Logits + 精确 Binoculars + 本地 Desklib DeBERTa 学术检测器（大样本 Best-of-16）：**

```bash
python core.py --mode contrastive --backend llama_cpp \
  --base-gguf /path/to/Qwen3-8B-Base-Q4_K_M.gguf \
  --instruct-gguf /path/to/Qwen3-8B-Instruct-Q4_K_M.gguf \
  --neural-detector desklib/ai-text-detector-academic-v1.01 --neural-device cpu \
  --binoculars-threshold 0.90 --attempts 16 \
  --input input.txt --output output.txt
```

**ZeroGPT 反贪心模式：**

```bash
python core.py --mode anti_greedy --input input.txt --output output.txt \
  --model hf.co/mradermacher/Qwen3-8B-Base-GGUF:Q4_K_M --p-skip 0.72 --min-ppl 3.8 --attempts 4
```

更多可选 CLI 参数：`--num-ctx 2048`、`--gpu-layers -1`、`--i-floor -11.5`、`--beta 0.020`、`--surrogate-probes 16`、`--no-telegraphic`、`--no-dynamic-alpha`、`--no-human-demos`、`--no-fact-guard`、`--no-judge`、`--blind-cad`。

---

## Understanding Card Badges / 结果卡片指标徽章说明

- **`Human XX%`**：综合跨家族代理人类似度（≥65% 绿色），整合了 Burstiness、反改写 3-gram 距离、无条件 GLTR / 精确 Binoculars 与神经检测器。
- **`Neural AI XX%`**（启用本地 DeBERTa/RoBERTa 时显示）：跨架构神经分类器给出的 AI 概率（≤30% 绿色）。
- **`Binoc 0.96`**（`llama.cpp` 后端自动显示）：无条件全词表 Binoculars 比值（≥阈值显示绿色）。
- **`GLTR T1 XX%`**（启用跨家族 Ollama 代理时显示）：无条件 Top-1 命中率（越低越偏人类，≤48% 绿色）。
- **`Fid XX% · J8`**：确定性数字/引用/作者/术语召回得分 + 本地只读裁判 0–10 打分。
- **`Burst 0.xx`**：句长与句级 PPL 突发性综合分。

---

## Project Structure / 项目结构

```text
local-base-humanizer/
├── app.py                     # Local HTTP/SSE server, /api/capabilities & dual-backend endpoints
├── core.py                    # Decoding engines (Anti-Greedy & Human-Demo Dynamic Contrastive) + CLI
├── backends.py                # Keep-alive OllamaClient, GGUF discovery & exact full-vocab LlamaCppBackend
├── human_demos.py             # Curated human-style demo pool, custom corpus parser & TF-IDF retriever
├── surrogates.py              # Burstiness, unconditional Ollama GLTR, Binoculars & optional DeBERTa/RoBERTa detector
├── requirements-optional.txt  # Optional Step-2 accelerators (llama-cpp-python, torch/transformers)
├── tests/
│   └── test_pipeline.py       # Zero-dependency unit & end-to-end HTTP/SSE test suite
├── static/
│   ├── index.html             # Dual-backend, multi-signal Studio GUI
│   ├── styles.css             # Dark/Light editorial UI theme
│   └── app.js                 # SSE streaming, backend/surrogate controls, lock/regen, i18n
├── start.bat / stop.bat / studio.bat / studio_ctl.ps1
├── LICENSE
└── README.md
```

---

## License

MIT License
