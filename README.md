# Local Base-Model Humanizer Studio

A zero-dependency local GUI and CLI workbench for rewriting English academic and technical prose using **local Ollama unaligned base models** (`lab-qwen3-4b-base:latest` or any local base model), featuring **two specialized token-level `logprobs` decoding engines** targeting both **ZeroGPT** and **GPTZero (Model 5.0b)**.

一款零第三方依赖的本地可视化工作台（GUI）与命令行工具，内置**双本地基础模型 `logprobs` 解码引擎**，分别针对 **ZeroGPT** 与 **GPTZero（Model 5.0b 含 AI Paraphrased 检测头）** 进行了算法优化。

![Local Base-Model Humanizer Studio — Dual-Engine GUI](docs/screenshot.png)

---

## Dual-Engine Architecture / 双解码引擎对比

| Mode / 模式 | Target / 适用场景 | Decoding Mechanism / 核心原理 | Key Metric / 关键指标 |
| :--- | :--- | :--- | :--- |
| **ZeroGPT 反贪心 (`anti_greedy`)** | ZeroGPT 等困惑度/词法分类器 | 单基础模型（`raw: true`）在内容词首以概率 `p_skip` 跳过 Top-1，结合 85+ AI 高危词根硬黑名单与双风格 0%-AI 上下文示范 + 已锁定段落自举（Self-Bootstrap） | `PPL ≥ 3.8` |
| **GPTZero 对比反 RLHF (`contrastive`)** | GPTZero Model 5.0b（同时对抗 `AI Generated` 与 `AI Paraphrased` 双检测头） | 同词表双模型逐词元对比解码：`score(t) = log P_base(t) - α · log P_instruct(t)`（例如 `lab-qwen3-4b-base` 减去 `α · lab-qwen3-4b`），在不强行挖空 Rank-1 的前提下精准抵消 RLHF/指令微调偏好词，同时保持预训练语料合理性 | `InstPPL ≥ 350` & `BasePPL ≥ 6.5` |

---

## Key Features / 核心特性

- **Zero External Dependencies / 零依赖即开即用**：纯 Python 3.10+ 标准库驱动本地 Web GUI（`ThreadingHTTPServer` + SSE 实时流式推送），无需 `pip install`。
- **One-Click Engine Switch / 一键双引擎切换**：左侧栏一键切换 **ZeroGPT 反贪心** 与 **GPTZero 对比解码**，参数面板和结果指标徽章自动切换。
- **Same-Vocab Contrastive Anti-RLHF Decoding / 同词表对比反 RLHF 解码**：支持自定义 `Base Model`、`Instruct Model`、`α` 权重（默认 `0.48`）、`Min InstPPL`（默认 `350`）与温度 `Temp`，自动屏蔽重复引用循环并校验完整句与括号配对。
- **Hard AI-Buzzword Blacklist & Auto-PPL Retry / 黑名单掩码与自动重采样**：内置可编辑的 85+ AI 高危词根黑名单；任意段落未达目标 `PPL` 或 `InstPPL` 时自动更换种子重采样。
- **Paragraph-Level Lock & Single-Paragraph Regen / 分段锁定与单段一键重生成**：锁定已通过检测的段落（🔒），仅对未通过的段落一键重生成（↻）；在反贪心模式下支持把已锁定段落作为 Self-Bootstrap 上下文示范。
- **Structure Preservation & Bilingual UI / 自动保留结构与双语界面**：自动保留章节标题（`3.3.1 ...`、`# ...`）与行首引导小标题（`RQ1. ...`），支持中/英双语切换、暗/亮主题、一键复制全文、导出 `.txt` 以及顶部快捷打开 ZeroGPT / GPTZero。

---

## Prerequisites / 环境准备

1. **Python 3.10+**
2. **Ollama**（v0.4+，支持 `/api/generate` 返回 `logprobs`）运行在本地 `http://127.0.0.1:11434`
3. 本地 Ollama 模型：
   - **ZeroGPT 反贪心模式**：至少准备一个未对齐的 **Base** 模型（推荐 `lab-qwen3-4b-base:latest`）。
   - **GPTZero 对比解码模式**：准备一组**同家族、同词表**的 Base + Instruct 模型对（例如 `lab-qwen3-4b-base:latest` + `lab-qwen3-4b:latest` 或 `qwen3:4b`）。

---

## Quick Start / 快速上手

### 1. Launch GUI / 启动可视化界面

Windows 用户可直接双击 [`launch.bat`](launch.bat)，或在终端运行：

```bash
python app.py
```

默认自动打开浏览器访问 `http://127.0.0.1:8765`。

### 2. CLI Mode / 命令行模式

**ZeroGPT 反贪心模式：**

```bash
python core.py --mode anti_greedy --input input.txt --output output.txt \
  --model lab-qwen3-4b-base:latest --p-skip 0.72 --min-ppl 3.8 --attempts 3
```

**GPTZero 对比反 RLHF 模式：**

```bash
python core.py --mode contrastive --input input.txt --output output.txt \
  --model lab-qwen3-4b-base:latest --instruct-model lab-qwen3-4b:latest \
  --alpha 0.48 --min-inst-ppl 350 --temp 0.50 --attempts 3
```

---

## Project Structure / 项目结构

```text
local-base-humanizer/
├── app.py          # Local HTTP/SSE server & dual-mode API endpoints
├── core.py         # Anti-greedy + Contrastive Anti-RLHF decoding engines & CLI
├── launch.bat      # Windows one-click launcher
├── static/
│   ├── index.html  # Dual-engine Studio GUI markup
│   ├── styles.css  # Dark/Light editorial UI theme
│   └── app.js      # SSE streaming, mode switcher, paragraph lock/regen, i18n
├── docs/
│   └── screenshot.png
├── LICENSE
└── README.md
```

---

## License

MIT License
