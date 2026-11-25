# Local Base-Model Humanizer Studio

A zero-dependency local GUI and CLI workbench for rewriting English academic and technical prose using **local Ollama unaligned base models** (`lab-qwen3-4b-base:latest` or any local base model), featuring **two specialized token-level `logprobs` decoding engines** targeting both **ZeroGPT** and **GPTZero (Model 5.0b)**.

一款零第三方依赖的本地可视化工作台（GUI）与命令行工具，内置**双本地基础模型 `logprobs` 解码引擎**，分别针对 **ZeroGPT** 与 **GPTZero（Model 5.0b 含 AI Paraphrased 检测头）** 进行了算法优化。

![Local Base-Model Humanizer Studio — Dual-Engine GUI](docs/screenshot.png)

---

## Dual-Engine Architecture / 双解码引擎对比

| Mode / 模式 | Target / 适用场景 | Decoding Mechanism / 核心原理 | Key Metric / 关键指标 |
| :--- | :--- | :--- | :--- |
| **ZeroGPT 反贪心 (`anti_greedy`)** | ZeroGPT 等困惑度/词法分类器 | 单基础模型（`raw: true`）在内容词首以概率 `p_skip` 跳过 Top-1，结合自适应合理性门控、BPE 级 85+ AI 词根黑名单、**数值/引用防幻觉白名单**与已锁定段落自举（Self-Bootstrap） | `PPL ≥ 3.8` & `Fid ≥ 85%` |
| **GPTZero 对比反 RLHF (`contrastive`)** | GPTZero Model 5.0b（同时对抗 `AI Generated` 与 `AI Paraphrased` 双检测头） | **Stage-0 去 AI 骨架电报体要点抽取** + 同词表双模型平滑对比解码：`score(t) = log P_base(t) - α · max(log P_instruct(t), i_floor)`，配合**前缀受限数值/引用白名单**、自适应合理性门控、自适应 `α` 递增重采样与**本地只读忠实度裁判重排（Judge）** | `InstPPL ≥ 350` & `BasePPL ≥ 6.5` & `0 幻觉数字/引用` |

---

## Key Features / 核心特性

- **Zero External Dependencies / 零依赖即开即用**：纯 Python 3.10+ 标准库驱动本地 Web GUI（`ThreadingHTTPServer` + SSE 实时流式推送），无需 `pip install`。
- **One-Click Engine Switch / 一键双引擎切换**：左侧栏一键切换 **ZeroGPT 反贪心** 与 **GPTZero 对比解码**，参数面板和结果指标徽章自动切换。
- **Strict Numeric & Citation Fact Guard / 数值与引用防幻觉白名单**：在逐词元解码层对数字和 `[k]` 引用实施前缀受限采样，同时禁止编造原文不存在的图表号、页码、货币符号及单数字过度复写，在高熵 `BasePPL ≥ 6.5` 区间也能保证 0 伪造数字与 0 伪造引用。
- **Stage-0 Telegraphic Decomposition & Read-Only Faithfulness Judge / 去骨架预处理与只读裁判重排**：对比模式先将原 AI 段落拆解为结果前置、无 AI 动词骨架的无损电报体要点（防止触发 GPTZero `AI Paraphrased`），多轮采样后结合确定性事实召回与本地 Instruct 只读打分裁判择优（绝不改写输出词元，零污染检测率）。
- **VRAM-Safe `num_ctx=2048` & 8B-Base Support / 12 GB 显存双模型常驻优化**：显式限制 KV 缓存上下文窗口为 `2048`（避免 Ollama 默认 `16384` 导致双模型频繁换出降速），在 12 GB 显存（如 RTX 5070）上可流畅常驻 `Qwen3-8B-Base` + `lab-qwen3-4b`，并自动兼容 Qwen / Llama / Gemma 指令模板。
- **Automatic Length Preservation / 长文本自动分块保字数**：超长段落（>175 词）自动按句边界切分为 ~110–160 词均衡子块并按 `length_ratio`（默认 `1.00×`，支持 `0.75×–1.35×` 调节）动态控制生成预算与句号收尾，同时在界面实时展示改写字数、保留百分比与事实忠实度徽章（`Fid XX%`）。
- **Paragraph-Level Lock & Single-Paragraph Regen / 分段锁定与单段一键重生成**：锁定已通过检测的段落（🔒），仅对未通过的段落一键重生成（↻）；在反贪心模式下支持把已锁定段落作为 Self-Bootstrap 上下文示范。
- **Structure Preservation & Bilingual UI / 自动保留结构与双语界面**：自动保留章节标题（`3.3.1 ...`、`# ...`）与行首引导小标题（`RQ1. ...`），支持中/英双语切换、暗/亮主题、一键复制全文、导出 `.txt` 以及顶部快捷打开 ZeroGPT / GPTZero。

---

## Prerequisites / 环境准备

1. **Python 3.10+**
2. **Ollama**（v0.4+，支持 `/api/generate` 返回 `logprobs`）运行在本地 `http://127.0.0.1:11434`
3. 本地 Ollama 模型：
   - **ZeroGPT 反贪心模式**：至少准备一个未对齐的 **Base** 模型（推荐 `hf.co/mradermacher/Qwen3-8B-Base-GGUF:Q4_K_M` 或 `lab-qwen3-4b-base:latest`）。
   - **GPTZero 对比解码模式**：准备一组**同家族、同词表**的 Base + Instruct 模型对（推荐 12 GB 显存使用 `hf.co/mradermacher/Qwen3-8B-Base-GGUF:Q4_K_M` + `lab-qwen3-4b:latest`，或 `lab-qwen3-4b-base:latest` + `lab-qwen3-4b:latest`）。

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
  --model hf.co/mradermacher/Qwen3-8B-Base-GGUF:Q4_K_M --p-skip 0.72 --min-ppl 3.8 --length-ratio 1.0 --attempts 3
```

**GPTZero 对比反 RLHF 模式（默认启用 Stage-0 电报体 + 严格数值/引用白名单 + 只读忠实度裁判）：**

```bash
python core.py --mode contrastive --input input.txt --output output.txt \
  --model hf.co/mradermacher/Qwen3-8B-Base-GGUF:Q4_K_M --instruct-model lab-qwen3-4b:latest \
  --alpha 0.56 --min-inst-ppl 350 --temp 0.62 --length-ratio 1.0 --attempts 4
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
