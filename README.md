# Local Base-Model Humanizer Studio

A zero-dependency local GUI and CLI workbench for rewriting English academic and technical prose using **local Ollama unaligned base models** (`Qwen3-8B-Base`, `lab-qwen3-4b-base:latest`, or any local base model), featuring **two specialized token-level `logprobs` decoding engines** targeting both **ZeroGPT** and **GPTZero (Model 5.0b)**.

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
3. 本地 Ollama 模型（可通过 `ollama pull` 拉取）：
   - **ZeroGPT 反贪心模式**：至少准备一个未对齐的 **Base** 模型（推荐 `hf.co/mradermacher/Qwen3-8B-Base-GGUF:Q4_K_M` 或 `lab-qwen3-4b-base:latest`）。
   - **GPTZero 对比解码模式**：准备一组**同家族、同词表**的 Base + Instruct 模型对：
     - **12 GB 显存（如 RTX 4070 / 5070）推荐**：`hf.co/mradermacher/Qwen3-8B-Base-GGUF:Q4_K_M`（~5.0 GB） + `lab-qwen3-4b:latest`（~2.5 GB），或 `lab-qwen3-4b-base:latest` + `lab-qwen3-4b:latest`。
     - **24 GB 显存推荐**：同尺寸 `Qwen3-8B-Base` + `qwen3:8b`，或 `Qwen3-14B-Base` + `qwen3:14b`。

```bash
# 示例：拉取推荐的 8B Base 与 4B Instruct 模型（同属 Qwen3 词表）
ollama pull hf.co/mradermacher/Qwen3-8B-Base-GGUF:Q4_K_M
ollama pull qwen3:4b
```

---

## Quick Start / 快速上手

### 1. Launch / Stop GUI / 一键启动与关闭可视化界面

Windows 用户可直接双击项目根目录下的脚本（后台静默运行，不残留黑框）：
- [`start.bat`](start.bat)（或 [`launch.bat`](launch.bat)）：**一键启动**本地工作台并自动打开浏览器访问 `http://127.0.0.1:8765`（若已在运行则直接打开页面，不会重复占端口）。
- [`stop.bat`](stop.bat)：**一键关闭**后台工作台服务并释放 `8765` 端口。
- [`studio.bat`](studio.bat)：**交互式管理控制台**（实时查看运行状态，支持一键启动 / 关闭 / 重启 / 打开网页）。

或在终端运行：

```bash
python app.py
# 如 8765 端口被占用，可指定其他端口：
python app.py --port 8766
```

---

## GUI Usage Guide / GUI 详细使用操作说明

工作台采用**顶部工具栏 + 三栏式布局**（左：模式与解码参数；中：原文输入；右：实时改写结果与分段管理）。

### Step 1：检查 Ollama 连接与选择模型

1. 查看右上角状态胶囊：显示绿点 **`Ollama 已连接 (N)`** 表示本地接口正常；若显示红点，请先启动 Ollama，核对左侧 **Ollama 地址**（默认 `http://127.0.0.1:11434`），再点击顶部 **`↻`** 按钮刷新模型列表。
2. 选择 **基础模型（Base 未对齐模型）**：下拉列表会自动将名称含 `base` 的未对齐底座模型置顶并标星 **`★`**，首次加载默认优先选中容量更大的 Base 模型（如 `★ hf.co/mradermacher/Qwen3-8B-Base-GGUF:Q4_K_M`）。
3. 若使用 **GPTZero 对比解码**，再在下方选择 **对比指令模型（同词表 Instruct）**（如 `lab-qwen3-4b:latest` 或 `qwen3:4b`）。**务必保证两者属于同一模型家族/词表**（例如同为 Qwen3 系列），否则对比 logprobs 无法对齐。

### Step 2：选择解码引擎模式

在左侧栏顶部 **解码引擎模式** 中一键切换：

- **`ZeroGPT 反贪心`（`Base + Blacklist`）**：单模型运行，速度最快（8B-Base 单段约 5 秒），适合应对 ZeroGPT 等基于困惑度和高频词法的检测器。
- **`GPTZero 对比解码`（`Base − α·Instruct`）**：双模型逐词元对比解码，专门针对 **GPTZero Model 5.0b**（同时绕过 `AI Generated` 与 `AI Paraphrased` 双检测头），单段约 10–18 秒。

### Step 3：配置解码参数与保护开关

一般保持默认值即可直接开跑；需要微调时可参考下表：

| 控件名称 | 适用模式 | 默认值 | 作用与调节建议 |
| :--- | :--- | :---: | :--- |
| **Top-1 跳过率 (`p_skip`)** | 反贪心 | `0.72` | 在英文内容词首跳过模型最高概率词的比例。检测仍偏高时调至 `0.74–0.78`；语句偏怪时降至 `0.62–0.68`。 |
| **目标最低困惑度 (`Min PPL`)** | 反贪心 | `3.8` | 单段 `PPL` 低于此值时自动换种子重采样，通常 `3.8–4.5` 即可。 |
| **AI 高危词元黑名单** | 反贪心 | 85 词根 | 点击展开折叠面板即可自由增删逗号分隔词根，解码时在 BPE 词元层硬屏蔽；点 `恢复默认` 一键还原。 |
| **已锁定段落自举 (`Self-Bootstrap`)** | 反贪心 | 开启 | 勾选后，重新运行时会自动将已锁定（🔒）的过检段落作为上下文风格示范，让未通过段落向已过检风格靠拢。 |
| **反 RLHF 权重 (`α`)** | 对比解码 | `0.56` | 控制 `log P_base − α·log P_instruct` 的抵消强度。推荐 `0.48–0.64`；若 GPTZero 仍偏 AI 可略升至 `0.60–0.66`，未达标时后台也会在重试轮次中自动阶梯递增 `α`。 |
| **目标指令困惑度 (`Min InstPPL`)** | 对比解码 | `350` | 达标门控阈值。实测 `InstPPL ≥ 350` 且 `BasePPL ≥ 6.5` 时可稳定穿过 GPTZero 5.0b 双检测头。 |
| **采样温度 (`Temp`)** | 对比解码 | `0.62` | 候选池 softmax 温度，推荐 `0.55–0.68`。 |
| **Stage-0 去 AI 骨架电报体预处理** | 对比解码 | **开启** | **强烈建议保持勾选**。生成前先由本地 Instruct 把原 AI 段落拆解为结果前置、去 AI 动词的无损电报体要点，彻底打碎原句式骨架，避免触发 GPTZero `AI Paraphrased`。 |
| **本地只读忠实度裁判重排 (`Judge`)** | 对比解码 | **开启** | 多轮采样后由本地 Instruct 只读打分（0–10 分）挑出事实最准、最通顺的候选，**绝不改写任何词元**，零污染检测率。 |
| **数值/引用防幻觉白名单 (`Fact Guard`)** | 双模式通用 | **开启** | **强烈建议保持勾选**。逐 token 约束数字前缀与 `[k]` 引用，禁止编造原文不存在的数字、引用编号、图表号、页码、货币符号及单数字重复刷屏。 |
| **目标字数比例 (`Length`)** | 双模式通用 | `1.00×` | 控制输出与原文字数比例（`0.75×–1.35×`）；超过 175 词的长段落会自动按句边界均衡分块改写再无缝拼接。 |
| **单段最大重试 / 候选词池 (`K`) / 种子** | 双模式通用 | `4` / `6` / `401` | 重试次数越多越容易挑出“高忠实度 + 达标 PPL”兼得的候选；点击 `🎲` 可随机切换基础种子；点 `重置` 恢复所有默认参数。 |

### Step 4：输入原文并启动流式改写

1. 将英文学术/技术文本粘贴到中间 **原文输入** 框（或点击 **`加载示例`** 体验）。
   - 支持**空行分段**以及从 Word/网页复制的**单换行长段**自动切分；
   - 自动识别 `# 标题` 与 `3.3.1 ...` 类数字章节标题并**原样保留**，不参与改写；
   - 自动保留段落首句的引导短标题（如 `RQ1. ...`、`Workload-aware cluster tuning.`）。
2. 点击 **`▶ 开始改写`**，右侧会以 **SSE 实时流式**逐段显示采样进度与当前 `PPL` / `InstPPL`。
3. 生成过程中可随时点击红色 **`■ 停止`** 按钮立即中止后台推理任务。

### Step 5：看懂指标徽章 & 人工检测闭环（最佳实践）

生成完成后，右侧 **分段落卡片** 视图中每一段都会展示关键指标徽章：

- **`InstPPL 412`**（对比模式）或 **`PPL 4.82`**（反贪心模式）：绿色代表已达目标阈值，琥珀色代表偏低建议重生成。
- **`BasePPL 7.8`**（对比模式附加信息）：反映基础模型视角下的流畅度与熵值，通常 `6.5–11.0` 是兼顾可读性与过检率的甜点区。
- **`Fid 92% · J8`**（事实忠实度徽章）：`Fid` 为确定性数字/引用/作者/实词召回与零幻觉综合得分（≥85% 显示绿色），`J` 为本地只读裁判 0–10 分打分。
- **`88/91w (97%) · 12.4s`**：改写后词数 / 原文词数（保留率）与耗时。

**推荐的“高质量 + 高过检率”人工闭环操作：**

1. **快速核对事实**：利用每个卡片内置的可编辑文本框，花 10–20 秒快速扫一眼数字、比较方向（如“高于/低于”）和专有术语，直接手动微调个别措辞。
   > ⚠️ **切忌把改写后的段落再交给 ChatGPT / Claude / DeepSeek 润色**：前沿对齐模型会瞬间把文本拉回低熵 AI 分布，导致检测率立刻反弹回 100% AI。
2. **外部检测与分段锁定**：点击顶部 **`ZeroGPT ↗`** 或 **`GPTZero ↗`** 快捷按钮打开检测页，粘贴检查：
   - **已通过的段落**：回到工作台点击该卡片右上角的 **`🔓 锁定`**（变为绿色 **`🔒 已锁定`**）。
   - **未通过或质量不满意的段落**：直接点击该卡片的 **`↻ 重生成`**（自动更换新随机种子单独重跑，不影响其他段落）；或在锁定通过段落后再次点击 **`▶ 开始改写`**，系统会自动跳过所有已锁定段落。

### Step 6：切换合并视图与导出

- 点击右侧顶部的 **`合并全文`**，自动按原有标题与段落空行拼合完整文档，并在顶部展示整篇总词数、字数保留百分比与平均 `PPL` / `InstPPL`。
- 点击 **`复制全文`** 一键复制到剪贴板，或点击 **`导出 .txt`** 下载 `humanized_output.txt`。
- 顶部支持 **`EN / 中文`** 界面双语切换与 **`☀ / ☾`** 暗色/亮色主题切换。

---

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

可选高级 CLI 参数：`--num-ctx 2048`、`--i-floor -11.5`、`--beta 0.020`、`--no-telegraphic`、`--no-fact-guard`、`--no-judge`、`--blind-cad`。

---

## Troubleshooting / 常见问题

- **生成突然变得极慢（每个词要好几秒）**：通常是 Base + Instruct 两个模型加上 KV 缓存超出了显卡剩余显存，导致 Ollama 每生成一个 token 都把模型换出再重载。虽然本工具已将 `num_ctx` 限制为 `2048`，仍建议在 12 GB 显存上使用 `8B-Base + 4B-Instruct`（总计 ~7.5 GB），并先用 `ollama ps` 确认没有残留占用显存的其他大模型。
- **打开 `http://127.0.0.1:8765` 看到旧版本参数**：Windows 下如果旧终端里仍有旧版 `app.py` 占用 8765 端口，请关闭旧终端，或用 `python app.py --port 8766` 启动新端口。
- **GPTZero 仍判定 `AI Paraphrased`**：确认已开启 **`Stage-0 去 AI 骨架电报体预处理`**；将 `α` 略调高至 `0.58–0.64`、`单段最大重试` 设为 `4–5`，然后对该段点击 `↻ 重生成`。
- **界面显示 `Ollama 未连接`**：确认已运行 Ollama 桌面端或命令行 `ollama serve`，且 `http://127.0.0.1:11434/api/tags` 可访问。

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
