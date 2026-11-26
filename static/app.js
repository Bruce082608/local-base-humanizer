const I18N = {
  zh: {
    subtitle: "本地基础模型双引擎降 AI 率工作台（Ollama / llama.cpp 全词表 + 跨家族代理重排）",
    ollamaChecking: "检测 Ollama...",
    ollamaOnline: "Ollama 已连接",
    ollamaOffline: "Ollama 未连接",
    llamaCppReady: "llama.cpp 就绪",
    llamaCppMissing: "未安装 llama-cpp-python",
    openZeroGpt: "ZeroGPT ↗",
    openGptZero: "GPTZero ↗",
    engineMode: "解码引擎模式",
    modeAntiGreedy: "ZeroGPT 反贪心",
    modeContrastive: "GPTZero 对比解码",
    modeDescAntiGreedy: "单基础模型逐词元跳过 Rank-1 + 黑名单掩码，适合 ZeroGPT。",
    modeDescContrastive: "人类示范引导 + 动态 Burstiness 对比解码，抵消 RLHF 偏移并结合跨家族代理重排。",
    modelConfig: "推理后端与模型",
    backendLabel: "推理后端",
    ollamaUrl: "Ollama 地址",
    baseModel: "基础模型（Base 未对齐模型）",
    instructModel: "对比指令模型（同词表 Instruct）",
    instructHint: "需与基础模型同家族/同词表（建议同尺寸如 4B+4B 或 8B+8B）。",
    baseGgufLabel: "Base GGUF 模型路径",
    instructGgufLabel: "Instruct GGUF 模型路径（同词表）",
    gpuLayersLabel: "GPU Offload 层数 (-1 = 全部)",
    llamaCppHint: "自动扫描本地 ./models 与 Ollama GGUF；全词表精确计算无 Top-20 截断，可将重试提升至 12–16。",
    decodingParams: "解码参数",
    reset: "重置",
    pSkipLabel: "Top-1 跳过率 (p_skip)",
    pSkipHint: "越高越偏离大模型高频路径，建议 0.65–0.78",
    minPplLabel: "目标最低困惑度 (Min PPL)",
    minPplHint: "低于此值时自动更换随机种子重采样",
    alphaLabel: "反 RLHF 基准权重 (α)",
    alphaHint: "log P_base − α(t)·log P_instruct，推荐 0.46–0.62",
    minInstPplLabel: "目标指令困惑度 (Min InstPPL)",
    minInstPplHint: "启用下方跨家族/神经代理后将优先以代理分数与 Burstiness 为准",
    tempLabel: "采样温度 (Temp)",
    telegraphicLabel: "Stage-0 多视角去骨架电报体",
    dynamicAlphaLabel: "动态 Burstiness 句级 α(t) 调制",
    humanDemosLabel: "真实人类示范对比 (仅喂给 Base)",
    judgeLabel: "本地只读忠实度裁判重排 (Judge)",
    cadBlindLabel: "Blind-Instruct CAD (高忠实/轻改写)",
    factGuardLabel: "数值/引用/专名防幻觉白名单 (Fact Guard)",
    lengthRatioLabel: "目标字数比例 (Length)",
    lengthRatioHint: "长段落自动按句分块改写，保持与原文接近字数",
    attemptsLabel: "单段最大重试 (N)",
    poolLabel: "候选词池 (K)",
    seedLabel: "基础随机种子 (Seed)",
    bootstrapLabel: "启用已锁定段落自举示范 (Self-Bootstrap)",
    surrogateTitle: "跨家族代理裁判 (Step 2)",
    surrogateOllamaLabel: "无条件 GLTR 代理模型 (Ollama)",
    surrogateOllamaHint: "推荐选择与 Base 异家族的小模型（如 llama3.2 / mistral / gemma），独立无条件评估候选段。",
    neuralDetectorLabel: "本地神经 AI 检测器 (DeBERTa/RoBERTa)",
    neuralDeviceLabel: "检测器设备",
    surrogateProbesLabel: "GLTR 探针数",
    binocThreshLabel: "Binoculars 达标阈值",
    binocThreshHint: "llama.cpp 模式下自动计算无条件精确 Binoculars（越高越像人类，通常 ≥ 0.90）。",
    customDemosTitle: "人类风格范文库",
    customDemosHint: "支持纯人类段落（自动提取要点）、Notes:/Paragraph: 对或 JSONL；动态隔离范文实体，绝不泄漏。",
    blacklistTitle: "AI / 古语漂移词元黑名单",
    blacklistHint: "逗号分隔词根，在解码层硬屏蔽",
    restoreDefault: "恢复默认",
    sourceTitle: "原文输入",
    loadSample: "加载示例",
    clear: "清空",
    runAll: "开始改写",
    stop: "停止",
    outputTitle: "改写结果",
    cardsView: "分段落卡片",
    mergedView: "合并全文",
    copyAll: "复制全文",
    exportTxt: "导出 .txt",
    emptyTitle: "准备就绪",
    emptyDesc: "选择解码引擎与推理后端（Ollama 或 llama.cpp 全词表），可选配跨家族代理或本地 DeBERTa 检测器，粘贴英文文本后点击「开始改写」。",
    lock: "锁定",
    locked: "已锁定",
    regen: "重生成",
    copy: "复制",
    copied: "已复制到剪贴板",
    exported: "已导出为 humanized_output.txt",
    emptyInputWarn: "请先输入需要改写的英文文本",
    missingGgufWarn: "请先指定本地 Base / Instruct GGUF 文件路径",
    doneToast: "全部段落改写完成",
    cancelledToast: "已停止生成",
    paraPrefix: "段落",
    runningAttempt: "采样中",
    wordsUnit: "词",
    parasUnit: "段",
    avgPpl: "平均 PPL",
    avgInstPpl: "平均 InstPPL",
    avgHuman: "平均 Human",
    sizeMismatchMsg: "提示：当前 Base (~{b}B) 明显大于 Instruct (~{i}B)，差值会混入正向能力对比项。显存允许时优先配对同尺寸模型（如 4B+4B 或 8B+8B）。",
    llamaMissingBanner: "当前 Python 环境未安装 llama-cpp-python / numpy。请运行：pip install llama-cpp-python numpy",
    neuralMissingBanner: "当前 Python 环境未安装 torch / transformers。如需启用 DeBERTa/RoBERTa 检测器请运行：pip install torch transformers sentencepiece",
    crossFamilyTag: "✦ 异家族",
    sameFamilyTag: "同家族",
    surrogateOffOpt: "关闭 (仅 Burstiness / Binoculars)",
    manualGgufOpt: "— 手动输入本地 .gguf 路径 —",
    builtinUnit: "内置",
    customUnit: "自定义",
  },
  en: {
    subtitle: "Dual-Engine Local Humanizer Studio (Ollama / llama.cpp Full-Vocab + Cross-Family Surrogates)",
    ollamaChecking: "Checking Ollama...",
    ollamaOnline: "Ollama Connected",
    ollamaOffline: "Ollama Offline",
    llamaCppReady: "llama.cpp Ready",
    llamaCppMissing: "llama-cpp-python missing",
    openZeroGpt: "ZeroGPT ↗",
    openGptZero: "GPTZero ↗",
    engineMode: "Decoding Engine",
    modeAntiGreedy: "ZeroGPT Anti-Greedy",
    modeContrastive: "GPTZero Contrastive",
    modeDescAntiGreedy: "Single base model rank-1 skipping + AI-buzzword blacklist for ZeroGPT.",
    modeDescContrastive: "Human-demo guided + dynamic Burstiness contrastive decoding with cross-family surrogate reranking.",
    modelConfig: "Backend & Models",
    backendLabel: "Inference Backend",
    ollamaUrl: "Ollama URL",
    baseModel: "Base Model (Unaligned)",
    instructModel: "Contrastive Instruct Model (Same Vocab)",
    instructHint: "Must share vocabulary with Base (same-size pairs like 4B+4B or 8B+8B recommended).",
    baseGgufLabel: "Base GGUF Path",
    instructGgufLabel: "Instruct GGUF Path (Same Vocab)",
    gpuLayersLabel: "GPU Offload Layers (-1 = all)",
    llamaCppHint: "Auto-scans ./models & local Ollama GGUFs; exact full-vocab logits (no Top-20 cliff) — try N=12-16 retries.",
    decodingParams: "Decoding Parameters",
    reset: "Reset",
    pSkipLabel: "Rank-1 Skip Prob (p_skip)",
    pSkipHint: "Higher skips more top-1 LLM tokens (recommended 0.65–0.78)",
    minPplLabel: "Min Target Perplexity (PPL)",
    minPplHint: "Resamples automatically with a fresh seed when PPL is lower",
    alphaLabel: "Base Anti-RLHF Weight (α)",
    alphaHint: "log P_base − α(t)·log P_instruct (recommended 0.46–0.62)",
    minInstPplLabel: "Min Target InstPPL",
    minInstPplHint: "When cross-family/neural surrogates are active, surrogate score + Burstiness take priority",
    tempLabel: "Temperature (Temp)",
    telegraphicLabel: "Stage-0 Multi-Perspective Telegraphic Notes",
    dynamicAlphaLabel: "Dynamic Sentence Burstiness α(t)",
    humanDemosLabel: "Real Human-Style Demos (Base-only Contrast)",
    judgeLabel: "Read-Only Local Faithfulness Judge",
    cadBlindLabel: "Blind-Instruct CAD (Higher Fidelity)",
    factGuardLabel: "Strict Numeric/Citation/Entity Fact Guard",
    lengthRatioLabel: "Target Length Ratio",
    lengthRatioHint: "Long paragraphs auto-split on sentences to preserve word count",
    attemptsLabel: "Max Retries (N)",
    poolLabel: "Candidate Pool (K)",
    seedLabel: "Base Seed",
    bootstrapLabel: "Self-Bootstrap from locked paragraphs",
    surrogateTitle: "Cross-Family Surrogates (Step 2)",
    surrogateOllamaLabel: "Unconditional GLTR Surrogate (Ollama)",
    surrogateOllamaHint: "Pick a different-family small Ollama model (e.g. llama3.2 / mistral / gemma) for unconditional reranking.",
    neuralDetectorLabel: "Local Neural AI Detector (DeBERTa/RoBERTa)",
    neuralDeviceLabel: "Detector Device",
    surrogateProbesLabel: "GLTR Probes",
    binocThreshLabel: "Binoculars Threshold",
    binocThreshHint: "Exact unconditional Binoculars computed in llama.cpp mode (higher = more human, typically ≥ 0.90).",
    customDemosTitle: "Human Style Demo Corpus",
    customDemosHint: "Paste pre-2022 human paragraphs (blank-line separated), Notes:/Paragraph: pairs, or JSONL. Entities auto-isolated.",
    blacklistTitle: "AI / Archaic Drift Token Blacklist",
    blacklistHint: "Comma-separated roots masked during decoding",
    restoreDefault: "Defaults",
    sourceTitle: "Source Text",
    loadSample: "Load Sample",
    clear: "Clear",
    runAll: "Rewrite All",
    stop: "Stop",
    outputTitle: "Rewritten Output",
    cardsView: "Paragraph Cards",
    mergedView: "Merged Text",
    copyAll: "Copy All",
    exportTxt: "Export .txt",
    emptyTitle: "Ready to Rewrite",
    emptyDesc: "Choose engine & backend (Ollama or exact llama.cpp), optionally enable cross-family/neural surrogates, paste text and click Rewrite All.",
    lock: "Lock",
    locked: "Locked",
    regen: "Regen",
    copy: "Copy",
    copied: "Copied to clipboard",
    exported: "Exported humanized_output.txt",
    emptyInputWarn: "Please paste some English text first",
    missingGgufWarn: "Please set local Base / Instruct GGUF file paths first",
    doneToast: "All paragraphs rewritten",
    cancelledToast: "Generation cancelled",
    paraPrefix: "Paragraph",
    runningAttempt: "Sampling",
    wordsUnit: "words",
    parasUnit: "paras",
    avgPpl: "Avg PPL",
    avgInstPpl: "Avg InstPPL",
    avgHuman: "Avg Human",
    sizeMismatchMsg: "Note: Base (~{b}B) is much larger than Instruct (~{i}B), adding a positive scale-up contrast term. Pair equal sizes (e.g. 4B+4B or 8B+8B) when VRAM allows.",
    llamaMissingBanner: "llama-cpp-python / numpy not installed in this Python env. Run: pip install llama-cpp-python numpy",
    neuralMissingBanner: "torch / transformers not installed. To enable local DeBERTa/RoBERTa detection, run: pip install torch transformers sentencepiece",
    crossFamilyTag: "✦ Cross-Family",
    sameFamilyTag: "Same-Family",
    surrogateOffOpt: "Off (Burstiness / Binoculars only)",
    manualGgufOpt: "— Enter custom .gguf path below —",
    builtinUnit: "built-in",
    customUnit: "custom",
  },
};

const SAMPLE_TEXT = `Edge and federated forecasting. Wind-farm operators could evaluate federated training on edge servers with different network conditions and different local weather and generation patterns. The wind-farm case in [6] and the load studies in [7] and [8] motivate a comparison of local-only, synchronous federated and asynchronous federated models under identical forecast horizons.

Workload-aware cluster tuning. The effects of worker count in [2] and the sensitivity to caching and repartitioning in [9] motivate automated tuning for different forecast workloads. Future studies could compare default and tuned configurations while holding data, model, hardware budget and accuracy requirements constant.

Joint wind and solar forecasting. A joint system could align wind generation, PV output and weather data by site and event time, then share a feature pipeline while retaining separate forecasts where site characteristics require them. The wind and PV studies [4], [6] provide starting points for a comparison with separate models.`;

const state = {
  lang: "zh",
  theme: "dark",
  view: "cards",
  mode: "contrastive",
  backend: "ollama",
  defaults: null,
  capabilities: null,
  ollamaModels: [],
  ollamaOnline: false,
  modelsLoaded: false,
  antiGreedyBlacklist: "",
  contrastiveBlacklist: "",
  blocks: [],
  activeJobId: null,
  abortController: null,
};

const $ = (id) => document.getElementById(id);

function t(key) {
  return (I18N[state.lang] && I18N[state.lang][key]) || key;
}

function parseList(raw) {
  return (raw || "")
    .split(/[\n,]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function getBlacklist() {
  return parseList($("blacklistText").value);
}

function syncCurrentBlacklistToState() {
  if (state.mode === "contrastive") {
    state.contrastiveBlacklist = $("blacklistText").value;
  } else {
    state.antiGreedyBlacklist = $("blacklistText").value;
  }
}

function updateBlacklistCount() {
  $("blacklistCount").textContent = String(getBlacklist().length);
}

function updateHumanDemosBadge() {
  const builtinN = (state.capabilities && state.capabilities.builtin_human_demos_count) || 8;
  const raw = ($("customHumanDemosText").value || "").trim();
  let customN = 0;
  if (raw) {
    if (raw.startsWith("{")) {
      customN = raw.split(/\n+/).filter((l) => l.trim().startsWith("{")).length;
    } else {
      customN = raw.split(/\n\s*(?:---+\s*\n|\n)+/).filter((b) => b.trim().split(/\s+/).length >= 15).length;
    }
  }
  $("humanDemosBadge").textContent =
    customN > 0 ? `${builtinN} ${t("builtinUnit")} + ${customN} ${t("customUnit")}` : `${builtinN} ${t("builtinUnit")}`;
}

function getSelectedNeuralDetector() {
  const sel = $("neuralDetectorSelect").value;
  if (sel === "__custom__") return ($("neuralDetectorInput").value || "").trim();
  return sel;
}

function updateSurrogateBadge() {
  const tags = ["Burst"];
  if (state.backend === "llama_cpp" && state.mode === "contrastive") tags.push("Binoc");
  if ($("surrogateModelSelect").value) tags.push("GLTR");
  if (getSelectedNeuralDetector()) tags.push("Neural");
  $("surrogateActiveBadge").textContent = tags.join("+");

  const neuralSel = getSelectedNeuralDetector();
  const capWarn = $("neuralCapHint");
  if (neuralSel && state.capabilities && !state.capabilities.neural_detector_available) {
    capWarn.textContent = t("neuralMissingBanner");
    capWarn.classList.remove("hidden");
  } else {
    capWarn.classList.add("hidden");
  }
}

function updateSizeMismatchWarning() {
  const warnEl = $("sizeMismatchWarn");
  if (state.mode !== "contrastive" || state.backend !== "ollama") {
    warnEl.classList.add("hidden");
    return;
  }
  const baseName = $("modelSelect").value;
  const instName = $("instructModelSelect").value;
  const bm = state.ollamaModels.find((m) => m.name === baseName);
  const im = state.ollamaModels.find((m) => m.name === instName);
  if (bm && im && bm.param_b && im.param_b && bm.param_b >= im.param_b * 1.4) {
    warnEl.textContent = t("sizeMismatchMsg").replace("{b}", bm.param_b).replace("{i}", im.param_b);
    warnEl.classList.remove("hidden");
  } else {
    warnEl.classList.add("hidden");
  }
}

function updateTopStatusPill() {
  const pill = $("ollamaStatus");
  const textEl = pill.querySelector(".status-text");
  if (state.backend === "llama_cpp") {
    const ok = Boolean(state.capabilities && state.capabilities.llama_cpp_available);
    pill.className = `status-pill ${ok ? "status-online" : "status-offline"}`;
    const ver = state.capabilities && state.capabilities.llama_cpp_version ? ` (${state.capabilities.llama_cpp_version})` : "";
    textEl.textContent = ok ? `${t("llamaCppReady")}${ver}` : t("llamaCppMissing");
    return;
  }
  if (state.ollamaOnline) {
    pill.className = "status-pill status-online";
    textEl.textContent = `${t("ollamaOnline")} (${state.ollamaModels.length})`;
  } else {
    pill.className = "status-pill status-offline";
    textEl.textContent = t("ollamaOffline");
  }
}

function applyBackendUI() {
  const isLlamaCpp = state.backend === "llama_cpp";
  $("ollamaConfigFields").classList.toggle("hidden", isLlamaCpp);
  $("llamaCppConfigFields").classList.toggle("hidden", !isLlamaCpp);
  $("binocularsField").classList.toggle("hidden", !(isLlamaCpp && state.mode === "contrastive"));

  const lcWarn = $("llamaCppStatusHint");
  if (isLlamaCpp && state.capabilities && !state.capabilities.llama_cpp_available) {
    lcWarn.textContent = t("llamaMissingBanner");
    lcWarn.classList.remove("hidden");
  } else {
    lcWarn.classList.add("hidden");
  }
  updateTopStatusPill();
  updateSizeMismatchWarning();
  updateSurrogateBadge();
}

function applyModeUI() {
  const isContrastive = state.mode === "contrastive";
  $("btnModeAntiGreedy").classList.toggle("active", !isContrastive);
  $("btnModeContrastive").classList.toggle("active", isContrastive);
  $("instructModelField").classList.toggle("hidden", !isContrastive);
  $("instructGgufField").classList.toggle("hidden", !isContrastive);
  $("antiGreedyControls").classList.toggle("hidden", isContrastive);
  $("contrastiveControls").classList.toggle("hidden", !isContrastive);
  $("bootstrapWrap").classList.toggle("hidden", isContrastive);
  $("humanDemosPanel").classList.toggle("hidden", !isContrastive);
  $("modeDesc").textContent = isContrastive ? t("modeDescContrastive") : t("modeDescAntiGreedy");

  $("blacklistText").value = isContrastive ? state.contrastiveBlacklist : state.antiGreedyBlacklist;
  updateBlacklistCount();
  applyBackendUI();
  updateOutputStats();
  renderCards();
}

function applyI18n() {
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (I18N[state.lang][key]) el.textContent = I18N[state.lang][key];
  });
  $("btnLang").textContent = state.lang === "zh" ? "EN" : "中文";
  rebuildSurrogateOllamaOptions();
  updateHumanDemosBadge();
  applyModeUI();
  updateInputStats();
  updateOutputStats();
}

function showToast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => el.classList.add("hidden"), 2600);
}

function updateInputStats() {
  const txt = $("sourceInput").value.trim();
  const words = txt ? txt.split(/\s+/).length : 0;
  const paras = txt ? txt.split(/\n\s*\n/).filter((x) => x.trim()).length : 0;
  $("inputStats").textContent = `${words} ${t("wordsUnit")} · ${paras} ${t("parasUnit")}`;
}

function getMergedText() {
  const lines = [];
  for (let i = 0; i < state.blocks.length; i++) {
    const b = state.blocks[i];
    if (b.kind === "heading") {
      if (lines.length) lines.push("");
      lines.push(b.text);
    } else {
      const prevIsHeading = i > 0 && state.blocks[i - 1].kind === "heading";
      if (lines.length && !prevIsHeading) lines.push("");
      lines.push((b.text || "").trim());
    }
  }
  return lines.join("\n").trim();
}

function updateOutputStats() {
  const merged = getMergedText();
  const words = merged ? merged.split(/\s+/).length : 0;
  const srcTxt = $("sourceInput").value.trim();
  const srcWords = srcTxt ? srcTxt.split(/\s+/).length : 0;
  const wordPart =
    srcWords > 0 && words > 0
      ? `${words}/${srcWords} ${t("wordsUnit")} (${Math.round((100 * words) / srcWords)}%)`
      : `${words} ${t("wordsUnit")}`;

  const paras = state.blocks.filter((b) => b.kind === "para");
  const humanVals = paras.filter((b) => typeof b.human_score_pct === "number" && b.human_score_pct > 0).map((b) => b.human_score_pct);
  const humanPart = humanVals.length
    ? ` · ${t("avgHuman")} ${Math.round(humanVals.reduce((a, c) => a + c, 0) / humanVals.length)}%`
    : "";

  const hasInst = paras.some((b) => typeof b.inst_ppl === "number" && b.inst_ppl > 0);
  if (hasInst) {
    const vals = paras.filter((b) => typeof b.inst_ppl === "number" && b.inst_ppl > 0).map((b) => b.inst_ppl);
    const avg = vals.length ? (vals.reduce((a, c) => a + c, 0) / vals.length).toFixed(0) : "—";
    $("outputStats").textContent = `${wordPart}${humanPart} · ${t("avgInstPpl")} ${avg}`;
  } else {
    const vals = paras.filter((b) => typeof b.ppl === "number" && b.ppl > 0).map((b) => b.ppl);
    const avg = vals.length ? (vals.reduce((a, c) => a + c, 0) / vals.length).toFixed(2) : "—";
    $("outputStats").textContent = `${wordPart}${humanPart} · ${t("avgPpl")} ${avg}`;
  }
  $("mergedOutput").value = merged;
}

async function loadDefaults() {
  const resp = await fetch("/api/defaults");
  const d = await resp.json();
  state.defaults = d;
  state.mode = d.mode || "contrastive";
  state.backend = d.backend || "ollama";
  state.antiGreedyBlacklist = (d.banned_substrs || []).join(", ");
  state.contrastiveBlacklist = (d.contrastive_banned_substrs || []).join(", ");

  $("backendSelect").value = state.backend;
  $("ollamaUrl").value = d.ollama_url;
  $("pSkip").value = d.p_skip;
  $("pSkipVal").textContent = Number(d.p_skip).toFixed(2);
  $("minPpl").value = d.min_ppl;
  $("minPplVal").textContent = Number(d.min_ppl).toFixed(1);
  $("alphaSlider").value = d.alpha;
  $("alphaVal").textContent = Number(d.alpha).toFixed(2);
  $("minInstPpl").value = d.min_inst_ppl;
  $("minInstPplVal").textContent = String(d.min_inst_ppl);
  $("contrastiveTemp").value = d.contrastive_temp;
  $("contrastiveTempVal").textContent = Number(d.contrastive_temp).toFixed(2);
  const lr = d.length_ratio ?? 1.0;
  $("lengthRatio").value = lr;
  $("lengthRatioVal").textContent = `${Number(lr).toFixed(2)}×`;
  $("attempts").value = d.attempts;
  $("poolSize").value = d.pool_size;
  $("seedInput").value = d.seed;
  $("useTelegraphic").checked = d.use_telegraphic ?? true;
  $("dynamicAlpha").checked = d.dynamic_alpha ?? true;
  $("useHumanDemos").checked = d.use_human_demos ?? true;
  $("constrainFacts").checked = d.constrain_facts ?? true;
  $("useJudge").checked = d.use_judge ?? true;
  $("cadBlind").checked = d.cad_blind ?? false;
  $("surrogateProbes").value = d.surrogate_probes ?? 16;
  $("binocularsThreshold").value = d.binoculars_threshold ?? 0.90;
  $("binocularsThresholdVal").textContent = Number(d.binoculars_threshold ?? 0.90).toFixed(2);
  $("gpuLayersInput").value = d.n_gpu_layers ?? -1;
}

async function refreshCapabilities() {
  try {
    const resp = await fetch("/api/capabilities");
    const caps = await resp.json();
    state.capabilities = caps;
    populateGgufSelectors(caps.local_ggufs || []);
    updateHumanDemosBadge();
    applyBackendUI();
  } catch (_) {}
}

function populateGgufSelectors(ggufs) {
  const bSel = $("baseGgufSelect");
  const iSel = $("instructGgufSelect");
  const prevB = $("baseGgufInput").value;
  const prevI = $("instructGgufInput").value;

  bSel.innerHTML = "";
  iSel.innerHTML = "";

  const emptyB = document.createElement("option");
  emptyB.value = "";
  emptyB.textContent = t("manualGgufOpt");
  bSel.appendChild(emptyB);

  const emptyI = document.createElement("option");
  emptyI.value = "";
  emptyI.textContent = t("manualGgufOpt");
  iSel.appendChild(emptyI);

  ggufs.forEach((g) => {
    const gb = g.size ? ` (${(g.size / 1e9).toFixed(1)} GB)` : "";
    const optB = document.createElement("option");
    optB.value = g.path;
    optB.textContent = (g.is_base ? "★ " : "") + g.label + gb;
    bSel.appendChild(optB);

    const optI = document.createElement("option");
    optI.value = g.path;
    optI.textContent = g.label + gb;
    iSel.appendChild(optI);
  });

  if (!prevB) {
    const prefB = ggufs.find((g) => g.is_base) || ggufs[0];
    if (prefB) {
      bSel.value = prefB.path;
      $("baseGgufInput").value = prefB.path;
    }
  }
  if (!prevI) {
    const prefI = ggufs.find((g) => !g.is_base) || ggufs[0];
    if (prefI) {
      iSel.value = prefI.path;
      $("instructGgufInput").value = prefI.path;
    }
  }
}

function rebuildSurrogateOllamaOptions() {
  const sel = $("surrogateModelSelect");
  const prev = sel.value;
  sel.innerHTML = "";

  const offOpt = document.createElement("option");
  offOpt.value = "";
  offOpt.textContent = t("surrogateOffOpt");
  sel.appendChild(offOpt);

  if (!state.ollamaModels.length) return;
  const baseName = $("modelSelect").value;
  const baseObj = state.ollamaModels.find((m) => m.name === baseName);
  const baseFam = baseObj ? baseObj.family : "other";

  const sorted = [...state.ollamaModels].sort((a, b) => {
    const aDiff = a.family !== baseFam && a.family !== "other" ? 0 : 1;
    const bDiff = b.family !== baseFam && b.family !== "other" ? 0 : 1;
    if (aDiff !== bDiff) return aDiff - bDiff;
    return (a.size || 0) - (b.size || 0);
  });

  sorted.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.name;
    const isCross = m.family !== baseFam && m.family !== "other";
    const tag = isCross ? `${t("crossFamilyTag")} · ${m.family}` : `${t("sameFamilyTag")} · ${m.family}`;
    const gb = m.size ? ` (${(m.size / 1e9).toFixed(1)} GB)` : "";
    opt.textContent = `${m.name}${gb} [${tag}]`;
    sel.appendChild(opt);
  });

  if (prev && sorted.some((m) => m.name === prev)) {
    sel.value = prev;
  }
  updateSurrogateBadge();
}

async function refreshModels() {
  const pill = $("ollamaStatus");
  const textEl = pill.querySelector(".status-text");
  if (state.backend === "ollama") {
    pill.className = "status-pill status-checking";
    textEl.textContent = t("ollamaChecking");
  }

  const url = encodeURIComponent($("ollamaUrl").value.trim());
  try {
    const resp = await fetch(`/api/models?ollama_url=${url}`);
    const data = await resp.json();
    const baseSel = $("modelSelect");
    const instSel = $("instructModelSelect");
    const prevBase = baseSel.value;
    const prevInst = instSel.value;

    if (data.ok && data.models.length > 0) {
      state.ollamaOnline = true;
      state.ollamaModels = data.models;
      baseSel.innerHTML = "";
      instSel.innerHTML = "";

      data.models.forEach((m) => {
        const gb = m.size ? ` (${(m.size / 1e9).toFixed(1)} GB)` : "";
        const optB = document.createElement("option");
        optB.value = m.name;
        optB.textContent = (m.is_base ? "★ " : "") + m.name + gb;
        baseSel.appendChild(optB);

        const optI = document.createElement("option");
        optI.value = m.name;
        optI.textContent = m.name + gb;
        instSel.appendChild(optI);
      });

      const prefBase =
        (state.modelsLoaded && data.models.find((m) => m.name === prevBase)) ||
        data.models.find((m) => m.is_base) ||
        data.models.find((m) => m.name === (state.defaults && state.defaults.model)) ||
        data.models[0];
      if (prefBase) baseSel.value = prefBase.name;

      const prefInst =
        (state.modelsLoaded && data.models.find((m) => m.name === prevInst && !m.is_base)) ||
        data.models.find((m) => m.name === (state.defaults && state.defaults.instruct_model)) ||
        data.models.find((m) => m.name === "qwen3:4b") ||
        data.models.find((m) => !m.is_base) ||
        data.models[0];
      if (prefInst) instSel.value = prefInst.name;
      state.modelsLoaded = true;
    } else {
      state.ollamaOnline = false;
      state.ollamaModels = [];
    }
  } catch (_) {
    state.ollamaOnline = false;
    state.ollamaModels = [];
  }
  rebuildSurrogateOllamaOptions();
  updateTopStatusPill();
  updateSizeMismatchWarning();
}

function getRequestParams() {
  syncCurrentBlacklistToState();
  const gpuLayersRaw = parseInt($("gpuLayersInput").value, 10);
  const seedRaw = parseInt($("seedInput").value, 10);
  return {
    mode: state.mode,
    backend: state.backend,
    ollama_url: $("ollamaUrl").value.trim(),
    model: $("modelSelect").value,
    instruct_model: $("instructModelSelect").value,
    base_gguf: $("baseGgufInput").value.trim(),
    instruct_gguf: $("instructGgufInput").value.trim(),
    n_gpu_layers: Number.isNaN(gpuLayersRaw) ? -1 : gpuLayersRaw,
    length_ratio: parseFloat($("lengthRatio").value) || 1.0,
    p_skip: parseFloat($("pSkip").value),
    min_ppl: parseFloat($("minPpl").value),
    alpha: parseFloat($("alphaSlider").value),
    min_inst_ppl: parseFloat($("minInstPpl").value),
    contrastive_temp: parseFloat($("contrastiveTemp").value),
    attempts: parseInt($("attempts").value, 10) || 4,
    pool_size: parseInt($("poolSize").value, 10) || 6,
    seed: Number.isNaN(seedRaw) ? 401 : seedRaw,
    use_telegraphic: $("useTelegraphic").checked,
    dynamic_alpha: $("dynamicAlpha").checked,
    use_human_demos: $("useHumanDemos").checked,
    constrain_facts: $("constrainFacts").checked,
    use_judge: $("useJudge").checked,
    cad_blind: $("cadBlind").checked,
    custom_human_demos_text: $("customHumanDemosText").value || "",
    surrogate_ollama_model: $("surrogateModelSelect").value || "",
    surrogate_probes: parseInt($("surrogateProbes").value, 10) || 16,
    neural_detector_model: getSelectedNeuralDetector(),
    neural_detector_device: $("neuralDeviceSelect").value || "cpu",
    binoculars_threshold: parseFloat($("binocularsThreshold").value) || 0.90,
    banned_substrs: parseList(state.antiGreedyBlacklist),
    contrastive_banned_substrs: parseList(state.contrastiveBlacklist),
    use_bootstrap: $("useBootstrap").checked,
  };
}

function renderCards() {
  const box = $("cardsContainer");
  if (!state.blocks.length) {
    box.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">✦</div>
        <p class="empty-title">${t("emptyTitle")}</p>
        <p class="empty-desc">${t("emptyDesc")}</p>
      </div>`;
    return;
  }

  box.innerHTML = "";
  let paraCounter = 0;
  const minPpl = parseFloat($("minPpl").value);
  const minInstPpl = parseFloat($("minInstPpl").value);
  const binoThresh = parseFloat($("binocularsThreshold").value) || 0.90;

  state.blocks.forEach((b, idx) => {
    if (b.kind === "heading") {
      const h = document.createElement("div");
      h.className = "heading-card";
      h.textContent = b.text;
      box.appendChild(h);
      return;
    }

    paraCounter += 1;
    const isContrastiveBlock = b.mode === "contrastive" || (typeof b.inst_ppl === "number" && b.inst_ppl > 0);
    const card = document.createElement("div");
    card.className = `para-card${b.running ? " is-running" : ""}${b.locked ? " is-locked" : ""}`;
    card.dataset.index = String(idx);

    let pillClass = "ppl-warn";
    let pillLabel = "PPL —";
    let extraMeta = "";
    const extraBadges = [];

    const srcWords = b.source ? b.source.trim().split(/\s+/).filter(Boolean).length : 0;
    const outWords = b.text ? b.text.trim().split(/\s+/).filter(Boolean).length : 0;
    const wordMeta =
      !b.running && srcWords > 0 && outWords > 0
        ? ` · ${outWords}/${srcWords}w (${Math.round((100 * outWords) / srcWords)}%)`
        : "";

    if (b.running) {
      pillClass = "ppl-busy";
      pillLabel = `${t("runningAttempt")} ${b.statusText || "..."}`;
    } else if (isContrastiveBlock && typeof b.inst_ppl === "number") {
      pillClass = b.inst_ppl >= minInstPpl ? "ppl-good" : "ppl-warn";
      pillLabel = `InstPPL ${b.inst_ppl.toFixed(0)}`;
      const burstPart = typeof b.burstiness_score === "number" ? ` · Burst ${b.burstiness_score.toFixed(2)}` : "";
      extraMeta = `· BasePPL ${typeof b.ppl === "number" ? b.ppl.toFixed(1) : "—"}${burstPart}${wordMeta}${b.elapsed ? ` · ${b.elapsed}s` : ""}`;
    } else if (typeof b.ppl === "number") {
      pillClass = b.ppl >= minPpl ? "ppl-good" : "ppl-warn";
      pillLabel = `PPL ${b.ppl.toFixed(2)}`;
      const burstPart = typeof b.burstiness_score === "number" ? ` · Burst ${b.burstiness_score.toFixed(2)}` : "";
      extraMeta = `${burstPart}${wordMeta}${b.elapsed ? ` · ${b.elapsed}s` : ""}${b.skip_ratio ? ` · skip ${b.skip_ratio}%` : ""}`;
    }

    if (!b.running) {
      if (typeof b.human_score_pct === "number" && b.human_score_pct > 0) {
        const hc = b.human_score_pct >= 65 ? "ppl-good" : b.human_score_pct >= 45 ? "ppl-warn" : "ppl-bad";
        extraBadges.push(`<span class="ppl-pill ${hc}">Human ${b.human_score_pct}%</span>`);
      }
      if (typeof b.neural_ai_prob === "number") {
        const aiPct = Math.round(b.neural_ai_prob * 100);
        const nc = aiPct <= 30 ? "ppl-good" : aiPct <= 55 ? "ppl-warn" : "ppl-bad";
        extraBadges.push(`<span class="ppl-pill ${nc}">Neural AI ${aiPct}%</span>`);
      }
      if (typeof b.binoculars === "number") {
        const bc = b.binoculars >= binoThresh ? "ppl-good" : "ppl-warn";
        extraBadges.push(`<span class="ppl-pill ${bc}">Binoc ${b.binoculars.toFixed(2)}</span>`);
      }
      if (typeof b.gltr_top1 === "number") {
        const t1Pct = Math.round(b.gltr_top1 * 100);
        const gc = t1Pct <= 48 ? "ppl-good" : "ppl-warn";
        extraBadges.push(`<span class="ppl-pill ${gc}">GLTR T1 ${t1Pct}%</span>`);
      }
      if (typeof b.fidelity_pct === "number") {
        const fc = b.fidelity_pct >= 85 ? "ppl-good" : "ppl-warn";
        extraBadges.push(
          `<span class="ppl-pill ${fc}">Fid ${b.fidelity_pct}%${typeof b.judge_score === "number" ? ` · J${b.judge_score}` : ""}</span>`
        );
      }
    }

    card.innerHTML = `
      <div class="card-top">
        <div class="card-meta">
          <span class="card-label">${t("paraPrefix")} #${paraCounter}</span>
          <span class="ppl-pill ${pillClass}">${pillLabel}</span>
          ${extraBadges.join("")}
          <span class="stat-mini">${extraMeta}</span>
        </div>
        <div class="card-actions">
          <button class="btn btn-xs ${b.locked ? "btn-locked" : "btn-ghost"}" data-action="lock">
            ${b.locked ? "🔒 " + t("locked") : "🔓 " + t("lock")}
          </button>
          <button class="btn btn-xs btn-outline" data-action="regen" ${b.running || state.activeJobId ? "disabled" : ""}>
            ↻ ${t("regen")}
          </button>
          <button class="btn btn-xs btn-ghost" data-action="copy">${t("copy")}</button>
        </div>
      </div>
      <textarea class="card-textarea" spellcheck="false"></textarea>
    `;

    const ta = card.querySelector(".card-textarea");
    ta.value = b.text || "";
    ta.addEventListener("input", () => {
      state.blocks[idx].text = ta.value;
      updateOutputStats();
    });

    card.querySelector('[data-action="lock"]').addEventListener("click", () => {
      state.blocks[idx].locked = !state.blocks[idx].locked;
      renderCards();
    });

    card.querySelector('[data-action="regen"]').addEventListener("click", () => {
      regenerateSingle(idx, paraCounter - 1);
    });

    card.querySelector('[data-action="copy"]').addEventListener("click", async () => {
      await navigator.clipboard.writeText(state.blocks[idx].text || "");
      showToast(t("copied"));
    });

    box.appendChild(card);
  });
}

function updateCardLive(idx) {
  const card = $("cardsContainer").querySelector(`.para-card[data-index="${idx}"]`);
  if (!card) return;
  const b = state.blocks[idx];
  const ta = card.querySelector(".card-textarea");
  if (ta) ta.value = b.text || "";
  const pill = card.querySelector(".ppl-pill");
  if (pill) {
    pill.className = "ppl-pill ppl-busy";
    pill.textContent = `${t("runningAttempt")} ${b.statusText || "..."}`;
  }
  updateOutputStats();
}

async function consumeSSE(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sepIdx;
    while ((sepIdx = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, sepIdx);
      buffer = buffer.slice(sepIdx + 2);

      let eventName = "message";
      let dataStr = "";
      for (const line of rawEvent.split("\n")) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        else if (line.startsWith("data:")) dataStr += line.slice(5).trim();
      }
      if (dataStr) {
        try {
          onEvent(eventName, JSON.parse(dataStr));
        } catch (err) {
          console.error("Bad SSE JSON:", err, dataStr);
        }
        if (eventName === "done" || eventName === "cancelled" || eventName === "error") {
          try { await reader.cancel(); } catch (_) {}
          return;
        }
      }
    }
  }
}

function setRunningUI(isRunning, progressText = "") {
  $("btnRun").disabled = isRunning;
  $("btnStop").classList.toggle("hidden", !isRunning);
  $("runProgress").textContent = progressText;
}

function validateBeforeRun(params) {
  if (params.backend === "llama_cpp") {
    if (!params.base_gguf || (params.mode === "contrastive" && !params.instruct_gguf)) {
      showToast(t("missingGgufWarn"));
      return false;
    }
  }
  return true;
}

async function runAll() {
  const rawText = $("sourceInput").value.trim();
  if (!rawText) {
    showToast(t("emptyInputWarn"));
    return;
  }
  const params = getRequestParams();
  if (!validateBeforeRun(params)) return;

  const lockedBlocks = {};
  state.blocks.forEach((b, idx) => {
    if (b.kind === "para" && b.locked && b.text) {
      lockedBlocks[String(idx)] = {
        mode: b.mode,
        backend: b.backend,
        text: b.text,
        ppl: b.ppl,
        inst_ppl: b.inst_ppl,
        skip_ratio: b.skip_ratio,
        fidelity_pct: b.fidelity_pct,
        judge_score: b.judge_score,
        human_score_pct: b.human_score_pct,
        burstiness_score: b.burstiness_score,
        binoculars: b.binoculars,
        neural_ai_prob: b.neural_ai_prob,
        gltr_top1: b.gltr_top1,
        overlap_3gram: b.overlap_3gram,
      };
    }
  });

  const jobId = `job-${Date.now()}`;
  state.activeJobId = jobId;
  state.abortController = new AbortController();
  setRunningUI(true, "Starting...");

  try {
    const resp = await fetch("/api/rewrite/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id: jobId,
        text: rawText,
        locked_blocks: lockedBlocks,
        ...params,
      }),
      signal: state.abortController.signal,
    });

    await consumeSSE(resp, (evName, data) => {
      if (evName === "init") {
        state.blocks = data.blocks.map((blk, i) => ({
          index: i,
          kind: blk.kind,
          mode: lockedBlocks[String(i)]?.mode || state.mode,
          backend: lockedBlocks[String(i)]?.backend || state.backend,
          source: blk.text,
          text: blk.kind === "heading" ? blk.text : (lockedBlocks[String(i)]?.text || ""),
          ppl: lockedBlocks[String(i)]?.ppl ?? null,
          inst_ppl: lockedBlocks[String(i)]?.inst_ppl ?? null,
          skip_ratio: lockedBlocks[String(i)]?.skip_ratio ?? null,
          fidelity_pct: lockedBlocks[String(i)]?.fidelity_pct ?? null,
          judge_score: lockedBlocks[String(i)]?.judge_score ?? null,
          human_score_pct: lockedBlocks[String(i)]?.human_score_pct ?? null,
          burstiness_score: lockedBlocks[String(i)]?.burstiness_score ?? null,
          binoculars: lockedBlocks[String(i)]?.binoculars ?? null,
          neural_ai_prob: lockedBlocks[String(i)]?.neural_ai_prob ?? null,
          gltr_top1: lockedBlocks[String(i)]?.gltr_top1 ?? null,
          locked: Boolean(lockedBlocks[String(i)]),
          running: false,
        }));
        renderCards();
        updateOutputStats();
      } else if (evName === "block_start") {
        const b = state.blocks[data.index];
        if (b) {
          b.running = true;
          b.mode = data.mode || state.mode;
          b.backend = data.backend || state.backend;
          b.text = data.seed_start || "";
          b.statusText = "1/...";
          renderCards();
          $("runProgress").textContent = `Paragraph #${data.para_idx + 1} (${b.mode}/${b.backend})...`;
        }
      } else if (evName === "progress") {
        const b = state.blocks[data.index];
        if (b) {
          if (data.text) b.text = data.text;
          if (data.event === "attempt_start") {
            b.statusText = `${data.attempt}/${data.attempts}`;
          } else if (data.event === "token") {
            b.statusText = data.inst_ppl
              ? `${data.attempt}/${data.attempts} · InstPPL ${data.inst_ppl}`
              : `${data.attempt}/${data.attempts} · PPL ${data.ppl}`;
          }
          updateCardLive(data.index);
        }
      } else if (evName === "block_done") {
        state.blocks[data.index] = {
          ...state.blocks[data.index],
          ...data,
          running: false,
          statusText: "",
        };
        renderCards();
        updateOutputStats();
      } else if (evName === "done") {
        showToast(t("doneToast"));
      } else if (evName === "cancelled") {
        showToast(t("cancelledToast"));
      } else if (evName === "error") {
        showToast(`Error: ${data.message}`);
      }
    });
  } catch (err) {
    if (err.name !== "AbortError") {
      showToast(`Error: ${err.message}`);
    }
  } finally {
    state.activeJobId = null;
    state.abortController = null;
    state.blocks.forEach((b) => (b.running = false));
    setRunningUI(false, "");
    renderCards();
    updateOutputStats();
  }
}

async function regenerateSingle(idx, paraIdx) {
  const b = state.blocks[idx];
  if (!b || b.kind !== "para" || state.activeJobId) return;

  const params = getRequestParams();
  if (!validateBeforeRun(params)) return;
  params.seed = Math.floor(1000 + Math.random() * 90000);

  const jobId = `one-${Date.now()}`;
  state.activeJobId = jobId;
  state.abortController = new AbortController();
  b.running = true;
  b.locked = false;
  b.mode = state.mode;
  b.backend = state.backend;
  b.statusText = "...";
  setRunningUI(true, `Regenerating #${paraIdx + 1}...`);
  renderCards();

  let bootstrapDemo = null;
  if (state.mode === "anti_greedy" && $("useBootstrap").checked) {
    const lockedOther = state.blocks.find(
      (other, oIdx) => oIdx !== idx && other.kind === "para" && other.locked && other.text
    );
    if (lockedOther) {
      bootstrapDemo = { source: lockedOther.source, text: lockedOther.text };
    }
  }

  try {
    const resp = await fetch("/api/rewrite/one", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id: jobId,
        index: idx,
        para_idx: paraIdx,
        source: b.source,
        bootstrap_demo: bootstrapDemo,
        ...params,
      }),
      signal: state.abortController.signal,
    });

    await consumeSSE(resp, (evName, data) => {
      if (evName === "progress") {
        if (data.text) b.text = data.text;
        b.statusText = data.inst_ppl
          ? `${data.attempt}/${data.attempts} · InstPPL ${data.inst_ppl}`
          : data.ppl
          ? `${data.attempt}/${data.attempts} · PPL ${data.ppl}`
          : `${data.attempt}/${data.attempts}`;
        updateCardLive(idx);
      } else if (evName === "block_done") {
        state.blocks[idx] = { ...b, ...data, running: false, statusText: "" };
        renderCards();
        updateOutputStats();
      } else if (evName === "error") {
        showToast(`Error: ${data.message}`);
      }
    });
  } catch (err) {
    if (err.name !== "AbortError") showToast(`Error: ${err.message}`);
  } finally {
    state.activeJobId = null;
    state.abortController = null;
    b.running = false;
    setRunningUI(false, "");
    renderCards();
    updateOutputStats();
  }
}

async function stopActiveJob() {
  if (!state.activeJobId) return;
  const jid = state.activeJobId;
  try {
    await fetch("/api/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jid }),
    });
  } catch (_) {}
  if (state.abortController) state.abortController.abort();
}

function bindEvents() {
  $("btnModeAntiGreedy").addEventListener("click", () => {
    syncCurrentBlacklistToState();
    state.mode = "anti_greedy";
    applyModeUI();
  });
  $("btnModeContrastive").addEventListener("click", () => {
    syncCurrentBlacklistToState();
    state.mode = "contrastive";
    applyModeUI();
  });

  $("backendSelect").addEventListener("change", () => {
    state.backend = $("backendSelect").value;
    applyBackendUI();
  });

  $("baseGgufSelect").addEventListener("change", () => {
    if ($("baseGgufSelect").value) $("baseGgufInput").value = $("baseGgufSelect").value;
  });
  $("instructGgufSelect").addEventListener("change", () => {
    if ($("instructGgufSelect").value) $("instructGgufInput").value = $("instructGgufSelect").value;
  });

  $("modelSelect").addEventListener("change", () => {
    rebuildSurrogateOllamaOptions();
    updateSizeMismatchWarning();
  });
  $("instructModelSelect").addEventListener("change", updateSizeMismatchWarning);

  $("surrogateModelSelect").addEventListener("change", updateSurrogateBadge);
  $("neuralDetectorSelect").addEventListener("change", () => {
    const isCustom = $("neuralDetectorSelect").value === "__custom__";
    $("neuralDetectorInput").classList.toggle("hidden", !isCustom);
    updateSurrogateBadge();
  });
  $("neuralDetectorInput").addEventListener("input", updateSurrogateBadge);
  $("customHumanDemosText").addEventListener("input", updateHumanDemosBadge);

  $("sourceInput").addEventListener("input", updateInputStats);
  $("blacklistText").addEventListener("input", () => {
    syncCurrentBlacklistToState();
    updateBlacklistCount();
  });
  $("pSkip").addEventListener("input", () => ($("pSkipVal").textContent = Number($("pSkip").value).toFixed(2)));
  $("minPpl").addEventListener("input", () => ($("minPplVal").textContent = Number($("minPpl").value).toFixed(1)));
  $("alphaSlider").addEventListener("input", () => ($("alphaVal").textContent = Number($("alphaSlider").value).toFixed(2)));
  $("minInstPpl").addEventListener("input", () => ($("minInstPplVal").textContent = $("minInstPpl").value));
  $("contrastiveTemp").addEventListener("input", () => ($("contrastiveTempVal").textContent = Number($("contrastiveTemp").value).toFixed(2)));
  $("lengthRatio").addEventListener("input", () => ($("lengthRatioVal").textContent = `${Number($("lengthRatio").value).toFixed(2)}×`));
  $("binocularsThreshold").addEventListener("input", () => {
    $("binocularsThresholdVal").textContent = Number($("binocularsThreshold").value).toFixed(2);
    renderCards();
  });

  $("btnRandSeed").addEventListener("click", () => {
    $("seedInput").value = String(Math.floor(100 + Math.random() * 9900));
  });

  $("btnResetParams").addEventListener("click", () => {
    if (!state.defaults) return;
    const d = state.defaults;
    $("pSkip").value = d.p_skip;
    $("pSkipVal").textContent = Number(d.p_skip).toFixed(2);
    $("minPpl").value = d.min_ppl;
    $("minPplVal").textContent = Number(d.min_ppl).toFixed(1);
    $("alphaSlider").value = d.alpha;
    $("alphaVal").textContent = Number(d.alpha).toFixed(2);
    $("minInstPpl").value = d.min_inst_ppl;
    $("minInstPplVal").textContent = String(d.min_inst_ppl);
    $("contrastiveTemp").value = d.contrastive_temp;
    $("contrastiveTempVal").textContent = Number(d.contrastive_temp).toFixed(2);
    const lr = d.length_ratio ?? 1.0;
    $("lengthRatio").value = lr;
    $("lengthRatioVal").textContent = `${Number(lr).toFixed(2)}×`;
    $("attempts").value = d.attempts;
    $("poolSize").value = d.pool_size;
    $("seedInput").value = d.seed;
    $("useTelegraphic").checked = d.use_telegraphic ?? true;
    $("dynamicAlpha").checked = d.dynamic_alpha ?? true;
    $("useHumanDemos").checked = d.use_human_demos ?? true;
    $("constrainFacts").checked = d.constrain_facts ?? true;
    $("useJudge").checked = d.use_judge ?? true;
    $("cadBlind").checked = d.cad_blind ?? false;
    $("surrogateProbes").value = d.surrogate_probes ?? 16;
    $("binocularsThreshold").value = d.binoculars_threshold ?? 0.90;
    $("binocularsThresholdVal").textContent = Number(d.binoculars_threshold ?? 0.90).toFixed(2);
  });

  $("btnResetBlacklist").addEventListener("click", () => {
    if (!state.defaults) return;
    if (state.mode === "contrastive") {
      state.contrastiveBlacklist = (state.defaults.contrastive_banned_substrs || []).join(", ");
      $("blacklistText").value = state.contrastiveBlacklist;
    } else {
      state.antiGreedyBlacklist = (state.defaults.banned_substrs || []).join(", ");
      $("blacklistText").value = state.antiGreedyBlacklist;
    }
    updateBlacklistCount();
  });

  $("btnRefreshModels").addEventListener("click", () => {
    refreshModels();
    refreshCapabilities();
  });
  $("ollamaUrl").addEventListener("change", refreshModels);

  $("btnLoadSample").addEventListener("click", () => {
    $("sourceInput").value = SAMPLE_TEXT;
    updateInputStats();
  });

  $("btnClearInput").addEventListener("click", () => {
    $("sourceInput").value = "";
    state.blocks = [];
    updateInputStats();
    updateOutputStats();
    renderCards();
  });

  $("btnRun").addEventListener("click", runAll);
  $("btnStop").addEventListener("click", stopActiveJob);

  $("tabCards").addEventListener("click", () => {
    state.view = "cards";
    $("tabCards").classList.add("active");
    $("tabMerged").classList.remove("active");
    $("cardsContainer").classList.remove("hidden");
    $("mergedOutput").classList.add("hidden");
  });

  $("tabMerged").addEventListener("click", () => {
    state.view = "merged";
    $("tabMerged").classList.add("active");
    $("tabCards").classList.remove("active");
    $("cardsContainer").classList.add("hidden");
    $("mergedOutput").classList.remove("hidden");
    updateOutputStats();
  });

  $("btnCopyAll").addEventListener("click", async () => {
    const txt = getMergedText();
    if (!txt) return;
    await navigator.clipboard.writeText(txt);
    showToast(t("copied"));
  });

  $("btnExport").addEventListener("click", () => {
    const txt = getMergedText();
    if (!txt) return;
    const blob = new Blob([txt + "\n"], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "humanized_output.txt";
    a.click();
    URL.revokeObjectURL(a.href);
    showToast(t("exported"));
  });

  $("btnTheme").addEventListener("click", () => {
    state.theme = state.theme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", state.theme);
    $("btnTheme").textContent = state.theme === "dark" ? "☀" : "☾";
  });

  $("btnLang").addEventListener("click", () => {
    state.lang = state.lang === "zh" ? "en" : "zh";
    applyI18n();
  });
}

(async function init() {
  bindEvents();
  await loadDefaults();
  await Promise.all([refreshModels(), refreshCapabilities()]);
  applyI18n();
})();
