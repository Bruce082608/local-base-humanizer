const I18N = {
  zh: {
    subtitle: "Ollama 本地基础模型双引擎降 AI 率工作台（ZeroGPT & GPTZero）",
    ollamaChecking: "检测 Ollama...",
    ollamaOnline: "Ollama 已连接",
    ollamaOffline: "Ollama 未连接",
    openZeroGpt: "ZeroGPT ↗",
    openGptZero: "GPTZero ↗",
    engineMode: "解码引擎模式",
    modeAntiGreedy: "ZeroGPT 反贪心",
    modeContrastive: "GPTZero 对比解码",
    modeDescAntiGreedy: "单基础模型逐词元跳过 Rank-1 + 黑名单掩码，适合 ZeroGPT。",
    modeDescContrastive: "同词表双模型对比（Base − α·Instruct），抵消 RLHF 偏好词，同时绕过 GPTZero 生成与改写检测。",
    modelConfig: "模型与连接",
    ollamaUrl: "Ollama 地址",
    baseModel: "基础模型（Base 未对齐模型）",
    instructModel: "对比指令模型（同词表 Instruct）",
    instructHint: "需与基础模型同家族/同词表（如 Qwen3-4B Base + Instruct）。",
    decodingParams: "解码参数",
    reset: "重置",
    pSkipLabel: "Top-1 跳过率 (p_skip)",
    pSkipHint: "越高越偏离大模型高频路径，建议 0.65–0.78",
    minPplLabel: "目标最低困惑度 (Min PPL)",
    minPplHint: "低于此值时自动更换随机种子重采样",
    alphaLabel: "反 RLHF 权重 (α)",
    alphaHint: "log P_base − α·log P_instruct，推荐 0.48–0.64",
    minInstPplLabel: "目标指令困惑度 (Min InstPPL)",
    minInstPplHint: "InstPPL ≥ 350 时可同时绕过 GPTZero 的生成与改写检测",
    tempLabel: "采样温度 (Temp)",
    telegraphicLabel: "Stage-0 去 AI 骨架电报体预处理",
    judgeLabel: "本地只读忠实度裁判重排 (Judge)",
    factGuardLabel: "数值/引用防幻觉白名单 (Fact Guard)",
    lengthRatioLabel: "目标字数比例 (Length)",
    lengthRatioHint: "长段落自动按句分块改写，保持与原文接近字数",
    attemptsLabel: "单段最大重试",
    poolLabel: "候选词池 (K)",
    seedLabel: "基础随机种子 (Seed)",
    bootstrapLabel: "启用已锁定段落自举示范 (Self-Bootstrap)",
    blacklistTitle: "AI 高危词元黑名单",
    blacklistHint: "逗号分隔词根，解码时在 logprobs 层硬屏蔽",
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
    emptyDesc: "在左侧选择「ZeroGPT 反贪心」或「GPTZero 对比解码」模式，粘贴英文文本后点击「开始改写」。生成后可锁定已通过的段落，并对单个段落一键重生成。",
    lock: "锁定",
    locked: "已锁定",
    regen: "重生成",
    copy: "复制",
    copied: "已复制到剪贴板",
    exported: "已导出为 humanized_output.txt",
    emptyInputWarn: "请先输入需要改写的英文文本",
    doneToast: "全部段落改写完成",
    cancelledToast: "已停止生成",
    paraPrefix: "段落",
    runningAttempt: "采样中",
    wordsUnit: "词",
    parasUnit: "段",
    avgPpl: "平均 PPL",
    avgInstPpl: "平均 InstPPL",
  },
  en: {
    subtitle: "Dual-Engine Local Base-Model Humanizer Studio (ZeroGPT & GPTZero)",
    ollamaChecking: "Checking Ollama...",
    ollamaOnline: "Ollama Connected",
    ollamaOffline: "Ollama Offline",
    openZeroGpt: "ZeroGPT ↗",
    openGptZero: "GPTZero ↗",
    engineMode: "Decoding Engine",
    modeAntiGreedy: "ZeroGPT Anti-Greedy",
    modeContrastive: "GPTZero Contrastive",
    modeDescAntiGreedy: "Single base model rank-1 skipping + AI-buzzword blacklist for ZeroGPT.",
    modeDescContrastive: "Same-vocab contrastive decoding (Base − α·Instruct) cancels RLHF shifts to clear both GPTZero AI-Generated & AI-Paraphrased heads.",
    modelConfig: "Models & Connection",
    ollamaUrl: "Ollama URL",
    baseModel: "Base Model (Unaligned)",
    instructModel: "Contrastive Instruct Model (Same Vocab)",
    instructHint: "Must share vocabulary with the base model (e.g. Qwen3-4B Base + Instruct).",
    decodingParams: "Decoding Parameters",
    reset: "Reset",
    pSkipLabel: "Rank-1 Skip Prob (p_skip)",
    pSkipHint: "Higher skips more top-1 LLM tokens (recommended 0.65–0.78)",
    minPplLabel: "Min Target Perplexity (PPL)",
    minPplHint: "Resamples automatically with a fresh seed when PPL is lower",
    alphaLabel: "Anti-RLHF Weight (α)",
    alphaHint: "log P_base − α·log P_instruct (recommended 0.48–0.64)",
    minInstPplLabel: "Min Target InstPPL",
    minInstPplHint: "InstPPL ≥ 350 reliably clears GPTZero 5.0b AI & Paraphrase detectors",
    tempLabel: "Temperature (Temp)",
    telegraphicLabel: "Stage-0 Telegraphic Notes (Anti-Paraphrase)",
    judgeLabel: "Read-Only Local Faithfulness Judge",
    factGuardLabel: "Strict Numeric & Citation Fact Guard",
    lengthRatioLabel: "Target Length Ratio",
    lengthRatioHint: "Long paragraphs auto-split on sentences to preserve word count",
    attemptsLabel: "Max Retries",
    poolLabel: "Candidate Pool (K)",
    seedLabel: "Base Seed",
    bootstrapLabel: "Self-Bootstrap from locked paragraphs",
    blacklistTitle: "AI Buzzword Blacklist",
    blacklistHint: "Comma-separated roots masked at token logprobs decoding",
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
    emptyDesc: "Pick ZeroGPT Anti-Greedy or GPTZero Contrastive mode on the left, paste English text, and click Rewrite All. Lock passing paragraphs and regenerate individual cards.",
    lock: "Lock",
    locked: "Locked",
    regen: "Regen",
    copy: "Copy",
    copied: "Copied to clipboard",
    exported: "Exported humanized_output.txt",
    emptyInputWarn: "Please paste some English text first",
    doneToast: "All paragraphs rewritten",
    cancelledToast: "Generation cancelled",
    paraPrefix: "Paragraph",
    runningAttempt: "Sampling",
    wordsUnit: "words",
    parasUnit: "paras",
    avgPpl: "Avg PPL",
    avgInstPpl: "Avg InstPPL",
  },
};

const SAMPLE_TEXT = `Edge and federated forecasting. Wind-farm operators could evaluate federated training on edge servers with different network conditions and different local weather and generation patterns. The wind-farm case in [6] and the load studies in [7] and [8] motivate a comparison of local-only, synchronous federated and asynchronous federated models under identical forecast horizons.

Workload-aware cluster tuning. The effects of worker count in [2] and the sensitivity to caching and repartitioning in [9] motivate automated tuning for different forecast workloads. Future studies could compare default and tuned configurations while holding data, model, hardware budget and accuracy requirements constant.

Joint wind and solar forecasting. A joint system could align wind generation, PV output and weather data by site and event time, then share a feature pipeline while retaining separate forecasts where site characteristics require them. The wind and PV studies [4], [6] provide starting points for a comparison with separate models.`;

const state = {
  lang: "zh",
  theme: "dark",
  view: "cards",
  mode: "anti_greedy", // "anti_greedy" | "contrastive"
  defaults: null,
  modelsLoaded: false,
  blocks: [],
  activeJobId: null,
  abortController: null,
};

const $ = (id) => document.getElementById(id);

function t(key) {
  return (I18N[state.lang] && I18N[state.lang][key]) || key;
}

function applyModeUI() {
  const isContrastive = state.mode === "contrastive";
  $("btnModeAntiGreedy").classList.toggle("active", !isContrastive);
  $("btnModeContrastive").classList.toggle("active", isContrastive);
  $("instructModelField").classList.toggle("hidden", !isContrastive);
  $("antiGreedyControls").classList.toggle("hidden", isContrastive);
  $("contrastiveControls").classList.toggle("hidden", !isContrastive);
  $("bootstrapWrap").classList.toggle("hidden", isContrastive);
  $("blacklistPanel").classList.toggle("hidden", isContrastive);
  $("modeDesc").textContent = isContrastive ? t("modeDescContrastive") : t("modeDescAntiGreedy");
  updateOutputStats();
  renderCards();
}

function applyI18n() {
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (I18N[state.lang][key]) el.textContent = I18N[state.lang][key];
  });
  $("btnLang").textContent = state.lang === "zh" ? "EN" : "中文";
  applyModeUI();
  updateInputStats();
  updateOutputStats();
}

function showToast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => el.classList.add("hidden"), 2400);
}

function getBlacklist() {
  return $("blacklistText")
    .value.split(/[\n,]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function updateBlacklistCount() {
  $("blacklistCount").textContent = String(getBlacklist().length);
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
  const hasInst = paras.some((b) => typeof b.inst_ppl === "number" && b.inst_ppl > 0);
  if (hasInst) {
    const vals = paras.filter((b) => typeof b.inst_ppl === "number" && b.inst_ppl > 0).map((b) => b.inst_ppl);
    const avg = vals.length ? (vals.reduce((a, c) => a + c, 0) / vals.length).toFixed(0) : "—";
    $("outputStats").textContent = `${wordPart} · ${t("avgInstPpl")} ${avg}`;
  } else {
    const vals = paras.filter((b) => typeof b.ppl === "number" && b.ppl > 0).map((b) => b.ppl);
    const avg = vals.length ? (vals.reduce((a, c) => a + c, 0) / vals.length).toFixed(2) : "—";
    $("outputStats").textContent = `${wordPart} · ${t("avgPpl")} ${avg}`;
  }
  $("mergedOutput").value = merged;
}

async function loadDefaults() {
  const resp = await fetch("/api/defaults");
  const d = await resp.json();
  state.defaults = d;
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
  $("constrainFacts").checked = d.constrain_facts ?? true;
  $("useJudge").checked = d.use_judge ?? true;
  $("blacklistText").value = d.banned_substrs.join(", ");
  updateBlacklistCount();
}

async function refreshModels() {
  const pill = $("ollamaStatus");
  const textEl = pill.querySelector(".status-text");
  pill.className = "status-pill status-checking";
  textEl.textContent = t("ollamaChecking");

  const url = encodeURIComponent($("ollamaUrl").value.trim());
  try {
    const resp = await fetch(`/api/models?ollama_url=${url}`);
    const data = await resp.json();
    const baseSel = $("modelSelect");
    const instSel = $("instructModelSelect");
    const prevBase = baseSel.value;
    const prevInst = instSel.value;

    if (data.ok && data.models.length > 0) {
      pill.className = "status-pill status-online";
      textEl.textContent = `${t("ollamaOnline")} (${data.models.length})`;
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
      pill.className = "status-pill status-offline";
      textEl.textContent = t("ollamaOffline");
    }
  } catch (_) {
    pill.className = "status-pill status-offline";
    textEl.textContent = t("ollamaOffline");
  }
}

function getRequestParams() {
  return {
    mode: state.mode,
    ollama_url: $("ollamaUrl").value.trim(),
    model: $("modelSelect").value,
    instruct_model: $("instructModelSelect").value,
    length_ratio: parseFloat($("lengthRatio").value) || 1.0,
    p_skip: parseFloat($("pSkip").value),
    min_ppl: parseFloat($("minPpl").value),
    alpha: parseFloat($("alphaSlider").value),
    min_inst_ppl: parseFloat($("minInstPpl").value),
    contrastive_temp: parseFloat($("contrastiveTemp").value),
    attempts: parseInt($("attempts").value, 10) || 3,
    pool_size: parseInt($("poolSize").value, 10) || 6,
    seed: parseInt($("seedInput").value, 10) || 401,
    use_telegraphic: $("useTelegraphic").checked,
    constrain_facts: $("constrainFacts").checked,
    use_judge: $("useJudge").checked,
    banned_substrs: getBlacklist(),
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
      extraMeta = `· BasePPL ${typeof b.ppl === "number" ? b.ppl.toFixed(1) : "—"}${wordMeta}${b.elapsed ? ` · ${b.elapsed}s` : ""}`;
    } else if (typeof b.ppl === "number") {
      pillClass = b.ppl >= minPpl ? "ppl-good" : "ppl-warn";
      pillLabel = `PPL ${b.ppl.toFixed(2)}`;
      extraMeta = `${wordMeta}${b.elapsed ? ` · ${b.elapsed}s` : ""}${b.skip_ratio ? ` · skip ${b.skip_ratio}%` : ""}`;
    }

    const fidBadge =
      !b.running && typeof b.fidelity_pct === "number"
        ? `<span class="ppl-pill ${b.fidelity_pct >= 85 ? "ppl-good" : "ppl-warn"}">Fid ${b.fidelity_pct}%${typeof b.judge_score === "number" ? ` · J${b.judge_score}` : ""}</span>`
        : "";

    card.innerHTML = `
      <div class="card-top">
        <div class="card-meta">
          <span class="card-label">${t("paraPrefix")} #${paraCounter}</span>
          <span class="ppl-pill ${pillClass}">${pillLabel}</span>
          ${fidBadge}
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

async function runAll() {
  const rawText = $("sourceInput").value.trim();
  if (!rawText) {
    showToast(t("emptyInputWarn"));
    return;
  }

  const lockedBlocks = {};
  state.blocks.forEach((b, idx) => {
    if (b.kind === "para" && b.locked && b.text) {
      lockedBlocks[String(idx)] = {
        mode: b.mode,
        text: b.text,
        ppl: b.ppl,
        inst_ppl: b.inst_ppl,
        skip_ratio: b.skip_ratio,
        fidelity_pct: b.fidelity_pct,
        judge_score: b.judge_score,
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
        ...getRequestParams(),
      }),
      signal: state.abortController.signal,
    });

    await consumeSSE(resp, (evName, data) => {
      if (evName === "init") {
        state.blocks = data.blocks.map((blk, i) => ({
          index: i,
          kind: blk.kind,
          mode: lockedBlocks[String(i)]?.mode || state.mode,
          source: blk.text,
          text: blk.kind === "heading" ? blk.text : (lockedBlocks[String(i)]?.text || ""),
          ppl: lockedBlocks[String(i)]?.ppl ?? null,
          inst_ppl: lockedBlocks[String(i)]?.inst_ppl ?? null,
          skip_ratio: lockedBlocks[String(i)]?.skip_ratio ?? null,
          fidelity_pct: lockedBlocks[String(i)]?.fidelity_pct ?? null,
          judge_score: lockedBlocks[String(i)]?.judge_score ?? null,
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
          b.text = data.seed_start || "";
          b.statusText = "1/...";
          renderCards();
          $("runProgress").textContent = `Paragraph #${data.para_idx + 1} (${b.mode})...`;
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

  const jobId = `one-${Date.now()}`;
  state.activeJobId = jobId;
  state.abortController = new AbortController();
  b.running = true;
  b.locked = false;
  b.mode = state.mode;
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

  const params = getRequestParams();
  params.seed = Math.floor(1000 + Math.random() * 90000);

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
    state.mode = "anti_greedy";
    applyModeUI();
  });
  $("btnModeContrastive").addEventListener("click", () => {
    state.mode = "contrastive";
    applyModeUI();
  });

  $("sourceInput").addEventListener("input", updateInputStats);
  $("blacklistText").addEventListener("input", updateBlacklistCount);
  $("pSkip").addEventListener("input", () => ($("pSkipVal").textContent = Number($("pSkip").value).toFixed(2)));
  $("minPpl").addEventListener("input", () => ($("minPplVal").textContent = Number($("minPpl").value).toFixed(1)));
  $("alphaSlider").addEventListener("input", () => ($("alphaVal").textContent = Number($("alphaSlider").value).toFixed(2)));
  $("minInstPpl").addEventListener("input", () => ($("minInstPplVal").textContent = $("minInstPpl").value));
  $("contrastiveTemp").addEventListener("input", () => ($("contrastiveTempVal").textContent = Number($("contrastiveTemp").value).toFixed(2)));
  $("lengthRatio").addEventListener("input", () => ($("lengthRatioVal").textContent = `${Number($("lengthRatio").value).toFixed(2)}×`));

  $("btnRandSeed").addEventListener("click", () => {
    $("seedInput").value = String(Math.floor(100 + Math.random() * 9900));
  });

  $("btnResetParams").addEventListener("click", () => {
    if (!state.defaults) return;
    $("pSkip").value = state.defaults.p_skip;
    $("pSkipVal").textContent = Number(state.defaults.p_skip).toFixed(2);
    $("minPpl").value = state.defaults.min_ppl;
    $("minPplVal").textContent = Number(state.defaults.min_ppl).toFixed(1);
    $("alphaSlider").value = state.defaults.alpha;
    $("alphaVal").textContent = Number(state.defaults.alpha).toFixed(2);
    $("minInstPpl").value = state.defaults.min_inst_ppl;
    $("minInstPplVal").textContent = String(state.defaults.min_inst_ppl);
    $("contrastiveTemp").value = state.defaults.contrastive_temp;
    $("contrastiveTempVal").textContent = Number(state.defaults.contrastive_temp).toFixed(2);
    const lr = state.defaults.length_ratio ?? 1.0;
    $("lengthRatio").value = lr;
    $("lengthRatioVal").textContent = `${Number(lr).toFixed(2)}×`;
    $("attempts").value = state.defaults.attempts;
    $("poolSize").value = state.defaults.pool_size;
    $("seedInput").value = state.defaults.seed;
    $("useTelegraphic").checked = state.defaults.use_telegraphic ?? true;
    $("constrainFacts").checked = state.defaults.constrain_facts ?? true;
    $("useJudge").checked = state.defaults.use_judge ?? true;
  });

  $("btnResetBlacklist").addEventListener("click", () => {
    if (!state.defaults) return;
    $("blacklistText").value = state.defaults.banned_substrs.join(", ");
    updateBlacklistCount();
  });

  $("btnRefreshModels").addEventListener("click", refreshModels);
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
    refreshModels();
  });
}

(async function init() {
  bindEvents();
  await loadDefaults();
  applyI18n();
  await refreshModels();
})();
