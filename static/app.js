const I18N = {
  zh: {
    subtitle: "Ollama 基础模型逐词元反贪心降 AI 率工作台",
    ollamaChecking: "检测 Ollama...",
    ollamaOnline: "Ollama 已连接",
    ollamaOffline: "Ollama 未连接",
    openZeroGpt: "打开 ZeroGPT ↗",
    modelConfig: "模型与连接",
    ollamaUrl: "Ollama 地址",
    baseModel: "本地模型（推荐 Base 未对齐模型）",
    baseHint: "带 base 标识的预训练模型抗检测效果最佳。",
    decodingParams: "反贪心解码参数",
    reset: "重置",
    pSkipLabel: "Top-1 跳过率 (p_skip)",
    pSkipHint: "越高越偏离大模型高频路径，建议 0.65–0.78",
    minPplLabel: "目标最低困惑度 (Min PPL)",
    minPplHint: "低于此值时自动更换随机种子重采样",
    attemptsLabel: "单段最大重试",
    poolLabel: "候选词池 (2..K)",
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
    emptyDesc: "在左侧粘贴英文文本并点击「开始改写」。生成后你可以锁定已通过 ZeroGPT 检测的段落，并对仍被标黄的单个段落一键重新生成。",
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
  },
  en: {
    subtitle: "Token-Level Anti-Greedy Humanizer Studio for Local Ollama Base Models",
    ollamaChecking: "Checking Ollama...",
    ollamaOnline: "Ollama Connected",
    ollamaOffline: "Ollama Offline",
    openZeroGpt: "Open ZeroGPT ↗",
    modelConfig: "Model & Connection",
    ollamaUrl: "Ollama URL",
    baseModel: "Local Model (Unaligned Base Recommended)",
    baseHint: "Pre-trained base models produce the most natural non-RLHF distribution.",
    decodingParams: "Anti-Greedy Decoding",
    reset: "Reset",
    pSkipLabel: "Rank-1 Skip Prob (p_skip)",
    pSkipHint: "Higher skips more top-1 LLM tokens (recommended 0.65–0.78)",
    minPplLabel: "Min Target Perplexity (PPL)",
    minPplHint: "Automatically resamples with a fresh seed when PPL is lower",
    attemptsLabel: "Max Retries",
    poolLabel: "Candidate Pool (2..K)",
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
    emptyDesc: "Paste English text on the left and click Rewrite All. Lock paragraphs that pass ZeroGPT and regenerate individual flagged paragraphs with one click.",
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
  },
};

const SAMPLE_TEXT = `3.3 Real-World Applications
The selected cases demonstrate different uses of distributed frameworks: serving forecasts across PV sites, parallelizing wind-power calculations, and preparing smart-meter data for analytics. Evidence ranges from implemented research clusters to proposed service architectures. These levels must be distinguished when assessing practical deployment [1]-[7].

3.3.1 Photovoltaic Forecasting and Sensor Data Pipelines
Sun et al. implemented a multi-site PV forecasting system using Kafka for message distribution, HDFS for storage, and Flink with Alink for forecasting and incremental model updates [1]. Experiments used three servers and one year of data from six NREL PV sites, with corresponding NSRDB weather data at 15-minute intervals. For scalability testing, the site datasets were replicated to simulate up to 60 concurrent sites. At that load, average forecasting and update latencies were 0.522 s and 4.520 s, respectively (Table 6 of [1]).

Preda et al. connect PV logger measurements and weather information to a forecasting and adaptive-switching application [2]. Their implementation discussion identifies Kafka producers and Spark consumers, while the forecasting examples use R and SVM regression. Increasing the input set from one to five parameters reduced the reported RMSE from approximately 10.92 to 6.77. The article describes HDFS as a possible storage solution, but does not establish distributed speedup through a controlled cluster benchmark [2].`;

const state = {
  lang: "zh",
  theme: "dark",
  view: "cards",
  defaults: null,
  blocks: [], // [{index, kind, source, text, ppl, skip_ratio, elapsed, attempt, locked, running, statusText}]
  activeJobId: null,
  abortController: null,
};

const $ = (id) => document.getElementById(id);

function t(key) {
  return (I18N[state.lang] && I18N[state.lang][key]) || key;
}

function applyI18n() {
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (I18N[state.lang][key]) el.textContent = I18N[state.lang][key];
  });
  $("btnLang").textContent = state.lang === "zh" ? "EN" : "中文";
  updateInputStats();
  updateOutputStats();
  renderCards();
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
  const ppls = state.blocks
    .filter((b) => b.kind === "para" && typeof b.ppl === "number" && b.ppl > 0)
    .map((b) => b.ppl);
  const avg = ppls.length ? (ppls.reduce((a, c) => a + c, 0) / ppls.length).toFixed(2) : "—";
  $("outputStats").textContent = `${words} ${t("wordsUnit")} · ${t("avgPpl")} ${avg}`;
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
  $("attempts").value = d.attempts;
  $("poolSize").value = d.pool_size;
  $("seedInput").value = d.seed;
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
    const sel = $("modelSelect");
    const prev = sel.value;

    if (data.ok && data.models.length > 0) {
      pill.className = "status-pill status-online";
      textEl.textContent = `${t("ollamaOnline")} (${data.models.length})`;
      sel.innerHTML = "";
      data.models.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m.name;
        const gb = m.size ? ` (${(m.size / 1e9).toFixed(1)} GB)` : "";
        opt.textContent = (m.is_base ? "★ " : "") + m.name + gb;
        sel.appendChild(opt);
      });
      const preferred =
        data.models.find((m) => m.name === prev) ||
        data.models.find((m) => m.name === (state.defaults && state.defaults.model)) ||
        data.models.find((m) => m.is_base) ||
        data.models[0];
      if (preferred) sel.value = preferred.name;
    } else {
      pill.className = "status-pill status-offline";
      textEl.textContent = t("ollamaOffline");
    }
  } catch (e) {
    pill.className = "status-pill status-offline";
    textEl.textContent = t("ollamaOffline");
  }
}

function getRequestParams() {
  return {
    ollama_url: $("ollamaUrl").value.trim(),
    model: $("modelSelect").value,
    p_skip: parseFloat($("pSkip").value),
    min_ppl: parseFloat($("minPpl").value),
    attempts: parseInt($("attempts").value, 10) || 3,
    pool_size: parseInt($("poolSize").value, 10) || 6,
    seed: parseInt($("seedInput").value, 10) || 401,
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

  state.blocks.forEach((b, idx) => {
    if (b.kind === "heading") {
      const h = document.createElement("div");
      h.className = "heading-card";
      h.textContent = b.text;
      box.appendChild(h);
      return;
    }

    paraCounter += 1;
    const card = document.createElement("div");
    card.className = `para-card${b.running ? " is-running" : ""}${b.locked ? " is-locked" : ""}`;
    card.dataset.index = String(idx);

    const pplClass = b.running
      ? "ppl-busy"
      : typeof b.ppl === "number" && b.ppl >= minPpl
      ? "ppl-good"
      : "ppl-warn";

    const pplLabel = b.running
      ? `${t("runningAttempt")} ${b.statusText || "..."}`
      : typeof b.ppl === "number"
      ? `PPL ${b.ppl.toFixed(2)}`
      : "PPL —";

    const extraMeta =
      !b.running && b.elapsed
        ? `· ${b.elapsed}s${b.skip_ratio ? ` · skip ${b.skip_ratio}%` : ""}`
        : "";

    card.innerHTML = `
      <div class="card-top">
        <div class="card-meta">
          <span class="card-label">${t("paraPrefix")} #${paraCounter}</span>
          <span class="ppl-pill ${pplClass}">${pplLabel}</span>
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

  // Preserve locked paragraphs if source matches previously parsed blocks
  const lockedBlocks = {};
  state.blocks.forEach((b, idx) => {
    if (b.kind === "para" && b.locked && b.text) {
      lockedBlocks[String(idx)] = { text: b.text, ppl: b.ppl, skip_ratio: b.skip_ratio };
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
          source: blk.text,
          text: blk.kind === "heading" ? blk.text : (lockedBlocks[String(i)]?.text || ""),
          ppl: lockedBlocks[String(i)]?.ppl ?? null,
          skip_ratio: lockedBlocks[String(i)]?.skip_ratio ?? null,
          locked: Boolean(lockedBlocks[String(i)]),
          running: false,
        }));
        renderCards();
        updateOutputStats();
      } else if (evName === "block_start") {
        const b = state.blocks[data.index];
        if (b) {
          b.running = true;
          b.text = data.seed_start || "";
          b.statusText = "1/...";
          renderCards();
          $("runProgress").textContent = `Paragraph #${data.para_idx + 1} ...`;
        }
      } else if (evName === "progress") {
        const b = state.blocks[data.index];
        if (b) {
          if (data.text) b.text = data.text;
          if (data.event === "attempt_start") {
            b.statusText = `${data.attempt}/${data.attempts}`;
          } else if (data.event === "token") {
            b.statusText = `${data.attempt}/${data.attempts} · PPL ${data.ppl}`;
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
  b.statusText = "...";
  setRunningUI(true, `Regenerating #${paraIdx + 1}...`);
  renderCards();

  // If bootstrap is enabled and another paragraph is locked, pass it as style demo!
  let bootstrapDemo = null;
  if ($("useBootstrap").checked) {
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
        b.statusText = data.ppl ? `${data.attempt}/${data.attempts} · PPL ${data.ppl}` : `${data.attempt}/${data.attempts}`;
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
  $("sourceInput").addEventListener("input", updateInputStats);
  $("blacklistText").addEventListener("input", updateBlacklistCount);
  $("pSkip").addEventListener("input", () => ($("pSkipVal").textContent = Number($("pSkip").value).toFixed(2)));
  $("minPpl").addEventListener("input", () => ($("minPplVal").textContent = Number($("minPpl").value).toFixed(1)));

  $("btnRandSeed").addEventListener("click", () => {
    $("seedInput").value = String(Math.floor(100 + Math.random() * 9900));
  });

  $("btnResetParams").addEventListener("click", () => {
    if (!state.defaults) return;
    $("pSkip").value = state.defaults.p_skip;
    $("pSkipVal").textContent = Number(state.defaults.p_skip).toFixed(2);
    $("minPpl").value = state.defaults.min_ppl;
    $("minPplVal").textContent = Number(state.defaults.min_ppl).toFixed(1);
    $("attempts").value = state.defaults.attempts;
    $("poolSize").value = state.defaults.pool_size;
    $("seedInput").value = state.defaults.seed;
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
