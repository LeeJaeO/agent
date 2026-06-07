const state = {
  defaults: null,
  pollTimer: null,
  inputFiles: [],
  jobImageDir: null,
  jobInputFiles: [],
};

const els = {
  agentRoot: document.getElementById("agentRoot"),
  statusPill: document.getElementById("statusPill"),
  imageDir: document.getElementById("imageDir"),
  outputDir: document.getElementById("outputDir"),
  depthBackend: document.getElementById("depthBackend"),
  segBackend: document.getElementById("segBackend"),
  textFile: document.getElementById("textFile"),
  runButton: document.getElementById("runButton"),
  stopButton: document.getElementById("stopButton"),
  refreshButton: document.getElementById("refreshButton"),
  commandPreview: document.getElementById("commandPreview"),
  imageCount: document.getElementById("imageCount"),
  imageList: document.getElementById("imageList"),
  logBox: document.getElementById("logBox"),
  jobElapsed: document.getElementById("jobElapsed"),
  currentImageSurface: document.getElementById("currentImageSurface"),
  currentImagePreview: document.getElementById("currentImagePreview"),
  currentImageName: document.getElementById("currentImageName"),
  currentImageProgress: document.getElementById("currentImageProgress"),
  currentImageStatus: document.getElementById("currentImageStatus"),
  metricVolume: document.getElementById("metricVolume"),
  metricTruck: document.getElementById("metricTruck"),
  metricProcessed: document.getElementById("metricProcessed"),
  metricElapsed: document.getElementById("metricElapsed"),
  resultPath: document.getElementById("resultPath"),
  pdfLink: document.getElementById("pdfLink"),
  categoryRow: document.getElementById("categoryRow"),
  resultRows: document.getElementById("resultRows"),
  outputImageCount: document.getElementById("outputImageCount"),
  outputImageGrid: document.getElementById("outputImageGrid"),
  pdfCustomerFields: Array.from(document.querySelectorAll("[data-pdf-key]")),
};

function shellQuote(value) {
  const text = String(value || "");
  if (/^[A-Za-z0-9_./:=,+-]+$/.test(text)) return text;
  return "'" + text.replaceAll("'", "'\\''") + "'";
}

function formatM3(value) {
  const number = Number(value || 0);
  return `${number.toFixed(3)} m³`;
}

function formatSeconds(value) {
  const number = Number(value || 0);
  return `${number.toFixed(1)}s`;
}

function fileUrl(path) {
  return `/api/download?path=${encodeURIComponent(path)}`;
}

function commandArg(command, flag) {
  const parts = String(command || "").trim().split(/\s+/);
  const index = parts.indexOf(flag);
  if (index < 0 || index + 1 >= parts.length) return null;
  return parts[index + 1].trim().replace(/^['"]|['"]$/g, "");
}

function imageDirFromCommand(command) {
  return commandArg(command, "--image-dir");
}

function firstInputImage(files) {
  const file = files?.[0];
  if (!file) return null;
  return {
    index: 1,
    total: files.length,
    name: file.name,
    path: file.path,
    status_label: "준비 중",
  };
}

async function fallbackCurrentImage(job) {
  const imageDir = imageDirFromCommand(job?.command) || els.imageDir.value.trim();
  if (!imageDir) return firstInputImage(state.inputFiles);

  if (state.jobImageDir !== imageDir) {
    state.jobImageDir = imageDir;
    state.jobInputFiles = [];
    try {
      const payload = await api(`/api/images?path=${encodeURIComponent(imageDir)}`);
      state.jobInputFiles = payload.files || [];
    } catch (error) {
      console.error(error);
    }
  }

  return firstInputImage(state.jobInputFiles.length ? state.jobInputFiles : state.inputFiles);
}

function currentImageFromResults(results, job) {
  const files = results?.summary?.files || [];
  if (!files.length) return null;

  const imageDir = imageDirFromCommand(job?.command);
  const summaryInputDir = results?.summary?.input_dir;
  if (imageDir && summaryInputDir && imageDir !== summaryInputDir) return null;

  const file = files[files.length - 1];
  const total = Number(results?.summary?.num_files || files.length);
  const index = Math.min(files.length, total || files.length);
  const path = file.path || file.file;
  const name = file.file || (path ? path.split(/[\/]/).pop() : "-");
  const isRunning = job?.status === "running" || job?.status === "stopping";
  return {
    index,
    total: total || files.length,
    name,
    path,
    status_label: isRunning && index >= total ? "PDF 생성 중" : "마지막 처리 이미지",
  };
}

async function syncFormToJob(job) {
  const command = job?.command || "";
  const imageDir = imageDirFromCommand(command);
  const textFile = commandArg(command, "--text-file");
  const outputDir = commandArg(command, "--output") || job?.output_dir;
  let imageChanged = false;

  if (imageDir && els.imageDir.value.trim() !== imageDir) {
    els.imageDir.value = imageDir;
    imageChanged = true;
  }
  if (textFile && els.textFile.value.trim() !== textFile) {
    els.textFile.value = textFile;
  }
  if (outputDir && els.outputDir.value.trim() !== outputDir) {
    els.outputDir.value = outputDir;
  }

  if (imageChanged) {
    state.jobImageDir = null;
    state.jobInputFiles = [];
    await refreshImages();
  }
  renderCommand();
}

function setStatus(status) {
  const labelMap = {
    running: "실행 중",
    completed: "완료",
    failed: "실패",
    stopped: "중지됨",
    stopping: "중지 중",
  };
  els.statusPill.textContent = labelMap[status] || "대기";
  els.statusPill.className = `status-pill ${status || ""}`;
  const isRunning = status === "running" || status === "stopping";
  els.runButton.disabled = isRunning;
  els.stopButton.disabled = !isRunning;
}

function pdfCustomerPayload() {
  const customer = {};
  for (const field of els.pdfCustomerFields) {
    customer[field.dataset.pdfKey] = field.value.trim();
  }
  return customer;
}

function hasPdfCustomerPayload(customer) {
  return Object.values(customer || {}).some((value) => String(value || "").trim());
}

function currentPayload() {
  const pdfCustomer = pdfCustomerPayload();
  return {
    image_dir: els.imageDir.value.trim(),
    output_dir: els.outputDir.value.trim(),
    depth_backend: els.depthBackend.value,
    seg_backend: els.segBackend.value,
    text_file: els.textFile.value.trim(),
    pdf_customer: pdfCustomer,
  };
}

function renderCommand() {
  const payload = currentPayload();
  const args = [
    "python",
    "-u",
    "pipeline.py",
    "--image-dir",
    payload.image_dir,
    "--text-file",
    payload.text_file,
    "--depth-backend",
    payload.depth_backend,
    "--seg-backend",
    payload.seg_backend,
    "--output",
    payload.output_dir,
    "--json-only",
    "--pdf",
  ];
  if (hasPdfCustomerPayload(payload.pdf_customer)) {
    args.push("--pdf-customer-json", "agent_ui/runs/<job>_customer.json");
  }

  els.commandPreview.textContent = [
    `cd ${shellQuote(state.defaults?.agent_root || "")}`,
    "conda activate volume_est",
    args.map(shellQuote).join(" "),
  ].join("\n");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || response.statusText);
  }
  return payload;
}

function renderImages(payload) {
  state.inputFiles = payload.files || [];
  els.imageCount.textContent = String(payload.count || 0);
  els.imageList.innerHTML = "";
  if (!payload.exists) {
    els.imageList.innerHTML = '<div class="empty-state">폴더 없음</div>';
    return;
  }
  if (!payload.files?.length) {
    els.imageList.innerHTML = '<div class="empty-state">이미지 없음</div>';
    return;
  }
  for (const file of payload.files) {
    const row = document.createElement("a");
    row.className = "image-item";
    row.href = fileUrl(file.path);
    row.target = "_blank";
    row.rel = "noreferrer";
    row.innerHTML = `<strong>${file.stem}</strong><span>${Math.round(file.size_bytes / 1024)} KB</span>`;
    els.imageList.appendChild(row);
  }
}

async function refreshImages() {
  const path = encodeURIComponent(els.imageDir.value.trim());
  const payload = await api(`/api/images?path=${path}`);
  renderImages(payload);
}

function updateCurrentImage(image, status) {
  if (!image) {
    els.currentImageSurface.classList.add("hidden");
    els.currentImagePreview.removeAttribute("src");
    return;
  }

  els.currentImageSurface.classList.remove("hidden");
  els.currentImageName.textContent = image.name || "-";
  els.currentImageProgress.textContent = image.index && image.total ? `${image.index}/${image.total}` : "-";
  els.currentImageStatus.textContent = image.status_label || (status === "running" ? "처리 중" : "마지막 처리 이미지");
  if (image.path) {
    els.currentImagePreview.src = fileUrl(image.path);
    els.currentImagePreview.classList.remove("hidden");
  } else {
    els.currentImagePreview.removeAttribute("src");
    els.currentImagePreview.classList.add("hidden");
  }
}

function updatePdfLink(path) {
  if (!path) {
    els.pdfLink.classList.add("hidden");
    els.pdfLink.removeAttribute("href");
    return;
  }
  els.pdfLink.href = fileUrl(path);
  els.pdfLink.classList.remove("hidden");
}

function renderOutputImages(files = []) {
  els.outputImageCount.textContent = String(files.length || 0);
  els.outputImageGrid.innerHTML = "";
  if (!files.length) {
    els.outputImageGrid.innerHTML = '<div class="empty-state">저장 이미지 없음</div>';
    return;
  }
  for (const file of files) {
    const link = document.createElement("a");
    link.className = "output-image-card";
    link.href = fileUrl(file.path);
    link.target = "_blank";
    link.rel = "noreferrer";
    link.innerHTML = `
      <img src="${fileUrl(file.path)}" alt="${file.relative_path}" loading="lazy" />
      <span>${file.relative_path}</span>
    `;
    els.outputImageGrid.appendChild(link);
  }
}

function renderSplitTime(summary) {
  const processing = Number(summary.total_processing_seconds ?? summary.processing_seconds ?? 0);
  const save = Number(summary.total_save_seconds ?? summary.save_seconds ?? 0);
  if (!processing && !save) return "-";
  return `${formatSeconds(processing)} / ${formatSeconds(save)}`;
}

function renderResults(payload) {
  updatePdfLink(payload?.quote_pdf);
  renderOutputImages(payload?.output_images || []);
  if (!payload || !payload.exists) {
    els.metricVolume.textContent = "-";
    els.metricTruck.textContent = "-";
    els.metricProcessed.textContent = "-";
    els.metricElapsed.textContent = "-";
    els.resultPath.textContent = "-";
    els.categoryRow.innerHTML = "";
    els.resultRows.innerHTML = '<tr><td colspan="5" class="empty-state">결과 없음</td></tr>';
    return;
  }

  const summary = payload.summary || {};
  els.resultPath.textContent = payload.path || "-";
  els.metricVolume.textContent = formatM3(summary.total_volume_m3);
  els.metricTruck.textContent = summary.recommended_truck || "-";
  els.metricProcessed.textContent =
    summary.num_processed !== undefined
      ? `${summary.num_processed}/${summary.num_files}`
      : `${summary.objects?.length || 0}`;
  els.metricElapsed.textContent = renderSplitTime(summary);

  els.categoryRow.innerHTML = "";
  const categories = summary.category_counts || {};
  for (const [name, count] of Object.entries(categories)) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = `${name} ${count}`;
    els.categoryRow.appendChild(chip);
  }

  els.resultRows.innerHTML = "";
  const files = summary.files || [];
  if (!files.length) {
    const objects = summary.objects || [];
    for (const obj of objects) {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${obj.object || obj.object_category || "-"}</td>
        <td>${formatM3(obj.volume_m3)}</td>
        <td>${obj.object_category || "-"}</td>
        <td>-</td>
        <td>-</td>
      `;
      els.resultRows.appendChild(row);
    }
    if (!objects.length) {
      els.resultRows.innerHTML = '<tr><td colspan="5" class="empty-state">결과 없음</td></tr>';
    }
    return;
  }

  for (const file of files) {
    const row = document.createElement("tr");
    const categoriesText = (file.detected_categories || []).join(", ") || "-";
    row.innerHTML = `
      <td>${file.file || "-"}</td>
      <td>${formatM3(file.volume_m3)}</td>
      <td>${file.num_detected_objects || 0}개 · ${categoriesText}</td>
      <td>${formatSeconds(file.processing_seconds)}</td>
      <td>${formatSeconds(file.save_seconds)}</td>
    `;
    els.resultRows.appendChild(row);
  }
}

async function refreshResults() {
  const output = encodeURIComponent(els.outputDir.value.trim());
  const payload = await api(`/api/results?output=${output}`);
  renderResults(payload);
}

async function pollJob() {
  const payload = await api("/api/job");
  const job = payload.job;
  if (!job) {
    setStatus(null);
    updateCurrentImage(null);
    return;
  }

  await syncFormToJob(job);
  setStatus(job.status);
  let currentImage = job.current_image;
  if (!currentImage) {
    currentImage = currentImageFromResults(payload.results, job);
  }
  if (!currentImage && (job.status === "running" || job.status === "stopping")) {
    currentImage = await fallbackCurrentImage(job);
  }
  updateCurrentImage(currentImage, job.status);
  els.jobElapsed.textContent = formatSeconds(job.elapsed_seconds);
  els.logBox.textContent = job.log || "";
  els.logBox.scrollTop = els.logBox.scrollHeight;
  if (payload.results) renderResults(payload.results);

  if (job.status !== "running" && job.status !== "stopping") {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    await refreshResults();
  }
}

async function runPipeline() {
  renderCommand();
  const payload = currentPayload();
  els.logBox.textContent = "";
  updateCurrentImage(null);
  state.jobImageDir = null;
  state.jobInputFiles = [];
  await refreshImages();
  renderResults({ exists: false, output_images: [] });
  const result = await api("/api/run", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  setStatus(result.job.status);
  if (!state.pollTimer) {
    state.pollTimer = setInterval(() => pollJob().catch(console.error), 2000);
  }
  await pollJob();
}

async function stopPipeline() {
  await api("/api/stop", { method: "POST", body: "{}" });
  await pollJob();
}

async function boot() {
  state.defaults = await api("/api/defaults");
  els.agentRoot.textContent = state.defaults.agent_root;
  els.imageDir.value = state.defaults.image_dir;
  els.outputDir.value = state.defaults.output_dir;
  els.textFile.value = state.defaults.text_file || "";
  for (const name of state.defaults.depth_backends) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    els.depthBackend.appendChild(option);
  }
  for (const name of state.defaults.seg_backends) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    els.segBackend.appendChild(option);
  }
  els.depthBackend.value = "anycalib_unidepth";
  els.segBackend.value = "sam3";

  for (const element of [
    els.imageDir,
    els.outputDir,
    els.depthBackend,
    els.segBackend,
    els.textFile,
  ]) {
    element.addEventListener("input", renderCommand);
    element.addEventListener("change", renderCommand);
  }
  els.imageDir.addEventListener("change", () => refreshImages().catch(showError));
  els.outputDir.addEventListener("change", () => refreshResults().catch(showError));
  for (const element of els.pdfCustomerFields) {
    element.addEventListener("input", renderCommand);
    element.addEventListener("change", renderCommand);
  }

  els.refreshButton.addEventListener("click", () => refreshImages().catch(showError));
  els.runButton.addEventListener("click", () => runPipeline().catch(showError));
  els.stopButton.addEventListener("click", () => stopPipeline().catch(showError));

  renderCommand();
  await refreshImages();
  await refreshResults();
  await pollJob();
}

function showError(error) {
  setStatus("failed");
  els.logBox.textContent = `ERROR: ${error.message || error}`;
}

boot().catch(showError);
