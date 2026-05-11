/**
 * Zero-Knowledge Voice -- Dashboard Frontend
 * Handles mic recording, file upload, metrics display,
 * PII highlighting, and benchmark execution.
 */

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// DOM
const statusBadge   = $("#status-badge");
const statusText    = $(".status-text");
const micButton     = $("#mic-button");
const micLabel      = $("#mic-label");
const waveCanvas    = $("#waveform-canvas");
const dropZone      = $("#drop-zone");
const fileInput     = $("#file-input");
const fileInfo      = $("#file-info");
const fileName      = $("#file-name");
const btnProcess    = $("#btn-process");
const processingBar = $("#processing-bar");
const resultsSection = $("#results-section");
const textOriginal   = $("#text-original");
const textRedacted   = $("#text-redacted");
const piiDetails     = $("#pii-details");
const entityList     = $("#entity-list");
const btnBenchmark   = $("#btn-benchmark");
const benchStatus    = $("#bench-status");
const benchResults   = $("#bench-results");
const benchMaxFiles  = $("#bench-max-files");

// State
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let audioContext = null;
let analyser = null;
let animFrame = null;
let selectedFile = null;

// ── Health Check ──
async function checkHealth() {
    try {
        const r = await fetch("/api/health");
        if (r.ok) {
            statusBadge.classList.add("online");
            statusText.textContent = "Models Ready";
        }
    } catch {
        statusBadge.classList.remove("online");
        statusText.textContent = "Offline";
    }
}
checkHealth();
setInterval(checkHealth, 20000);

// ── Tabs ──
$$(".tab").forEach((t) =>
    t.addEventListener("click", () => {
        $$(".tab").forEach((x) => x.classList.remove("active"));
        $$(".tab-panel").forEach((p) => p.classList.remove("active"));
        t.classList.add("active");
        $(`#panel-${t.dataset.tab}`).classList.add("active");
    })
);

// ── Waveform ──
function drawWaveform(an) {
    const ctx = waveCanvas.getContext("2d");
    const buf = new Float32Array(an.fftSize);
    (function draw() {
        animFrame = requestAnimationFrame(draw);
        an.getFloatTimeDomainData(buf);
        const w = waveCanvas.width, h = waveCanvas.height;
        ctx.clearRect(0, 0, w, h);
        const grad = ctx.createLinearGradient(0, 0, w, 0);
        grad.addColorStop(0, "hsl(245,72%,64%)");
        grad.addColorStop(1, "hsl(188,95%,48%)");
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        const step = w / buf.length;
        for (let i = 0; i < buf.length; i++) {
            const y = (buf[i] + 1) / 2 * h;
            i === 0 ? ctx.moveTo(0, y) : ctx.lineTo(i * step, y);
        }
        ctx.stroke();
    })();
}
function stopWaveform() {
    if (animFrame) cancelAnimationFrame(animFrame);
    waveCanvas.getContext("2d").clearRect(0, 0, waveCanvas.width, waveCanvas.height);
}

// ── Microphone ──
async function startRec() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 } });
        audioContext = new AudioContext();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        audioContext.createMediaStreamSource(stream).connect(analyser);
        drawWaveform(analyser);

        mediaRecorder = new MediaRecorder(stream, {
            mimeType: MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm",
        });
        audioChunks = [];
        mediaRecorder.ondataavailable = (e) => { if (e.data.size) audioChunks.push(e.data); };
        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach((t) => t.stop());
            stopWaveform();
            if (audioContext) { audioContext.close(); audioContext = null; }
            if (audioChunks.length) await uploadBlob(new Blob(audioChunks, { type: "audio/webm" }), "recording.webm");
        };
        mediaRecorder.start();
        isRecording = true;
        micButton.classList.add("recording");
        micLabel.textContent = "Recording -- click to stop";
    } catch (e) {
        micLabel.textContent = "Microphone access denied";
    }
}
function stopRec() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
    isRecording = false;
    micButton.classList.remove("recording");
    micLabel.textContent = "Click to start recording";
}
micButton.addEventListener("click", () => isRecording ? stopRec() : startRec());

// ── File Upload ──
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => { e.preventDefault(); dropZone.classList.remove("dragover"); if (e.dataTransfer.files[0]) pickFile(e.dataTransfer.files[0]); });
fileInput.addEventListener("change", () => { if (fileInput.files[0]) pickFile(fileInput.files[0]); });

function pickFile(f) {
    selectedFile = f;
    fileName.textContent = f.name;
    fileInfo.style.display = "flex";
}
btnProcess.addEventListener("click", async () => {
    if (!selectedFile) return;
    btnProcess.disabled = true;
    btnProcess.textContent = "Processing...";
    await uploadBlob(selectedFile, selectedFile.name);
    btnProcess.disabled = false;
    btnProcess.textContent = "Process";
});

// ── API ──
async function uploadBlob(blob, name) {
    processingBar.style.display = "flex";
    resultsSection.style.display = "block";
    const fd = new FormData();
    fd.append("file", blob, name);
    try {
        const r = await fetch("/api/transcribe", { method: "POST", body: fd });
        if (!r.ok) throw new Error((await r.json()).error || "Failed");
        displayResults(await r.json());
    } catch (e) {
        textOriginal.innerHTML = `<p style="color:var(--red)">Error: ${esc(e.message)}</p>`;
        textRedacted.innerHTML = "";
    } finally {
        processingBar.style.display = "none";
    }
}

// ── Display Results ──
function displayResults(d) {
    resultsSection.style.display = "block";
    resultsSection.classList.add("fade-in");

    // Metric cards
    $("#mv-latency").textContent = d.latency_ms || "--";
    $("#mv-rtf").textContent = d.rtf != null ? d.rtf.toFixed(2) : "--";
    $("#mv-words").textContent = d.word_count || 0;
    $("#mv-pii").textContent = d.pii_entities ? d.pii_entities.length : 0;
    $("#mv-duration").textContent = d.audio_duration_sec || "--";
    $("#mv-lang").textContent = (d.language || "en").toUpperCase();

    // Original text with PII highlights
    if (d.pii_entities && d.pii_entities.length > 0) {
        textOriginal.innerHTML = highlightPII(d.raw_text, d.pii_entities);
    } else {
        textOriginal.innerHTML = `<p>${esc(d.raw_text) || '<span class="placeholder-text">No speech detected</span>'}</p>`;
    }

    // Redacted
    textRedacted.innerHTML = `<p>${esc(d.redacted_text) || '<span class="placeholder-text">No speech detected</span>'}</p>`;

    // PII chips
    if (d.pii_entities && d.pii_entities.length) {
        piiDetails.style.display = "block";
        entityList.innerHTML = d.pii_entities.map((e) =>
            `<div class="entity-chip">
                <span class="chip-type" style="color:var(--pii-${typeVar(e.type)})">${e.type}</span>
                <span class="chip-value">${esc(e.text)}</span>
                <span class="chip-score">${(e.score*100).toFixed(0)}%</span>
            </div>`
        ).join("");
    } else {
        piiDetails.style.display = "none";
    }

    // Session
    if (d.session) {
        $("#sv-total").textContent = d.session.total_transcriptions;
        $("#sv-pii").textContent = d.session.total_pii_detected;
        $("#sv-latency").textContent = d.session.avg_latency_ms + " ms";
    }
}

function highlightPII(text, entities) {
    if (!entities || !entities.length) return `<p>${esc(text)}</p>`;
    
    // Sort entities by start index
    const sorted = [...entities].sort((a, b) => a.start - b.start);
    
    let html = "";
    let lastIdx = 0;
    
    for (const e of sorted) {
        // Skip if this entity overlaps with the previous one
        if (e.start < lastIdx) continue;
        
        // Add text before the entity
        html += esc(text.substring(lastIdx, e.start));
        
        // Add the highlighted entity
        const entityText = text.substring(e.start, e.end);
        html += `<span class="pii-tag" data-type="${e.type}" title="${e.type} (${(e.score*100).toFixed(0)}%)">${esc(entityText)}</span>`;
        
        lastIdx = e.end;
    }
    
    // Add remaining text
    html += esc(text.substring(lastIdx));
    
    return `<p>${html}</p>`;
}

function typeVar(t) {
    return { PERSON:"person", PHONE_NUMBER:"phone", EMAIL_ADDRESS:"email", CREDIT_CARD:"credit",
             US_SSN:"ssn", LOCATION:"location", DATE_TIME:"datetime", IP_ADDRESS:"ip", 
             BROKEN_EMAIL:"email", GOVT_ID:"ssn", SPELLED_NUM:"credit",
             ZIP_CODE:"location", ACCOUNT_NUM:"ssn" }[t] || "person";
}

function esc(s) {
    if (!s) return "";
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
}

// ── Benchmark ──
btnBenchmark.addEventListener("click", async () => {
    const maxFiles = parseInt(benchMaxFiles.value) || 20;
    btnBenchmark.disabled = true;
    benchStatus.style.display = "flex";
    benchResults.style.display = "none";
    $("#bench-status-text").textContent = `Running benchmark on ${maxFiles} utterances...`;

    try {
        const r = await fetch(`/api/benchmark?max_files=${maxFiles}`, { method: "POST" });
        if (!r.ok) throw new Error((await r.json()).error || "Benchmark failed");
        const d = await r.json();
        showBenchmark(d);
    } catch (e) {
        $("#bench-status-text").textContent = `Error: ${e.message}`;
    } finally {
        btnBenchmark.disabled = false;
        benchStatus.style.display = "none";
    }
});

function showBenchmark(d) {
    benchResults.style.display = "block";
    benchResults.classList.add("fade-in");

    const pct = (v) => v != null ? (v * 100).toFixed(2) + "%" : "--";
    const sign = (v) => v != null ? ((v > 0 ? "+" : "") + (v * 100).toFixed(2) + "%") : "--";

    $("#bv-wer").textContent = pct(d.corpus_wer);
    $("#bv-cer").textContent = pct(d.corpus_cer);
    $("#bv-baseline").textContent = pct(d.paper_baseline_wer);
    $("#bv-delta").textContent = sign(d.delta_vs_paper);
    $("#bv-latency").textContent = d.avg_latency_ms ? d.avg_latency_ms + "ms" : "--";
    $("#bv-utterances").textContent = d.num_utterances || "--";

    // Color delta
    if (d.delta_vs_paper != null) {
        const el = $("#bv-delta");
        el.style.color = d.delta_vs_paper <= 0.01 ? "var(--green)" : d.delta_vs_paper <= 0.03 ? "var(--amber)" : "var(--red)";
    }

    $("#bd-wer-mean").textContent = pct(d.mean_wer);
    $("#bd-wer-std").textContent = pct(d.std_wer);
    $("#bd-wer-median").textContent = pct(d.median_wer);
    $("#bd-wer-min").textContent = pct(d.min_wer);
    $("#bd-wer-max").textContent = pct(d.max_wer);
    $("#bd-total-time").textContent = d.total_time_sec ? d.total_time_sec + "s" : "--";
    $("#bd-avg-lat").textContent = d.avg_latency_ms ? d.avg_latency_ms + "ms" : "--";
    $("#bd-model").textContent = d.model_size || "base";
    $("#bd-cer-mean").textContent = pct(d.mean_cer);
    $("#bd-cer-std").textContent = pct(d.std_cer);
}
