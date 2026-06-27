/* ═══════════════════════════════════════════════════════════════════
   AI Knowledge Assistant — Client Script
   ═══════════════════════════════════════════════════════════════════ */

// ── DOM refs ──────────────────────────────────────────────────────
const form        = document.getElementById("chat-form");
const input       = document.getElementById("message-input");
const messages    = document.getElementById("messages");
const thinking    = document.getElementById("thinking");
const sendBtn     = document.getElementById("send-btn");
const fileInput   = document.getElementById("file-input");
const dropZone    = document.getElementById("drop-zone");
const browseBtn   = document.getElementById("browse-btn");
const fileList    = document.getElementById("file-list");
const uploadBtn   = document.getElementById("upload-btn");
const ingestBtn   = document.getElementById("ingest-btn");
const statusBar   = document.getElementById("status-bar");
const docCount    = document.getElementById("doc-count");

// ── State ─────────────────────────────────────────────────────────
let currentMode   = "chat";          // "chat" | "rag" | "graph"
let selectedFiles = [];              // File objects queued for upload
let isIngested    = false;           // Whether documents have been ingested

const MODE_ENDPOINTS = {
	chat:  "/chat",
	rag:   "/rag-chat",
	graph: "/graph-chat",
};

// ── Init ──────────────────────────────────────────────────────────
refreshDocCount();

// ── Mode Tabs ─────────────────────────────────────────────────────
document.querySelectorAll(".mode-tab").forEach((tab) => {
	tab.addEventListener("click", () => switchToMode(tab.dataset.mode));
});

function switchToMode(mode) {
	document.querySelectorAll(".mode-tab").forEach((t) => {
		const isActive = t.dataset.mode === mode;
		t.classList.toggle("active", isActive);
		t.setAttribute("aria-selected", String(isActive));
	});
	currentMode = mode;
}

// ── File Selection (browse + drag-and-drop) ───────────────────────
browseBtn.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("click", (e) => {
	if (e.target !== browseBtn) fileInput.click();
});

fileInput.addEventListener("change", () => {
	addFiles(Array.from(fileInput.files));
	fileInput.value = "";
});

dropZone.addEventListener("dragover", (e) => {
	e.preventDefault();
	dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
	dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
	e.preventDefault();
	dropZone.classList.remove("drag-over");
	const files = Array.from(e.dataTransfer.files).filter((f) =>
		f.name.toLowerCase().endsWith(".pdf")
	);
	addFiles(files);
});

function addFiles(files) {
	for (const file of files) {
		if (!file.name.toLowerCase().endsWith(".pdf")) continue;
		if (selectedFiles.some((f) => f.name === file.name)) continue;
		selectedFiles.push(file);
	}
	renderFileList();
}

function removeFile(index) {
	selectedFiles.splice(index, 1);
	renderFileList();
}

function renderFileList() {
	fileList.innerHTML = "";
	selectedFiles.forEach((file, idx) => {
		const item = document.createElement("div");
		item.className = "file-item";
		const sizeKB = (file.size / 1024).toFixed(1);
		item.innerHTML = `
			<span class="file-icon">
				<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
			</span>
			<span class="file-name">${file.name}</span>
			<span class="file-size">${sizeKB} KB</span>
			<button type="button" class="remove-btn" title="Remove">&times;</button>
		`;
		item.querySelector(".remove-btn").addEventListener("click", () => removeFile(idx));
		fileList.appendChild(item);
	});
	uploadBtn.disabled = selectedFiles.length === 0;
}

// ── Upload (auto-chains into ingest) ──────────────────────────────
uploadBtn.addEventListener("click", async () => {
	if (selectedFiles.length === 0) return;

	uploadBtn.disabled = true;
	ingestBtn.disabled = true;
	showStatus("⬆ Uploading…", "info");

	const formData = new FormData();
	selectedFiles.forEach((f) => formData.append("files", f));

	try {
		const res = await fetch("/upload", { method: "POST", body: formData });
		const data = await res.json();

		if (data.saved && data.saved.length > 0) {
			selectedFiles = [];
			renderFileList();
			refreshDocCount();

			// ── Auto-ingest after successful upload ──
			showStatus(`✓ Uploaded ${data.saved.length} file(s). Now ingesting into knowledge base…`, "info");
			await triggerIngest();
		} else {
			showStatus(data.errors?.join("; ") || "No files were saved.", "error");
		}
	} catch (err) {
		showStatus("Upload failed: " + err.message, "error");
	} finally {
		uploadBtn.disabled = selectedFiles.length === 0;
		ingestBtn.disabled = false;
	}
});

// ── Ingest (shared logic) ─────────────────────────────────────────
async function triggerIngest() {
	try {
		const res = await fetch("/ingest", { method: "POST" });
		const data = await res.json();

		if (data.status === "success") {
			isIngested = true;
			showStatus(`✓ ${data.message} You're now in RAG mode — ask away!`, "success");
			switchToMode("rag");
		} else if (data.status === "warning") {
			showStatus(`⚠ ${data.message}`, "warning");
		} else {
			showStatus(`✗ ${data.message}`, "error");
		}
	} catch (err) {
		showStatus("Ingestion failed: " + err.message, "error");
	}
}

ingestBtn.addEventListener("click", async () => {
	ingestBtn.disabled = true;
	showStatus("Ingesting documents… This may take a minute.", "info");
	await triggerIngest();
	ingestBtn.disabled = false;
});

// ── Status bar helper ─────────────────────────────────────────────
function showStatus(message, type) {
	statusBar.hidden = false;
	statusBar.textContent = message;
	statusBar.className = "status-bar " + type;
}

// ── Document count ────────────────────────────────────────────────
async function refreshDocCount() {
	try {
		const res = await fetch("/documents");
		const data = await res.json();
		docCount.textContent = `${data.count} doc${data.count !== 1 ? "s" : ""}`;
	} catch {
		docCount.textContent = "? docs";
	}
}

// ── Chat ──────────────────────────────────────────────────────────
function addMessage(text, role, mode, sources) {
	const wrapper = document.createElement("div");
	wrapper.className = `message ${role}`;

	let html = "";

	if (role === "bot" && mode) {
		html += `<div class="mode-indicator ${mode}">${mode === "graph" ? "GraphRAG" : mode.toUpperCase()}</div>`;
	}

	html += `<div class="message-content">${escapeHtml(text)}</div>`;

	if (role === "bot" && sources && sources.length > 0) {
		html += `<div class="source-badges">`;
		sources.forEach((src) => {
			const name = src.split(/[/\\]/).pop();
			html += `<span class="source-badge" title="${escapeHtml(src)}">${escapeHtml(name)}</span>`;
		});
		html += `</div>`;
	}

	wrapper.innerHTML = html;
	messages.appendChild(wrapper);
	messages.scrollTop = messages.scrollHeight;
}

function escapeHtml(str) {
	const div = document.createElement("div");
	div.textContent = str;
	return div.innerHTML;
}

function setThinking(isThinking) {
	thinking.classList.toggle("visible", isThinking);
	thinking.setAttribute("aria-hidden", String(!isThinking));
	input.disabled = isThinking;
	sendBtn.disabled = isThinking;
}

form.addEventListener("submit", async (event) => {
	event.preventDefault();

	const userMessage = input.value.trim();
	if (!userMessage) return;

	// Hint: if user is in Chat mode but docs are ingested, nudge them
	if (currentMode === "chat" && isIngested) {
		addMessage(userMessage, "user");
		addMessage(
			"💡 You're in Chat mode (no document context). Switch to RAG or GraphRAG to query your uploaded documents.",
			"bot", "chat", []
		);
		input.value = "";
		return;
	}

	addMessage(userMessage, "user");
	input.value = "";
	setThinking(true);

	const endpoint = MODE_ENDPOINTS[currentMode];

	try {
		const response = await fetch(endpoint, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ message: userMessage }),
		});

		const data = await response.json();
		addMessage(
			data.response || "No response returned.",
			"bot",
			data.mode || currentMode,
			data.sources || []
		);
	} catch (error) {
		addMessage("Unable to reach the backend. Is the server running?", "bot", currentMode, []);
	} finally {
		setThinking(false);
	}
});
