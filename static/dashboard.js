/* ═══════════════════════════════════════════════════════════════════
   AI Knowledge Assistant — Dashboard Script
   ═══════════════════════════════════════════════════════════════════ */

// ── DOM refs ──────────────────────────────────────────────────────
const totalChats   = document.getElementById("total-chats");
const totalDocs    = document.getElementById("total-docs");
const totalChunks  = document.getElementById("total-chunks");
const ingestedDocs = document.getElementById("ingested-docs");
const docsBody     = document.getElementById("docs-body");
const historyBody  = document.getElementById("history-body");
const modeChart    = document.getElementById("mode-chart");
const topQuestions = document.getElementById("top-questions");
const refreshBtn   = document.getElementById("refresh-btn");

// ── Mode colours ──────────────────────────────────────────────────
const MODE_COLORS = {
	chat:  { bg: "rgba(56, 189, 248, 0.18)", fg: "#38bdf8", label: "Chat" },
	rag:   { bg: "rgba(52, 211, 153, 0.18)", fg: "#34d399", label: "RAG" },
	graph: { bg: "rgba(167, 139, 250, 0.18)", fg: "#a78bfa", label: "GraphRAG" },
};

// ── Init ──────────────────────────────────────────────────────────
loadDashboard();
refreshBtn.addEventListener("click", loadDashboard);

async function loadDashboard() {
	refreshBtn.disabled = true;
	await Promise.all([loadAnalytics(), loadDocuments(), loadHistory()]);
	refreshBtn.disabled = false;
}

// ── Analytics ─────────────────────────────────────────────────────
async function loadAnalytics() {
	try {
		const res = await fetch("/api/analytics");
		const data = await res.json();

		// Stat cards
		animateNumber(totalChats, data.total_chats);
		animateNumber(totalDocs, data.total_documents);
		animateNumber(totalChunks, data.total_chunks);
		animateNumber(ingestedDocs, data.ingested_documents);

		// Chats per mode bar chart
		renderModeChart(data.chats_per_mode);

		// Top questions
		renderTopQuestions(data.top_questions);
	} catch (err) {
		console.error("Failed to load analytics:", err);
	}
}

// ── Documents table ───────────────────────────────────────────────
async function loadDocuments() {
	try {
		const res = await fetch("/api/documents-db");
		const data = await res.json();

		if (!data.documents || data.documents.length === 0) {
			docsBody.innerHTML = '<tr><td colspan="5" class="empty-row">No documents uploaded yet.</td></tr>';
			return;
		}

		docsBody.innerHTML = data.documents.map((doc) => {
			const size = formatSize(doc.file_size);
			const status = statusBadge(doc.status);
			const date = formatDate(doc.uploaded_at);
			return `<tr>
				<td class="cell-filename">${escapeHtml(doc.filename)}</td>
				<td>${size}</td>
				<td>${status}</td>
				<td>${doc.chunks}</td>
				<td>${date}</td>
			</tr>`;
		}).join("");
	} catch (err) {
		docsBody.innerHTML = '<tr><td colspan="5" class="empty-row">Failed to load documents.</td></tr>';
	}
}

// ── Chat history table ────────────────────────────────────────────
async function loadHistory() {
	try {
		const res = await fetch("/api/chat-history?limit=100");
		const data = await res.json();

		if (!data.history || data.history.length === 0) {
			historyBody.innerHTML = '<tr><td colspan="4" class="empty-row">No chats yet. Start a conversation!</td></tr>';
			return;
		}

		historyBody.innerHTML = data.history.map((chat) => {
			const q = truncate(chat.question, 60);
			const a = truncate(chat.answer, 80);
			const mode = modeBadge(chat.mode);
			const time = formatDate(chat.created_at);
			return `<tr>
				<td class="cell-question" title="${escapeHtml(chat.question)}">${escapeHtml(q)}</td>
				<td class="cell-answer" title="${escapeHtml(chat.answer)}">${escapeHtml(a)}</td>
				<td>${mode}</td>
				<td class="cell-time">${time}</td>
			</tr>`;
		}).join("");
	} catch (err) {
		historyBody.innerHTML = '<tr><td colspan="4" class="empty-row">Failed to load chat history.</td></tr>';
	}
}

// ── Bar chart (CSS-based) ─────────────────────────────────────────
function renderModeChart(chatsPerMode) {
	if (!chatsPerMode || Object.keys(chatsPerMode).length === 0) {
		modeChart.innerHTML = '<p class="empty-row">No chat data yet.</p>';
		return;
	}

	const maxVal = Math.max(...Object.values(chatsPerMode), 1);

	modeChart.innerHTML = Object.entries(chatsPerMode).map(([mode, count]) => {
		const colors = MODE_COLORS[mode] || { bg: "rgba(148,163,184,0.18)", fg: "#94a3b8", label: mode };
		const pct = Math.max((count / maxVal) * 100, 4);
		return `
			<div class="bar-row">
				<span class="bar-label">${colors.label}</span>
				<div class="bar-track">
					<div class="bar-fill" style="width:${pct}%; background:${colors.fg};">${count}</div>
				</div>
			</div>`;
	}).join("");
}

// ── Top questions list ────────────────────────────────────────────
function renderTopQuestions(questions) {
	if (!questions || questions.length === 0) {
		topQuestions.innerHTML = '<p class="empty-row">No questions asked yet.</p>';
		return;
	}

	topQuestions.innerHTML = questions.map((q, i) => `
		<div class="question-row">
			<span class="question-rank">${i + 1}</span>
			<span class="question-text" title="${escapeHtml(q.question)}">${escapeHtml(truncate(q.question, 70))}</span>
			<span class="question-count">${q.count}×</span>
		</div>
	`).join("");
}

// ── Helpers ───────────────────────────────────────────────────────
function escapeHtml(str) {
	const div = document.createElement("div");
	div.textContent = str;
	return div.innerHTML;
}

function truncate(str, len) {
	if (!str) return "";
	return str.length > len ? str.slice(0, len) + "…" : str;
}

function formatSize(bytes) {
	if (!bytes || bytes === 0) return "—";
	if (bytes < 1024) return bytes + " B";
	if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
	return (bytes / 1048576).toFixed(1) + " MB";
}

function formatDate(dateStr) {
	if (!dateStr) return "—";
	try {
		const d = new Date(dateStr);
		return d.toLocaleDateString("en-IN", {
			day: "2-digit", month: "short", year: "numeric",
			hour: "2-digit", minute: "2-digit",
		});
	} catch {
		return dateStr;
	}
}

function statusBadge(status) {
	const map = {
		uploaded: { cls: "badge-amber", label: "Uploaded" },
		ingested: { cls: "badge-green", label: "Ingested" },
		failed:   { cls: "badge-red",   label: "Failed" },
	};
	const s = map[status] || { cls: "badge-amber", label: status };
	return `<span class="status-badge ${s.cls}">${s.label}</span>`;
}

function modeBadge(mode) {
	const colors = MODE_COLORS[mode] || { bg: "rgba(148,163,184,0.18)", fg: "#94a3b8", label: mode };
	return `<span class="mode-badge" style="background:${colors.bg}; color:${colors.fg};">${colors.label}</span>`;
}

function animateNumber(el, target) {
	const start = parseInt(el.textContent) || 0;
	if (start === target) { el.textContent = target; return; }
	const duration = 600;
	const startTime = performance.now();

	function step(now) {
		const progress = Math.min((now - startTime) / duration, 1);
		const eased = 1 - Math.pow(1 - progress, 3);
		el.textContent = Math.round(start + (target - start) * eased);
		if (progress < 1) requestAnimationFrame(step);
	}
	requestAnimationFrame(step);
}
