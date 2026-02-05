/**
 * scurl API Web Interface
 */

const form = document.getElementById('fetch-form');
const urlInput = document.getElementById('url-input');
const pasteInput = document.getElementById('paste-input');
const submitBtn = document.getElementById('submit-btn');
const pasteSubmitBtn = document.getElementById('paste-submit-btn');
const loading = document.getElementById('loading');
const errorContainer = document.getElementById('error');
const errorMessage = document.getElementById('error-message');
const resultContainer = document.getElementById('result');
const resultMeta = document.getElementById('result-meta');
const resultContent = document.getElementById('result-content');
const copyBtn = document.getElementById('copy-btn');
const downloadBtn = document.getElementById('download-btn');
const copyFeedback = document.getElementById('copy-feedback');
const renderCheckbox = document.getElementById('render-checkbox');
const injectionStatus = document.getElementById('injection-status');
const tabs = document.querySelectorAll('.tab');
const tabPanels = document.querySelectorAll('.tab-panel');

let activeTab = 'url';
let currentSource = '';
let currentMarkdown = '';

// Tab switching
tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    const target = tab.dataset.tab;
    activeTab = target;
    tabs.forEach(t => t.classList.toggle('active', t.dataset.tab === target));
    tabPanels.forEach(p => p.classList.toggle('active', p.id === `tab-${target}`));
    if (target === 'url') urlInput.focus();
    else pasteInput.focus();
  });
});

function setSubmitting(busy) {
  loading.classList.toggle('visible', busy);
  errorContainer.classList.remove('visible');
  if (busy) resultContainer.classList.remove('visible');
  submitBtn.disabled = busy;
  pasteSubmitBtn.disabled = busy;
}

function showError(message) {
  setSubmitting(false);
  errorMessage.textContent = message;
  errorContainer.classList.add('visible');
  resultContainer.classList.remove('visible');
}

function showResult(source, markdown, injection) {
  setSubmitting(false);
  currentSource = source;
  currentMarkdown = markdown;
  resultMeta.textContent = `Source: ${source}`;
  resultContent.textContent = markdown;

  // Show injection analysis badge
  if (injection) {
    const pct = (injection.score * 100).toFixed(1);
    if (injection.flagged) {
      const signals = injection.signals.length ? injection.signals.join(', ') : 'semantic';
      injectionStatus.innerHTML =
        `<span class="injection-badge flagged">Prompt injection detected (score: ${pct}%, signals: ${signals})</span>`;
    } else {
      injectionStatus.innerHTML =
        `<span class="injection-badge clean">No injection detected (score: ${pct}%)</span>`;
    }
    injectionStatus.style.display = 'block';
  } else {
    injectionStatus.style.display = 'none';
  }

  errorContainer.classList.remove('visible');
  resultContainer.classList.add('visible');
}

function showCopyFeedback() {
  copyFeedback.classList.add('visible');
  setTimeout(() => copyFeedback.classList.remove('visible'), 2000);
}

async function apiCall(endpoint, body) {
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  const data = await response.json();

  if (!response.ok) {
    if (response.status === 429) {
      const retryAfter = data.retryAfter || 60;
      throw new Error(`Rate limit exceeded. Please wait ${retryAfter} seconds.`);
    }
    throw new Error(data.error || 'An unknown error occurred');
  }

  return data;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  if (activeTab === 'url') {
    const url = urlInput.value.trim();
    if (!url) return;
    setSubmitting(true);
    try {
      const data = await apiCall('/api/fetch', { url, render: renderCheckbox.checked });
      showResult(data.url, data.markdown, data.injection);
    } catch (error) {
      showError(error.message);
    }
  } else {
    const html = pasteInput.value.trim();
    if (!html) return;
    setSubmitting(true);
    try {
      const data = await apiCall('/api/convert', { html });
      showResult('pasted content', data.markdown, data.injection);
    } catch (error) {
      showError(error.message);
    }
  }
});

copyBtn.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(currentMarkdown);
    showCopyFeedback();
  } catch {
    const textarea = document.createElement('textarea');
    textarea.value = currentMarkdown;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
    showCopyFeedback();
  }
});

downloadBtn.addEventListener('click', () => {
  let filename = 'converted.md';
  try {
    filename = new URL(currentSource).hostname.replace(/\./g, '_') + '.md';
  } catch {}
  const blob = new Blob([currentMarkdown], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

urlInput.focus();

// Code example tabs
const codeTabs = document.querySelectorAll('.code-tab');
const codePanels = document.querySelectorAll('.code-panel');

codeTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    const lang = tab.dataset.lang;
    codeTabs.forEach(t => t.classList.toggle('active', t.dataset.lang === lang));
    codePanels.forEach(p => p.classList.toggle('active', p.dataset.lang === lang));
  });
});

// Copy code buttons
document.querySelectorAll('.copy-code-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    const code = btn.parentElement.querySelector('code').textContent;
    try {
      await navigator.clipboard.writeText(code);
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => {
        btn.textContent = 'Copy';
        btn.classList.remove('copied');
      }, 2000);
    } catch {
      // Fallback
      const textarea = document.createElement('textarea');
      textarea.value = code;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => {
        btn.textContent = 'Copy';
        btn.classList.remove('copied');
      }, 2000);
    }
  });
});
