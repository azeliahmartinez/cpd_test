// load shared partials, then activate icons and set active link
(async function injectPartials() {
  const slots = document.querySelectorAll('[data-include]');
  await Promise.all(Array.from(slots).map(async el => {
    const url = el.getAttribute('data-include');
    const html = await fetch(url).then(r => r.text());
    el.outerHTML = html;
  }));
  if (window.feather) feather.replace();

  const path = (window.location.pathname.split('/').pop() || 'dashboard.html');
  document.querySelectorAll('.ef-sidebar a').forEach(link => {
    const href = link.getAttribute('href');
    if (href && href.endsWith(path)) link.classList.add('active');
  });
})();

let selectedFile = null;

document.addEventListener('DOMContentLoaded', () => {
  const dz = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const status = document.getElementById('uploadStatus');
  const progressBar = document.getElementById('progressBar');

  const dateEl  = document.getElementById('videoDate');
  const titleEl = document.getElementById('videoTitle');

  // Restore known details if present
  const storedDate   = localStorage.getItem('uploadDate');
  const storedTitle  = localStorage.getItem('videoTitle');
  const storedName   = localStorage.getItem('savedName');

  if (storedDate && dateEl)  dateEl.textContent = `Uploaded ${storedDate}`;
  if (storedTitle && titleEl) titleEl.textContent = storedTitle;

  // ---------- Upload page logic ----------
  if (dz && fileInput) {
    dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
    dz.addEventListener('drop', e => {
      e.preventDefault();
      dz.classList.remove('drag');
      handleFiles(e.dataTransfer.files);
    });
    fileInput.addEventListener('change', e => handleFiles(e.target.files));
  }

  function handleFiles(files) {
    if (!files || !files[0]) return;
    selectedFile = files[0];
    if (status) status.textContent = `Selected: ${selectedFile.name}`;
    if (analyzeBtn) analyzeBtn.disabled = false;
  }

  // ✅ NEW: Do upload -> THEN run /api/analyze (with nice progress + messages) -> THEN redirect
  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async () => {
      if (!selectedFile) return;

      analyzeBtn.disabled = true;
      animateProgress(0);
      setStatus('Uploading video…');

      // 1) Upload
      const fd = new FormData();
      fd.append('video', selectedFile);

      let uploadRes;
      try {
        uploadRes = await fetch('/api/upload', { method: 'POST', body: fd }).then(r => r.json());
      } catch (e) {
        setStatus('Upload failed. Please try again.');
        analyzeBtn.disabled = false;
        return;
      }

      if (!uploadRes?.ok) {
        setStatus('Upload failed. Please try again.');
        analyzeBtn.disabled = false;
        return;
      }

      // Save basics for analysis page header
      if (uploadRes.uploadDate) localStorage.setItem('uploadDate', uploadRes.uploadDate);
      if (uploadRes.filename)   localStorage.setItem('videoTitle', uploadRes.filename);
      if (uploadRes.url)        localStorage.setItem('videoUrl', uploadRes.url);
      if (uploadRes.savedName)  localStorage.setItem('savedName', uploadRes.savedName);

      animateProgress(35);
      setStatus('Analyzing video…');

      // 2) Analyze (call Python via backend)
      let analyzeRes;
      try {
        // small staged progress updates while waiting
        const progTimer = stagedProgress([55, 72, 88], [700, 1200, 1600]);

        analyzeRes = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ savedName: uploadRes.savedName, nFrames: 5 })
        }).then(r => r.json());

        clearTimeouts(progTimer);
      } catch (e) {
        setStatus('Analysis failed. Please try again.');
        analyzeBtn.disabled = false;
        return;
      }

      if (!analyzeRes?.ok || !analyzeRes.data) {
        setStatus('Analysis failed. Please try again.');
        analyzeBtn.disabled = false;
        return;
      }

      setStatus('Predicting engagement…');
      animateProgress(95);

      // 3) Store analysis for immediate render on analysis.html
      try {
        localStorage.setItem('analysisData', JSON.stringify(analyzeRes.data));
      } catch (_) { /* ignore quota errors */ }

      setStatus('Done! Opening results…');
      animateProgress(100);
      setTimeout(() => {
        window.location.href = '/analysis.html';
      }, 300);
    });
  }

  // ---------- Analysis page logic ----------
  const donutCanvas = document.getElementById('donutChart');

  // If we already have results (because we analyzed BEFORE redirect), render them immediately.
  const cached = safeParse(localStorage.getItem('analysisData'));
  if (donutCanvas && cached) {
    renderAnalysis(cached);
  } else if (donutCanvas && storedName) {
    // Fallback: if user navigates here directly, run analyze now
    runAnalyze(storedName);
  }

  const extractedFramesBtn = document.getElementById('extractedFramesBtn');
  if (extractedFramesBtn) {
    extractedFramesBtn.addEventListener('click', async () => {
      const saved = localStorage.getItem('savedName');
      if (!saved) return alert('No uploaded video found.');
      await runAnalyze(saved);
    });
  }

  const downloadBtn = document.getElementById('downloadCSVBtn');
  if (downloadBtn) {
    downloadBtn.addEventListener('click', () => alert('Download stub. Connect to backend to enable.'));
  }

  // helpers
  function setStatus(msg) {
    if (status) status.textContent = msg;
  }

  function animateProgress(target) {
    const bar = document.getElementById('progressBar');
    if (!bar) return;
    bar.style.width = `${target}%`;
  }

  function stagedProgress(targets = [], delays = []) {
    const tids = [];
    targets.forEach((t, i) => {
      const id = setTimeout(() => animateProgress(t), delays[i] || 800);
      tids.push(id);
    });
    return tids;
  }

  function clearTimeouts(ids = []) {
    ids.forEach(id => clearTimeout(id));
  }
});

// ---- shared analysis renderers ----
async function runAnalyze(savedName) {
  const body = { savedName, nFrames: 5 };
  const res = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  }).then(r => r.json()).catch(() => null);

  if (!res?.ok || !res.data) return;
  try { localStorage.setItem('analysisData', JSON.stringify(res.data)); } catch (_) {}
  renderAnalysis(res.data);
}

function renderAnalysis(data) {
  // Update headers
  const titleEl = document.getElementById('videoTitle');
  const dateEl  = document.getElementById('videoDate');
  if (titleEl) titleEl.textContent = data.title || localStorage.getItem('videoTitle') || 'Uploaded Video';
  if (dateEl)  dateEl.textContent  = data.date ? `Uploaded ${data.date}` : (localStorage.getItem('uploadDate') || '');

  // Engagement Overview
  const scoreEl = document.getElementById('engagementScore');
  const labelEl = document.getElementById('engagementLabel');
  if (scoreEl && typeof data.engagementIndex === 'number') {
    scoreEl.textContent = String(data.engagementIndex);
  }
  if (labelEl && data.engagementLabel) {
    labelEl.textContent = data.engagementLabel;
  }

  // Recommendations
  const recoList = document.getElementById('recoList');
  if (recoList && Array.isArray(data.recommendations)) {
    recoList.innerHTML = '';
    data.recommendations.forEach(t => {
      const li = document.createElement('li');
      li.textContent = t;
      recoList.appendChild(li);
    });
  }

  // Donut + Legend from RF probabilities
  const probs = data.probabilities || {};
  drawDonut(probs);
  drawProbLegend(probs);
}

function drawDonut(states = {}) {
  const c = document.getElementById('donutChart');
  if (!c) return;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);

  const labels = Object.keys(states);
  const vals = Object.values(states);
  const total = vals.reduce((a, b) => a + b, 0) || 1;
  const colors = ['#12865C', '#17A673', '#8FD6B5', '#C8E8DA'];
  let angle = -Math.PI / 2;
  const cx = c.width / 2, cy = c.height / 2, r = Math.min(cx, cy) - 20;

  vals.forEach((v, i) => {
    const slice = (v / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.fillStyle = colors[i % colors.length];
    ctx.arc(cx, cy, r, angle, angle + slice);
    ctx.closePath();
    ctx.fill();
    angle += slice;
  });

  ctx.globalCompositeOperation = 'destination-out';
  ctx.beginPath();
  ctx.arc(cx, cy, r * 0.6, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalCompositeOperation = 'source-over';
}

function drawProbLegend(states = {}) {
  const container = document.getElementById('probLegend');
  if (!container) return;
  container.innerHTML = '';

  const colors = ['#12865C', '#17A673', '#8FD6B5', '#C8E8DA'];
  const labels = Object.keys(states);
  const vals = Object.values(states);

  labels.forEach((label, i) => {
    const row = document.createElement('div');
    row.style.display = 'flex';
    row.style.alignItems = 'center';
    row.style.gap = '8px';
    row.style.marginTop = '6px';

    const swatch = document.createElement('span');
    swatch.style.display = 'inline-block';
    swatch.style.width = '10px';
    swatch.style.height = '10px';
    swatch.style.borderRadius = '3px';
    swatch.style.background = colors[i % colors.length];

    const text = document.createElement('span');
    text.className = 'muted small';
    const pct = (Math.round((vals[i] || 0) * 10) / 10).toFixed(1);
    text.textContent = `${label}: ${pct}%`;

    row.appendChild(swatch);
    row.appendChild(text);
    container.appendChild(row);
  });
}

function safeParse(s) {
  try { return JSON.parse(s); } catch { return null; }
}
