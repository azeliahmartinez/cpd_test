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

  const storedDate   = localStorage.getItem('uploadDate');
  const storedTitle  = localStorage.getItem('videoTitle');
  const storedName   = localStorage.getItem('savedName');   // <- we’ll use this on analysis page

  if (storedDate && dateEl)  dateEl.textContent = `Uploaded ${storedDate}`;
  if (storedTitle && titleEl) titleEl.textContent = storedTitle;

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

  // ---- Upload -> store savedName for analysis ----
  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async () => {
      if (!selectedFile) return;

      if (progressBar) {
        progressBar.style.width = '20%';
        setTimeout(() => progressBar.style.width = '60%', 350);
        setTimeout(() => progressBar.style.width = '100%', 800);
      }

      const fd = new FormData();
      fd.append('video', selectedFile);

      const res = await fetch('/api/upload', { method: 'POST', body: fd }).then(r => r.json());

      if (res.ok) {
        if (res.uploadDate) localStorage.setItem('uploadDate', res.uploadDate);
        if (res.filename)   localStorage.setItem('videoTitle', res.filename);
        if (res.url)        localStorage.setItem('videoUrl', res.url);
        if (res.savedName)  localStorage.setItem('savedName', res.savedName); // <- important
      }

      window.location.href = '/analysis.html';
    });
  }

  // ---- ANALYSIS PAGE: auto-run analyze and draw donut ----
  const donutCanvas = document.getElementById('donutChart');
  if (donutCanvas && storedName) {
    runAnalyze(storedName);
  }

  const extractedFramesBtn = document.getElementById('extractedFramesBtn');
  if (extractedFramesBtn) {
    extractedFramesBtn.addEventListener('click', async () => {
      if (!storedName) return alert('No uploaded video found.');
      await runAnalyze(storedName);
    });
  }

  const downloadBtn = document.getElementById('downloadCSVBtn');
  if (downloadBtn) {
    downloadBtn.addEventListener('click', () => alert('Download stub. Connect to backend to enable.'));
  }
});

async function runAnalyze(savedName) {
  const body = { savedName, nFrames: 5 };
  const res = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  }).then(r => r.json());

  if (!res?.ok || !res.data) {
    console.error(res);
    return;
  }

  // Update headers
  const titleEl = document.getElementById('videoTitle');
  const dateEl  = document.getElementById('videoDate');
  if (titleEl) titleEl.textContent = res.data.title || localStorage.getItem('videoTitle') || 'Uploaded Video';
  if (dateEl)  dateEl.textContent  = res.data.date ? `Uploaded ${res.data.date}` : (localStorage.getItem('uploadDate') || '');

  // Engagement Overview (index + label)
  const scoreEl = document.getElementById('engagementScore');
  const labelEl = document.getElementById('engagementLabel');
  if (scoreEl && typeof res.data.engagementIndex === 'number') {
    scoreEl.textContent = String(res.data.engagementIndex);
  }
  if (labelEl && res.data.engagementLabel) {
    labelEl.textContent = res.data.engagementLabel;
  }

  // Recommendations
  const recoList = document.getElementById('recoList');
  if (recoList && Array.isArray(res.data.recommendations)) {
    recoList.innerHTML = '';
    res.data.recommendations.forEach(t => {
      const li = document.createElement('li');
      li.textContent = t;
      recoList.appendChild(li);
    });
  }

  // Donut + Legend from RF probabilities
  const probs = res.data.probabilities || {};
  drawDonut(probs);
  drawProbLegend(probs);
}

// Donut expects an object of { label: percent, ... }
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

// Compact legend under the donut
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

