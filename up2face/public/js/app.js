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
  const storedName   = localStorage.getItem('savedName');

  if (storedDate && dateEl)  dateEl.textContent = `Uploaded ${storedDate}`;
  if (storedTitle && titleEl) titleEl.textContent = storedTitle;

  // Upload page logic 
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

  // Progress controller with % text inside bar
  const Progress = (() => {
    let pct = 0;
    let timer = null;
    const bar = document.getElementById('progressBar');
    const statusEl = document.getElementById('uploadStatus');

    function setPct(newPct) {
      pct = Math.max(0, Math.min(100, newPct));
      if (bar) {
        bar.style.width = `${pct}%`;
        bar.setAttribute('data-pct', `${Math.round(pct)}%`);
      }
      if (statusEl) {
        const label = statusEl.dataset.label || 'Processing';
        statusEl.textContent = `${label}…`;
      }
    }

    function label(text) {
      if (!statusEl) return;
      statusEl.dataset.label = text;
      statusEl.textContent = `${text}…`;
    }

    function startIndeterminate(maxHold = 94, stepMs = 400, stepInc = 1.2) {
      stop();
      timer = setInterval(() => {
        const remaining = Math.max(0, maxHold - pct);
        const inc = Math.max(0.4, Math.min(stepInc, remaining * 0.08));
        setPct(pct + inc);
      }, stepMs);
    }

    function rampTo(target, durationMs = 700) {
      const start = pct;
      const delta = target - start;
      const t0 = performance.now();
      function tick(t) {
        const p = Math.min(1, (t - t0) / durationMs);
        const ease = 1 - Math.pow(1 - p, 3);
        setPct(start + delta * ease);
        if (p < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    }

    function stop() {
      if (timer) { clearInterval(timer); timer = null; }
    }

    function complete() {
      stop();
      rampTo(100, 400);
      if (bar) bar.classList.add('done');
    }

    return { setPct, label, startIndeterminate, rampTo, stop, complete };
  })();

  // Upload -> Analyze (with percentage) -> Redirect
  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async () => {
      if (!selectedFile) return;

      analyzeBtn.disabled = true;

      // Phase 1: Upload
      Progress.label('Uploading video');
      Progress.rampTo(15, 500);

      const fd = new FormData();
      fd.append('video', selectedFile);

      let uploadRes;
      try {
        uploadRes = await fetch('/api/upload', { method: 'POST', body: fd }).then(r => r.json());
      } catch {
        Progress.label('Upload failed');
        analyzeBtn.disabled = false;
        return;
      }

      if (!uploadRes?.ok) {
        Progress.label('Upload failed');
        analyzeBtn.disabled = false;
        return;
      }

      // Save basics for analysis page header
      if (uploadRes.uploadDate) localStorage.setItem('uploadDate', uploadRes.uploadDate);
      if (uploadRes.filename)   localStorage.setItem('videoTitle', uploadRes.filename);
      if (uploadRes.url)        localStorage.setItem('videoUrl', uploadRes.url);
      if (uploadRes.savedName)  localStorage.setItem('savedName', uploadRes.savedName);

      Progress.rampTo(35, 500);

      // Phase 2: Analyze
      Progress.label('Analyzing video');
      Progress.startIndeterminate(94, 350, 1.5);

      let analyzeRes;
      try {
        analyzeRes = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ savedName: uploadRes.savedName, nFrames: 5 })
        }).then(r => r.json());
      } catch {
        Progress.stop();
        Progress.label('Analysis failed');
        analyzeBtn.disabled = false;
        return;
      }

      if (!analyzeRes?.ok || !analyzeRes.data) {
        Progress.stop();
        Progress.label('Analysis failed');
        analyzeBtn.disabled = false;
        return;
      }

      // Phase 3: Predicting / Finishing
      Progress.label('Predicting engagement');
      Progress.rampTo(97, 500);
      Progress.complete();

      try {
        localStorage.setItem('analysisData', JSON.stringify(analyzeRes.data));
      } catch { /* ignore quota */ }

      setTimeout(() => {
        window.location.href = '/analysis.html';
      }, 350);
    });
  }

  // Analysis page logic
  const donutCanvas = document.getElementById('donutChart');

  const cached = safeParse(localStorage.getItem('analysisData'));
  if (donutCanvas && cached) {
    renderAnalysis(cached);
  } else if (donutCanvas && storedName) {
    runAnalyze(storedName);
  }

  // Download functionality
  const extractedFramesBtn = document.getElementById('extractedFramesBtn');
  if (extractedFramesBtn) {
    extractedFramesBtn.addEventListener('click', async () => {
      const videoName = localStorage.getItem('savedName');
      if (!videoName) {
        alert('No uploaded video found.');
        return;
      }

      extractedFramesBtn.disabled = true;
      extractedFramesBtn.innerHTML = '<i data-feather="download"></i> Downloading...';
      if (window.feather) feather.replace();

      try {
        const response = await fetch(`/api/download-frames?video=${encodeURIComponent(videoName)}`);
        
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.error || 'Download failed');
        }

        // Create download link
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `frames_${videoName.replace('.mp4', '')}.zip`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

      } catch (error) {
        console.error('Download error:', error);
        alert(`Download failed: ${error.message}`);
      } finally {
        extractedFramesBtn.disabled = false;
        extractedFramesBtn.innerHTML = '<i data-feather="image"></i> Download Extracted Frames';
        if (window.feather) feather.replace();
      }
    });
  }

  const downloadCSVBtn = document.getElementById('downloadCSVBtn');
  if (downloadCSVBtn) {
    downloadCSVBtn.addEventListener('click', async () => {
      const videoName = localStorage.getItem('savedName');
      if (!videoName) {
        alert('No uploaded video found.');
        return;
      }

      downloadCSVBtn.disabled = true;
      downloadCSVBtn.innerHTML = '<i data-feather="download"></i> Downloading...';
      if (window.feather) feather.replace();

      try {
        const response = await fetch(`/api/download-csv?video=${encodeURIComponent(videoName)}`);
        
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.error || 'Download failed');
        }

        // Create download link
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `landmarks_${videoName.replace('.mp4', '')}.zip`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

      } catch (error) {
        console.error('Download error:', error);
        alert(`Download failed: ${error.message}`);
      } finally {
        downloadCSVBtn.disabled = false;
        downloadCSVBtn.innerHTML = '<i data-feather="download"></i> Download CSVs Data';
        if (window.feather) feather.replace();
      }
    });
  }
});

// shared analysis renderers
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

// function to show exact timestamp in the video playback
function attachCustomTimestamp(video) {
  const label = document.createElement('div');
  label.className = 'timestamp-overlay';
  label.textContent = '00:00.000';
  video.parentElement.style.position = 'relative';
  video.parentElement.appendChild(label);

  video.addEventListener('timeupdate', () => {
    const sec = video.currentTime;
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    const ms = ((sec % 1) * 1000).toFixed(0).padStart(3, '0');
    label.textContent = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${ms}`;
  });
}

// function to load uploaded video
function loadUploadedVideo() {
  const container = document.getElementById('videoContainer');
  if (!container) return;

  const videoUrl = localStorage.getItem('videoUrl'); 
  if (!videoUrl) {
    container.innerHTML = '<p class="muted">No video uploaded yet. Please upload a video first.</p>';
    return;
  }

  container.innerHTML = '';
  const v = document.createElement('video');
  v.src = videoUrl;           
  v.controls = true;
  v.preload = 'metadata';
  v.playsInline = true;
  v.style.width = '100%';
  v.style.borderRadius = '8px';
  v.style.border = '1px solid #DDE5DD';
  v.style.background = '#000';

  v.addEventListener('loadedmetadata', () => console.log('[Video] duration:', v.duration));
  v.addEventListener('error', () => console.error('[Video] element error for', videoUrl));

  container.appendChild(v);
  attachCustomTimestamp(v);
}

// function for loading extracted frames
async function loadExtractedFrames() {
  const savedName = localStorage.getItem('savedName');
  const nFrames   = Number(localStorage.getItem('nFrames') || 5);

  const res = await fetch('/api/frames', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ savedName, nFrames })
  }).then(r => r.json()).catch(() => null);

  if (!res?.ok || !res.frames) return;

  const grid = document.getElementById('framesGrid');
  if (!grid) return;

  grid.innerHTML = '';
  res.frames.forEach(src => {
    const img = document.createElement('img');
    img.src = src;
    img.alt = 'Extracted Frame';
    img.classList.add('frame-thumb');
    grid.appendChild(img);
    img.onerror = () => console.warn('Image failed to load:', src);
  });

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
  const confEl  = document.getElementById('engagementConfidence');

  let idx = (typeof data.engagementIndex === 'number') ? data.engagementIndex : null;
  let label = data.engagementLabel || null;
  let confidence = (typeof data.confidencePercent === 'number') ? data.confidencePercent : null;

  if ((!idx || !label || confidence == null) && data.probabilities && Object.keys(data.probabilities).length) {
    const entries = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
    const [topLabel, topPct] = entries[0];

    if (!label) label = topLabel;
    if (idx == null) {
      const labelToIdx = { 'Disengaged': 0, 'Low': 1, 'Engaged': 2, 'Highly Engaged': 3 };
      idx = (labelToIdx[topLabel] !== undefined) ? labelToIdx[topLabel] : null;
    }
    if (confidence == null) confidence = Math.round(topPct);
  }

  if (scoreEl && idx != null) scoreEl.textContent = String(idx);
  if (labelEl && label) labelEl.textContent = label;
  if (confEl && typeof confidence === 'number') confEl.textContent = `• Confidence ${confidence}%`;

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
  const colors = ['#E63946', '#F4A261', '#8FD6B5', '#8ECAE6'];
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

  const colors = ['#E63946', '#F4A261', '#8FD6B5', '#8ECAE6'];
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

// function for loading key engagement moments (timestamps)
async function loadKeyEngagementMoments() {
  const savedName = localStorage.getItem('savedName');  
  const nFrames   = Number(localStorage.getItem('nFrames') || 5);

  const res = await fetch('/api/key-moments', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ savedName, nFrames })
  }).then(r => r.json());

  if (!res?.ok) return;

  const list = document.getElementById('momentsList');
  if (!list) return;

  // helper function from mm:ss.mmm to seconds
  function parseMSms(str) {
    const [mm, rest='0'] = String(str).split(':');
    const [ss, ms='0']  = rest.split('.');
    return (+mm * 60) + (+ss) + (+ms / 1000);
  }

  // build the list 
  list.innerHTML = '';
  res.keyMoments.forEach(m => {
    const li = document.createElement('li');
    li.innerHTML = `<time>${m.t}</time> <a href="#" class="jump">Jump to Video</a>`;
    list.appendChild(li);
  });

  // click then jump
  document.getElementById('momentsList')?.addEventListener('click', (e) => {
    const a = e.target.closest('a.jump');
    if (!a) return;
    e.preventDefault();

    const timeText = a.previousElementSibling?.textContent || '00:00.000';
    const sec = parseMSms(timeText);

    const video = document.querySelector('#videoContainer video');
    if (!video) return;

    video.currentTime = sec;
    video.pause();
    document.getElementById('videoContainer')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  });
}

if (window.location.pathname.endsWith('analysis.html')) {
  document.addEventListener('DOMContentLoaded', () => {
    loadUploadedVideo();
    loadExtractedFrames();
    loadKeyEngagementMoments();
  });
}

function safeParse(s) {
  try { return JSON.parse(s); } catch { return null; }
}