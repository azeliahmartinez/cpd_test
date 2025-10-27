// load shared partials, then activate icons and set active link
(async function injectPartials() {
  const slots = document.querySelectorAll('[data-include]');
  await Promise.all(Array.from(slots).map(async el => {
    const url = el.getAttribute('data-include');
    const html = await fetch(url).then(r => r.text());
    el.outerHTML = html;
  }));
  if (window.feather) feather.replace();

  // sidebar active state
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

  if (dz && fileInput) {
    dz.addEventListener('click', () => fileInput.click());
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

  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async () => {
      if (!selectedFile) return;
      if (progressBar) {
        progressBar.style.width = '20%';
        setTimeout(()=> progressBar.style.width='60%', 400);
        setTimeout(()=> progressBar.style.width='100%', 900);
      }
      await fetch('/api/upload', { method: 'POST' });
      setTimeout(()=> window.location.href = '/analysis.html', 1100);
    });
  }

  // analysis page buttons
  const reanalyzeBtn = document.getElementById('reanalyzeBtn');
  if (reanalyzeBtn) {
    reanalyzeBtn.addEventListener('click', async () => {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ videoId: 'demo' })
      }).then(r => r.json());

      if (res?.data) {
        document.getElementById('videoTitle').textContent = res.data.title;
        document.getElementById('videoDuration').textContent = res.data.duration;
        document.getElementById('videoDate').textContent = res.data.date;
        document.getElementById('engagementScore').textContent = `${res.data.score}%`;

        const recoList = document.getElementById('recoList');
        recoList.innerHTML = '';
        res.data.recommendations.forEach(t => {
          const li = document.createElement('li');
          li.textContent = t;
          recoList.appendChild(li);
        });

        drawTrend(res.data.trend);
        drawDonut(res.data.states);

        const moments = document.getElementById('momentsList');
        moments.innerHTML = '';
        res.data.keyMoments.forEach(m => {
          const li = document.createElement('li');
          li.innerHTML = `<time>${m.t}</time> ${m.note} <a href="#" class="jump">Jump to Video</a>`;
          moments.appendChild(li);
        });
      }
    });
  }

  const downloadBtn = document.getElementById('downloadBtn');
  if (downloadBtn) {
    downloadBtn.addEventListener('click', () => alert('Download stub. Connect to backend to enable.'));
  }
});

function drawTrend(points = []) {
  const c = document.getElementById('trendChart');
  if (!c) return;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);

  ctx.strokeStyle = '#DDE5DD';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(40, 10); ctx.lineTo(40, c.height - 30); ctx.lineTo(c.width - 10, c.height - 30);
  ctx.stroke();

  if (!points.length) return;

  const max = 100, min = 0;
  const plotW = c.width - 60, plotH = c.height - 50;
  ctx.beginPath();
  ctx.strokeStyle = '#12865C';
  ctx.lineWidth = 2;

  points.forEach((v, i) => {
    const x = 40 + (i * (plotW / (points.length - 1 || 1)));
    const y = 10 + (1 - (v - min) / (max - min)) * plotH;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawDonut(states = { Focused: 40, Engaged: 30, Neutral: 20, Distracted: 10 }) {
  const c = document.getElementById('donutChart');
  if (!c) return;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);

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
