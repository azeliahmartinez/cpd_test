const express = require('express');
const path = require('path');
const fs = require('fs');
const multer = require('multer');
const { spawn } = require('child_process');
const archiver = require('archiver');

const app = express();

// ensure uploads directory exists
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });

// serve static assets
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));
app.use('/uploads', express.static(uploadDir));

// basic multer storage
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    const safe = file.originalname.replace(/\s+/g, '_');
    cb(null, Date.now() + '_' + safe);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 5 * 1024 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowedExts = ['.mp4', '.mov', '.avi'];
    const ext = path.extname(file.originalname).toLowerCase();

    if (allowedExts.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error('Unsupported format. Allowed: .mp4, .mov, .avi'), false);
    }
  }
});

// mount extracted frames so browser can access images
const projectRoot = path.resolve(__dirname, '..');
const pyRoot = path.resolve(projectRoot, 'Facial-Expression-Changepoint-Detection');
const extractedRoot = path.resolve(pyRoot, 'predictions', 'extracted_frames');

// serve all extracted frames under /frames
app.use('/frames', express.static(extractedRoot));

// helper function to get sub directories sorted by mtime desc 
function getLatestSubdir(rootDir) {
  if (!fs.existsSync(rootDir)) return null;
  const entries = fs.readdirSync(rootDir, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => {
      const p = path.join(rootDir, d.name);
      return { name: d.name, path: p, mtime: fs.statSync(p).mtimeMs };
    })
    .sort((a, b) => b.mtime - a.mtime);
  return entries[0]?.path || null;
}

// helper function to read 'time_sec' column from a CSV file 
function readTimesFromCsv(csvPath) {
  const out = [];
  const txt = fs.readFileSync(csvPath, 'utf8').trim();
  if (!txt) return out;
  const lines = txt.split(/\r?\n/);
  if (!lines.length) return out;

  const header = lines[0].split(',').map(s => s.trim());
  const tIdx = header.indexOf('time_sec');
  if (tIdx === -1) return out;

  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',');
    const sec = parseFloat((cols[tIdx] || '').trim());
    if (!Number.isNaN(sec)) out.push(sec);
  }
  return out;
}

// helper function to turn seconds to m:ss.ms
function toMSms(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = ((sec % 1) * 1000).toFixed(0).padStart(3, '0');
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${ms}`;
}

// helper function to get latest sub directory by mtime
function getLatestSubdir(rootDir) {
  if (!fs.existsSync(rootDir)) return null;
  const entries = fs.readdirSync(rootDir, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => {
      const p = path.join(rootDir, d.name);
      return { name: d.name, path: p, mtime: fs.statSync(p).mtimeMs };
    })
    .sort((a, b) => b.mtime - a.mtime);
  return entries[0] || null;
}

// upload endpoint 
app.post('/api/upload', upload.single('video'), (req, res) => {
  const uploadDate = new Date();
  const formattedDate = uploadDate.toLocaleString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });

  return res.json({
    ok: true,
    message: 'Upload successful.',
    filename: req.file?.originalname || 'video.mp4',
    savedName: req.file?.filename,
    url: req.file ? `/uploads/${req.file.filename}` : null,
    uploadDate: formattedDate
  });
});

// analyze endpoint 
app.post('/api/analyze', async (req, res) => {
  let responseSent = false; // Prevent double response
  const sendResponse = (data) => {
    if (responseSent) return;
    responseSent = true;
    res.json(data);
  };

  try {
    const savedName = (req.body && req.body.savedName) || null;
    if (!savedName) {
      return sendResponse({ ok: false, error: 'savedName is required' });
    }

    const nFrames = Number(req.body.nFrames || 5);

    console.log(`[ANALYSIS] Starting analysis for: ${savedName}`);
    console.log(`[ANALYSIS] Current directory: ${__dirname}`);

    // Match your directory structure: CPD_TEST/Facial-Expression-Changepoint-Detection
    const projectRoot = path.resolve(__dirname, '..');
    const pyRoot = path.resolve(projectRoot, 'Facial-Expression-Changepoint-Detection');
    console.log(`[ANALYSIS] Python root: ${pyRoot}`);

    const videoPath = path.resolve(__dirname, 'uploads', savedName);
    const scriptPath = path.resolve(pyRoot, 'predict_engagement.py');
    const modelDir  = path.resolve(pyRoot, 'output_ml', 'models');
    const outDir    = path.resolve(pyRoot, 'predictions');

    console.log(`[ANALYSIS] Video path: ${videoPath}`);
    console.log(`[ANALYSIS] Script path: ${scriptPath}`);
    console.log(`[ANALYSIS] Model dir: ${modelDir}`);
    console.log(`[ANALYSIS] Output dir: ${outDir}`);

    if (!fs.existsSync(videoPath)) {
      console.log(`[ERROR] Video not found: ${videoPath}`);
      return sendResponse({ ok: false, error: 'Video file not found' });
    }
    if (!fs.existsSync(scriptPath)) {
      console.log(`[ERROR] Python script not found: ${scriptPath}`);
      return sendResponse({ ok: false, error: 'Python analysis script not found' });
    }
    if (!fs.existsSync(modelDir)) {
      console.log(`[ERROR] Model directory not found: ${modelDir}`);
      return sendResponse({ ok: false, error: 'Model directory not found' });
    }
    if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

    const args = [
      scriptPath,
      '--video', videoPath,
      '--model-dir', modelDir,
      '--n-frames', String(nFrames),
      '--out-dir', outDir
    ];

    const commandsToTry = process.platform === 'darwin' ? ['python3', 'python'] : ['python', 'python3'];
    console.log(`[ANALYSIS] Working directory: ${pyRoot}`);
    console.log(`[ANALYSIS] Platform: ${process.platform}`);
    console.log(`[ANALYSIS] Candidate python commands: ${commandsToTry.join(', ')}`);

    let py;
    let pythonCommand = null;

    // Try starting python with preferred names
    for (const cmd of commandsToTry) {
      try {
        py = spawn(cmd, args, {
          cwd: pyRoot,
          stdio: ['pipe', 'pipe', 'pipe'],
          env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
        });
        pythonCommand = cmd;
        console.log(`[SUCCESS] Using Python command: ${cmd}`);
        break;
      } catch (err) {
        console.log(`[WARNING] Command '${cmd}' failed to spawn: ${err.message}`);
      }
    }

    if (!py) {
      return sendResponse({
        ok: false,
        error: 'Python not found. Install Python 3 and ensure it is in PATH.',
        details: `Tried commands: ${commandsToTry.join(', ')}`
      });
    }

    let stdout = '';
    let stderr = '';
    py.stdout.on('data', (d) => {
      const data = d.toString();
      stdout += data;
      console.log(`[PYTHON] STDOUT: ${data.trim()}`);
    });

    py.stderr.on('data', (d) => {
      const data = d.toString();
      stderr += data;
      console.log(`[PYTHON] STDERR: ${data.trim()}`);
    });

    py.on('error', (err) => {
      console.log(`[ERROR] Python spawn error: ${err.message}`);
      return sendResponse({
        ok: false,
        error: 'Failed to start Python process',
        details: err.message,
        pythonCommand
      });
    });

    py.on('close', (code) => {
      if (responseSent) return;

      console.log(`[PYTHON] Python process exited with code: ${code}`);
      if (code !== 0) {
        console.log('[PYTHON] Full STDOUT:\n' + stdout);
        console.log('[PYTHON] Full STDERR:\n' + stderr);
        return sendResponse({
          ok: false,
          error: 'Python analysis failed',
          details: stderr || stdout,
          exitCode: code,
          pythonCommand
        });
      }

      // ---- Build result from Python stdout ----
      const result = {
        ok: true,
        data: {
          title: savedName,
          date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
          engagementIndex: null,
          engagementLabel: null,
          probabilities: {},
          topLabel: null,
          confidencePercent: null,
          keyMoments: [],
          recommendations: [
            'Insert short breaks every 20–25 minutes.',
            'Add interactive quizzes at active points.',
            'Provide summaries after dips in focus.'
          ]
        }
      };

      // Predicted engagement line
      const predLine = stdout.split('\n').find(l => /Predicted engagement:/i.test(l));
      if (predLine) {
        console.log(`[ANALYSIS] Found prediction line: ${predLine}`);
        const m = predLine.match(/Predicted engagement:\s*(\d+)\s*[–-]\s*(.+)\s*$/i);
        if (m) {
          result.data.engagementIndex = Number(m[1]);
          result.data.engagementLabel = m[2].trim();
          console.log(`[ANALYSIS] Parsed engagement: ${m[1]} - ${m[2]}`);
        }
      }

      // Class probabilities block
      const probsBlockStart = stdout.indexOf('Class probabilities');
      if (probsBlockStart !== -1) {
        const lines = stdout.slice(probsBlockStart).split('\n').map(s => s.trim());
        const probLines = lines.filter(l => /^\d+\s*\(.+\):\s*\d*\.\d+$/i.test(l));
        console.log(`[ANALYSIS] Found ${probLines.length} probability lines`);

        probLines.forEach(line => {
          const mm = line.match(/^(\d+)\s*\((.+?)\):\s*(\d*\.\d+)$/);
          if (mm) {
            const label = mm[2].trim();
            const valPct = parseFloat(mm[3]) * 100; // 0.49 -> 49
            const rounded = Math.round(valPct * 10) / 10;
            result.data.probabilities[label] = rounded;
            console.log(`[ANALYSIS] Probability: ${label} = ${Math.round(rounded)}%`);
          }
        });
      }

      // Normalize / fallback if needed
      if (Object.keys(result.data.probabilities).length === 0 && result.data.engagementIndex !== null) {
        console.log(`[WARNING] No probabilities parsed; synthesizing from predicted class ${result.data.engagementIndex}`);
        const labels = ['Disengaged', 'Low', 'Engaged', 'Highly Engaged']; // normalized labels
        labels.forEach((label, idx) => {
          result.data.probabilities[label] = (idx === result.data.engagementIndex) ? 70 : 10;
        });
      }

      // Derive top label / confidence
      const entries = Object.entries(result.data.probabilities);
      if (entries.length) {
        entries.sort((a, b) => b[1] - a[1]);
        const [topLabel, topPct] = entries[0];
        result.data.topLabel = topLabel;
        result.data.confidencePercent = Math.round(topPct);
      }

      console.log(`[SUCCESS] Analysis complete:`, result.data);
      return sendResponse(result);
    });

  } catch (err) {
    if (responseSent) return;
    console.log(`[ERROR] Server error in /api/analyze: ${err.message}`);
    return sendResponse({
      ok: false,
      error: 'Server error during analysis',
      details: err.message
    });
  }
});

// download extracted frames as ZIP endpoint
app.get('/api/download-frames', (req, res) => {
  try {
    const videoName = req.query.video;
    if (!videoName) {
      return res.status(400).json({ ok: false, error: 'Video name is required' });
    }

    const projectRoot = path.resolve(__dirname, '..');
    const pyRoot = path.resolve(projectRoot, 'Facial-Expression-Changepoint-Detection');
    const videoStem = path.parse(videoName).name;
    const framesDir = path.resolve(pyRoot, 'predictions', 'extracted_frames', '5_frames', videoStem);

    console.log(`[DOWNLOAD] Looking for frames in: ${framesDir}`);

    if (!fs.existsSync(framesDir)) {
      return res.status(404).json({ ok: false, error: 'No extracted frames found for this video' });
    }

    const frames = fs.readdirSync(framesDir).filter(file => 
      file.endsWith('.png') || file.endsWith('.jpg') || file.endsWith('.jpeg')
    );

    if (frames.length === 0) {
      return res.status(404).json({ ok: false, error: 'No frame images found' });
    }

    const archive = archiver('zip', {
      zlib: { level: 9 }
    });

    res.attachment(`frames_${videoStem}.zip`);

    archive.on('error', (err) => {
      console.log(`[DOWNLOAD] Archive error: ${err}`);
      res.status(500).json({ ok: false, error: 'Failed to create download archive' });
    });

    archive.pipe(res);

    frames.forEach(frame => {
      const framePath = path.join(framesDir, frame);
      archive.file(framePath, { name: frame });
    });

    archive.finalize();

    console.log(`[DOWNLOAD] Sent ${frames.length} frames as ZIP`);

  } catch (err) {
    console.log(`[DOWNLOAD] Error: ${err.message}`);
    res.status(500).json({ ok: false, error: 'Download failed', details: err.message });
  }
});

// download CSV data as ZIP endpoint
app.get('/api/download-csv', (req, res) => {
  try {
    const videoName = req.query.video;
    if (!videoName) {
      return res.status(400).json({ ok: false, error: 'Video name is required' });
    }

    const projectRoot = path.resolve(__dirname, '..');
    const pyRoot = path.resolve(projectRoot, 'Facial-Expression-Changepoint-Detection');
    const videoStem = path.parse(videoName).name;
    const csvDir = path.resolve(pyRoot, 'predictions', 'raw_landmarks', '5_frames', videoStem);

    console.log(`[DOWNLOAD] Looking for CSVs in: ${csvDir}`);

    if (!fs.existsSync(csvDir)) {
      return res.status(404).json({ ok: false, error: 'No CSV data found for this video' });
    }

    const csvFiles = fs.readdirSync(csvDir).filter(file => file.endsWith('.csv'));

    if (csvFiles.length === 0) {
      return res.status(404).json({ ok: false, error: 'No CSV files found' });
    }

    const archive = archiver('zip', {
      zlib: { level: 9 }
    });

    res.attachment(`landmarks_${videoStem}.zip`);

    archive.on('error', (err) => {
      console.log(`[DOWNLOAD] Archive error: ${err}`);
      res.status(500).json({ ok: false, error: 'Failed to create download archive' });
    });

    archive.pipe(res);

    csvFiles.forEach(csvFile => {
      const csvPath = path.join(csvDir, csvFile);
      archive.file(csvPath, { name: csvFile });
    });

    // Add prediction summary if it exists
    const summaryPath = path.resolve(pyRoot, 'predictions', `prediction_summary_${videoStem}.txt`);
    if (fs.existsSync(summaryPath)) {
      archive.file(summaryPath, { name: `prediction_summary.txt` });
    }

    archive.finalize();

    console.log(`[DOWNLOAD] Sent ${csvFiles.length} CSV files as ZIP`);

  } catch (err) {
    console.log(`[DOWNLOAD] Error: ${err.message}`);
    res.status(500).json({ ok: false, error: 'Download failed', details: err.message });
  }
});

// extracted frames endpoint
app.post('/api/frames', (req, res) => {
  try {
    const savedName = (req.body && req.body.savedName) || null;
    if (!savedName) return res.json({ ok: false, error: 'savedName is required' });
    const nFrames = Number(req.body.nFrames || 5);

    const projectRoot = path.resolve(__dirname, '..');
    const pyRoot     = path.resolve(projectRoot, 'Facial-Expression-Changepoint-Detection');
    const framesRoot = path.resolve(pyRoot, 'predictions', 'extracted_frames');
    const tierRoot   = path.resolve(framesRoot, `${nFrames}_frames`);

    if (!app._framesStaticMounted) { 
      app.use('/frames', express.static(tierRoot)); app._framesStaticMounted = true; 
    }

    const baseNoExt = path.basename(savedName, path.extname(savedName));

    let runDirPath = path.join(tierRoot, baseNoExt);
    let runDirName = baseNoExt;

    if (!fs.existsSync(runDirPath)) {
      const candidates = fs.readdirSync(tierRoot, { withFileTypes: true })
        .filter(d => d.isDirectory())
        .map(d => d.name);
      const found = candidates.find(n => n.toLowerCase().includes(baseNoExt.toLowerCase()));
      if (found) {
        runDirName = found;
        runDirPath = path.join(tierRoot, found);
      }
    }

    if (!fs.existsSync(runDirPath)) {
      const latest = getLatestSubdir(tierRoot);
      if (!latest) return res.json({ ok: true, frames: [] });
      runDirName = latest.name;
      runDirPath = latest.path;
    }

    // collect images and sort by frame number if present
    const files = fs.readdirSync(runDirPath)
      .filter(f => /\.(png|jpg|jpeg)$/i.test(f))
      .sort((a, b) => {
        const na = (a.match(/(\d+)/) || [0, 0])[1];
        const nb = (b.match(/(\d+)/) || [0, 0])[1];
        return Number(na) - Number(nb);
      });

    const frames = files.map(f =>
      `/frames/${nFrames}_frames/${encodeURIComponent(runDirName)}/${encodeURIComponent(f)}`
    );

    return res.json({ ok: true, frames, runFolder: runDirName });
  } catch (err) {
    console.log('[FRAMES] error:', err.message);
    return res.status(500).json({ ok: false, error: 'Failed to list frames' });
  }
});

// key moments endpoint
app.post('/api/key-moments', (req, res) => {
  try {
    const savedName = (req.body && req.body.savedName) || null;
    if (!savedName) {
      return res.json({ ok: false, error: 'savedName is required' });
    }
    const nFrames = Number(req.body.nFrames || 5);

    // match /api/analyze path resolution
    const projectRoot = path.resolve(__dirname, '..');
    const pyRoot     = path.resolve(projectRoot, 'Facial-Expression-Changepoint-Detection');
    const framesRoot = path.resolve(pyRoot, 'predictions', 'raw_landmarks', `${nFrames}_frames`);

    // checks folder named after the savedName 
    const baseNoExt = path.basename(savedName, path.extname(savedName));
    let runDir = path.join(framesRoot, baseNoExt);

    // if that doesn't exist, falls back to "latest" folder under {nFrames}_frames
    if (!fs.existsSync(runDir)) {
      runDir = getLatestSubdir(framesRoot);
    }
    if (!runDir || !fs.existsSync(runDir)) {
      return res.json({ ok: true, keyMoments: [] });
    }

    // collect time_sec from all CSVs in the folder
    const files = fs.readdirSync(runDir).filter(f => f.toLowerCase().endsWith('.csv'));
    let times = [];
    for (const f of files) {
      const csvPath = path.join(runDir, f);
      try {
        times.push(...readTimesFromCsv(csvPath));
      } catch (e) {
        // ignore a bad csv, continue
      }
    }

    // de-dupe (by rounded second), sort asc, and format
    const uniq = Array.from(new Set(times.map(t => t.toFixed(2))))
      .map(s => parseFloat(s))
      .sort((a, b) => a - b);

    const keyMoments = uniq.map(s => ({ t: toMSms(s) }));

    return res.json({ ok: true, keyMoments, runDir });
  } catch (err) {
    console.log('[KEY-MOMENTS] error:', err.message);
    return res.status(500).json({ ok: false, error: 'Failed to read key moments' });
  }
});

// fallback
app.get(/.*/, (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Up2Face running at http://localhost:${PORT}`));