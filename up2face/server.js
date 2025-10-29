const express = require('express');
const path = require('path');
const fs = require('fs');
const multer = require('multer');
const { spawn } = require('child_process');
const os = require('os');

const app = express();

// ensure uploads dir exists
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
  limits: { fileSize: 5 * 1024 * 1024 * 1024 }, // 5GB
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

// helper function to get subdirs sorted by mtime desc 
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

// helper function to turn seconds to m:ss
function toMinSec(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
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

    const keyMoments = uniq.map(s => ({ t: toMinSec(s) }));

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
