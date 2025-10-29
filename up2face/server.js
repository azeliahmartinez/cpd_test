const express = require('express');
const path = require('path');
const fs = require('fs');
const multer = require('multer');
const { spawn } = require('child_process');
const archiver = require('archiver');

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
  limits: { fileSize: 5 * 1024 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const ok = /video\/(mp4|quicktime|x-msvideo)/.test(file.mimetype);
    cb(ok ? null : new Error('Unsupported format'), ok);
  }
});

// ---- Upload endpoint ----
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

// ---- NEW: Download extracted frames as ZIP ----
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

// ---- NEW: Download CSV data as ZIP ----
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

// ---- Analyze endpoint (unchanged) ----
app.post('/api/analyze', async (req, res) => {
  let responseSent = false;

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

    const projectRoot = path.resolve(__dirname, '..');
    const pyRoot = path.resolve(projectRoot, 'Facial-Expression-Changepoint-Detection');
    
    const videoPath = path.resolve(__dirname, 'uploads', savedName);
    const scriptPath = path.resolve(pyRoot, 'predict_engagement.py');
    const modelDir  = path.resolve(pyRoot, 'output_ml', 'models');
    const outDir    = path.resolve(pyRoot, 'predictions');

    // Check if files exist
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

    let pythonCommand = 'python';
    if (process.platform === 'darwin') {
      pythonCommand = 'python3';
    }
    const commandsToTry = process.platform === 'darwin' ? ['python3', 'python'] : ['python', 'python3'];

    console.log(`[ANALYSIS] Running: ${pythonCommand} ${args.join(' ')}`);

    let py;
    let spawnError = null;

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
        spawnError = err;
        console.log(`[WARNING] Command '${cmd}' failed: ${err.message}`);
        continue;
      }
    }

    if (!py) {
      console.log(`[ERROR] All Python commands failed. Last error: ${spawnError?.message}`);
      return sendResponse({ 
        ok: false, 
        error: 'Python not found. Please install Python and ensure it is in your PATH.',
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

    py.on('close', (code) => {
      if (responseSent) return;
      
      console.log(`[PYTHON] Python process exited with code: ${code}`);

      if (code !== 0) {
        return sendResponse({ 
          ok: false, 
          error: 'Python analysis failed',
          details: stderr || stdout,
          exitCode: code,
          pythonCommand: pythonCommand
        });
      }

      // Parse the Python output
      const result = {
        ok: true,
        data: {
          title: savedName,
          date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
          engagementIndex: null,
          engagementLabel: null,
          probabilities: {},
          keyMoments: [],
          recommendations: [
            'Insert short breaks every 20–25 minutes.',
            'Add interactive quizzes at active points.',
            'Provide summaries after dips in focus.'
          ]
        }
      };

      // Parse predicted engagement
      const predLine = stdout.split('\n').find(l => /Predicted engagement:/i.test(l));
      if (predLine) {
        const m = predLine.match(/Predicted engagement:\s*(\d+)\s*[–-]\s*(.+)\s*$/i);
        if (m) {
          result.data.engagementIndex = Number(m[1]);
          result.data.engagementLabel = m[2].trim();
        }
      }

      // Parse probabilities
      const probsBlockStart = stdout.indexOf('Class probabilities');
      if (probsBlockStart !== -1) {
        const lines = stdout.slice(probsBlockStart).split('\n').map(s => s.trim());
        const probLines = lines.filter(l => /^\d+\s*\(.+\):\s*\d*\.\d+$/i.test(l));
        
        probLines.forEach(line => {
          const mm = line.match(/^(\d+)\s*\((.+?)\):\s*(\d*\.\d+)$/);
          if (mm) {
            const label = mm[2].trim();
            const val = parseFloat(mm[3]) * 100;
            result.data.probabilities[label] = Math.round(val * 10) / 10;
          }
        });
      }

      // If no probabilities found, create default ones
      if (Object.keys(result.data.probabilities).length === 0 && result.data.engagementIndex !== null) {
        const labels = ['Disengaged', 'Low Engagement', 'Engaged', 'Highly Engaged'];
        result.data.probabilities = {};
        labels.forEach((label, index) => {
          const prob = index === result.data.engagementIndex ? 70 : 10;
          result.data.probabilities[label] = prob;
        });
      }

      console.log(`[SUCCESS] Analysis complete`);
      return sendResponse(result);
    });

    py.on('error', (err) => {
      if (responseSent) return;
      console.log(`[ERROR] Python spawn error: ${err.message}`);
      return sendResponse({ 
        ok: false, 
        error: 'Failed to start Python process',
        details: err.message,
        pythonCommand: pythonCommand
      });
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

// fallback
app.get(/.*/, (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Up2Face running at http://localhost:${PORT}`));