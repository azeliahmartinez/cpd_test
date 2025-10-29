const express = require('express');
const path = require('path');
const fs = require('fs');
const multer = require('multer');
const { spawn } = require('child_process');

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

// ---- Analyze endpoint (calls your Python) ----
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

    // FIXED: Use absolute path that works with your directory structure
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

    // Determine Python command based on platform
    let pythonCommand = 'python';
    if (process.platform === 'darwin') { // macOS
      pythonCommand = 'python3';
    }
    // Try 'python3' first, fallback to 'python'
    const commandsToTry = process.platform === 'darwin' ? ['python3', 'python'] : ['python', 'python3'];

    console.log(`[ANALYSIS] Running: ${pythonCommand} ${args.join(' ')}`);
    console.log(`[ANALYSIS] Working directory: ${pyRoot}`);
    console.log(`[ANALYSIS] Platform: ${process.platform}`);

    let py;
    let spawnError = null;

    // Try different Python commands
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
      console.log(`[PYTHON] Full STDOUT:\n${stdout}`);
      console.log(`[PYTHON] Full STDERR:\n${stderr}`);

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
        console.log(`[ANALYSIS] Found prediction line: ${predLine}`);
        const m = predLine.match(/Predicted engagement:\s*(\d+)\s*[–-]\s*(.+)\s*$/i);
        if (m) {
          result.data.engagementIndex = Number(m[1]);
          result.data.engagementLabel = m[2].trim();
          console.log(`[ANALYSIS] Parsed engagement: ${m[1]} - ${m[2]}`);
        }
      }

      // Parse probabilities
      const probsBlockStart = stdout.indexOf('Class probabilities');
      if (probsBlockStart !== -1) {
        const lines = stdout.slice(probsBlockStart).split('\n').map(s => s.trim());
        const probLines = lines.filter(l => /^\d+\s*\(.+\):\s*\d*\.\d+$/i.test(l));
        console.log(`[ANALYSIS] Found ${probLines.length} probability lines`);
        
        probLines.forEach(line => {
          const mm = line.match(/^(\d+)\s*\((.+?)\):\s*(\d*\.\d+)$/);
          if (mm) {
            const label = mm[2].trim();
            const val = parseFloat(mm[3]) * 100;
            result.data.probabilities[label] = Math.round(val * 10) / 10;
            console.log(`[ANALYSIS] Probability: ${label} = ${val}%`);
          }
        });
      }

      // If no probabilities found, create default ones based on engagement index
      if (Object.keys(result.data.probabilities).length === 0 && result.data.engagementIndex !== null) {
        console.log(`[WARNING] No probabilities found, creating defaults based on engagement index: ${result.data.engagementIndex}`);
        const labels = ['Disengaged', 'Low Engagement', 'Engaged', 'Highly Engaged'];
        result.data.probabilities = {};
        labels.forEach((label, index) => {
          // Give highest probability to the predicted class, distribute rest
          const prob = index === result.data.engagementIndex ? 70 : 10;
          result.data.probabilities[label] = prob;
        });
      }

      console.log(`[SUCCESS] Analysis complete:`, result.data);
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