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
  try {
    const savedName = (req.body && req.body.savedName) || null;
    if (!savedName) {
      return res.status(400).json({ ok: false, error: 'savedName is required' });
    }

    const nFrames = Number(req.body.nFrames || 5);

    // IMPORTANT: match your directory structure exactly
    // CPD_TEST/
    // ├─ Facial-Expression-Changepoint-Detection/
    // │  ├─ predict_engagement.py
    // │  └─ output_ml/models/engagement_rf.joblib
    // └─ up2face/
    const projectRoot = path.resolve(__dirname, '..');
    const pyRoot = path.resolve(projectRoot, 'Facial-Expression-Changepoint-Detection');

    const videoPath = path.resolve(__dirname, 'uploads', savedName);
    const scriptPath = path.resolve(pyRoot, 'predict_engagement.py');
    const modelDir  = path.resolve(pyRoot, 'output_ml', 'models');
    const outDir    = path.resolve(pyRoot, 'predictions');

    if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

    const args = [
      scriptPath,
      '--video', videoPath,
      '--model-dir', modelDir,
      '--n-frames', String(nFrames),
      '--out-dir', outDir
    ];

    const py = spawn('python3', args, { cwd: pyRoot });

    let stdout = '';
    let stderr = '';
    py.stdout.on('data', (d) => (stdout += d.toString()));
    py.stderr.on('data', (d) => (stderr += d.toString()));

    py.on('close', (code) => {
      if (code !== 0) {
        return res.status(500).json({ ok: false, error: 'Python error', details: stderr || stdout });
      }

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

      const predLine = stdout.split('\n').find(l => /Predicted engagement:/i.test(l));
      if (predLine) {
        const m = predLine.match(/Predicted engagement:\s*(\d+)\s*[–-]\s*(.+)\s*$/i);
        if (m) {
          result.data.engagementIndex = Number(m[1]);
          result.data.engagementLabel = m[2].trim();
        }
      }

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

      return res.json(result);
    });
  } catch (err) {
    return res.status(500).json({ ok: false, error: String(err) });
  }
});

// fallback
app.get(/.*/, (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Up2Face running at http://localhost:${PORT}`));

