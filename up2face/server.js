const express = require('express');
const path = require('path');
const fs = require('fs');
const multer = require('multer');

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

// server.js
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

app.post('/api/analyze', async (req, res) => {
  return res.json({
    ok: true,
    data: {
      title: 'Sample Lecture',
      date: '2025-03-15',
      engagement: 78,
      recommendations: [
        'Insert short breaks every 20–25 minutes.',
        'Add interactive quizzes at active points.',
        'Provide summaries after dips in focus.'
      ],
      trend: [72,85,60,95,70],
      states: { Focused: 40, Engaged: 30, Neutral: 20, Distracted: 10 },
      keyMoments: [
        { t: '10:20'},
        { t: '20:45'},
        { t: '30:10'},
        { t: '40:05'}
      ]
    }
  });
});

// fallback
app.get(/.*/, (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Up2Face running at http://localhost:${PORT}`));
