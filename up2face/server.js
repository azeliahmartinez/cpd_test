const express = require('express');
const path = require('path');
const app = express();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

app.post('/api/upload', (req, res) => {
  return res.status(200).json({ ok: true, message: 'File received (stub).' });
});

app.post('/api/analyze', async (req, res) => {
  return res.json({
    ok: true,
    data: {
      title: 'Sample Lecture',
      duration: '45:30',
      date: '2025-03-15',
      score: 78,
      recommendations: [
        'Insert short breaks every 20–25 minutes.',
        'Add interactive quizzes at active points.',
        'Provide summaries after dips in focus.'
      ],
      trend: [72,85,60,95,70],
      states: { Focused: 40, Engaged: 30, Neutral: 20, Distracted: 10 },
      keyMoments: [
        { t: '10:20', note: 'Gaze shift to notes — deep focus' },
        { t: '20:45', note: 'Phone glance — quick recovery' },
        { t: '30:10', note: 'Active note-taking — high engagement' },
        { t: '40:05', note: 'Fidgeting — attention dip' }
      ]
    }
  });
});

app.get(/.*/, (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Up2Face running at http://localhost:${PORT}`));
