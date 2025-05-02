const express = require('express');
const axios = require('axios');
const path = require('path');
const app = express();

app.use(express.static('public'));
app.use(express.json());

// API to generate blog content
app.post('/api/generate', async (req, res) => {
  const { prompt } = req.body;
  try {
    const response = await axios.post('http://localhost:11434/api/generate', {
      model: 'tinyllama',
      prompt,
      stream: false,
    });
    res.json({ content: response.data.response });
  } catch (error) {
    console.error('Blog generation failed:', error.message);
    res.status(500).json({ error: 'Blog generation failed. See server logs.' });
  }
});


// Serve index.html
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/index.html'));
});

const PORT = 3000;
app.listen(PORT, () => console.log(`🚀 Server running at http://localhost:${PORT}`));
