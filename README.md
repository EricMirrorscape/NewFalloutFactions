# Post-Apocalyptic Terminal - Rules Assistant Bot

AI-powered rules assistant for tabletop gaming.

## 🚀 Deploy to Railway

### Option 1: One-Click Deploy
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new)

### Option 2: Manual Deploy

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Connect to Railway**
   - Go to [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository

3. **Add Environment Variable**
   - In Railway dashboard, go to your project
   - Click "Variables" tab
   - Add: `OPENAI_API_KEY` = `sk-your-api-key-here`

4. **Deploy**
   - Railway will automatically build and deploy
   - Click "Generate Domain" to get your public URL

## 📁 Project Structure

```
fallout-factions-1764268794_deploy/
├── backend/
│   ├── app.py              # FastAPI server
│   └── requirements.txt    # Python dependencies
├── data/
│   ├── chunks.db           # Rule chunks database
│   └── rulebook.index      # FAISS vector index
├── frontend/
│   ├── index.html          # UI
│   ├── script.js           # Chat logic
│   ├── style.css           # Styling
│   └── assets/             # Images
├── Dockerfile              # Container config
├── railway.json            # Railway config
└── .gitignore              # Git ignore rules
```

## ⚠️ Important

- **Never commit your `.env` file** - it contains your API key
- Set `OPENAI_API_KEY` in Railway's environment variables instead
- The `.gitignore` file is already configured to block `.env`

## 💰 API Costs

Each question costs approximately $0.01-0.05 in OpenAI API usage.

---

Built with [Rules Bot Maker](https://github.com/your-repo/rules-bot-maker)
