# 🚀 GitHub Deployment & Live Hosting Guide

This guide details how to publish your **ML Interview Lab & 3D Pipeline Studio** to GitHub and make it live on the web.

---

## 1. Push to GitHub

From the project root folder (`ml-interview-lab`), run the following commands in your terminal or PowerShell:

```bash
# 1. Initialize Git repository
git init

# 2. Add all files to staging
git add .

# 3. Commit your files
git commit -m "feat: ML Interview Lab 3D Studio, Data Pipelines & Model Serving"

# 4. Rename default branch to main
git branch -M main

# 5. Link your GitHub repository (replace with your actual GitHub repo URL)
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git

# 6. Push to GitHub
git push -u origin main
```

---

## 2. Live Deployment Options

### Option A: GitHub Pages (Instant Free Hosting for 3D Studio)
We included an automated workflow: `.github/workflows/deploy.yml`.

1. Go to your repository on GitHub.
2. Click **Settings** > **Pages** (in the left sidebar).
3. Under **Build and deployment** > **Source**, select **GitHub Actions**.
4. Push a commit or trigger the action under **Actions** > **Deploy 3D Studio to GitHub Pages**.
5. Your 3D Studio will be live at:
   `https://<YOUR_USERNAME>.github.io/<YOUR_REPO_NAME>/`

---

### Option B: Vercel / Netlify (Zero Config Static Web Hosting)
1. Log in to [Vercel](https://vercel.com) or [Netlify](https://netlify.com).
2. Import your GitHub repository.
3. Set **Root Directory** to `web`.
4. Click **Deploy** — your live interactive 3D Web Studio is now online with an HTTPS URL.

---

### Option C: Render / Railway / Fly.io (Full-Stack Backend + 3D UI)
To run both the Python FastAPI server, SQLite database, and the 3D frontend live in the cloud:

1. Connect your GitHub repository to [Render](https://render.com).
2. Create a new **Web Service**.
3. Set **Runtime** to **Docker** (it will auto-detect `Dockerfile`).
4. Set **Port** to `8000`.
5. Deploy! Both the 3D Studio and the `/api/predict` endpoints will be active online.

---

### Option D: HuggingFace Spaces (ML Community Hosting)
1. Create a new Space on [HuggingFace Spaces](https://huggingface.co/spaces).
2. Select **Docker** as the Space SDK.
3. Push the repository contents to your HuggingFace Space Git remote.

---

## 3. Local Development

To run the full stack locally:

```bash
# 1. Activate Python virtual environment
.venv\Scripts\activate      # Windows
source .venv/bin/activate    # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch unified server
python server.py
```
Open **`http://localhost:8000`** in your browser to view the 3D Studio.
Interactive Swagger API documentation is available at **`http://localhost:8000/docs`**.
