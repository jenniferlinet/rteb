# Deploy on Streamlit Community Cloud (free, resume-friendly)

Streamlit gives you a public URL like `https://rteb-retrieval.streamlit.app` — ideal for LinkedIn and your resume.

## 1. Push this repo to GitHub

Ensure **`uploaded_pdf.pdf`** and **`.env`** are **not** committed (they are in `.gitignore`).

```bash
git add main.py requirements.txt .gitignore .streamlit/ .env.example DEPLOY.md README.md
git commit -m "Prepare RTEB app for Streamlit Cloud deployment"
git push origin main
```

Your repo: https://github.com/jenniferlinet/rteb

## 2. Create the live app

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with **GitHub**.
2. Click **Create app**.
3. **Repository:** `jenniferlinet/rteb`
4. **Branch:** `main`
5. **Main file path:** `main.py`
6. Click **Advanced settings** → **Secrets** and paste:

```toml
RTEB_CLOUD = "1"
```

(This uses lighter embedding models so the app fits free-tier memory.)

7. Click **Deploy**.

First boot takes **3–8 minutes** (downloads PyTorch and Hugging Face models). If the app sleeps, the next visitor may wait on cold start.

## 3. Add the link to your resume / LinkedIn

**Project title:** RTEB Retrieval Dashboard  

**One line:** Interactive PDF search comparing TF-IDF, semantic embeddings, and hybrid retrieval with cross-encoder reranking and extractive summaries.

**Link:** your Streamlit URL from step 2.

**LinkedIn (Featured → Add link):**  
Title: `RTEB – Hybrid PDF Retrieval Demo`  
URL: `https://YOUR-APP-NAME.streamlit.app`

## Troubleshooting

| Issue | Fix |
|--------|-----|
| App crashes on start | Confirm `RTEB_CLOUD = "1"` in Secrets; reboot app from Manage app. |
| `ModuleNotFoundError` | Check `requirements.txt` is on `main` and redeploy. |
| Out of memory | Free tier is limited; keep `RTEB_CLOUD=1`; avoid huge PDFs in demos. |
| Internal PDFs | Never commit real customer docs — upload only in the live app session. |

## Other hosts

Same command works on Render/Railway/Fly with **≥2 GB RAM** and `streamlit run main.py --server.port=$PORT --server.address=0.0.0.0`.
