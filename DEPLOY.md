# Deploy on Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **Create app** → Repository: `jenniferlinet/rteb` → Branch: `main` → Main file: **`main.py`** (repo root — not `main/main.py`).
3. **Advanced settings → Secrets:**

```toml
RTEB_CLOUD = "1"
```

4. **Deploy** (first build often takes 3–8 minutes).

Your app URL will look like: `https://rteb-jenniferlinet.streamlit.app`
