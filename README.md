# ai-resume-assistant# 📄 ATS Resume Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) friendliness and suggests concrete improvements, powered by Google Gemini Flash.

## Features
- Upload a resume as **PDF or DOCX**
- Optional **job description** for tailored keyword matching
- ATS score out of 100 with a 5-category breakdown
- Strengths, weaknesses, matched and missing keywords, formatting issues
- Prioritised improvements with example rewrites
- Downloadable text report

> **Note:** The score is an AI-based estimate, not the output of a real ATS. Use it as guidance.

## Project structure
```
app.py             # the whole app
requirements.txt   # dependencies
README.md
```

## Run locally
```bash
git clone https://github.com/<your-username>/ats-resume-checker.git
cd ats-resume-checker
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Get a free Gemini API key at https://aistudio.google.com/apikey and paste it in the sidebar, or store it as a secret (below).

### Optional: store the key in a secrets file
Create `.streamlit/secrets.toml` (already git-ignored):
```toml
GEMINI_API_KEY = "your-key-here"
# GEMINI_MODEL = "gemini-flash-latest"   # optional override
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (public, or private with access granted).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → select your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## Configuration
| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | secrets / env var / sidebar | none |
| `GEMINI_MODEL` | secrets / env var / sidebar | `gemini-flash-latest` |

`gemini-flash-latest` is an alias for Google's newest Flash model. Pin an exact model name if you need stable behaviour.

## Limitations
- Scanned/image-only resumes can't be read (an ATS can't read them either).
- Resume text is sent to the Gemini API, so don't upload documents you aren't comfortable sharing.
- Files over 5 MB are rejected.
