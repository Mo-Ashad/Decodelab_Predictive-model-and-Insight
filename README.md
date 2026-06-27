# Predictive Model & Insight Generator

Simple Flask backend for the frontend dashboard. Run locally for development.

Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# Run locally (default binds to 0.0.0.0:5000)
python app.py
```

Open http://127.0.0.1:5000 or http://<your-server-ip>:5000

Optional environment variables (for server deployment):

```powershell
set FLASK_RUN_HOST=0.0.0.0
set PORT=5000
set FLASK_DEBUG=0
python app.py
