"""
Start the website:   python run.py
Then open            http://localhost:8000

Works from any folder: it switches to the project folder first so all relative paths resolve.
"""
import os
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    os.chdir(ROOT)
    print("\n  FinTech RL Audit  ->  open http://localhost:8000  (press CTRL+C to stop)\n")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, app_dir=str(ROOT))
