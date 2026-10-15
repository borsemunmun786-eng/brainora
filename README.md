# Brainora AI Quiz Generator

A Flask study app that turns text-based PDF notes into quizzes, flashcards, and topic-level study recommendations. Quiz history, goals, and revision tasks are stored in SQLite.

## Local Setup

```powershell
python -m pip install -r requirements.txt
python setup_nltk.py
python app.py
```

Open `http://127.0.0.1:5000`. PDFs must contain extractable text and be no larger than 16 MB. The app uses a development-only secret by default; configure `SECRET_KEY` for any shared or deployed instance.

## Tests

```powershell
python -m unittest test_app
python test_logic.py
```

The Flask workflow test covers PDF upload, quiz generation, submission, and saved history. The logic smoke test exercises topic detection and question generation.

## Deployment

The app currently supports a **single instance with a persistent local volume**. SQLite and filesystem sessions are not suitable for multiple replicas or ephemeral storage. Set `APP_DATA_DIR` to a writable persistent mount so the database and session files survive restarts.

For Render, configure:

- **Build command:** `pip install -r requirements.txt && python setup_nltk.py`
- **Start command:** `APP_ENV=production gunicorn app:app --bind 0.0.0.0:$PORT`
- **Environment variables:** `SECRET_KEY` set to a randomly generated value, and `APP_DATA_DIR` set to the persistent disk mount path.

## Alternative: PythonAnywhere

PythonAnywhere supports Flask apps through its WSGI service. After signing in, open a Bash console and run:

```bash
git clone https://github.com/borsemunmun786-eng/brainora.git
cd brainora
mkvirtualenv --python=/usr/bin/python3.13 brainora-env
pip install -r requirements.txt
mkdir -m 700 -p ~/.nltk_data
NLTK_DATA="$HOME/.nltk_data" NLTK_ALLOW_PROXIED_URLOPEN=1 python setup_nltk.py
```

Create a web app with **Manual configuration** and the same Python version. Set its virtualenv to `/home/<username>/.virtualenvs/brainora-env`. In the web app's WSGI configuration file, use:

```python
import sys
project_path = "/home/<username>/brainora"
if project_path not in sys.path:
	sys.path.insert(0, project_path)
from wsgi import application
```

The WSGI entry point creates a private secret, stores the SQLite database and filesystem sessions under `~/.brainora`, and loads NLTK data from `~/.nltk_data`. NLTK downloads use `NLTK_ALLOW_PROXIED_URLOPEN=1` only during setup because PythonAnywhere routes outbound requests through a proxy. Replace `<username>` with your PythonAnywhere username. Free accounts may have CPU and outbound-network limits.

Generate a secret with:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

The NLTK resources are installed into `.nltk_data` during the build and must be included in the deployed app files. The web process does not download them at import time. For multiple app instances, migrate SQLite to a managed database and filesystem sessions to a shared session store such as Redis before scaling.

## Project Files

- `app.py`: Flask pages and API routes
- `database.py`: SQLite persistence
- `pdf_extractor.py`: PDF text extraction and topic detection
- `question_generator.py`: NLTK-based MCQ generation
- `agent_service.py`, `revision_service.py`: study coaching and revision planning
- `setup_nltk.py`: explicit NLTK resource provisioning
- `templates/index.html`: application interface