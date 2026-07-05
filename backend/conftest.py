"""
conftest.py — ensures backend/ is on sys.path so that
`import config`, `from memory.gateway import ...`, etc. all resolve
correctly when pytest is run from the backend/ directory.

Also loads backend/.env (and the project-root .env as a fallback) via
python-dotenv so that MISTRAL_API_KEY and other vars are available to
tests without requiring a manual `source .env` step.
"""
import sys
import os

# Add the backend/ directory to sys.path if it isn't already there
_backend_dir = os.path.dirname(__file__)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

# Load environment variables from backend/.env (override=False so that
# values already present in the process environment take precedence).
try:
    from dotenv import load_dotenv

    # backend/.env takes priority; fall back to project-root .env
    _backend_env = os.path.join(_backend_dir, ".env")
    _root_env = os.path.join(_backend_dir, "..", ".env")

    if os.path.isfile(_backend_env):
        load_dotenv(_backend_env, override=False)
    elif os.path.isfile(_root_env):
        load_dotenv(_root_env, override=False)
except ImportError:
    # python-dotenv not installed — env vars must be set externally
    pass
