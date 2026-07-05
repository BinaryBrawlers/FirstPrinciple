"""
conftest.py — ensures backend/ is on sys.path so that
`import config`, `from memory.gateway import ...`, etc. all resolve
correctly when pytest is run from the backend/ directory.
"""
import sys
import os

# Add the backend/ directory to sys.path if it isn't already there
_backend_dir = os.path.dirname(__file__)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
