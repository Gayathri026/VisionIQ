import sys
from pathlib import Path

# Add visioniq directory to sys.path so 'backend.app...' is importable
visioniq_dir = Path(__file__).resolve().parent.parent.parent
if str(visioniq_dir) not in sys.path:
    sys.path.insert(0, str(visioniq_dir))

# Also ensure current backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
