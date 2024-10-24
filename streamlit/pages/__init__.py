import sys
from pathlib import Path

# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# You can also import utils directly here if needed
import utils
import detect
import recognize
import tracking
import face_alignment
from qdrant_db import utils as db_utils