"""pipeline.config -- re-exports root config for package-level access."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import *  # noqa: F401, F403
