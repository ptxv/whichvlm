from pathlib import Path

from cli import app

# Preserve whichvlm.* imports without restoring a nested source directory.
__path__ = [str(Path(__file__).parent)]


if __name__ == "__main__":
    app()
