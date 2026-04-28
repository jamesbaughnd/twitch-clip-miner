

import sys
from pathlib import Path
from src.gui_review import ClipReviewer

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
    else:
        # Auto-pick the most recent clips folder
        base = Path("data/clips")
        if base.exists():
            folders = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if folders:
                folder = folders[0]
            else:
                print("No clips folders found in data/clips")
                sys.exit(1)
        else:
            print("data/clips/directory does not exist yet.")
            sys.exit(1)

    if not folder.exists():
        print(f"Folder not found: {folder}")
        sys.exit(1)

    ClipReviewer(folder)