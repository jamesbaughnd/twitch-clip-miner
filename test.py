from pathlib import Path
from src.vision import analyze_visual_engagement

# Replace with an actual video path that has a face
vid = Path("data/vods/v2741532426.mp4")
score = analyze_visual_engagement(vid, start_s=10, end_s=20)
print(f"Visual score: {score}")