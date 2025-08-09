# count_timer_headless.py

import cv2
import time
from ultralytics import YOLO
from supabase import create_client
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# 1) Configuration
STREAM_URL   = 'http://100.64.61.3:8080/?action=stream'  # your MJPEG stream URL
MODEL_PATH   = 'sheet_counter/production_run/weights/best.pt'
CONF_THRESH  = 0.5         # detection confidence threshold

# Define your counting zone:
X_MIN, X_MAX  = 200, 440   # horizontal bounds of the machine exit
LINE_Y        = 300        # y-coordinate of your counting line
TRACK_DIST2   = 80**2      # squared distance for tracking matches

# Supabase configuration
SUPABASE_URL = 'https://pzndsucdxloknrgecijj.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB6bmRzdWNkeGxva25yZ2VjaWpqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDA3NjY0OTcsImV4cCI6MjA1NjM0MjQ5N30.M9ITlEE4KHiScjIgP3lceygmwxLySHiaQBSrOda-b54'
# ─────────────────────────────────────────────────────────────────────────────

# Initialize Supabase client
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# Retrieve last count on startup (or start at 0)
resp = sb.table("sheet_counts") \
         .select("count") \
         .order("id", desc=True) \
         .limit(1) \
         .execute()
if resp.data and len(resp.data) > 0:
    total_count = resp.data[0]["count"]
else:
    total_count = 0

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

# Tracking state
next_id     = 0
tracks      = {}   # track_id -> (cx, cy, top_y)
counted_ids = set()

# Open video source
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print(f"ERROR: Unable to open source {STREAM_URL}")
    exit(1)

# Keep buffer size to minimum so we always grab the freshest frame
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# For FPS calculation
frame_count = 0
t_start     = time.time()

try:
    while True:
        # ── Drop any stale frame in the buffer ────────────────────────────────
        cap.grab()

        # ── Read the freshest frame and timestamp it ──────────────────────────
        ret, frame = cap.read()
        if not ret:
            break  # stream ended or file over
        capture_ts = datetime.utcnow().isoformat()

        # ── Start timing this iteration ──────────────────────────────────────
        t0 = time.time()

        # ─────────────────────────────────────────────────────────────────────
        # 2) Run detection
        results = model.predict(frame, conf=CONF_THRESH, imgsz=320, verbose=False)[0]

        detections = []
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            top_y = y1

            # Only consider this detection if its centroid X is in our band
            if X_MIN <= cx <= X_MAX:
                detections.append((cx, cy, top_y))

        # ─────────────────────────────────────────────────────────────────────
        # 3) Tracking + top-edge crossing logic
        new_tracks = {}
        for cx, cy, ty in detections:
            # match to existing track
            best_id, best_d2 = None, float('inf')
            for tid, (px, py, pty) in tracks.items():
                d2 = (cx-px)**2 + (cy-py)**2
                if d2 < best_d2:
                    best_id, best_d2 = tid, d2

            if best_d2 < TRACK_DIST2:
                track_id = best_id
            else:
                track_id = next_id
                next_id += 1

            # count when top edge crosses the line from above to at/below
            prev_ty = tracks.get(track_id, (cx, cy, 0))[2]
            if prev_ty < LINE_Y <= ty and track_id not in counted_ids:
                total_count += 1
                counted_ids.add(track_id)
                print(f"▶️ Counted sheet {track_id} at top_y={ty}, total={total_count}")

                # Persist to Supabase with the true capture timestamp
                sb.table("sheet_counts").insert({
                    "count": total_count,
                    "recorded_at": capture_ts
                }).execute()

            new_tracks[track_id] = (cx, cy, ty)

        tracks = new_tracks

        # ── End timing & compute FPS ─────────────────────────────────────────
        t1 = time.time()
        frame_count += 1
        fps = frame_count / (t1 - t_start) if (t1 - t_start) > 0 else 0

        # 4) Print running count and FPS to terminal (overwrite same line)
        print(f"\r▶️ Sheets: {total_count} │ FPS: {fps:.1f}", end="", flush=True)

except KeyboardInterrupt:
    # allow graceful exit with Ctrl+C
    pass
finally:
    cap.release()
    print("\nStream closed. Exiting.")

