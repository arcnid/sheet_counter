# count_sheets_headless.py

import cv2
import time
import threading
from ultralytics import YOLO
from supabase import create_client
from datetime import datetime
import sys

# ─────────────────────────────────────────────────────────────────────────────
# 1) Configuration
STREAM_URL   = 'http://100.64.61.3:8080/?action=stream'  # your MJPEG stream URL
MODEL_PATH   = 'sheet_counter/production_run/weights/best.pt'
CONF_THRESH  = 0.5         # detection confidence threshold
SAMPLE_CLIP  = 'clip.mp4'
MODEL_VIEW   = True

# Render / performance knobs
PROCESS_EVERY = 2   # 1 = infer every frame, 2 = every other frame, 3 = every 3rd, ...
DRAW_EVERY    = 2   # kept for parity; unused in headless
IMG_SIZE      = 320 # keep your original imgsz

# Define your counting zone (unchanged)
X_MIN, X_MAX  = 200, 440   # horizontal bounds of the machine exit
LINE_Y        = 300        # y-coordinate of the counting line
TRACK_DIST2   = 80**2

# Global cooldown to suppress rapid re-counts (NEW)
COUNT_COOLDOWN_S = 10.0

# Supabase configuration (UNCHANGED)
SUPABASE_URL = 'https://pzndsucdxloknrgecijj.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB6bmRzdWNkeGxva25yZ2VjaWpqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDA3NjY0OTcsImV4cCI6MjA1NjM0MjQ5N30.M9ITlEE4KHiScjIgP3lceygmwxLySHiaQBSrOda-b54'
# ─────────────────────────────────────────────────────────────────────────────

# Initialize Supabase client (UNCHANGED)
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# Retrieve last count on startup (UNCHANGED)
resp = sb.table("sheet_counts") \
         .select("count") \
         .order("id", desc=True) \
         .limit(1) \
         .execute()
if resp.data and len(resp.data) > 0:
    total_count = resp.data[0]["count"]
else:
    total_count = 0

# Load YOLOv8 model (UNCHANGED)
model = YOLO(MODEL_PATH)

# Tracking state (UNCHANGED core logic)
next_id     = 0
tracks      = {}   # track_id -> (cx, cy, top_y)
counted_ids = set()

# Cooldown timer (NEW)
last_count_ts = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Latest-frame-only capture thread (prevents backlog / catch-up bursts)
class LatestFrame:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.ts = 0.0
        self.id = -1
        self.ok = False

latest = LatestFrame()
stop_flag = False

def capture_loop():
    global stop_flag
    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
    # keep buffer minimal
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fid = -1

    if not cap.isOpened():
        print(f"ERROR: Unable to open source {STREAM_URL}", flush=True)
        sys.exit(1)  # terminate so Docker restarts

    while not stop_flag:
        ok, f = cap.read()
        if not ok:
            print("ERROR: Stream read failed.", flush=True)
            sys.exit(1)  # terminate so Docker restarts
        ts = time.time()
        fid += 1
        with latest.lock:
            latest.frame = f
            latest.ts = ts
            latest.id = fid
            latest.ok = True

    cap.release()

cap_thread = threading.Thread(target=capture_loop, daemon=True)
cap_thread.start()

def get_latest(timeout_s=0.5):
    """Return the most recent frame without blocking for old frames."""
    start = time.time()
    last_id = -2
    while time.time() - start < timeout_s:
        with latest.lock:
            f, ts, fid, ok = latest.frame, latest.ts, latest.id, latest.ok
        if not ok:
            time.sleep(0.01)
            continue
        if fid != last_id:
            return f, ts, fid
        time.sleep(0.002)
    # Timeout; return whatever we have (keeps loop responsive)
    with latest.lock:
        return latest.frame, latest.ts, latest.id

# ─────────────────────────────────────────────────────────────────────────────
# Main loop (headless)
frame_count = 0
t_start = time.time()

# cache last detections for display on skipped frames (kept for parity)
last_boxes_xyxy = None

try:
    while True:
        # Grab newest frame (no backlog)
        frame, capture_ts, fid = get_latest()
        if frame is None:
            # Stream not ready / broken; terminate so Docker restarts
            print("ERROR: No frames received from stream.", flush=True)
            sys.exit(1)

        frame_count += 1
        do_infer = (frame_count % PROCESS_EVERY == 0)

        # Run detection only on selected frames
        if do_infer:
            # ─────────────────────────────────────────────────────────────
            # 2) Run detection (UNCHANGED params except using IMG_SIZE var)
            results = model.predict(frame, conf=CONF_THRESH, imgsz=IMG_SIZE, verbose=False)[0]

            # Build detections list (UNCHANGED counting pre-steps)
            detections = []
            boxes_xyxy = []
            for box in results.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                boxes_xyxy.append((x1, y1, x2, y2))
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                top_y = y1
                if X_MIN <= cx <= X_MAX:
                    detections.append((cx, cy, top_y))
            last_boxes_xyxy = boxes_xyxy  # cached but unused (headless)

            # 4) Tracking + top-edge crossing logic (UNCHANGED core counting)
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

                # ─────────────────────────────────────────────────────────
                # Cooldown gate (NEW): require LINE_Y crossing AND cooldown
                crossed = (prev_ty < LINE_Y <= ty)
                cooldown_over = ((capture_ts - last_count_ts) >= COUNT_COOLDOWN_S)
                if crossed and track_id not in counted_ids:
                    if cooldown_over:
                        total_count += 1
                        counted_ids.add(track_id)
                        last_count_ts = capture_ts  # start cooldown
                        print(f"▶️ Counted sheet {track_id} at top_y={ty}, total={total_count}", flush=True)

                        # Persist to Supabase with the true capture timestamp (UNCHANGED)
                        sb.table("sheet_counts").insert({
                            "count": total_count,
                            "recorded_at": datetime.utcnow().isoformat()
                        }).execute()
                    else:
                        # Optional: debug log to make it obvious we suppressed a hit
                        remaining = COUNT_COOLDOWN_S - (capture_ts - last_count_ts)
                        if remaining < 0: remaining = 0
                        print(f"⏱️ Cooldown active ({remaining:.1f}s left). Suppressed track {track_id} crossing.", flush=True)
                # ─────────────────────────────────────────────────────────

                new_tracks[track_id] = (cx, cy, ty)

            tracks = new_tracks

        # simple heartbeat to logs
        if frame_count % 100 == 0:
            t_now = time.time()
            fps = frame_count / (t_now - t_start) if (t_now - t_start) > 0 else 0.0
            print(f"[Heartbeat] frames={frame_count}, fps={fps:.1f}, total_count={total_count}", flush=True)

except KeyboardInterrupt:
    pass
finally:
    stop_flag = True
    # no GUI to destroy; exiting cleanly
    sys.exit(0)
