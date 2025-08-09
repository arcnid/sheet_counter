# count_sheets_optimized.py

import os
import cv2
import time
import threading
import queue
from ultralytics import YOLO
from supabase import create_client
from datetime import datetime

# Optional: tame CPU oversubscription (safe even if torch isn't available)
try:
    import torch
    # Limit PyTorch threads (tune if needed)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# OpenCV runtime tweaks
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass
try:
    cv2.setNumThreads(1)
except Exception:
    pass

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

# UI / display throttling (purely visual; does NOT affect detection/counting)
SHOW_WINDOW   = True
DRAW_EVERY_N  = 3     # draw & imshow every N frames (set to 1 to draw every frame)

# ─────────────────────────────────────────────────────────────────────────────

# Initialize Supabase client
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# Background writer to keep network I/O off the hot path
write_q: "queue.Queue[dict]" = queue.Queue(maxsize=256)

def _writer_thread():
    while True:
        payload = write_q.get()
        if payload is None:  # shutdown signal if you ever use it
            break
        try:
            sb.table("sheet_counts").insert(payload).execute()
        except Exception:
            # Swallow errors to avoid killing the loop; optionally log
            pass
        finally:
            write_q.task_done()

threading.Thread(target=_writer_thread, daemon=True).start()

# Retrieve last count on startup (or start at 0)
try:
    resp = sb.table("sheet_counts") \
             .select("count") \
             .order("id", desc=True) \
             .limit(1) \
             .execute()
    if resp.data and len(resp.data) > 0:
        total_count = resp.data[0]["count"]
    else:
        total_count = 0
except Exception:
    total_count = 0  # fail-safe if DB is unreachable at start

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
    raise SystemExit(1)

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
        # 2) Run detection  (logic unchanged; just wrapped with no_grad)
        if TORCH_AVAILABLE:
            with torch.no_grad():
                results = model.predict(frame, conf=CONF_THRESH, imgsz=320, verbose=False)[0]
        else:
            results = model.predict(frame, conf=CONF_THRESH, imgsz=320, verbose=False)[0]

        detections = []
        # We'll throttle drawing (purely visual). Counting logic untouched.
        do_draw = SHOW_WINDOW and (DRAW_EVERY_N <= 1 or (frame_count % DRAW_EVERY_N == 0))

        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            top_y = y1

            # Draw every detection (throttled)
            if do_draw:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Only consider this detection if its centroid X is in our band
            if X_MIN <= cx <= X_MAX:
                detections.append((cx, cy, top_y))

        # 3) Draw the counting line segment (middle only)
        if do_draw:
            cv2.line(frame,
                    (X_MIN, LINE_Y),
                    (X_MAX, LINE_Y),
                    (255, 0, 0), 2)
            # Vertical caps
            cv2.line(frame, (X_MIN, LINE_Y-10), (X_MIN, LINE_Y+10), (255,0,0), 1)
            cv2.line(frame, (X_MAX, LINE_Y-10), (X_MAX, LINE_Y+10), (255,0,0), 1)

        # ─────────────────────────────────────────────────────────────────────
        # 4) Tracking + top-edge crossing logic (UNCHANGED)
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

                # Enqueue Supabase write (non-blocking)
                payload = {"count": total_count, "recorded_at": capture_ts}
                try:
                    write_q.put_nowait(payload)
                except queue.Full:
                    # Drop on overload rather than blocking FPS
                    pass

            new_tracks[track_id] = (cx, cy, ty)

        tracks = new_tracks

        # ── End timing & compute FPS ─────────────────────────────────────────
        t1 = time.time()
        frame_count += 1
        fps = frame_count / (t1 - t_start) if (t1 - t_start) > 0 else 0

        # 5) Overlay running count and FPS (throttled)
        if do_draw:
            cv2.putText(frame, f"Count: {total_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if SHOW_WINDOW:
                cv2.imshow('Sheet Counter', frame)
                # Only check keys when drawing
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
                    break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

