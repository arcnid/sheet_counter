# count_sheets.py

import cv2
import time
import threading
from ultralytics import YOLO
from supabase import create_client
from datetime import datetime
from collections import deque  # <-- added
import os  # <-- added

# ─────────────────────────────────────────────────────────────────────────────
# 1) Configuration
STREAM_URL   = 'http://100.64.61.3:8080/?action=stream'  # your MJPEG stream URL
MODEL_PATH   = 'sheet_counter/production_run/weights/best.pt'
CONF_THRESH  = 0.5         # detection confidence threshold
SAMPLE_CLIP  = 'clip.mp4'
MODEL_VIEW = True

# Headless servers (EC2) don’t have a display; skip any UI calls.
HEADLESS = os.environ.get("HEADLESS", "1") == "1"  # <-- added

# Render / performance knobs
PROCESS_EVERY = 2   # 1 = infer every frame, 2 = every other frame, 3 = every 3rd, ...
DRAW_EVERY    = 2   # draw HUD every N frames (reduces GUI cost)
IMG_SIZE      = 320 # keep your original imgsz; you can try 320/384 later

# Define your counting zone (unchanged)
X_MIN, X_MAX  = 200, 440   # horizontal bounds of the machine exit
LINE_Y        = 300        # y-coordinate of the counting line
TRACK_DIST2   = 80**2

# ── New stability knobs (lightweight; won’t disrupt your flow) ───────────────
CROSS_HYST          = 8     # must come from at least 8 px above the line
COUNT_BAND_BELOW    = 40    # only count if the top edge is within 40 px below the line
LOCAL_COOLDOWN_S    = 0.60  # block re-counts near the same x for this many seconds
COOLDOWN_DX         = 35    # “near” in x for cooldown
MIN_GLOBAL_COOLDOWN_S = 5.0 # GLOBAL: block any second count for 5 seconds

# Console logging frequency (seconds)
STATUS_EVERY_S      = float(os.environ.get("STATUS_EVERY_S", "2.0"))
# ─────────────────────────────────────────────────────────────────────────────

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
print(f"✅ YOLO model loaded from: {MODEL_PATH}")

# Tracking state (UNCHANGED core logic)
next_id     = 0
tracks      = {}   # track_id -> (cx, cy, top_y)
counted_ids = set()

# ── New lightweight de-dupe buffer ───────────────────────────────────────────
recent_crossings = deque()  # items: (timestamp, x)

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
    """Open the stream; if it fails to open or any read fails, EXIT so Docker restarts."""
    global stop_flag
    fid = -1

    # Try to open once; if not opened, exit non-zero (let Docker restart)
    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"❌ Unable to open source {STREAM_URL}. Exiting for Docker restart…")
        os._exit(2)  # immediate exit so Docker restarts

    print("✅ Stream opened.")

    while not stop_flag:
        ok, f = cap.read()
        ts = time.time()

        # If the stream read fails (disconnect / camera offline), exit to trigger restart
        if not ok or f is None:
            print("❌ Stream read failed (camera disconnected?). Exiting for Docker restart…")
            try:
                cap.release()
            except Exception:
                pass
            os._exit(3)

        fid += 1
        with latest.lock:
            latest.frame = f
            latest.ts = ts
            latest.id = fid
            latest.ok = True

    # Normal shutdown path
    try:
        cap.release()
    except Exception:
        pass

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
    # Timeout; return whatever we have (keeps UI live)
    with latest.lock:
        return latest.frame, latest.ts, latest.id

# ── Cooldown helper ──────────────────────────────────────────────────────────
def cooldown_ok(x, ts):
    # drop stale entries
    while recent_crossings and ts - recent_crossings[0][0] > LOCAL_COOLDOWN_S:
        recent_crossings.popleft()
    # reject if any recent crossing too close in X
    for t0, x0 in recent_crossings:
        if abs(x - x0) <= COOLDOWN_DX:
            return False
    return True

# ─────────────────────────────────────────────────────────────────────────────
# Main loop with process/draw throttling
frame_count = 0
t_start = time.time()

# cache last detections for display on skipped frames
last_boxes_xyxy = None
last_count_ts = -1e9  # GLOBAL: timestamp of last credited count
last_status_ts = time.time()
last_boxes_count = 0
last_in_zone_count = 0

try:
    while True:
        # Grab newest frame (no backlog)
        frame, capture_ts, fid = get_latest()
        if frame is None:
            # Stream not ready yet
            time.sleep(0.01)
            continue

        frame_count += 1
        do_infer = (frame_count % PROCESS_EVERY == 0)
        do_draw  = (frame_count % DRAW_EVERY == 0)

        # Run detection only on selected frames
        if do_infer:
            # ─────────────────────────────────────────────────────────────
            # 2) Run detection (UNCHANGED params except using IMG_SIZE var)
            results = model.predict(frame, conf=CONF_THRESH, imgsz=IMG_SIZE, verbose=False)[0]

            # Build detections list (UNCHANGED counting pre-steps)
            detections = []
            boxes_xyxy = []
            in_zone = 0
            for box in results.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                boxes_xyxy.append((x1, y1, x2, y2))
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                top_y = y1
                if X_MIN <= cx <= X_MAX:
                    detections.append((cx, cy, top_y))
                    in_zone += 1
            last_boxes_xyxy = boxes_xyxy  # cache for skipped frames
            last_boxes_count = len(boxes_xyxy)
            last_in_zone_count = in_zone

            # 3) Draw the counting line segment (we'll draw later if do_draw)
            # ─────────────────────────────────────────────────────────────
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
                # Use a safer default prev_ty so first-sighting can count ONLY near the line,
                # and add both local and GLOBAL cooldowns to suppress re-ID doubles and bursts.
                prev_ty = tracks.get(track_id, (cx, cy, LINE_Y - CROSS_HYST))[2]

                crossed = (prev_ty < (LINE_Y - CROSS_HYST)) and (ty >= LINE_Y)
                near_enough_below = (ty <= LINE_Y + COUNT_BAND_BELOW)
                global_ok = (capture_ts - last_count_ts) >= MIN_GLOBAL_COOLDOWN_S

                if crossed and near_enough_below and (track_id not in counted_ids) and cooldown_ok(cx, capture_ts) and global_ok:
                    total_count += 1
                    counted_ids.add(track_id)
                    recent_crossings.append((capture_ts, cx))
                    last_count_ts = capture_ts  # update GLOBAL cooldown anchor
                    print(f"▶️ Counted sheet {track_id} at top_y={ty}, total={total_count}")
                elif crossed and near_enough_below:
                    # Helpful debug: why didn't we count?
                    reason = []
                    if track_id in counted_ids: reason.append("already-counted-id")
                    if not cooldown_ok(cx, capture_ts): reason.append("local-cooldown")
                    if not global_ok: reason.append("global-cooldown")
                    if reason:
                        print(f"⏸️  Blocked count at x={cx}, ty={ty}: {', '.join(reason)}")

                new_tracks[track_id] = (cx, cy, ty)

            tracks = new_tracks

        # ── FPS calc (kept as in your code) ───────────────────────────────
        t_now = time.time()
        fps = frame_count / (t_now - t_start) if (t_now - t_start) > 0 else 0.0

        # ── Console STATUS every STATUS_EVERY_S seconds ───────────────────
        if (t_now - last_status_ts) >= STATUS_EVERY_S:
            since = t_now - last_count_ts if last_count_ts > 0 else float('inf')
            print(
                f"STATUS fps={fps:.1f} total={total_count} "
                f"boxes={last_boxes_count} in_zone={last_in_zone_count} "
                f"since_last_count={since:.1f}s"
            )
            last_status_ts = t_now

        # ── Draw (throttled) ──────────────────────────────────────────────
        if do_draw and not HEADLESS:  # <-- guard UI on servers
            # Draw boxes (from this frame’s inference or last cached)
            boxes_to_draw = last_boxes_xyxy if last_boxes_xyxy is not None else []
            for (x1, y1, x2, y2) in boxes_to_draw:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # small dot is cheaper than circle fill
                cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 3, (0, 0, 255), -1)

            # Draw the counting line segment (unchanged)
            cv2.line(frame, (X_MIN, LINE_Y), (X_MAX, LINE_Y), (255, 0, 0), 2)
            cv2.line(frame, (X_MIN, LINE_Y-10), (X_MIN, LINE_Y+10), (255,0,0), 1)
            cv2.line(frame, (X_MAX, LINE_Y-10), (X_MAX, LINE_Y+10), (255,0,0), 1)
            # Visualize the below-line band used to accept first-sighting counts
            cv2.line(frame, (X_MIN, LINE_Y + COUNT_BAND_BELOW), (X_MAX, LINE_Y + COUNT_BAND_BELOW), (255, 0, 0), 1)

            # Overlay running count and FPS (unchanged content)
            cv2.putText(frame, f"Count: {total_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Display result
            cv2.imshow('Sheet Counter', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
                break

except KeyboardInterrupt:
    pass
finally:
    stop_flag = True
    cv2.destroyAllWindows()
