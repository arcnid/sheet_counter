import cv2, time, threading, queue, os
from ultralytics import YOLO
from datetime import datetime

STREAM_URL = 'http://100.64.61.3:8080/?action=stream'
MODEL_PATH = 'sheet_counter/production_run/weights/best.pt'
CONF = 0.5
IMG = 320   # keep 320–384; you can still use your 288 if you prefer

X_MIN, X_MAX = 200, 440
LINE_Y = 300
TRACK_DIST2 = 80**2

# ---------------- perf hygiene ----------------
cv2.setNumThreads(1)  # avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")

# ---------------- async DB writer (optional) -------------
from supabase import create_client
# Supabase configuration
SUPABASE_URL = 'https://pzndsucdxloknrgecijj.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB6bmRzdWNkeGxva25yZ2VjaWpqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDA3NjY0OTcsImV4cCI6MjA1NjM0MjQ5N30.M9ITlEE4KHiScjIgP3lceygmwxLySHiaQBSrOda-b54'
# ─────────────────────────────────────────────────────────────────────────────
sb = create_client(SUPABASE_URL, SUPABASE_KEY)
dbq = queue.Queue(maxsize=1000)

def db_writer():
    buf, last = [], time.monotonic()
    while True:
        try:
            item = dbq.get(timeout=0.2)
            buf.append(item)
        except queue.Empty:
            pass
        if buf and (len(buf) >= 10 or time.monotonic() - last >= 1.0):
            try:
                sb.table("sheet_counts").insert(buf).execute()
                buf.clear()
                last = time.monotonic()
            except Exception:
                time.sleep(0.2)

threading.Thread(target=db_writer, daemon=True).start()

# ---------------- capture thread: latest-frame only -------------
latest = {"frame": None, "ts_ns": 0, "id": -1}
cap_errs = 0
stop = False

def now_ns():
    return time.monotonic_ns()

def capture_loop():
    global cap_errs
    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fid = -1
    while not stop:
        ok, f = cap.read()
        if not ok:
            cap_errs += 1
            time.sleep(0.05)
            continue
        fid += 1
        # single-slot overwrite; no lock needed for small dict writes (GIL)
        latest["frame"] = f
        latest["ts_ns"] = now_ns()       # capture timestamp (monotonic)
        latest["id"] = fid               # frame token

threading.Thread(target=capture_loop, daemon=True).start()

# ---------------- YOLO + tracking (your logic, instrumented) -------------
model = YOLO(MODEL_PATH)

total_count = 0
next_id, tracks, counted_ids = 0, {}, set()

# metrics
proc_frames = 0
dropped_frames = 0      # how many frame IDs we skipped since last processed
last_seen_id = -1
t0_run = time.monotonic()
ema_fps = None

def get_latest(timeout_s=0.5):
    start = time.monotonic()
    baseline_id = latest["id"]
    while time.monotonic() - start < timeout_s:
        cur_id = latest["id"]
        if cur_id != baseline_id and latest["frame"] is not None:
            return latest["frame"], latest["ts_ns"], cur_id
        time.sleep(0.002)
    # stale; return what we have anyway (stay live, don’t block)
    if latest["frame"] is not None:
        return latest["frame"], latest["ts_ns"], latest["id"]
    return None, 0, -1

while True:
    frame, cap_ts_ns, fid = get_latest()
    if frame is None:
        print("⚠️ No frames yet…")
        continue

    # --- backlog math: how many frames were overwritten since last process
    if last_seen_id >= 0 and fid > last_seen_id + 1:
        dropped_frames += (fid - last_seen_id - 1)
    last_seen_id = fid

    # OPTIONAL ROI to boost accuracy at same cost
    Y_MIN, Y_MAX = 150, 450
    roi = frame[Y_MIN:Y_MAX, X_MIN:X_MAX]

    t_infer0 = time.monotonic()
    res = model.predict(roi, conf=CONF, imgsz=IMG, max_det=10, verbose=False)[0]
    t_infer1 = time.monotonic()

    detections = []
    for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
        x1 += X_MIN; x2 += X_MIN; y1 += Y_MIN; y2 += Y_MIN
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        ty = int(y1)
        if X_MIN <= cx <= X_MAX:
            detections.append((cx, cy, ty))

    new_tracks = {}
    for cx, cy, ty in detections:
        best, bestd = None, 1e12
        for tid, (px, py, pty) in tracks.items():
            d2 = (cx-px)**2 + (cy-py)**2
            if d2 < bestd: best, bestd = tid, d2
        tid = best if bestd < TRACK_DIST2 else next_id; 
        if tid == next_id: next_id += 1

        prev_ty = tracks.get(tid, (cx, cy, ty))[2]
        if prev_ty < LINE_Y <= ty and tid not in counted_ids:
            total_count += 1
            counted_ids.add(tid)
            try:
                dbq.put_nowait({"count": total_count, "recorded_at": datetime.utcnow().isoformat()})
            except queue.Full:
                pass
        new_tracks[tid] = (cx, cy, ty)
    tracks = new_tracks

    # --- metrics
    proc_frames += 1
    dt = t_infer1 - t_infer0
    cur_fps = 1.0 / dt if dt > 0 else 0.0
    ema_fps = cur_fps if ema_fps is None else (0.9 * ema_fps + 0.1 * cur_fps)

    # end-to-end latency (how “live” we are): now - capture timestamp
    e2e_ms = (time.monotonic_ns() - cap_ts_ns) / 1e6

    # dropped-rate since start
    total_seen = proc_frames + dropped_frames
    drop_pct = (100.0 * dropped_frames / total_seen) if total_seen else 0.0

    # --- lightweight HUD (draw every frame or every N frames)
    cv2.putText(frame, f"Count: {total_count}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.putText(frame, f"FPS(cur/EMA): {cur_fps:.1f}/{(ema_fps or 0):.1f}", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"Latency: {e2e_ms:.0f} ms", (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"Dropped: {dropped_frames} ({drop_pct:.1f}%)", (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
    cv2.putText(frame, f"FrameID: {fid}", (10, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,255,180), 2)

    cv2.line(frame, (X_MIN, LINE_Y), (X_MAX, LINE_Y), (255,0,0), 2)
    cv2.line(frame, (X_MIN, LINE_Y-10), (X_MIN, LINE_Y+10), (255,0,0), 1)
    cv2.line(frame, (X_MAX, LINE_Y-10), (X_MAX, LINE_Y+10), (255,0,0), 1)

    cv2.imshow("Sheet Counter (Live)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

stop = True
cv2.destroyAllWindows()
