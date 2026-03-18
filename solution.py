import cv2
import json
import base64
import subprocess
import time
import os
import shutil
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
import imagehash

VIDEO_IN               = Path("video_sample_1.mov")
VIDEO_OUT              = Path("compressed_output.mp4")
REPORT_HTML_OUT        = Path("compression_report.html")
SEGMENTS_JSON_OUT      = Path("segments_kept.json")

PHASH_THRESHOLD        = 0.95
MOTION_KEEP_THRESH     = 0.15
MOTION_DISCARD_THRESH  = 0.05
CONTEXT_EVERY_SEC      = 3
OUTPUT_FPS             = 12
OUTPUT_CRF             = 28

FRAME_SKIP        = 3
ANALYSIS_WIDTH    = 320
HAAR_SCALE        = 1.1
HAAR_NEIGHBORS    = 3


def get_small(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if w <= ANALYSIS_WIDTH:
        return frame_bgr
    nh = int(h * ANALYSIS_WIDTH / w)
    return cv2.resize(frame_bgr, (ANALYSIS_WIDTH, nh), interpolation=cv2.INTER_LINEAR)


def compute_phash(frame: np.ndarray) -> imagehash.ImageHash:
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_img)


def phash_similarity(h1: imagehash.ImageHash, h2: imagehash.ImageHash) -> float:
    if h1 is None or h2 is None:
        return 0.0
    max_bits = len(h1.hash) ** 2
    distance = h1 - h2
    return 1.0 - distance / max_bits


def compute_motion_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    if prev_gray is None:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(magnitude))


def has_face(frame: np.ndarray, cascade: cv2.CascadeClassifier) -> bool:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray  = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=HAAR_SCALE,
        minNeighbors=HAAR_NEIGHBORS, minSize=(20, 20)
    )
    return len(faces) > 0


def should_keep_frame(frame: np.ndarray,
                      prev_frame,
                      prev_kept_hash,
                      last_kept_time_sec: float,
                      current_time_sec: float,
                      cascade: cv2.CascadeClassifier,
                      motion_discard_thresh: float = MOTION_DISCARD_THRESH) -> tuple:
    small = get_small(frame)

    curr_hash = compute_phash(small)
    if prev_kept_hash is not None:
        sim = phash_similarity(curr_hash, prev_kept_hash)
        if sim > PHASH_THRESHOLD:
            if current_time_sec - last_kept_time_sec >= CONTEXT_EVERY_SEC:
                return True, "context_frame", 0.0, False
            return False, "discarded_duplicate", 0.0, False

    prev_gray = None
    curr_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if prev_frame is not None:
        prev_small = get_small(prev_frame)
        prev_gray  = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)

    motion_score = compute_motion_score(prev_gray, curr_gray)

    if motion_score < motion_discard_thresh:
        if current_time_sec - last_kept_time_sec >= CONTEXT_EVERY_SEC:
            return True, "context_frame", motion_score, False
        return False, "discarded_static", motion_score, False

    face_found = has_face(small, cascade)
    if face_found:
        if motion_score > MOTION_KEEP_THRESH:
            return True, "face_and_motion", motion_score, True
        return True, "face_detected", motion_score, True

    if motion_score > MOTION_KEEP_THRESH:
        return True, "motion_above_threshold", motion_score, False

    if current_time_sec - last_kept_time_sec >= CONTEXT_EVERY_SEC:
        return True, "context_frame", motion_score, False

    return False, "discarded_static", motion_score, False


def auto_calibrate_motion_threshold(cap: cv2.VideoCapture,
                                     calibration_secs: float = 30.0) -> float:
    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    max_frames = int(fps * calibration_secs)
    scores     = []

    ret, prev_frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return MOTION_DISCARD_THRESH

    prev_small = get_small(prev_frame)
    prev_gray  = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)

    for i in range(max_frames - 1):
        ret, frame = cap.read()
        if not ret:
            break
        if i % FRAME_SKIP != 0:
            continue
        small     = get_small(frame)
        curr_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        scores.append(compute_motion_score(prev_gray, curr_gray))
        prev_gray = curr_gray

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not scores:
        return MOTION_DISCARD_THRESH

    mean_score = float(np.mean(scores))
    std_score  = float(np.std(scores))
    calibrated = max(0.02, min(mean_score + 0.5 * std_score, 0.20))
    print(f"[CALIBRATE] motion_discard_threshold = {calibrated:.4f} "
          f"(mean={mean_score:.4f}, std={std_score:.4f})")
    return calibrated


def frame_to_b64_thumb(frame: np.ndarray, width: int = 200) -> str:
    h, w = frame.shape[:2]
    nh   = int(h * width / w)
    thumb = cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return base64.b64encode(buf).decode("utf-8")


def write_frames_to_video(kept_frames: list, output_path: Path,
                          fps: float, frame_size: tuple):
    w, h = frame_size
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-preset", "veryfast",
        "-crf", str(OUTPUT_CRF),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path)
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for frame in kept_frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg encoding failed")
    print(f"[ENCODE] Done → {output_path}")


def generate_compression_report(segments: list, stats: dict, output_path: Path):
    orig_mb = stats["original_size_mb"]
    comp_mb = stats["compressed_size_mb"]
    red_pct = stats["reduction_pct"]
    bar_w   = max(2.0, 100.0 - red_pct)

    speed_x = round(stats["original_duration_sec"] /
                    max(stats["processing_time_sec"], 0.1), 1)

    reason_colors = {
        "face_detected":          "#4CAF50",
        "face_and_motion":        "#8BC34A",
        "motion_above_threshold": "#2196F3",
        "context_frame":          "#FF9800",
        "discarded_duplicate":    "#9E9E9E",
        "discarded_static":       "#607D8B",
    }

    thumb_html = ""
    for seg in segments:
        b64   = seg.get("thumbnail_b64", "")
        color = reason_colors.get(seg.get("reason_kept", ""), "#888")
        ts    = seg.get("start_sec", 0)
        label = seg.get("reason_kept", "")
        thumb_html += f"""
        <div class="thumb">
          <img src="data:image/jpeg;base64,{b64}" alt="{ts}s"/>
          <div class="thumb-label" style="border-top:3px solid {color}">
            <span>{ts:.1f}s</span>
            <span class="badge" style="background:{color}">{label}</span>
          </div>
        </div>"""

    disc = stats.get("frames_discarded_reasons", {})
    disc_rows = (
        f'<tr><td>Near-duplicate (pHash)</td>'
        f'<td>{disc.get("near_duplicate_phash", 0)}</td></tr>'
        f'<tr><td>Low motion / no face</td>'
        f'<td>{disc.get("low_motion_no_face", 0)}</td></tr>'
        f'<tr><td><b>Total discarded</b></td>'
        f'<td><b>{disc.get("total_discarded", 0)}</b></td></tr>'
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<title>Sentio Mind — Compression Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#0f1117;color:#e0e0e0;padding:24px}}
h1{{color:#00bcd4;margin-bottom:4px;font-size:1.8em}}
.subtitle{{color:#888;margin-bottom:24px;font-size:.9em}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:24px}}
.card{{background:#1a1d27;border-radius:10px;padding:18px;border:1px solid #2a2d3a}}
.card h3{{color:#00bcd4;font-size:.8em;text-transform:uppercase;margin-bottom:8px}}
.card .val{{font-size:2em;font-weight:700;color:#fff}}
.card .sub{{font-size:.8em;color:#888;margin-top:4px}}
.section{{background:#1a1d27;border-radius:10px;padding:20px;margin-bottom:20px;border:1px solid #2a2d3a}}
.section h2{{color:#00bcd4;margin-bottom:14px;font-size:1.1em}}
.bar-wrap{{background:#2a2d3a;border-radius:6px;height:22px;overflow:hidden;margin:6px 0}}
.bar{{height:100%;border-radius:6px;display:flex;align-items:center;padding-left:8px;font-size:.75em;font-weight:600;color:#fff}}
.storyboard{{display:flex;flex-wrap:wrap;gap:10px}}
.thumb{{width:200px}}
.thumb img{{width:200px;height:112px;object-fit:cover;border-radius:6px 6px 0 0;display:block}}
.thumb-label{{background:#1f2230;border-radius:0 0 6px 6px;padding:4px 6px;
              display:flex;justify-content:space-between;font-size:.72em}}
.badge{{border-radius:3px;padding:1px 5px;color:#fff;font-size:.85em}}
table{{width:100%;border-collapse:collapse;font-size:.83em}}
th{{color:#00bcd4;text-align:left;padding:6px 10px;border-bottom:1px solid #2a2d3a}}
td{{padding:5px 10px;border-bottom:1px solid #1e2030;color:#ccc}}
</style></head><body>
<h1>Sentio Mind — Smart Behavioral Video Compression</h1>
<p class="subtitle">Generated {time.strftime('%Y-%m-%d %H:%M:%S')} · {stats['source_video']}</p>

<div class="grid">
  <div class="card">
    <h3>Size Reduction</h3>
    <div class="val" style="color:#4CAF50">{red_pct:.1f}%</div>
    <div class="sub">{orig_mb:.1f} MB → {comp_mb:.1f} MB</div>
  </div>
  <div class="card">
    <h3>Frames Kept</h3>
    <div class="val" style="color:#2196F3">{stats['frames_kept']}</div>
    <div class="sub">of {stats['frames_original']} total</div>
  </div>
  <div class="card">
    <h3>Duration</h3>
    <div class="val">{stats['original_duration_sec']:.1f}s</div>
    <div class="sub">{stats['original_fps']:.0f} fps → {stats['output_fps']} fps</div>
  </div>
  <div class="card">
    <h3>Processing Speed</h3>
    <div class="val" style="color:#{'4CAF50' if speed_x >= 4 else 'FF5722'}">{speed_x}×</div>
    <div class="sub">real-time (target ≥ 4×) in {stats['processing_time_sec']}s</div>
  </div>
</div>

<div class="section"><h2>File Size Comparison</h2>
  <p style="margin-bottom:6px;font-size:.85em">Original: <b>{orig_mb:.2f} MB</b></p>
  <div class="bar-wrap"><div class="bar" style="width:100%;background:#e53935">{orig_mb:.2f} MB</div></div>
  <p style="margin-bottom:6px;margin-top:12px;font-size:.85em">Compressed: <b>{comp_mb:.2f} MB</b></p>
  <div class="bar-wrap"><div class="bar" style="width:{bar_w:.1f}%;background:#43a047">{comp_mb:.2f} MB</div></div>
</div>

<div class="section"><h2>Segment Storyboard</h2>
  <div class="storyboard">{thumb_html}</div>
</div>

<div class="section"><h2>Discarded Frames Breakdown</h2>
  <table><tr><th>Reason</th><th>Count</th></tr>{disc_rows}</table>
</div>
</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[REPORT] Saved → {output_path}")


if __name__ == "__main__":
    import sys
    video_in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else VIDEO_IN

    t_start = time.time()

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise RuntimeError("Haar cascade not found — check your opencv-python install")

    cap = cv2.VideoCapture(str(video_in_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_in_path}")

    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration   = total / fps_in
    orig_mb    = video_in_path.stat().st_size / 1_000_000

    print(f"Input : {video_in_path}  |  {total} frames  |  {duration:.1f}s  |  {orig_mb:.1f} MB")
    print(f"[SPEED] FRAME_SKIP={FRAME_SKIP}, ANALYSIS_WIDTH={ANALYSIS_WIDTH}px")

    motion_discard_thresh = auto_calibrate_motion_threshold(cap, calibration_secs=30.0)

    kept_frames  = []
    segments     = []
    prev_frame   = None
    prev_hash    = None
    last_kept_t  = -999.0
    cur_seg      = None
    disc_dup     = 0
    disc_stat    = 0
    frame_idx    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP != 0:
            prev_frame = frame
            frame_idx += 1
            continue

        ts = frame_idx / fps_in

        keep, reason, motion, face = should_keep_frame(
            frame, prev_frame, prev_hash, last_kept_t, ts,
            cascade, motion_discard_thresh
        )

        if keep:
            kept_frames.append(frame.copy())
            prev_hash   = compute_phash(get_small(frame))
            last_kept_t = ts

            if cur_seg is None or (ts - cur_seg["end_sec"]) > 2.5:
                if cur_seg:
                    segments.append(cur_seg)
                cur_seg = {
                    "segment_id":            len(segments) + 1,
                    "start_sec":             round(ts, 2),
                    "end_sec":               round(ts, 2),
                    "frames_in_segment":     1,
                    "reason_kept":           reason,
                    "face_count_in_segment": 1 if face else 0,
                    "motion_score_avg":      round(motion, 3),
                    "thumbnail_b64":         frame_to_b64_thumb(frame),
                }
            else:
                cur_seg["end_sec"]               = round(ts, 2)
                cur_seg["frames_in_segment"]     += 1
                cur_seg["face_count_in_segment"] += 1 if face else 0
                cur_seg["motion_score_avg"]       = round(
                    (cur_seg["motion_score_avg"] * (cur_seg["frames_in_segment"] - 1) + motion)
                    / cur_seg["frames_in_segment"], 3
                )
        else:
            if "duplicate" in reason:
                disc_dup  += 1
            else:
                disc_stat += 1

        prev_frame = frame
        frame_idx += 1

    if cur_seg:
        segments.append(cur_seg)
    cap.release()

    print(f"Kept {len(kept_frames)} / {total} frames across {len(segments)} segments")
    print("Writing compressed video ...")
    write_frames_to_video(kept_frames, VIDEO_OUT, OUTPUT_FPS, (fw, fh))

    comp_mb = VIDEO_OUT.stat().st_size / 1_000_000 if VIDEO_OUT.exists() else 0.0
    t_end   = time.time()

    stats = {
        "source_video":             str(video_in_path),
        "compressed_video":         str(VIDEO_OUT),
        "original_size_mb":         round(orig_mb, 2),
        "compressed_size_mb":       round(comp_mb, 2),
        "reduction_pct":            round((1 - comp_mb / (orig_mb + 1e-9)) * 100, 1),
        "original_duration_sec":    round(duration, 2),
        "compressed_duration_sec":  round(len(kept_frames) / OUTPUT_FPS, 2),
        "original_fps":             round(fps_in, 2),
        "output_fps":               OUTPUT_FPS,
        "frames_original":          total,
        "frames_kept":              len(kept_frames),
        "processing_time_sec":      round(t_end - t_start, 2),
        "segments":                 segments,
        "frames_discarded_reasons": {
            "near_duplicate_phash": disc_dup,
            "low_motion_no_face":   disc_stat,
            "total_discarded":      total - len(kept_frames),
        },
    }

    with open(SEGMENTS_JSON_OUT, "w") as f:
        json.dump(stats, f, indent=2)

    generate_compression_report(segments, stats, REPORT_HTML_OUT)

    speed_x = round(duration / max(t_end - t_start, 0.1), 1)
    print()
    print("=" * 55)
    print(f"  Done in {stats['processing_time_sec']}s  ({speed_x}× real-time)")
    print(f"  Size:     {orig_mb:.1f} MB  →  {comp_mb:.1f} MB  ({stats['reduction_pct']}% smaller)")
    print(f"  Duration: {duration:.1f}s  →  {stats['compressed_duration_sec']:.1f}s")
    print(f"  Segments: {len(segments)}")
    print(f"  Report  → {REPORT_HTML_OUT}")
    print(f"  JSON    → {SEGMENTS_JSON_OUT}")
    print("=" * 55)