"""
╔══════════════════════════════════════════════════════════════════╗
║                    EchoVision  –  main.py                        ║
║  Entry point. Wires together every module in src/:                ║
║                                                                    ║
║    camera → detector (COCO + navigation hazards, with automatic   ║
║    low-light enhancement) → obstacle_filter (dedupe + hazard       ║
║    tagging) → depth_estimator (distance) → direction_detector      ║
║    (left/center/right) → navigation_engine (prioritise) →          ║
║    sentence_generator (NLG) → speech_engine (spoken output)        ║
║                                                                    ║
║  Usage:                                                            ║
║      python main.py                        # webcam               ║
║      python main.py --source video.mp4      # video file           ║
║      python main.py --source 1              # second camera        ║
║      python main.py --save                  # save annotated video ║
║      python main.py --no-tts                # silent mode          ║
║      python main.py --no-low-light-boost     # disable auto-enhance║
║      python main.py --no-depth               # skip MiDaS (faster) ║
╚══════════════════════════════════════════════════════════════════╝
"""

import argparse
import time

import cv2

from src import config
from src.utils import log, draw_box, draw_hud, FPSMeter
from src.camera import VideoStream
from src.detector import ObjectDetector
from src.depth_estimator import DepthEstimator
from src.obstacle_filter import filter_detections
from src.navigation_engine import NavigationEngine
from src.sentence_generator import generate_announcement
from src.speech_engine import SpeechEngine


def parse_args():
    p = argparse.ArgumentParser(description="EchoVision – AI Voice Navigation Assistant")
    p.add_argument("--source", default="0",
                    help="Camera index (0, 1…) or path to a video file")
    p.add_argument("--save", action="store_true",
                    help="Save annotated output video to outputs/recordings/")
    p.add_argument("--no-tts", action="store_true", help="Disable voice output")
    p.add_argument("--no-low-light-boost", action="store_true",
                    help="Disable automatic low-light image enhancement")
    p.add_argument("--no-depth", action="store_true",
                    help="Skip MiDaS depth estimation (faster, no distances)")
    args = p.parse_args()
    try:
        args.source = int(args.source)
    except ValueError:
        pass
    return args


def main():
    args = parse_args()

    stream = VideoStream(args.source)
    detector = ObjectDetector(auto_low_light=not args.no_low_light_boost)
    depth_estimator = DepthEstimator() if not args.no_depth else None
    nav_engine = NavigationEngine()
    tts = SpeechEngine(enabled=not args.no_tts)
    tts.start()
    if not args.no_tts:
        tts.speak("EchoVision started.")

    writer = None
    if args.save:
        config.RECORDING_DIR.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = str(config.RECORDING_DIR / f"echovision_{int(time.time())}.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, stream.fps,
                                  (stream.width, stream.height))
        log.info(f"Recording to {out_path}")

    fps_meter = FPSMeter()
    frame_n = 0
    log.info("Running … press ESC to quit.")

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                continue
            frame_n += 1
            fps = fps_meter.tick()

            detections, infer_frame, low_light = detector.detect(frame)
            filtered = filter_detections(detections)

            dmap = None
            if depth_estimator is not None and filtered:
                try:
                    dmap = depth_estimator.depth_map(infer_frame)
                except Exception as e:
                    log.warning(f"Depth error on frame {frame_n}: {e}")

            events = nav_engine.process(filtered, infer_frame.shape[1],
                                         depth_estimator, dmap) if depth_estimator else \
                     nav_engine.process(filtered, infer_frame.shape[1], None, None)

            # ── Draw all filtered detections (not just the top events) ──
            from src.direction_detector import get_direction
            for det in filtered:
                distance = (depth_estimator.box_distance(dmap, det.bbox)
                            if (depth_estimator is not None and dmap is not None) else None)
                direction = get_direction(det.bbox, infer_frame.shape[1])
                kind = "hazard" if det.hazard else "coco"
                if distance is not None and distance <= config.NEAR_DISTANCE_M:
                    kind = "warning" if not det.hazard else "hazard"
                draw_box(infer_frame, *det.bbox, det.label, distance, direction, kind)

            draw_hud(infer_frame, frame_n, fps, len(filtered),
                     config.TTS_INTERVAL_FRAMES - (frame_n % config.TTS_INTERVAL_FRAMES),
                     low_light)

            # ── Voice announcement, throttled ────────────────────────────
            if frame_n % config.TTS_INTERVAL_FRAMES == 0 and events:
                sentence = generate_announcement(events)
                if sentence:
                    log.info(f"[TTS] {sentence}")
                    tts.speak(sentence)

            cv2.imshow("EchoVision", infer_frame)
            if writer:
                writer.write(infer_frame)

            if cv2.waitKey(1) & 0xFF == 27:   # ESC
                break
    finally:
        stream.stop()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        if not args.no_tts:
            tts.speak("EchoVision stopped.")
            time.sleep(1.2)
        tts.stop()
        log.info("EchoVision stopped.")


if __name__ == "__main__":
    main()
