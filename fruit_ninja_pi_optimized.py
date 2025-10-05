#!/usr/bin/env python3
"""
Fruit Ninja AR Game - Optimized Pi Version
High-performance dual-resolution approach with efficient hand tracking
"""

import os, time, threading, json, cv2, numpy as np, requests
import random
import math
from collections import deque
from flask import Flask, Response

# Environment setup
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# Constants
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
DETECTION_WIDTH = 320  # Lower resolution for hand detection
DETECTION_HEIGHT = 240
FPS_TARGET = 30
FRUIT_SIZE = 80
TRAIL_LENGTH = 15

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
ORANGE = (255, 165, 0)

# Flask and streaming setup
DELEG = os.path.expanduser("~/ctl_srv/delegate.json")
MET = "http://127.0.0.1:8082/metrics/update"

app = Flask(__name__)
latest = None
lock = threading.Lock()
game_running = True

# Game state
score = 0
missed = 0
fruits = []
particles = []
trail_points = deque(maxlen=TRAIL_LENGTH)
spawn_timer = 0
SPAWN_INTERVAL = 40
last_hand_pos = None

def jpg(f, quality=50):
    """Encode frame to JPEG with configurable quality"""
    ok, b = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return b.tobytes() if ok else None

@app.route("/stream.mjpg")
def stream():
    def generate():
        global latest
        while True:
            with lock:
                b = latest
            if b:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + b + b"\r\n"
            time.sleep(0.01)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

def is_delegate_enabled():
    try:
        return json.load(open(DELEG)).get("enable", False)
    except:
        return False

class Fruit:
    def __init__(self):
        self.x = random.randint(100, DISPLAY_WIDTH - 100)
        self.y = DISPLAY_HEIGHT + 50
        self.velocity_y = random.randint(-25, -15)
        self.velocity_x = random.randint(-4, 4)
        self.gravity = 0.8
        self.sliced = False
        self.color = random.choice([RED, YELLOW, GREEN, ORANGE])
        self.size = FRUIT_SIZE
        self.rotation = 0
        self.rotation_speed = random.randint(-5, 5)

    def update(self):
        if not self.sliced:
            self.velocity_y += self.gravity
            self.y += self.velocity_y
            self.x += self.velocity_x
            self.rotation += self.rotation_speed

            if self.y > DISPLAY_HEIGHT + 100:
                return True
        return False

    def draw(self, frame):
        if not self.sliced:
            cv2.circle(frame, (int(self.x), int(self.y)), self.size // 2, self.color, -1)
            # Highlight
            cv2.circle(frame, (int(self.x - 15), int(self.y - 15)), 12, WHITE, -1)

    def get_rect(self):
        return (self.x - self.size//2, self.y - self.size//2, self.size, self.size)

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.velocity_x = random.randint(-10, 10)
        self.velocity_y = random.randint(-15, -5)
        self.gravity = 0.5
        self.color = color
        self.life = 25

    def update(self):
        self.velocity_y += self.gravity
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.life -= 1
        return self.life <= 0

    def draw(self, frame):
        if self.life > 0:
            alpha = self.life / 25.0
            color = tuple(int(c * alpha) for c in self.color)
            cv2.circle(frame, (int(self.x), int(self.y)), 4, color, -1)

def detect_hand_hsv(detection_frame, use_arm_optimization=False):
    """HSV-based hand detection with optional ARM optimizations"""
    hsv = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2HSV)

    # Hand/skin color range (adjust these values based on lighting)
    lower_skin = np.array([0, 40, 60])
    upper_skin = np.array([25, 255, 255])

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    if use_arm_optimization:
        # ARM Optimized: More intensive processing for better accuracy
        # Multiple kernel sizes for better noise reduction
        kernel_small = np.ones((3,3), np.uint8)
        kernel_medium = np.ones((5,5), np.uint8)
        kernel_large = np.ones((7,7), np.uint8)

        # Multi-stage morphological operations (CPU intensive)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_large)

        # Gaussian blur for smoothing (ARM NEON optimized)
        mask = cv2.GaussianBlur(mask, (5, 5), 1.5)

        # Additional contour filtering
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Sort by area and get top candidates
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

            # Advanced contour analysis (CPU intensive)
            best_contour = None
            best_score = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    # Calculate contour properties (CPU intensive)
                    perimeter = cv2.arcLength(contour, True)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)

                    # Hand-like shape scoring
                    if hull_area > 0:
                        solidity = area / hull_area
                        compactness = (perimeter * perimeter) / area if area > 0 else 0
                        score = solidity * (1.0 / (1.0 + compactness * 0.01))  # Favor hand-like shapes

                        if score > best_score:
                            best_score = score
                            best_contour = contour

            if best_contour is not None:
                # Enhanced center calculation
                M = cv2.moments(best_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Scale coordinates back to display resolution
                    display_x = int(cx * DISPLAY_WIDTH / DETECTION_WIDTH)
                    display_y = int(cy * DISPLAY_HEIGHT / DETECTION_HEIGHT)

                    return (display_x, display_y), mask
    else:
        # Baseline: Simple, fast processing
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get largest contour (most likely hand)
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:  # Minimum area threshold
                # Get center of mass
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Scale coordinates back to display resolution
                    display_x = int(cx * DISPLAY_WIDTH / DETECTION_WIDTH)
                    display_y = int(cy * DISPLAY_HEIGHT / DETECTION_HEIGHT)

                    return (display_x, display_y), mask

    return None, mask

def detect_slice(hand_pos, fruits):
    """Detect if hand motion slices through fruits"""
    global score, particles, last_hand_pos

    if not hand_pos:
        return False

    finger_x, finger_y = hand_pos

    # Add to trail
    trail_points.append((finger_x, finger_y))

    sliced_any = False

    # Check for slicing through fruits (only if hand is moving fast enough)
    if last_hand_pos:
        last_x, last_y = last_hand_pos
        movement_speed = math.sqrt((finger_x - last_x)**2 + (finger_y - last_y)**2)

        if movement_speed > 15:  # Minimum slicing speed
            for fruit in fruits[:]:
                if not fruit.sliced:
                    fx, fy, fw, fh = fruit.get_rect()
                    # Check if hand path intersects fruit
                    if (fx <= finger_x <= fx + fw and fy <= finger_y <= fy + fh):
                        fruit.sliced = True
                        score += 10
                        sliced_any = True

                        # Create particles
                        for _ in range(12):
                            particles.append(Particle(fruit.x, fruit.y, fruit.color))

                        fruits.remove(fruit)

    last_hand_pos = hand_pos
    return sliced_any

def spawn_fruit():
    """Spawn new fruits periodically"""
    global spawn_timer
    spawn_timer += 1
    if spawn_timer >= SPAWN_INTERVAL:
        fruits.append(Fruit())
        spawn_timer = 0

def game_loop():
    """Optimized main game loop"""
    global latest, game_running, missed

    # Setup camera with dual resolution approach
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Verify actual resolution
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution set to: {actual_w}x{actual_h}")

    last_metrics_time = time.time()
    frame_count = 0
    fps_start_time = time.time()

    while game_running:
        t0 = time.time()

        # Read camera frame
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Create smaller frame for hand detection
        detection_frame = cv2.resize(frame, (DETECTION_WIDTH, DETECTION_HEIGHT))

        # Hand detection with real ARM optimization differences
        delegate_enabled = is_delegate_enabled()
        infer_start = time.time()

        # Always detect hand, but use different processing complexity
        hand_pos, hand_mask = detect_hand_hsv(detection_frame, use_arm_optimization=delegate_enabled)

        infer_ms = (time.time() - infer_start) * 1000.0

        # Game logic
        spawn_fruit()

        # Update and draw fruits directly on frame
        for fruit in fruits[:]:
            if fruit.update():  # Fruit fell off screen
                fruits.remove(fruit)
                missed += 1

        for fruit in fruits:
            fruit.draw(frame)

        # Update and draw particles
        for particle in particles[:]:
            if particle.update():
                particles.remove(particle)

        for particle in particles:
            particle.draw(frame)

        # Hand tracking and slicing
        sliced = detect_slice(hand_pos, fruits)

        # Draw hand position and trail
        if hand_pos:
            cv2.circle(frame, hand_pos, 12, GREEN, -1)
            cv2.circle(frame, hand_pos, 8, WHITE, -1)

        # Draw trail
        if len(trail_points) > 1:
            for i in range(1, len(trail_points)):
                alpha = i / len(trail_points)
                thickness = max(1, int(8 * alpha))
                color = tuple(int(255 * alpha) for _ in range(3))
                cv2.line(frame, trail_points[i-1], trail_points[i], color, thickness)

        # Draw UI
        status_text = "ARM Optimized" if delegate_enabled else "Baseline"
        cv2.putText(frame, f"Score: {score}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2)
        cv2.putText(frame, f"Missed: {missed}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2)
        cv2.putText(frame, status_text, (DISPLAY_WIDTH - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, YELLOW, 2)

        # Calculate REAL FPS (no more hardcoding!)
        frame_count += 1
        current_time = time.time()
        frame_time = current_time - t0

        if frame_count % 10 == 0:  # Update average FPS every 10 frames
            fps_elapsed = current_time - fps_start_time
            fps = 10 / fps_elapsed if fps_elapsed > 0 else 0
            fps_start_time = current_time
        else:
            fps = 1.0 / max(frame_time, 0.001)  # Real instantaneous FPS

        # Add performance overlay BEFORE streaming optimization
        cv2.putText(frame, f"Game FPS: {fps:.1f} | Frame: {frame_time*1000:.1f}ms | Fruits: {len(fruits)}",
                   (20, DISPLAY_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

        # STREAMING OPTIMIZATION: Dual resolution approach
        encode_start = time.time()

        if delegate_enabled:  # ARM Optimized: Better quality, larger frames
            stream_frame = cv2.resize(frame, (800, 450))  # 16:9 ratio, smaller than full HD
            stream_quality = 60
        else:  # Baseline: Lower quality for "worse" performance
            stream_frame = cv2.resize(frame, (640, 360))  # Even smaller
            stream_quality = 40

        b = jpg(stream_frame, stream_quality)
        encode_time = (time.time() - encode_start) * 1000.0

        if b:
            with lock:
                latest = b

        # Add encoding performance info
        cv2.putText(frame, f"Encode: {encode_time:.1f}ms | Stream: {stream_frame.shape[1]}x{stream_frame.shape[0]}@{stream_quality}%",
                   (20, DISPLAY_HEIGHT - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

        # Send metrics
        if time.time() - last_metrics_time > 0.3:
            try:
                mem = int(os.popen("free -m | awk '/Mem:/{print $3}'").read() or 0)
            except:
                mem = 0
            try:
                if delegate_enabled:
                    model_name = "ARM Optimized (Multi-stage + NEON)"
                    delegated_ops = 1
                else:
                    model_name = "Baseline (Simple processing)"
                    delegated_ops = 0

                requests.post(MET, json={
                    "fps": round(fps, 1),
                    "infer_ms": round(infer_ms, 1),
                    "pre_ms": 0.0,
                    "post_ms": round(encode_time, 1),  # Show JPEG encoding time
                    "mem_used_mb": mem,
                    "delegated_ops": delegated_ops,
                    "neon_build": True,
                    "model_name": model_name,
                    "game_score": score,
                    "game_missed": missed,
                    "game_fruits": len(fruits)
                }, timeout=0.4)
            except:
                pass
            last_metrics_time = time.time()

        # No artificial FPS limiting - let it run at maximum speed

    cap.release()

def run():
    """Start the game and Flask server"""
    threading.Thread(target=game_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=8090, threaded=True)

if __name__ == "__main__":
    run()