#!/usr/bin/env python3
"""
Fruit Ninja AR Game - Pi Streaming Version (Simplified)
Uses existing TFLite infrastructure, no MediaPipe dependency
"""

import os, time, threading, json, cv2, numpy as np, requests
import pygame
import random
import math
from collections import deque
from flask import Flask, Response

# Environment setup
cv2.setUseOptimized(True)
cv2.setNumThreads(4)
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Headless pygame

# Initialize pygame for headless operation
pygame.init()
pygame.display.set_mode((1, 1))  # Minimal display

# Constants
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS_TARGET = 15
FRUIT_SIZE = 80
TRAIL_LENGTH = 10

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
MODEL = os.path.expanduser("~/models/hand.tflite")

app = Flask(__name__)
latest = None
lock = threading.Lock()
game_running = True

# TFLite setup (from original ml_game.py)
tfl = None
load_delegate = None
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    tfl = Interpreter
except Exception:
    pass

# Game state
score = 0
missed = 0
fruits = []
particles = []
trail_points = deque(maxlen=TRAIL_LENGTH)
spawn_timer = 0
SPAWN_INTERVAL = 30
last_hand_pos = None

def jpg(f):
    ok, b = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
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

def make_interpreter(use_delegate):
    """Create TFLite interpreter (from original ml_game.py)"""
    if not tfl or not os.path.exists(MODEL):
        return None, None

    if use_delegate:
        os.environ.pop("TF_LITE_DISABLE_XNNPACK", None)
        threads = 4
    else:
        os.environ["TF_LITE_DISABLE_XNNPACK"] = "1"
        threads = 1

    delegates = []
    if use_delegate and load_delegate:
        for p in ["/opt/armnn/build/delegate/libarmnnDelegate.so", "libarmnnDelegate.so"]:
            try:
                delegates = [load_delegate(p, {})]
                break
            except Exception:
                pass

    try:
        interp = tfl(MODEL, experimental_delegates=delegates, num_threads=threads)
        interp.allocate_tensors()
        return interp, (interp.get_input_details(), interp.get_output_details(), bool(delegates))
    except Exception:
        return None, None

def run_infer(interp, meta, frame):
    """Run hand detection inference (simplified from original)"""
    if not interp or not meta:
        return 0.0, None

    input_details, output_details, using_delegate = meta

    # Resize frame for inference
    input_shape = input_details[0]['shape']
    h, w = input_shape[1], input_shape[2]
    frame_resized = cv2.resize(frame, (w, h))

    # Normalize
    if input_details[0]['dtype'] == np.float32:
        frame_resized = frame_resized.astype(np.float32) / 255.0

    # Run inference
    t0 = time.time()
    interp.set_tensor(input_details[0]['index'], frame_resized[None, ...])
    interp.invoke()
    infer_ms = (time.time() - t0) * 1000.0

    # Get output (simplified - just return center point if confident)
    output = interp.get_tensor(output_details[0]['index'])

    # Simple hand detection - look for high confidence detection
    if len(output.shape) >= 2 and output.size > 0:
        confidence = np.max(output)
        if confidence > 0.5:
            # Return approximate center of detected region
            h_idx, w_idx = np.unravel_index(np.argmax(output), output.shape[-2:])
            hand_x = int((w_idx / output.shape[-1]) * SCREEN_WIDTH)
            hand_y = int((h_idx / output.shape[-2]) * SCREEN_HEIGHT)
            return infer_ms, (hand_x, hand_y)

    return infer_ms, None

class Fruit:
    def __init__(self):
        self.x = random.randint(50, SCREEN_WIDTH - 50)
        self.y = SCREEN_HEIGHT + 30
        self.velocity_y = random.randint(-20, -12)
        self.velocity_x = random.randint(-3, 3)
        self.gravity = 0.6
        self.sliced = False
        self.color = random.choice([RED, YELLOW, GREEN, ORANGE])
        self.size = FRUIT_SIZE
        self.rotation = 0
        self.rotation_speed = random.randint(-3, 3)

    def update(self):
        if not self.sliced:
            self.velocity_y += self.gravity
            self.y += self.velocity_y
            self.x += self.velocity_x
            self.rotation += self.rotation_speed

            if self.y > SCREEN_HEIGHT + 50:
                return True
        return False

    def draw(self, surface):
        if not self.sliced:
            pygame.draw.circle(surface, self.color,
                             (int(self.x), int(self.y)), self.size // 2)
            # Highlight
            pygame.draw.circle(surface, WHITE,
                             (int(self.x - 10), int(self.y - 10)), 8)

    def get_rect(self):
        return pygame.Rect(self.x - self.size//2, self.y - self.size//2,
                          self.size, self.size)

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.velocity_x = random.randint(-8, 8)
        self.velocity_y = random.randint(-12, -4)
        self.gravity = 0.4
        self.color = color
        self.life = 30

    def update(self):
        self.velocity_y += self.gravity
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.life -= 1
        return self.life <= 0

    def draw(self, surface):
        if self.life > 0:
            alpha = self.life / 30.0
            color = tuple(int(c * alpha) for c in self.color)
            pygame.draw.circle(surface, color, (int(self.x), int(self.y)), 3)

def detect_slice(hand_pos, fruits):
    """Detect if hand motion slices through fruits"""
    global score, particles, last_hand_pos

    if not hand_pos:
        return

    finger_x, finger_y = hand_pos

    # Add to trail
    trail_points.append((finger_x, finger_y))

    # Check for slicing through fruits (only if hand is moving)
    if last_hand_pos:
        last_x, last_y = last_hand_pos
        movement = math.sqrt((finger_x - last_x)**2 + (finger_y - last_y)**2)

        if movement > 10:  # Only slice if hand is moving
            for fruit in fruits[:]:
                if not fruit.sliced:
                    fruit_rect = fruit.get_rect()
                    if fruit_rect.collidepoint(finger_x, finger_y):
                        fruit.sliced = True
                        score += 10

                        # Create particles
                        for _ in range(8):
                            particles.append(Particle(fruit.x, fruit.y, fruit.color))

                        fruits.remove(fruit)

    last_hand_pos = hand_pos

def spawn_fruit():
    """Spawn new fruits periodically"""
    global spawn_timer
    spawn_timer += 1
    if spawn_timer >= SPAWN_INTERVAL:
        fruits.append(Fruit())
        spawn_timer = 0

def game_loop():
    """Main game loop running in separate thread"""
    global latest, game_running, missed

    # Setup camera with high resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Verify actual resolution
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution set to: {actual_w}x{actual_h}")

    # Create virtual pygame surface
    surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.font.Font(None, 48)  # Larger font for HD resolution
    last_metrics_time = time.time()

    # Initialize TFLite
    delegate_enabled = is_delegate_enabled()
    interp, meta = make_interpreter(delegate_enabled)

    while game_running:
        t0 = time.time()

        # Check if delegate setting changed
        current_delegate = is_delegate_enabled()
        if current_delegate != delegate_enabled:
            delegate_enabled = current_delegate
            interp, meta = make_interpreter(delegate_enabled)

        # Read camera frame
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Hand detection
        infer_ms, hand_pos = run_infer(interp, meta, frame)

        # Clear virtual surface with camera background
        surface.fill(BLACK)

        # Convert camera frame to pygame surface and blit as background
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        surface.blit(frame_surface, (0, 0))

        # Game logic
        spawn_fruit()

        # Update and draw fruits
        for fruit in fruits[:]:
            if fruit.update():  # Fruit fell off screen
                fruits.remove(fruit)
                missed += 1

        for fruit in fruits:
            fruit.draw(surface)

        # Update and draw particles
        for particle in particles[:]:
            if particle.update():
                particles.remove(particle)

        for particle in particles:
            particle.draw(surface)

        # Hand tracking and slicing
        detect_slice(hand_pos, fruits)

        # Draw hand position
        if hand_pos:
            pygame.draw.circle(surface, GREEN, hand_pos, 8)

        # Draw trail
        if len(trail_points) > 1:
            for i in range(1, len(trail_points)):
                alpha = i / len(trail_points)
                color = tuple(int(255 * alpha) for _ in range(3))
                pygame.draw.circle(surface, color, trail_points[i], max(1, int(5 * alpha)))

        # Draw UI
        score_text = font.render(f"Score: {score}", True, WHITE)
        missed_text = font.render(f"Missed: {missed}", True, WHITE)
        surface.blit(score_text, (10, 10))
        surface.blit(missed_text, (10, 50))

        # Game status
        status_text = "ARM Optimized" if delegate_enabled else "Baseline"
        status_surface = font.render(status_text, True, YELLOW)
        surface.blit(status_surface, (SCREEN_WIDTH - 200, 10))

        # Convert pygame surface to OpenCV frame
        frame_array = pygame.surfarray.array3d(surface)
        frame_array = frame_array.swapaxes(0, 1)
        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)

        # Calculate FPS
        fps = 1.0 / max(1e-6, time.time() - t0)

        # Add overlays to final frame
        model_name = ("TFLite+ArmNN" if (delegate_enabled and meta and meta[-1])
                     else "TFLite (XNN ON)" if delegate_enabled
                     else "TFLite (XNN OFF)" if interp
                     else "No Hand Detection")

        cv2.putText(frame_array, f"{model_name} | FPS: {fps:.1f} | Infer: {infer_ms:.1f}ms",
                   (10, SCREEN_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame_array, f"Fruits: {len(fruits)} | Score: {score} | Missed: {missed}",
                   (10, SCREEN_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Update stream
        b = jpg(frame_array)
        if b:
            with lock:
                latest = b

        # Send metrics
        if time.time() - last_metrics_time > 0.3:
            try:
                mem = int(os.popen("free -m | awk '/Mem:/{print $3}'").read() or 0)
            except:
                mem = 0
            try:
                requests.post(MET, json={
                    "fps": round(fps, 1),
                    "infer_ms": round(infer_ms, 1),
                    "pre_ms": 0.0,
                    "post_ms": 1.0,
                    "mem_used_mb": mem,
                    "delegated_ops": int(delegate_enabled and meta and meta[-1] if meta else False),
                    "neon_build": True,
                    "model_name": model_name,
                    "game_score": score,
                    "game_missed": missed,
                    "game_fruits": len(fruits)
                }, timeout=0.4)
            except:
                pass
            last_metrics_time = time.time()

        # Target FPS control
        elapsed = time.time() - t0
        target_delay = 1.0 / FPS_TARGET
        if elapsed < target_delay:
            time.sleep(target_delay - elapsed)

    cap.release()

def run():
    """Start the game and Flask server"""
    threading.Thread(target=game_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=8090, threaded=True)

if __name__ == "__main__":
    run()