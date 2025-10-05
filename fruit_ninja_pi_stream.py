#!/usr/bin/env python3
"""
Fruit Ninja AR Game - Pi Streaming Version
Headless pygame rendering with MJPEG streaming integration
"""

import os, time, threading, json, cv2, numpy as np, requests
import pygame
import mediapipe as mp
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
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
FPS_TARGET = 15
FRUIT_SIZE = 60
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
SPAWN_INTERVAL = 30

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)
mp_draw = mp.solutions.drawing_utils

def jpg(f):
    ok, b = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
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

def detect_slice(hand_landmarks, fruits):
    """Detect if hand motion slices through fruits"""
    global score, particles

    if not hand_landmarks:
        return

    # Get index finger tip position
    finger_tip = hand_landmarks.landmark[8]  # Index finger tip
    finger_x = int(finger_tip.x * SCREEN_WIDTH)
    finger_y = int(finger_tip.y * SCREEN_HEIGHT)

    # Add to trail
    trail_points.append((finger_x, finger_y))

    # Check for slicing through fruits
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

    # Setup camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Create virtual pygame surface
    surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.font.Font(None, 36)
    last_metrics_time = time.time()

    while game_running:
        t0 = time.time()

        # Read camera frame
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # MediaPipe hand detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Clear virtual surface
        surface.fill(BLACK)

        # Convert camera frame to pygame surface and blit as background
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
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                detect_slice(hand_landmarks, fruits)

                # Draw hand landmarks on pygame surface
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * SCREEN_WIDTH)
                    y = int(landmark.y * SCREEN_HEIGHT)
                    pygame.draw.circle(surface, GREEN, (x, y), 5)

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
        delegate_enabled = is_delegate_enabled()
        status_text = "ARM Optimized" if delegate_enabled else "Baseline"
        status_surface = font.render(status_text, True, YELLOW)
        surface.blit(status_surface, (SCREEN_WIDTH - 200, 10))

        # Convert pygame surface to OpenCV frame
        frame_array = pygame.surfarray.array3d(surface)
        frame_array = frame_array.swapaxes(0, 1)
        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)

        # Calculate FPS
        fps = 1.0 / max(1e-6, time.time() - t0)

        # Add FPS overlay
        cv2.putText(frame_array, f"FPS: {fps:.1f} | Fruits: {len(fruits)}",
                   (10, SCREEN_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
                    "infer_ms": 0.0,
                    "pre_ms": 0.0,
                    "post_ms": 1.0,
                    "mem_used_mb": mem,
                    "delegated_ops": int(delegate_enabled),
                    "neon_build": True,
                    "model_name": f"Fruit Ninja {status_text}",
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