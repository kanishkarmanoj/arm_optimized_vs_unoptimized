#!/usr/bin/env python3
"""
Fruit Ninja AR Game - Optimized Pi Version
High-performance dual-resolution approach with MediaPipe hand tracking

Features:
- Threaded architecture: Camera, Game Loop, JPEG Encoder run independently
- Hand tracking: MediaPipe (Optimized) vs HSV color detection (Baseline)
- Performance: ~14 FPS with MediaPipe, ~25-30 FPS with HSV (after encoder threading)
"""

import os, time, threading, json, cv2, numpy as np, requests
import random
import math
from collections import deque
from queue import Queue
from flask import Flask, Response
import mediapipe as mp

# Environment setup for multi-threading
os.environ['OMP_NUM_THREADS'] = '4'  # OpenMP threads
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'  # TensorFlow intra-op parallelism
os.environ['TF_NUM_INTEROP_THREADS'] = '2'  # TensorFlow inter-op parallelism
os.environ['XNNPACK_FORCE_PTHREADPOOL_PARALLELISM'] = '1'  # Force XNNPACK multi-threading
os.environ['XNNPACK_NUM_THREADS'] = '4'  # XNNPACK thread pool size
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# MediaPipe Hands setup (initialized once for efficiency)
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Track only 1 hand for better FPS
    min_detection_confidence=0.4,  # Lowered for speed (default 0.5)
    min_tracking_confidence=0.4,   # Lowered for faster tracking
    model_complexity=0  # 0=lite, 1=full (lite is fastest on RPi)
)

# Constants - OPTIMIZED for Pi 4 performance
# Working resolution: 640x360 (no wasteful resizes, MJPG-friendly)
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360
DETECTION_WIDTH = 320  # MediaPipe detection resolution (upscaled to display)
DETECTION_HEIGHT = 180
FPS_TARGET = 30
FRUIT_SIZE = 50  # Scaled down for 640x360
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

# Load PNG assets for fruits and sword
ASSETS_PATH = os.path.join(os.path.dirname(__file__), "fruit_ninja_game", "assets")
fruit_images = {}
sword_image = None

def load_assets():
    """Load and resize all fruit and sword images"""
    global fruit_images, sword_image

    # Load fruit images (whole and cut versions)
    fruit_types = ["apple", "orange", "watermelon"]
    for fruit_type in fruit_types:
        # Load whole fruit
        whole_path = os.path.join(ASSETS_PATH, f"{fruit_type}.png")
        cut_path = os.path.join(ASSETS_PATH, f"{fruit_type}_cut.png")

        whole_img = cv2.imread(whole_path, cv2.IMREAD_UNCHANGED)
        cut_img = cv2.imread(cut_path, cv2.IMREAD_UNCHANGED)

        if whole_img is not None and cut_img is not None:
            # Resize to appropriate size (fruits are ~80px for better visibility)
            whole_img = cv2.resize(whole_img, (80, 80))
            cut_img = cv2.resize(cut_img, (80, 80))

            fruit_images[fruit_type] = {
                "whole": whole_img,
                "cut": cut_img
            }
            print(f"✓ Loaded {fruit_type} images")
        else:
            print(f"⚠️  Failed to load {fruit_type} images")

    # Load sword image
    sword_path = os.path.join(ASSETS_PATH, "sword.png")
    sword_img = cv2.imread(sword_path, cv2.IMREAD_UNCHANGED)
    if sword_img is not None:
        sword_image = cv2.resize(sword_img, (60, 60))
        print(f"✓ Loaded sword image")
    else:
        print(f"⚠️  Failed to load sword image")

# Camera capture thread state
latest_camera_frame = None
camera_frame_lock = threading.Lock()
camera_thread_running = False

# Encoder thread state
encoder_queue = Queue(maxsize=2)  # Small queue to prevent memory buildup
encoder_thread_running = False

# Game states
MENU = "menu"
PLAYING = "playing"
PAUSED = "paused"
GAME_OVER = "game_over"

# Game state
game_state = MENU
score = 0
missed = 0
fruits = []
particles = []
trail_points = deque(maxlen=TRAIL_LENGTH)
spawn_timer = 0
SPAWN_INTERVAL = 40
last_hand_pos = None
MAX_MISSES = 3  # Game over after 3 missed fruits

# Combo/Streak system
combo_count = 0
last_slice_time = 0
COMBO_TIMEOUT = 1.5  # seconds to maintain combo
max_combo = 0  # Track best combo

# Score popup animations
score_popups = []

# Countdown timer
countdown_timer = 0
countdown_active = False

# Pinch detection state
last_pinch_state = False  # Track previous pinch to detect release

# MediaPipe optimization: frame skipping with interpolation
mediapipe_frame_skip = 4  # Process every Nth frame (1=all, 2=half, 3=third, 4=quarter)
mediapipe_frame_counter = 0
mediapipe_last_pos = None
mediapipe_prev_pos = None

class ScorePopup:
    """Floating score text that appears when fruit is sliced"""
    def __init__(self, x, y, points, is_combo=False):
        self.x = x
        self.y = y
        self.points = points
        self.is_combo = is_combo
        self.life = 30  # frames to live
        self.velocity_y = -2  # Float upward

    def update(self):
        self.y += self.velocity_y
        self.life -= 1
        return self.life <= 0

    def draw(self, frame):
        if self.life > 0:
            alpha = self.life / 30.0
            color = YELLOW if self.is_combo else WHITE
            text = f"+{self.points}"

            # Draw text with fading effect (simulate alpha with color intensity)
            fade_color = tuple(int(c * alpha) for c in color)
            cv2.putText(frame, text, (int(self.x), int(self.y)),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, fade_color, 2)

def draw_heart(frame, x, y, filled=True):
    """Draw a heart shape at position (x, y)"""
    if filled:
        color = RED
    else:
        color = (100, 100, 100)  # Gray for empty hearts

    # Simple heart using circles and triangle
    # Top two circles
    cv2.circle(frame, (x - 8, y - 5), 8, color, -1)
    cv2.circle(frame, (x + 8, y - 5), 8, color, -1)
    # Bottom triangle (using polygon)
    pts = np.array([[x - 15, y - 5], [x, y + 12], [x + 15, y - 5]], np.int32)
    cv2.fillPoly(frame, [pts], color)

class Button:
    """Interactive button for menu/pause screens"""
    def __init__(self, x, y, width, height, text, font_scale=0.8):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.font_scale = font_scale
        self.is_hovered_state = False

    def is_hovered(self, pos):
        """Check if position is over button"""
        if pos is None:
            return False
        px, py = pos
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)

    def draw(self, frame, pos):
        """Draw button with hover effect"""
        self.is_hovered_state = self.is_hovered(pos)

        # Button colors
        if self.is_hovered_state:
            bg_color = (100, 200, 100)  # Green when hovered
            text_color = WHITE
            border_color = WHITE
        else:
            bg_color = (50, 50, 50)  # Dark gray
            text_color = WHITE
            border_color = (100, 100, 100)

        # Draw background
        cv2.rectangle(frame, (self.x, self.y),
                     (self.x + self.width, self.y + self.height),
                     bg_color, -1)

        # Draw border
        cv2.rectangle(frame, (self.x, self.y),
                     (self.x + self.width, self.y + self.height),
                     border_color, 2)

        # Draw text (centered)
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX,
                                    self.font_scale, 2)[0]
        text_x = self.x + (self.width - text_size[0]) // 2
        text_y = self.y + (self.height + text_size[1]) // 2
        cv2.putText(frame, self.text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, text_color, 2)

def overlay_png(background, overlay, x, y):
    """
    Overlay a PNG image with alpha channel onto background
    x, y = top-left position
    """
    h, w = overlay.shape[:2]

    # Ensure overlay fits within background bounds
    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return  # Skip if out of bounds

    # Extract alpha channel if present
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        rgb = overlay[:, :, :3]
    else:
        alpha = np.ones((h, w))
        rgb = overlay

    # Get region of interest
    roi = background[y:y+h, x:x+w]

    # Blend using alpha channel
    for c in range(3):
        roi[:, :, c] = rgb[:, :, c] * alpha + roi[:, :, c] * (1 - alpha)

def jpg(f, quality=60):
    """
    Encode frame to JPEG with configurable quality
    Quality 60 is a sweet spot: good compression, acceptable visual quality
    """
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
        self.fruit_type = random.choice(["apple", "orange", "watermelon"])
        self.size = 80  # Match image size
        self.rotation = 0
        self.rotation_speed = random.randint(-5, 5)
        self.slice_timer = 0  # Timer for showing cut animation

    def update(self):
        if not self.sliced:
            self.velocity_y += self.gravity
            self.y += self.velocity_y
            self.x += self.velocity_x
            self.rotation += self.rotation_speed

            if self.y > DISPLAY_HEIGHT + 100:
                return True
        else:
            # Sliced fruit animation - show cut version briefly
            self.slice_timer += 1
            if self.slice_timer > 3:  # Show for 3 frames
                return True
        return False

    def draw(self, frame):
        global fruit_images

        # Select image based on sliced state
        if self.fruit_type in fruit_images:
            if self.sliced:
                img = fruit_images[self.fruit_type]["cut"]
            else:
                img = fruit_images[self.fruit_type]["whole"]

            # Overlay PNG with transparency
            overlay_png(frame, img, int(self.x) - self.size//2, int(self.y) - self.size//2)
        else:
            # Fallback to colored circle if image not loaded
            color_map = {"apple": RED, "orange": ORANGE, "watermelon": GREEN}
            color = color_map.get(self.fruit_type, RED)
            cv2.circle(frame, (int(self.x), int(self.y)), self.size // 2, color, -1)

    def get_rect(self):
        return (self.x - self.size//2, self.y - self.size//2, self.size, self.size)

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.velocity_x = random.randint(-12, 12)
        self.velocity_y = random.randint(-18, -3)
        self.gravity = 0.6
        self.color = color
        self.life = 30
        self.size = random.randint(2, 7)  # Size variation

    def update(self):
        self.velocity_y += self.gravity
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.life -= 1
        self.size = max(1, self.size - 0.1)  # Shrink over time
        return self.life <= 0

    def draw(self, frame):
        if self.life > 0:
            alpha = self.life / 30.0
            color = tuple(int(c * alpha) for c in self.color)
            # Draw particle with current size
            cv2.circle(frame, (int(self.x), int(self.y)), int(self.size), color, -1)

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

def detect_hand_mediapipe(detection_frame):
    """
    MediaPipe-based hand detection with frame skipping + interpolation
    - Runs inference every 2nd frame only (cuts compute in half)
    - Interpolates position on skipped frames for smooth tracking
    - Boosts perceived FPS from 6 → 12+ with minimal accuracy loss
    Returns: (index_pos, thumb_pos, is_pinching, hand_landmarks)
    """
    global mediapipe_frame_counter, mediapipe_last_pos, mediapipe_prev_pos

    mediapipe_frame_counter += 1

    # Only run MediaPipe inference every Nth frame
    if mediapipe_frame_counter % mediapipe_frame_skip == 0:
        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = hands_detector.process(rgb_frame)

        if results.multi_hand_landmarks:
            # Get first hand (we only track 1 hand)
            hand_landmarks = results.multi_hand_landmarks[0]

            # Get index finger tip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            index_x = int(index_finger_tip.x * DISPLAY_WIDTH)
            index_y = int(index_finger_tip.y * DISPLAY_HEIGHT)

            # Get thumb tip (landmark 4)
            thumb_tip = hand_landmarks.landmark[4]
            thumb_x = int(thumb_tip.x * DISPLAY_WIDTH)
            thumb_y = int(thumb_tip.y * DISPLAY_HEIGHT)

            # Detect pinch (thumb tip to index finger tip distance)
            distance = math.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)
            is_pinching = distance < 20  # Pinch threshold - requires actual contact

            # Update position history for interpolation
            mediapipe_prev_pos = mediapipe_last_pos
            mediapipe_last_pos = (index_x, index_y, thumb_x, thumb_y, is_pinching)

            return (index_x, index_y), (thumb_x, thumb_y), is_pinching, hand_landmarks
        else:
            # No hand detected - clear history
            mediapipe_prev_pos = None
            mediapipe_last_pos = None
            return None, None, False, None
    else:
        # Skipped frame - interpolate between last two positions
        if mediapipe_last_pos and mediapipe_prev_pos:
            # Calculate interpolation factor (0.0 to 1.0)
            frame_offset = mediapipe_frame_counter % mediapipe_frame_skip
            t = frame_offset / mediapipe_frame_skip

            # Linear interpolation for index finger
            index_x = int(mediapipe_prev_pos[0] + (mediapipe_last_pos[0] - mediapipe_prev_pos[0]) * t)
            index_y = int(mediapipe_prev_pos[1] + (mediapipe_last_pos[1] - mediapipe_prev_pos[1]) * t)

            # Linear interpolation for thumb
            thumb_x = int(mediapipe_prev_pos[2] + (mediapipe_last_pos[2] - mediapipe_prev_pos[2]) * t)
            thumb_y = int(mediapipe_prev_pos[3] + (mediapipe_last_pos[3] - mediapipe_prev_pos[3]) * t)

            # Use last known pinch state
            is_pinching = mediapipe_last_pos[4] if len(mediapipe_last_pos) > 4 else False

            return (index_x, index_y), (thumb_x, thumb_y), is_pinching, None
        elif mediapipe_last_pos:
            # Only have one position, use it
            is_pinching = mediapipe_last_pos[4] if len(mediapipe_last_pos) > 4 else False
            index_pos = (mediapipe_last_pos[0], mediapipe_last_pos[1])
            thumb_pos = (mediapipe_last_pos[2], mediapipe_last_pos[3])
            return index_pos, thumb_pos, is_pinching, None
        else:
            # No position history
            return None, None, False, None

def detect_slice(hand_pos, fruits):
    """Detect if hand motion slices through fruits"""
    global score, particles, last_hand_pos, combo_count, last_slice_time, score_popups

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
                        sliced_any = True

                        # Combo system - check if slice is within timeout
                        current_time = time.time()
                        if current_time - last_slice_time < COMBO_TIMEOUT:
                            combo_count += 1
                        else:
                            combo_count = 1
                        last_slice_time = current_time

                        # Track max combo
                        global max_combo
                        max_combo = max(max_combo, combo_count)

                        # Calculate points with combo multiplier
                        base_points = 10
                        multiplier = min(combo_count, 5)  # Cap at 5x
                        points = base_points * multiplier
                        score += points

                        # Create score popup
                        is_combo = combo_count > 1
                        score_popups.append(ScorePopup(fruit.x, fruit.y, points, is_combo))

                        # Enhanced particles with fruit color
                        color_map = {"apple": RED, "orange": ORANGE, "watermelon": GREEN}
                        particle_color = color_map.get(fruit.fruit_type, RED)
                        # More particles - increased for better effect
                        particle_count = 30 if combo_count > 2 else 25
                        for _ in range(particle_count):
                            particles.append(Particle(fruit.x, fruit.y, particle_color))

                        # Don't remove immediately - let fruit.update() handle it
                        # This allows the cut animation to show for a few frames

    last_hand_pos = hand_pos
    return sliced_any

def spawn_fruit():
    """Spawn new fruits periodically"""
    global spawn_timer
    spawn_timer += 1
    if spawn_timer >= SPAWN_INTERVAL:
        fruits.append(Fruit())
        spawn_timer = 0

def reset_game():
    """Reset game state for new game"""
    global score, missed, fruits, particles, trail_points, spawn_timer, last_hand_pos
    global combo_count, last_slice_time, score_popups, max_combo, countdown_timer, countdown_active
    score = 0
    missed = 0
    fruits.clear()
    particles.clear()
    trail_points.clear()
    spawn_timer = 0
    last_hand_pos = None
    combo_count = 0
    last_slice_time = 0
    score_popups.clear()
    max_combo = 0
    countdown_timer = 90  # 3 seconds at 30 FPS
    countdown_active = True

def draw_menu(frame, hand_pos, is_pinching):
    """Draw main menu screen"""
    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (DISPLAY_WIDTH, DISPLAY_HEIGHT), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Title
    title = "FRUIT NINJA"
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)[0]
    title_x = (DISPLAY_WIDTH - title_size[0]) // 2
    cv2.putText(frame, title, (title_x, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, YELLOW, 3)

    # Start button
    start_button = Button(DISPLAY_WIDTH // 2 - 100, DISPLAY_HEIGHT // 2 - 30, 200, 60, "START GAME")
    start_button.draw(frame, hand_pos)

    # Instructions
    inst_text = "Pinch to select"
    inst_size = cv2.getTextSize(inst_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    inst_x = (DISPLAY_WIDTH - inst_size[0]) // 2
    cv2.putText(frame, inst_text, (inst_x, DISPLAY_HEIGHT - 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    return start_button.is_hovered_state

def draw_pause_overlay(frame, hand_pos, is_pinching):
    """Draw pause screen overlay"""
    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (DISPLAY_WIDTH, DISPLAY_HEIGHT), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Paused text
    paused_text = "PAUSED"
    paused_size = cv2.getTextSize(paused_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
    paused_x = (DISPLAY_WIDTH - paused_size[0]) // 2
    cv2.putText(frame, paused_text, (paused_x, 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, WHITE, 3)

    # Menu button
    menu_button = Button(DISPLAY_WIDTH // 2 - 100, DISPLAY_HEIGHT // 2 - 30, 200, 60, "Go to Menu?")
    menu_button.draw(frame, hand_pos)

    # Resume instruction
    resume_text = "Move hand away to resume"
    resume_size = cv2.getTextSize(resume_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    resume_x = (DISPLAY_WIDTH - resume_size[0]) // 2
    cv2.putText(frame, resume_text, (resume_x, DISPLAY_HEIGHT - 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    return menu_button.is_hovered_state

def draw_game_over(frame, hand_pos, is_pinching):
    """Draw game over screen"""
    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (DISPLAY_WIDTH, DISPLAY_HEIGHT), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Game Over text
    game_over_text = "GAME OVER"
    go_size = cv2.getTextSize(game_over_text, cv2.FONT_HERSHEY_DUPLEX, 1.3, 3)[0]
    go_x = (DISPLAY_WIDTH - go_size[0]) // 2
    cv2.putText(frame, game_over_text, (go_x, 80), cv2.FONT_HERSHEY_DUPLEX, 1.3, RED, 3)

    # Score display
    score_text = f"Final Score: {score}"
    score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    score_x = (DISPLAY_WIDTH - score_size[0]) // 2
    cv2.putText(frame, score_text, (score_x, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)

    # Retry button
    retry_button = Button(DISPLAY_WIDTH // 2 - 100, DISPLAY_HEIGHT // 2 - 30, 200, 60, "RETRY")
    retry_button.draw(frame, hand_pos)

    return retry_button.is_hovered_state

def camera_capture_thread():
    """
    Dedicated camera capture thread - runs independently of game loop
    This prevents camera I/O blocking from stalling the game
    """
    global latest_camera_frame, camera_thread_running

    # Setup camera with MJPG format for fast capture
    # Try /dev/video1 first (Brio 101), fallback to /dev/video0
    cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Verify camera settings
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fcc = ''.join([chr((fourcc >> (8*i)) & 0xFF) for i in range(4)])
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 60)
    print("CAMERA CAPTURE THREAD STARTED")
    print(f"  Format: {fcc}")
    print(f"  Resolution: {actual_w}x{actual_h}")
    print(f"  Target FPS: {actual_fps}")

    if fcc != 'MJPG':
        print(f"  ⚠️  WARNING: Using {fcc} instead of MJPG - expect slow performance!")
        print(f"  ⚠️  Run ./diagnose_camera.py to test camera capabilities")
    else:
        print(f"  ✓ MJPG active - good performance expected")
    print("=" * 60)

    camera_thread_running = True
    frame_count = 0
    slow_frame_warnings = 0

    while camera_thread_running:
        t0 = time.time()
        ret, frame = cap.read()
        read_time_ms = (time.time() - t0) * 1000

        if ret:
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)

            # Update shared frame (thread-safe)
            with camera_frame_lock:
                latest_camera_frame = frame

            frame_count += 1

            # Warn about slow camera reads (only first few times)
            if read_time_ms > 100 and slow_frame_warnings < 3:
                print(f"⚠️  Slow camera read: {read_time_ms:.1f}ms (frame {frame_count})")
                slow_frame_warnings += 1
                if slow_frame_warnings == 3:
                    print("   (suppressing further slow frame warnings)")

        # No sleep - capture as fast as possible

    cap.release()
    print("Camera capture thread stopped")

def encoder_thread():
    """
    Dedicated JPEG encoder thread - runs independently of game loop
    This prevents JPEG encoding from blocking the game loop
    Encoding can take 30-50ms per frame at 640x360, so offloading it
    allows the game loop to run much faster
    """
    global latest, encoder_thread_running

    print("=" * 60)
    print("JPEG ENCODER THREAD STARTED")
    print("  Encoding frames in background for streaming")
    print("=" * 60)

    encoder_thread_running = True
    encode_count = 0
    total_encode_time = 0

    while encoder_thread_running:
        try:
            # Get frame from queue (block with timeout)
            frame_data = encoder_queue.get(timeout=0.1)

            if frame_data is None:  # Shutdown signal
                break

            frame, quality = frame_data

            # Encode JPEG
            t0 = time.time()
            b = jpg(frame, quality)
            encode_time_ms = (time.time() - t0) * 1000.0

            if b:
                with lock:
                    latest = b

            encode_count += 1
            total_encode_time += encode_time_ms

            # Print stats every 100 frames
            if encode_count % 100 == 0:
                avg_time = total_encode_time / encode_count
                print(f"Encoder: {encode_count} frames, avg {avg_time:.1f}ms per frame")

        except Exception as e:
            if encoder_thread_running:  # Only print if not shutting down
                pass  # Timeout is normal when no frames

    print("JPEG encoder thread stopped")

def game_loop():
    """Optimized main game loop with state-based rendering"""
    global latest, game_running, missed, latest_camera_frame, game_state, last_pinch_state

    last_metrics_time = time.time()
    frame_count = 0
    fps_start_time = time.time()
    frames_without_camera = 0

    # Pause button area (top right)
    PAUSE_BUTTON_X = DISPLAY_WIDTH - 80
    PAUSE_BUTTON_Y = 10
    PAUSE_BUTTON_SIZE = 60

    print("=" * 60)
    print("GAME LOOP STARTED (non-blocking camera mode)")
    print("=" * 60)

    while game_running:
        t0 = time.time()

        # Get latest camera frame (NO BLOCKING - use whatever is available)
        with camera_frame_lock:
            frame = latest_camera_frame.copy() if latest_camera_frame is not None else None

        if frame is None:
            # No camera frame yet - wait a bit and try again
            frames_without_camera += 1
            if frames_without_camera % 30 == 1:
                print(f"⚠️  Waiting for camera... ({frames_without_camera} frames)")
            time.sleep(0.01)
            continue

        frames_without_camera = 0  # Reset counter

        # Create smaller frame for hand detection
        detection_frame = cv2.resize(frame, (DETECTION_WIDTH, DETECTION_HEIGHT))

        # Hand detection with real ARM optimization differences
        delegate_enabled = is_delegate_enabled()
        infer_start = time.time()

        # Hand detection: MediaPipe (ARM Optimized) vs HSV (Baseline)
        is_pinching = False
        hand_landmarks = None
        thumb_pos = None
        if delegate_enabled:
            # ARM Optimized: Use MediaPipe for accurate hand tracking
            hand_pos, thumb_pos, is_pinching, hand_landmarks = detect_hand_mediapipe(detection_frame)
        else:
            # Baseline: Use simple HSV color detection (no pinch support)
            hand_pos, hand_mask = detect_hand_hsv(detection_frame, use_arm_optimization=False)

        infer_ms = (time.time() - infer_start) * 1000.0

        # Detect pinch "click" (pinch release to prevent repeated triggers)
        pinch_clicked = False
        if is_pinching and not last_pinch_state:
            pinch_clicked = True
        last_pinch_state = is_pinching

        # STATE-BASED RENDERING
        if game_state == MENU:
            # MENU STATE
            button_hovered = draw_menu(frame, hand_pos, is_pinching)

            # Draw pinch visual feedback for menu interaction (thumb only)
            if delegate_enabled and thumb_pos:
                if is_pinching:
                    # Pinching: Blue outline with white fill (like a pressed button)
                    cv2.circle(frame, thumb_pos, 14, BLUE, 2)  # Outline
                    cv2.circle(frame, thumb_pos, 10, WHITE, -1)  # White fill
                else:
                    # Not pinching: Blue outline only (empty)
                    cv2.circle(frame, thumb_pos, 14, BLUE, 2)  # Outline only

            if pinch_clicked and button_hovered:
                # Start game
                reset_game()
                game_state = PLAYING
                print("Game started!")

        elif game_state == PLAYING:
            # PLAYING STATE

            # Add simple edge darkening for visual depth (fast)
            edge_darkness = 40
            frame[0:30, :] = cv2.subtract(frame[0:30, :], (edge_darkness, edge_darkness, edge_darkness))
            frame[DISPLAY_HEIGHT-30:DISPLAY_HEIGHT, :] = cv2.subtract(frame[DISPLAY_HEIGHT-30:DISPLAY_HEIGHT, :], (edge_darkness, edge_darkness, edge_darkness))
            frame[:, 0:40] = cv2.subtract(frame[:, 0:40], (edge_darkness, edge_darkness, edge_darkness))
            frame[:, DISPLAY_WIDTH-40:DISPLAY_WIDTH] = cv2.subtract(frame[:, DISPLAY_WIDTH-40:DISPLAY_WIDTH], (edge_darkness, edge_darkness, edge_darkness))

            # Countdown timer (3-2-1-GO!)
            global countdown_timer, countdown_active
            if countdown_active:
                countdown_timer -= 1
                if countdown_timer > 0:
                    # Display countdown
                    if countdown_timer > 60:
                        text = "3"
                    elif countdown_timer > 30:
                        text = "2"
                    else:
                        text = "1"

                    # Large centered countdown
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 3.0, 5)[0]
                    text_x = (DISPLAY_WIDTH - text_size[0]) // 2
                    text_y = (DISPLAY_HEIGHT + text_size[1]) // 2
                    cv2.putText(frame, text, (text_x, text_y),
                               cv2.FONT_HERSHEY_DUPLEX, 3.0, BLACK, 8)
                    cv2.putText(frame, text, (text_x, text_y),
                               cv2.FONT_HERSHEY_DUPLEX, 3.0, YELLOW, 5)
                    # Don't process game logic during countdown
                    continue
                elif countdown_timer == 0:
                    # Show "GO!"
                    text = "GO!"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 2.5, 5)[0]
                    text_x = (DISPLAY_WIDTH - text_size[0]) // 2
                    text_y = (DISPLAY_HEIGHT + text_size[1]) // 2
                    cv2.putText(frame, text, (text_x, text_y),
                               cv2.FONT_HERSHEY_DUPLEX, 2.5, BLACK, 8)
                    cv2.putText(frame, text, (text_x, text_y),
                               cv2.FONT_HERSHEY_DUPLEX, 2.5, GREEN, 5)
                else:
                    # Countdown finished
                    countdown_active = False

            # Game logic (only when countdown is done)
            if not countdown_active:
                spawn_fruit()

            # Update and draw fruits directly on frame
            for fruit in fruits[:]:
                if fruit.update():  # Fruit fell off screen or cut animation finished
                    fruits.remove(fruit)
                    if not fruit.sliced:  # Only count as missed if not sliced
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

            # Draw trail FIRST (so it appears behind the sword)
            if len(trail_points) > 1:
                for i in range(1, len(trail_points)):
                    alpha = i / len(trail_points)
                    thickness = max(1, int(8 * alpha))
                    color = tuple(int(255 * alpha) for _ in range(3))
                    cv2.line(frame, trail_points[i-1], trail_points[i], color, thickness)

            # Draw sword overlay at hand position (ON TOP of everything)
            if hand_pos and sword_image is not None:
                # Position sword so tip aligns with fingertip
                sword_w, sword_h = 60, 60
                sword_x = hand_pos[0] - sword_w // 2
                sword_y = hand_pos[1] - sword_h // 2

                # Overlay sword with transparency
                overlay_png(frame, sword_image, sword_x, sword_y)
            elif hand_pos:
                # Fallback to circles if sword not loaded
                cv2.circle(frame, hand_pos, 12, GREEN, -1)
                cv2.circle(frame, hand_pos, 8, WHITE, -1)

            # Update and draw score popups
            for popup in score_popups[:]:
                if popup.update():
                    score_popups.remove(popup)
            for popup in score_popups:
                popup.draw(frame)

            # Draw UI with enhanced styling
            # Background box for UI
            cv2.rectangle(frame, (5, 5), (250, 75), (0, 0, 0), -1)
            cv2.rectangle(frame, (5, 5), (250, 75), (100, 100, 100), 2)

            # Score with gold color
            cv2.putText(frame, f"Score: {score}", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, YELLOW, 2)

            # Hearts/Lives system (replace "Missed" counter)
            heart_x_start = 15
            heart_y = 55
            for i in range(MAX_MISSES):
                filled = i >= missed  # Heart is filled if not yet missed
                draw_heart(frame, heart_x_start + i * 35, heart_y, filled)

            # Mode badge removed - shown on website instead

            # Combo counter (center screen when combo > 1)
            if combo_count > 1:
                combo_text = f"COMBO x{combo_count}!"
                combo_size = cv2.getTextSize(combo_text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)[0]
                combo_x = (DISPLAY_WIDTH - combo_size[0]) // 2
                combo_y = 100
                # Draw with outline for visibility
                cv2.putText(frame, combo_text, (combo_x, combo_y),
                           cv2.FONT_HERSHEY_DUPLEX, 1.5, BLACK, 5)
                cv2.putText(frame, combo_text, (combo_x, combo_y),
                           cv2.FONT_HERSHEY_DUPLEX, 1.5, YELLOW, 3)

            # Check hover over pause button first to determine color
            pause_hovered = False
            if hand_pos:
                if (PAUSE_BUTTON_X - 10 <= hand_pos[0] <= PAUSE_BUTTON_X + PAUSE_BUTTON_SIZE and
                    PAUSE_BUTTON_Y <= hand_pos[1] <= PAUSE_BUTTON_Y + 30):
                    pause_hovered = True

            # Draw pause button (top right) with hover effect
            pause_color = YELLOW if pause_hovered else WHITE
            cv2.putText(frame, "PAUSE", (PAUSE_BUTTON_X - 10, PAUSE_BUTTON_Y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, pause_color, 2)

            # Pause game if hovering
            if pause_hovered:
                game_state = PAUSED
                print("Game paused!")

            # Check game over condition
            if missed >= MAX_MISSES:
                game_state = GAME_OVER
                print(f"Game Over! Final Score: {score}")

        elif game_state == PAUSED:
            # PAUSED STATE
            button_hovered = draw_pause_overlay(frame, hand_pos, is_pinching)

            # Draw pinch visual feedback for pause menu interaction (thumb only)
            if delegate_enabled and thumb_pos:
                if is_pinching:
                    # Pinching: Blue outline with white fill
                    cv2.circle(frame, thumb_pos, 14, BLUE, 2)
                    cv2.circle(frame, thumb_pos, 10, WHITE, -1)
                else:
                    # Not pinching: Blue outline only
                    cv2.circle(frame, thumb_pos, 14, BLUE, 2)

            if pinch_clicked and button_hovered:
                # Go to menu
                game_state = MENU
                print("Returning to menu...")
            elif hand_pos:
                # Resume if hand moves away from pause area
                if not (PAUSE_BUTTON_X <= hand_pos[0] <= PAUSE_BUTTON_X + PAUSE_BUTTON_SIZE and
                       PAUSE_BUTTON_Y <= hand_pos[1] <= PAUSE_BUTTON_Y + PAUSE_BUTTON_SIZE):
                    # Check if hand is not over the menu button either
                    menu_button_area = (DISPLAY_WIDTH // 2 - 100, DISPLAY_HEIGHT // 2 - 30, 200, 60)
                    if not (menu_button_area[0] <= hand_pos[0] <= menu_button_area[0] + menu_button_area[2] and
                           menu_button_area[1] <= hand_pos[1] <= menu_button_area[1] + menu_button_area[3]):
                        game_state = PLAYING
                        print("Game resumed!")

        elif game_state == GAME_OVER:
            # GAME OVER STATE
            button_hovered = draw_game_over(frame, hand_pos, is_pinching)

            # Draw pinch visual feedback for game over menu interaction (thumb only)
            if delegate_enabled and thumb_pos:
                if is_pinching:
                    # Pinching: Blue outline with white fill
                    cv2.circle(frame, thumb_pos, 14, BLUE, 2)
                    cv2.circle(frame, thumb_pos, 10, WHITE, -1)
                else:
                    # Not pinching: Blue outline only
                    cv2.circle(frame, thumb_pos, 14, BLUE, 2)

            if pinch_clicked and button_hovered:
                # Retry
                reset_game()
                game_state = PLAYING
                print("Retrying game!")

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

        # Determine stream quality BEFORE encoding
        if delegate_enabled:  # ARM Optimized: Slightly better quality
            stream_quality = 70
        else:  # Baseline: Standard quality
            stream_quality = 60

        # Add all performance overlays BEFORE encoding (so they appear in stream)
        cv2.putText(frame, f"FPS: {fps:.1f} | Frame: {frame_time*1000:.1f}ms | Fruits: {len(fruits)}",
                   (10, DISPLAY_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        # STREAMING OPTIMIZATION: Push frame to encoder thread (non-blocking)
        # This prevents JPEG encoding (30-50ms) from blocking the game loop
        try:
            encoder_queue.put_nowait((frame.copy(), stream_quality))
        except:
            pass  # Queue full, skip this frame (encoder will catch up)

        # Send metrics
        if time.time() - last_metrics_time > 0.3:
            try:
                mem = int(os.popen("free -m | awk '/Mem:/{print $3}'").read() or 0)
            except:
                mem = 0
            try:
                if delegate_enabled:
                    model_name = "ARM Optimized (MediaPipe Hands)"
                    delegated_ops = 1
                else:
                    model_name = "Baseline (HSV Color Detection)"
                    delegated_ops = 0

                requests.post(MET, json={
                    "fps": round(fps, 1),
                    "infer_ms": round(infer_ms, 1),
                    "pre_ms": 0.0,
                    "post_ms": 0.0,  # Encoding offloaded to separate thread
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

        # FPS limiter: Lock to 30 FPS for smooth, consistent performance
        target_frame_time = 1.0 / 30.0  # 33.3ms per frame
        elapsed = time.time() - t0
        sleep_time = target_frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

def run():
    """Start the camera capture thread, encoder thread, game loop, and Flask server"""
    global camera_thread_running, encoder_thread_running

    # Load assets FIRST
    print("=" * 60)
    print("LOADING ASSETS...")
    print("=" * 60)
    load_assets()
    print("=" * 60)

    # Start camera capture thread FIRST
    camera_thread = threading.Thread(target=camera_capture_thread, daemon=True, name="CameraCapture")
    camera_thread.start()

    # Start JPEG encoder thread
    jpeg_encoder_thread = threading.Thread(target=encoder_thread, daemon=True, name="JPEGEncoder")
    jpeg_encoder_thread.start()

    # Give threads time to initialize
    time.sleep(0.5)

    # Start game loop
    game_thread = threading.Thread(target=game_loop, daemon=True, name="GameLoop")
    game_thread.start()

    # Run Flask server (blocking)
    try:
        app.run(host="0.0.0.0", port=8090, threaded=True)
    finally:
        # Cleanup on exit
        global game_running
        game_running = False
        camera_thread_running = False
        encoder_thread_running = False
        # Send shutdown signal to encoder
        try:
            encoder_queue.put_nowait(None)
        except:
            pass

if __name__ == "__main__":
    run()