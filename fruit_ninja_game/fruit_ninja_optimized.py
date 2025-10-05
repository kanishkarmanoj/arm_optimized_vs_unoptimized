import cv2
import mediapipe as mp
import pygame
import random
import time
import numpy as np
from collections import deque

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS_TARGET = 60
FRUIT_SIZE = 80
TRAIL_LENGTH = 15

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)

# Setup display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Fruit Ninja - ARM OPTIMIZED")
clock = pygame.time.Clock()

# Initialize MediaPipe Hands (ARM OPTIMIZED)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)

# Initialize camera with optimizations
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Game variables
score = 0
missed = 0

# Sword trail with smoothing
trail_points = deque(maxlen=TRAIL_LENGTH)

# Fruit class
class Fruit:
    def __init__(self):
        self.x = random.randint(100, SCREEN_WIDTH - 100)
        self.y = SCREEN_HEIGHT + 50
        self.velocity_y = random.randint(-28, -18)
        self.velocity_x = random.randint(-5, 5)
        self.gravity = 0.8
        self.sliced = False
        self.color = random.choice([RED, YELLOW, GREEN])
        self.size = FRUIT_SIZE
        self.rotation = 0
        self.rotation_speed = random.randint(-5, 5)
        
    def update(self):
        if not self.sliced:
            self.velocity_y += self.gravity
            self.y += self.velocity_y
            self.x += self.velocity_x
            self.rotation += self.rotation_speed
            
            if self.y > SCREEN_HEIGHT + 100:
                return True
        return False
    
    def draw(self, surface):
        if not self.sliced:
            pygame.draw.circle(surface, self.color, 
                             (int(self.x), int(self.y)), self.size // 2)
            pygame.draw.circle(surface, WHITE, 
                             (int(self.x - 15), int(self.y - 15)), 12)
            pygame.draw.circle(surface, self.color, 
                             (int(self.x - 15), int(self.y - 15)), 8)
    
    def get_rect(self):
        return pygame.Rect(self.x - self.size//2, self.y - self.size//2, 
                          self.size, self.size)

# Particle effect
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.velocity_x = random.randint(-10, 10)
        self.velocity_y = random.randint(-15, -5)
        self.gravity = 0.5
        self.color = color
        self.size = random.randint(3, 8)
        self.lifetime = 30
        
    def update(self):
        self.velocity_y += self.gravity
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.lifetime -= 1
        return self.lifetime <= 0
    
    def draw(self, surface):
        if self.lifetime > 0:
            pygame.draw.circle(surface, self.color, 
                             (int(self.x), int(self.y)), self.size)

# Lists
fruits = []
particles = []
spawn_timer = 0
SPAWN_INTERVAL = 50

# Fonts
font_large = pygame.font.Font(None, 74)
font_small = pygame.font.Font(None, 36)

# FPS tracking
fps_display = 0
frame_count = 0
fps_start_time = time.time()

# Frame processing
frame_skip = 1
frame_counter = 0

# Main game loop
running = True
while running:
    frame_counter += 1
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
    
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    
    if frame_counter % frame_skip == 0:
        small_frame = cv2.resize(frame, (320, 240))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
    
        fingertip_x, fingertip_y = None, None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[8]
                fingertip_x = int(index_tip.x * SCREEN_WIDTH)
                fingertip_y = int(index_tip.y * SCREEN_HEIGHT)
                trail_points.append((fingertip_x, fingertip_y))
    
    screen.fill(BLACK)
    
    spawn_timer += 1
    if spawn_timer >= SPAWN_INTERVAL:
        fruits.append(Fruit())
        spawn_timer = 0
    
    fruits_to_remove = []
    for fruit in fruits:
        if fruit.update():
            fruits_to_remove.append(fruit)
            if not fruit.sliced:
                missed += 1
        fruit.draw(screen)
    
    for fruit in fruits_to_remove:
        fruits.remove(fruit)
    
    particles_to_remove = []
    for particle in particles:
        if particle.update():
            particles_to_remove.append(particle)
        particle.draw(screen)
    
    for particle in particles_to_remove:
        particles.remove(particle)
    
    if len(trail_points) >= 2:
        p1 = trail_points[-2]
        p2 = trail_points[-1]
        
        for fruit in fruits:
            if not fruit.sliced:
                fruit_rect = fruit.get_rect()
                if fruit_rect.clipline(p1, p2):
                    fruit.sliced = True
                    score += 10
                    for _ in range(15):
                        particles.append(Particle(fruit.x, fruit.y, fruit.color))
    
    if len(trail_points) > 1:
        for i in range(1, len(trail_points)):
            alpha = int(255 * (i / len(trail_points)))
            pygame.draw.line(screen, (0, alpha//3, 255//3), 
                           trail_points[i-1], trail_points[i], 12)
            pygame.draw.line(screen, (0, alpha, 255), 
                           trail_points[i-1], trail_points[i], 6)
    
    if fingertip_x and fingertip_y:
        pygame.draw.circle(screen, (0, 150, 255), 
                         (fingertip_x, fingertip_y), 15, 3)
        pygame.draw.circle(screen, (0, 255, 255), 
                         (fingertip_x, fingertip_y), 8)
    
    score_text = font_large.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))
    
    missed_text = font_small.render(f"Missed: {missed}", True, RED)
    screen.blit(missed_text, (10, 80))
    
    frame_count += 1
    if frame_count >= 10:
        fps_display = 10 / (time.time() - fps_start_time)
        fps_start_time = time.time()
        frame_count = 0
    
    fps_text = font_small.render(f"FPS: {fps_display:.1f}", True, GREEN)
    screen.blit(fps_text, (10, 120))
    
    label_text = font_small.render("ARM OPTIMIZED 🚀", True, GREEN)
    screen.blit(label_text, (SCREEN_WIDTH - 300, 10))
    
    pygame.display.flip()
    clock.tick(FPS_TARGET)

cap.release()
hands.close()
pygame.quit()