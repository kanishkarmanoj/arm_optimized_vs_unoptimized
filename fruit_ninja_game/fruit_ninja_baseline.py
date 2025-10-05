import cv2
import mediapipe as mp
import pygame
import random
import math 
import os
from collections import deque

# --- Initialization ---
pygame.init()

# --- Asset Paths (Unoptimized) ---
FRUIT_IMAGE_PATHS = [
    os.path.join('assets', 'apple.png'),
    os.path.join('assets', 'watermelon.png'),
    os.path.join('assets', 'orange.png')
]
SWORD_IMAGE_PATH = os.path.join('assets', 'sword.png')

# --- MediaPipe and Camera Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("❌ Error: Failed to grab initial frame from camera.")
    exit()

# --- Screen and Display Setup ---
SCREEN_HEIGHT, SCREEN_WIDTH, _ = frame.shape
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Fruit Ninja AR (UNOPTIMIZED)")
clock = pygame.time.Clock()

# --- Constants, Colors, and Fonts ---
TRAIL_LENGTH = 15
FRUIT_SIZE = 90
PINCH_THRESHOLD = 30 
WHITE = (255, 255, 255)
RED = (255, 69, 0)
GREEN = (0, 200, 0)
BLUE = (65, 105, 225)
GRAY = (100, 100, 100) 
TRANSPARENT_BLACK = (0, 0, 0, 150)
font_large = pygame.font.Font('freesansbold.ttf', 70)
font_small = pygame.font.Font('freesansbold.ttf', 40)
font_tiny = pygame.font.Font('freesansbold.ttf', 20) 

# --- Game State Management ---
game_state = 'MENU' 
session_high_score = 0
last_score = 0

# Game variables (will be reset)
score = 0
missed = 0
fruits = []
spawn_timer = 0
SPAWN_INTERVAL = 50
trail_points = deque(maxlen=TRAIL_LENGTH)

# Gesture control variables
pinch_detected_this_frame = False
pinch_was_active_last_frame = False

# --- UI Elements ---
play_button_rect = pygame.Rect(SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 - 50, 300, 100)
menu_button_rect = pygame.Rect(SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 + 70, 300, 100)
exit_button_rect = pygame.Rect(SCREEN_WIDTH - 130, 20, 110, 50)


# --- Classes and Helper Functions ---

def reset_game():
    """Resets all game variables to start a new round."""
    global score, missed, fruits, trail_points, spawn_timer
    score = 0
    missed = 0
    fruits = []
    trail_points.clear()
    spawn_timer = 0

def draw_button(rect, text, font, is_hovered):
    """Draws a button and changes its color if the user's hand is over it."""
    base_color = GRAY if text == "Menu" else GREEN
    color = BLUE if is_hovered else base_color
    pygame.draw.rect(screen, color, rect, border_radius=15)
    text_surf = font.render(text, True, WHITE)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)

class Fruit:
    def __init__(self):
        self.x = random.randint(100, SCREEN_WIDTH - 100)
        self.y = SCREEN_HEIGHT + 50
        self.velocity_y = random.randint(-22, -18)
        self.velocity_x = random.randint(-4, 4)
        self.gravity = 0.5
        self.sliced = False
        self.size = FRUIT_SIZE
        self.angle = 0
        self.rotation_speed = random.randint(-5, 5)
        self.image_path = random.choice(FRUIT_IMAGE_PATHS)
        self.cut_image_path = self.image_path.replace('.png', '_cut.png')

    def update(self):
        self.velocity_y += self.gravity
        self.y += self.velocity_y
        self.x += self.velocity_x
        if not self.sliced:
            self.angle += self.rotation_speed
        if self.y > SCREEN_HEIGHT + 100:
            return True
        return False

    def draw(self, surface):
        image_to_load = self.cut_image_path if self.sliced else self.image_path
        try:
            fruit_image = pygame.image.load(image_to_load).convert_alpha()
            fruit_image = pygame.transform.scale(fruit_image, (self.size, self.size))
            rotated_image = pygame.transform.rotate(fruit_image, self.angle)
            rect = rotated_image.get_rect(center=(self.x, self.y))
            surface.blit(rotated_image, rect)
        except pygame.error:
            pygame.draw.circle(surface, RED, (int(self.x), int(self.y)), self.size // 2)

    def get_rect(self):
        return pygame.Rect(self.x - self.size//2, self.y - self.size//2, self.size, self.size)


# --- Main Loop ---
running = True
is_hovering_exit = False # Define this outside the loop to ensure it always exists

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            running = False

    # --- Frame & Hand Processing ---
    success, frame = cap.read()
    if not success: continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    frame_surface = pygame.surfarray.make_surface(rgb_frame.swapaxes(0, 1))
    screen.blit(frame_surface, (0, 0))

    fingertip_pos = None
    pinch_detected_this_frame = False
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        
        fingertip_pos = (int(index_tip.x * SCREEN_WIDTH), int(index_tip.y * SCREEN_HEIGHT))
        thumb_pos = (int(thumb_tip.x * SCREEN_WIDTH), int(thumb_tip.y * SCREEN_HEIGHT))

        distance = math.sqrt((fingertip_pos[0] - thumb_pos[0])**2 + (fingertip_pos[1] - thumb_pos[1])**2)
        if distance < PINCH_THRESHOLD:
            pinch_detected_this_frame = True

    is_click = pinch_detected_this_frame and not pinch_was_active_last_frame
    pinch_was_active_last_frame = pinch_detected_this_frame
    
    # --- State Machine Logic ---

    # --- MENU STATE ---
    if game_state == 'MENU':
        title_surf = font_large.render("Fruit Ninja AR", True, WHITE)
        screen.blit(title_surf, (SCREEN_WIDTH // 2 - title_surf.get_width() // 2, 100))
        
        is_hovering_play = fingertip_pos and play_button_rect.collidepoint(fingertip_pos)
        draw_button(play_button_rect, "Play", font_large, is_hovering_play)

        if is_hovering_play and is_click:
            reset_game()
            game_state = 'PLAYING'

    # --- PLAYING STATE ---
    elif game_state == 'PLAYING':
        if fingertip_pos:
            trail_points.append(fingertip_pos)
        else:
            trail_points.clear()

        spawn_timer += 1
        if spawn_timer >= SPAWN_INTERVAL:
            fruits.append(Fruit())
            spawn_timer = 0
            
        fruits_to_remove = []
        for fruit in fruits:
            if fruit.update():
                if not fruit.sliced:
                    missed += 1
                fruits_to_remove.append(fruit)
            else:
                fruit.draw(screen)
        for f in fruits_to_remove: fruits.remove(f)

        if len(trail_points) >= 2:
            p1 = trail_points[-2]
            p2 = trail_points[-1]
            for fruit in fruits:
                if not fruit.sliced and fruit.get_rect().clipline(p1, p2):
                    fruit.sliced = True
                    score += 10
        
        score_surf = font_small.render(f"Score: {score}", True, WHITE)
        missed_surf = font_small.render(f"Missed: {missed}/3", True, RED)
        screen.blit(score_surf, (20, 20))
        screen.blit(missed_surf, (20, 70))
        
        is_hovering_exit = fingertip_pos and exit_button_rect.collidepoint(fingertip_pos)
        draw_button(exit_button_rect, "Menu", font_tiny, is_hovering_exit)

        if missed >= 3 or (is_hovering_exit and is_click):
            last_score = score
            if score > session_high_score:
                session_high_score = score
            game_state = 'GAME_OVER'

    # --- GAME OVER STATE ---
    elif game_state == 'GAME_OVER':
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill(TRANSPARENT_BLACK)
        screen.blit(overlay, (0, 0))
        
        go_surf = font_large.render("Game Over", True, RED)
        ls_surf = font_small.render(f"Last Score: {last_score}", True, WHITE)
        hs_surf = font_small.render(f"High Score: {session_high_score}", True, WHITE)
        screen.blit(go_surf, (SCREEN_WIDTH // 2 - go_surf.get_width() // 2, 150))
        screen.blit(ls_surf, (SCREEN_WIDTH // 2 - ls_surf.get_width() // 2, 250))
        screen.blit(hs_surf, (SCREEN_WIDTH // 2 - hs_surf.get_width() // 2, 300))

        is_hovering_menu = fingertip_pos and menu_button_rect.collidepoint(fingertip_pos)
        draw_button(menu_button_rect, "Menu", font_large, is_hovering_menu)

        if is_hovering_menu and is_click:
            game_state = 'MENU'

    # --- Universal Drawing (on top of all states) ---
    if fingertip_pos:
        # Check if we are in the playing state AND not hovering over the exit button
        if game_state == 'PLAYING' and not is_hovering_exit:
            try:
                sword_image = pygame.image.load(SWORD_IMAGE_PATH).convert_alpha()
                sword_image = pygame.transform.scale(sword_image, (150, 150))
                screen.blit(sword_image, sword_image.get_rect(center=fingertip_pos))
            except pygame.error: pass
        else: # In all other cases (Menu, Game Over, or hovering the exit button), draw the cursor
            pygame.draw.circle(screen, BLUE, fingertip_pos, 15)
            # Show pinch state with a smaller white circle
            if pinch_detected_this_frame:
                pygame.draw.circle(screen, WHITE, fingertip_pos, 7)

    pygame.display.flip()
    clock.tick(60)

# --- Cleanup ---
cap.release()
hands.close()
pygame.quit()