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
pygame.display.set_caption("Fruit Ninja AR (VERY UNOPTIMIZED)")
clock = pygame.time.Clock()

# --- Constants, Colors ---
TRAIL_LENGTH = 15
FRUIT_SIZE = 90
PINCH_THRESHOLD = 30 
WHITE = (255, 255, 255)
RED = (255, 69, 0)
GREEN = (0, 200, 0)
BLUE = (65, 105, 225)
GRAY = (100, 100, 100) 
TRANSPARENT_BLACK = (0, 0, 0, 150)

# --- Game State Management ---
game_state = 'MENU' 
session_high_score = 0
last_score = 0
score = 0
missed = 0
fruits = []
spawn_timer = 0
SPAWN_INTERVAL = 50
trail_points = deque(maxlen=TRAIL_LENGTH)
pinch_detected_this_frame = False
pinch_was_active_last_frame = False

# --- Classes and Helper Functions ---

def reset_game():
    global score, missed, fruits, trail_points, spawn_timer
    score = 0
    missed = 0
    fruits = []
    trail_points.clear()
    spawn_timer = 0

def draw_button(rect, text, is_hovered):
    font = pygame.font.Font('freesansbold.ttf', 40 if rect.width > 150 else 20)
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
        if not self.sliced: self.angle += self.rotation_speed
        return self.y > SCREEN_HEIGHT + 100

    def draw(self, surface):
        image_to_load = self.cut_image_path if self.sliced else self.image_path
        try:
            # --- MODIFIED: Ensure convert_alpha is explicitly called after loading ---
            fruit_image = pygame.image.load(image_to_load)
            if fruit_image.get_alpha() is None: # Check if it even has an alpha channel
                fruit_image = fruit_image.convert() # Use regular convert if no alpha
            else:
                fruit_image = fruit_image.convert_alpha() # Use convert_alpha if it has alpha

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
is_hovering_exit = False 
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
        index_tip, thumb_tip = hand_landmarks.landmark[8], hand_landmarks.landmark[4]
        fingertip_pos = (int(index_tip.x * SCREEN_WIDTH), int(index_tip.y * SCREEN_HEIGHT))
        thumb_pos = (int(thumb_tip.x * SCREEN_WIDTH), int(thumb_tip.y * SCREEN_HEIGHT))
        distance = math.hypot(fingertip_pos[0] - thumb_pos[0], fingertip_pos[1] - thumb_pos[1])
        if distance < PINCH_THRESHOLD: pinch_detected_this_frame = True

    is_click = pinch_detected_this_frame and not pinch_was_active_last_frame
    pinch_was_active_last_frame = pinch_detected_this_frame
    
    # --- UNOPTIMIZED: Re-calculate button rectangles on every single frame.
    play_button_rect = pygame.Rect(SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 - 50, 300, 100)
    menu_button_rect = pygame.Rect(SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 + 70, 300, 100)
    exit_button_rect = pygame.Rect(SCREEN_WIDTH - 130, 20, 110, 50)
    
    # --- State Machine Logic ---
    if game_state == 'MENU':
        # --- NEW UNOPTIMIZED: Re-render static title text every frame ---
        title_surf = pygame.font.Font('freesansbold.ttf', 70).render("Fruit Ninja AR", True, WHITE)
        screen.blit(title_surf, (SCREEN_WIDTH // 2 - title_surf.get_width() // 2, 100))

        is_hovering_play = fingertip_pos and play_button_rect.collidepoint(fingertip_pos)
        draw_button(play_button_rect, "Play", is_hovering_play)
        if is_hovering_play and is_click:
            reset_game()
            game_state = 'PLAYING'

    elif game_state == 'PLAYING':
        if fingertip_pos: trail_points.append(fingertip_pos)
        else: trail_points.clear()

        spawn_timer += 1
        if spawn_timer >= SPAWN_INTERVAL:
            fruits.append(Fruit())
            spawn_timer = 0
            
        for fruit in fruits[:]:
            if fruit.update():
                if not fruit.sliced: missed += 1
                fruits.remove(fruit)
            else: fruit.draw(screen)

        # --- UNOPTIMIZED: Brute-force slicing logic. Very slow.
        if len(trail_points) >= 2:
            p1 = trail_points[-2]
            p2 = trail_points[-1]
            for fruit in fruits:
                if not fruit.sliced:
                    for i in range(11): # Check 10 points along the slice path
                        lerp_x = p1[0] + (p2[0] - p1[0]) * (i / 10)
                        lerp_y = p1[1] + (p2[1] - p1[1]) * (i / 10)
                        if fruit.get_rect().collidepoint(lerp_x, lerp_y):
                            fruit.sliced = True
                            score += 10
                            break 
        
        # --- NEW UNOPTIMIZED: Re-render score and missed text every frame ---
        font_score_display = pygame.font.Font('freesansbold.ttf', 40)
        score_surf = font_score_display.render(f"Score: {score}", True, WHITE)
        missed_surf = font_score_display.render(f"Missed: {missed}/3", True, RED)
        screen.blit(score_surf, (20, 20))
        screen.blit(missed_surf, (20, 70))

        is_hovering_exit = fingertip_pos and exit_button_rect.collidepoint(fingertip_pos)
        draw_button(exit_button_rect, "Menu", is_hovering_exit)
        if missed >= 3 or (is_hovering_exit and is_click):
            last_score = score
            if score > session_high_score: session_high_score = score
            game_state = 'GAME_OVER'

    elif game_state == 'GAME_OVER':
        # --- NEW UNOPTIMIZED: Re-render game over text and scores every frame ---
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill(TRANSPARENT_BLACK)
        screen.blit(overlay, (0, 0))
        
        font_game_over = pygame.font.Font('freesansbold.ttf', 70)
        font_score_display = pygame.font.Font('freesansbold.ttf', 40)

        go_surf = font_game_over.render("Game Over", True, RED)
        ls_surf = font_score_display.render(f"Last Score: {last_score}", True, WHITE)
        hs_surf = font_score_display.render(f"High Score: {session_high_score}", True, WHITE)
        screen.blit(go_surf, (SCREEN_WIDTH // 2 - go_surf.get_width() // 2, 150))
        screen.blit(ls_surf, (SCREEN_WIDTH // 2 - ls_surf.get_width() // 2, 250))
        screen.blit(hs_surf, (SCREEN_WIDTH // 2 - hs_surf.get_width() // 2, 300))

        is_hovering_menu = fingertip_pos and menu_button_rect.collidepoint(fingertip_pos)
        draw_button(menu_button_rect, "Menu", is_hovering_menu)
        if is_hovering_menu and is_click: game_state = 'MENU'

    # --- Universal Drawing (on top of all states) ---
    if fingertip_pos:
        if game_state == 'PLAYING' and not is_hovering_exit:
            try:
                # --- MODIFIED: Ensure convert_alpha for sword ---
                sword_image = pygame.image.load(SWORD_IMAGE_PATH)
                if sword_image.get_alpha() is None:
                    sword_image = sword_image.convert()
                else:
                    sword_image = sword_image.convert_alpha()

                sword_image = pygame.transform.scale(sword_image, (150, 150))
                screen.blit(sword_image, sword_image.get_rect(center=fingertip_pos))
            except pygame.error: pass
        else:
            pygame.draw.circle(screen, BLUE, fingertip_pos, 15)
            if pinch_detected_this_frame: pygame.draw.circle(screen, WHITE, fingertip_pos, 7)

    pygame.display.flip()
    clock.tick(60)

# --- Cleanup ---
cap.release()
hands.close()
pygame.quit()