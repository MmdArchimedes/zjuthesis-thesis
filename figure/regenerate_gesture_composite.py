"""Regenerate gesture_11classes composite with 2x larger text labels."""
from PIL import Image, ImageDraw, ImageFont
import os

# ── Config ──
GESTURE_DIR = "gesture"
OUTPUT_PNG = "gesture_11classes.png"
OUTPUT_PDF = "gesture_11classes.pdf"

# Chinese-friendly font (use a font that supports Chinese characters)
# On Windows, Microsoft YaHei or SimHei works well
FONT_PATH = "C:/Windows/Fonts/simhei.ttf"  # 黑体, bold, good for labels
LABEL_FONT_SIZE = 56   # 2x the original (~28px)
TITLE_FONT_SIZE = 42

# Grid layout: 4 cols × 3 rows (cell 12 empty)
COLS = 4
ROWS = 3
CELL_W = 620   # cell width for each photo
CELL_H = 720   # cell height (includes label space)
LABEL_H = 80   # space reserved for text label below photo
MARGIN = 60    # margin around edges
GAP = 40       # gap between cells

# Image size inside cell (photo area)
PHOTO_W = CELL_W - 20
PHOTO_H = CELL_H - LABEL_H - 10

# Total image dimensions
IMG_W = 2 * MARGIN + COLS * CELL_W + (COLS - 1) * GAP
IMG_H = 2 * MARGIN + ROWS * CELL_H + (ROWS - 1) * GAP

# ── Gesture mapping (matching the original order in fig caption) ──
# (a) through (k)
gestures = [
    ("(a) 握拳",     "握拳.jpg"),
    ("(b) 单指向左", "单指向左.jpg"),
    ("(c) 单指向右", "单指向右.jpg"),
    ("(d) 二指手心", "二指手心.jpg"),
    ("(e) 二指手背", "二指手背.jpg"),
    ("(f) 三指手心", "三指手心.jpg"),
    ("(g) 三指手背", "三指手背.jpg"),
    ("(h) 四指手心", "四指手心.jpg"),
    ("(i) 四指手背", "四指手背.jpg"),
    ("(j) 五指手心", "五指手心.jpg"),
    ("(k) 五指手背", "五指手背.jpg"),
]

# ── Create canvas ──
canvas = Image.new("RGB", (IMG_W, IMG_H), "white")
draw = ImageDraw.Draw(canvas)

# Load font
try:
    font_label = ImageFont.truetype(FONT_PATH, LABEL_FONT_SIZE)
    font_title = ImageFont.truetype(FONT_PATH, TITLE_FONT_SIZE)
    print(f"Using font: {FONT_PATH} at {LABEL_FONT_SIZE}px")
except Exception as e:
    print(f"Font load error: {e}, using default")
    font_label = ImageFont.load_default()
    font_title = ImageFont.load_default()

# ── Place images ──
for idx, (label, filename) in enumerate(gestures):
    row = idx // COLS
    col = idx % COLS

    # Cell top-left position
    cell_x = MARGIN + col * (CELL_W + GAP)
    cell_y = MARGIN + row * (CELL_H + GAP)

    # Photo center position within cell
    photo_x = cell_x + (CELL_W - PHOTO_W) // 2
    photo_y = cell_y + 5  # small top padding

    # Load and resize photo
    img_path = os.path.join(GESTURE_DIR, filename)
    if os.path.exists(img_path):
        photo = Image.open(img_path)
        # Calculate resize to fit within PHOTO_W × PHOTO_H maintaining aspect ratio
        scale = min(PHOTO_W / photo.width, PHOTO_H / photo.height)
        new_w = int(photo.width * scale)
        new_h = int(photo.height * scale)
        photo = photo.resize((new_w, new_h), Image.LANCZOS)

        # Center photo within photo area
        paste_x = photo_x + (PHOTO_W - new_w) // 2
        paste_y = photo_y + (PHOTO_H - new_h) // 2
        canvas.paste(photo, (paste_x, paste_y))
    else:
        print(f"WARNING: {img_path} not found")

    # Draw label below photo, centered
    label_y = cell_y + CELL_H - LABEL_H + 10
    bbox = draw.textbbox((0, 0), label, font=font_label)
    text_w = bbox[2] - bbox[0]
    text_x = cell_x + (CELL_W - text_w) // 2
    draw.text((text_x, label_y), label, fill="black", font=font_label)

print(f"Canvas size: {IMG_W}×{IMG_H}")

# ── Save ──
canvas.save(OUTPUT_PNG, "PNG", dpi=(300, 300))
print(f"Saved: {OUTPUT_PNG}")

# Also save as PDF
canvas.save(OUTPUT_PDF, "PDF", resolution=300.0)
print(f"Saved: {OUTPUT_PDF}")
print("Done!")
