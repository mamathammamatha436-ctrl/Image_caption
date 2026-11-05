"""
setup_assets.py - Setup Logo and Sample Images for Testing
==========================================================
Downloads or creates high quality sample images and an attractive logo
for the CodSoft AI-Based Image Caption Generator project.
"""

import os
from pathlib import Path
import requests
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
SAMPLE_DIR = BASE_DIR / "sample_images"

ASSETS_DIR.mkdir(exist_ok=True)
SAMPLE_DIR.mkdir(exist_ok=True)

# Curated high-quality, free-to-use Wikimedia Commons public domain image URLs
SAMPLE_URLS = {
    "person_in_park.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/A_man_in_a_park_%28Unsplash%29.jpg/640px-A_man_in_a_park_%28Unsplash%29.jpg",
    "dog_playing.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Golden_retriever_running_with_a_stick.jpg/640px-Golden_retriever_running_with_a_stick.jpg",
    "group_people.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/A_group_of_young_people_sitting_on_grass.jpg/640px-A_group_of_young_people_sitting_on_grass.jpg",
    "landscape.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Altja_j%C3%B5gi_Lahemaal.jpg/640px-Altja_j%C3%B5gi_Lahemaal.jpg",
    "car_on_road.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Audi_A4_B8_Limousine_front_20100911.jpg/640px-Audi_A4_B8_Limousine_front_20100911.jpg",
}


def download_or_create_samples():
    """Attempts to download sample images, falling back to clean generated images if offline."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ImageCaptioningApp/1.0 (Educational Internship Project)"
    }

    for filename, url in SAMPLE_URLS.items():
        filepath = SAMPLE_DIR / filename
        if filepath.exists() and filepath.stat().st_size > 1000:
            print(f"Sample image already exists: {filename}")
            continue

        downloaded = False
        try:
            print(f"Downloading {filename} from {url}...")
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200 and len(response.content) > 2000:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                # Verify PIL can open it
                img = Image.open(filepath)
                img.verify()
                print(f"Successfully saved {filename} ({img.format})")
                downloaded = True
        except Exception as e:
            print(f"Download failed for {filename}: {e}. Creating fallback image...")

        if not downloaded:
            create_fallback_image(filepath, filename)


def create_fallback_image(filepath: Path, filename: str):
    """Creates a distinct, realistic graphic if internet download is unavailable."""
    width, height = 640, 480
    img = Image.new("RGB", (width, height), color=(240, 243, 246))
    draw = ImageDraw.Draw(img)

    if "park" in filename:
        # Sky and lush grass with tree
        draw.rectangle([(0, 0), (width, 240)], fill=(135, 206, 235))
        draw.rectangle([(0, 240), (width, height)], fill=(34, 139, 34))
        # Trunk
        draw.rectangle([(420, 160), (460, 300)], fill=(101, 67, 33))
        # Foliage
        draw.ellipse([(380, 80), (500, 200)], fill=(46, 125, 50))
        # Man figure standing in park
        draw.ellipse([(190, 180), (230, 220)], fill=(245, 199, 168))  # Head
        draw.rectangle([(180, 220), (240, 340)], fill=(30, 64, 175))   # Blue jacket
        draw.rectangle([(190, 340), (210, 430)], fill=(31, 41, 55))    # Pants left
        draw.rectangle([(220, 340), (240, 430)], fill=(31, 41, 55))    # Pants right

    elif "dog" in filename:
        # Green backyard with sunny sky and dog
        draw.rectangle([(0, 0), (width, 200)], fill=(176, 226, 255))
        draw.rectangle([(0, 200), (width, height)], fill=(50, 160, 50))
        # Dog body (golden brown)
        draw.ellipse([(200, 250), (420, 360)], fill=(210, 140, 60))
        # Dog head
        draw.ellipse([(380, 220), (470, 310)], fill=(210, 140, 60))
        # Dog ears
        draw.polygon([(400, 210), (430, 250), (390, 250)], fill=(170, 100, 40))
        # Legs
        draw.rectangle([(230, 340), (260, 420)], fill=(190, 120, 50))
        draw.rectangle([(360, 340), (390, 420)], fill=(190, 120, 50))
        # Ball/stick
        draw.ellipse([(450, 290), (480, 320)], fill=(255, 69, 0))

    elif "group" in filename:
        # Outdoor picnic scene with group of people
        draw.rectangle([(0, 0), (width, 220)], fill=(200, 230, 250))
        draw.rectangle([(0, 220), (width, height)], fill=(76, 175, 80))
        colors = [(220, 38, 38), (37, 99, 235), (234, 88, 12), (147, 51, 234)]
        positions = [140, 240, 340, 440]
        for x, c in zip(positions, colors):
            draw.ellipse([(x + 10, 200), (x + 50, 240)], fill=(245, 199, 168))  # Head
            draw.rectangle([(x, 240), (x + 60, 360)], fill=c)  # Shirt
            draw.rectangle([(x + 5, 360), (x + 55, 420)], fill=(40, 40, 40))  # Pants

    elif "landscape" in filename:
        # Mountains, sunset sky, lake
        draw.rectangle([(0, 0), (width, 220)], fill=(255, 179, 102))  # Sunset
        draw.ellipse([(260, 80), (380, 200)], fill=(255, 223, 128))  # Sun
        # Mountains
        draw.polygon([(0, 260), (180, 100), (360, 260)], fill=(80, 70, 90))
        draw.polygon([(240, 260), (440, 90), (640, 260)], fill=(65, 55, 75))
        # Blue lake
        draw.rectangle([(0, 260), (width, height)], fill=(28, 107, 160))

    elif "car" in filename:
        # Asphalt road and a sedan
        draw.rectangle([(0, 0), (width, 220)], fill=(180, 210, 240))  # Sky
        draw.rectangle([(0, 220), (width, height)], fill=(60, 64, 72))  # Asphalt road
        # Lane divider
        draw.line([(0, 420), (width, 420)], fill=(255, 230, 0), width=6)
        # Car body (red sporty car)
        draw.rectangle([(160, 260), (480, 340)], fill=(220, 38, 38))
        draw.polygon([(230, 260), (280, 200), (400, 200), (430, 260)], fill=(220, 38, 38))
        # Windows
        draw.polygon([(285, 205), (395, 205), (415, 255), (250, 255)], fill=(180, 220, 240))
        # Wheels
        draw.ellipse([(200, 320), (260, 380)], fill=(20, 20, 20))
        draw.ellipse([(380, 320), (440, 380)], fill=(20, 20, 20))

    img.save(filepath, "JPEG", quality=92)
    print(f"Created fallback image: {filename}")


def create_app_logo():
    """Generates an aesthetic project logo in assets/logo.png."""
    logo_path = ASSETS_DIR / "logo.png"
    size = (512, 512)
    img = Image.new("RGBA", size, color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Gradient-like background circular badge
    draw.ellipse([(20, 20), (492, 492)], fill=(30, 27, 75, 255))
    draw.ellipse([(36, 36), (476, 476)], outline=(99, 102, 241, 200), width=8)

    # Inner camera / eye aperture aesthetic
    draw.rounded_rectangle([(110, 150), (402, 370)], radius=32, fill=(67, 56, 202, 255))
    draw.ellipse([(196, 200), (316, 320)], fill=(99, 102, 241, 255), outline=(224, 231, 255, 255), width=6)
    draw.ellipse([(236, 240), (276, 280)], fill=(255, 255, 255, 255))
    # Top flash
    draw.rounded_rectangle([(160, 115), (230, 145)], radius=8, fill=(129, 140, 248, 255))

    img.save(logo_path, "PNG")
    print(f"Generated logo: {logo_path}")


if __name__ == "__main__":
    create_app_logo()
    download_or_create_samples()
    print("Setup completed successfully.")

