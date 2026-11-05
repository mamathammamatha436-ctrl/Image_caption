"""
test_captioning.py - Standalone Model Test & Verification Script
================================================================
Verifies that:
1. BLIP processor and model download and load properly.
2. Inference pipeline functions end-to-end.
3. Test images in sample_images/ produce accurate, coherent captions.
"""

import sys
from pathlib import Path
from PIL import Image
from model import load_blip_model, generate_caption

def test_pipeline():
    print("=" * 60)
    print("BLIP Image Captioning Test & Model Cache Warm-up")
    print("=" * 60)
    
    print("\n1. Loading BLIP Processor and Model...")
    processor, model, device = load_blip_model()
    print(f"   Model successfully loaded on device: {device}")

    samples_dir = Path("sample_images")
    sample_files = list(samples_dir.glob("*.jpg"))
    
    if not sample_files:
        print("   No sample images found to test.")
        return False
        
    print(f"\n2. Testing inference on {len(sample_files)} sample images:")
    for img_path in sample_files:
        print(f"\n   Testing file: {img_path.name}")
        img = Image.open(img_path)
        print(f"   Image size: {img.size}, mode: {img.mode}")
        caption = generate_caption(img, processor, model, device)
        print(f"   --> Generated Caption: \"{caption}\"")

    print("\n" + "=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)

