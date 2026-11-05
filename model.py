"""
model.py - AI-Based Image Captioning Model Handler
===================================================
This module handles loading the pre-trained BLIP (Bootstrapping Language-Image Pre-training)
model from Hugging Face Transformers and generating descriptive captions for images.

Model used: Salesforce/blip-image-captioning-base
Frameworks: PyTorch, Hugging Face Transformers, PIL
"""

import logging
from typing import Tuple, Optional
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default model identifier on Hugging Face Model Hub
DEFAULT_MODEL_ID = "Salesforce/blip-image-captioning-base"


def get_optimal_device() -> torch.device:
    """
    Detects and returns the best available compute device:
    - CUDA (NVIDIA GPU) if available
    - MPS (Apple Silicon GPU) if available
    - CPU as universal fallback
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using NVIDIA CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device for inference")
    return device


def load_blip_model(
    model_id: str = DEFAULT_MODEL_ID
) -> Tuple[BlipProcessor, BlipForConditionalGeneration, torch.device]:
    """
    Loads the BLIP Image Processor and Model from Hugging Face.
    
    Architecture summary:
    - BlipProcessor: Normalizes the input image (resizing, tensor conversion, ImageNet normalization).
    - BlipForConditionalGeneration: Vision-Language model featuring a Vision Transformer (ViT)
      image encoder and a Transformer-based text decoder for autoregressive text generation.
    
    Args:
        model_id: Hugging Face model repository path.
        
    Returns:
        tuple: (processor, model, device)
    """
    device = get_optimal_device()
    logger.info(f"Loading BLIP processor from {model_id}...")
    processor = BlipProcessor.from_pretrained(model_id)

    logger.info(f"Loading BLIP conditional generation model from {model_id}...")
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    
    # Move model to optimal device and set to evaluation mode (disables dropout)
    model.to(device)
    model.eval()
    
    logger.info("BLIP model successfully loaded and ready for inference.")
    return processor, model, device


def generate_caption(
    image: Image.Image,
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    device: torch.device,
    max_new_tokens: int = 50,
    num_beams: int = 4,
    conditional_prefix: Optional[str] = None
) -> str:
    """
    Generates a natural-language caption for the provided image.
    
    Steps:
    1. Validates and converts image to RGB (handles RGBA, grayscale, CMYK, etc.).
    2. Preprocesses image through BlipProcessor to produce normalized pixel tensors.
    3. Feeds pixel tensors to the BLIP model encoder-decoder.
    4. Generates output token IDs using Beam Search decoding.
    5. Decodes token IDs into human-readable English text.
    
    Args:
        image: PIL.Image object.
        processor: BlipProcessor instance.
        model: BlipForConditionalGeneration instance.
        device: torch.device (CPU or CUDA).
        max_new_tokens: Maximum number of words/tokens to generate.
        num_beams: Number of beams for beam search (higher = better quality, slightly slower).
        conditional_prefix: Optional starting prompt (e.g., "A photograph of").
        
    Returns:
        str: Generated caption formatted as a clean sentence.
    """
    try:
        # Step 1: Ensure image is in standard RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Step 2: Preprocess image (and optional conditional text)
        if conditional_prefix and conditional_prefix.strip():
            inputs = processor(
                images=image,
                text=conditional_prefix.strip(),
                return_tensors="pt"
            )
        else:
            inputs = processor(
                images=image,
                return_tensors="pt"
            )

        # Move input tensors to the target compute device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Step 3 & 4: Inference with torch.no_grad (saves memory and speeds up inference)
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                min_length=5,
                repetition_penalty=1.2,
                early_stopping=True
            )

        # Step 5: Decode token sequences to English string
        caption = processor.decode(output_tokens[0], skip_special_tokens=True).strip()

        # Format nicely: capitalize first letter and ensure ending punctuation
        if caption:
            caption = caption[0].upper() + caption[1:]
            if not caption.endswith((".", "!", "?")):
                caption += "."

        return caption

    except Exception as e:
        logger.error(f"Error during caption generation: {e}", exc_info=True)
        raise RuntimeError(f"Caption generation failed: {str(e)}")

