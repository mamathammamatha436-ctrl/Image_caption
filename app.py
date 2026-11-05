"""
app.py - AI-Based Image Caption Generator Frontend
===================================================
A modern, responsive Streamlit web application for AI-powered Image Captioning.
Combines Computer Vision & NLP using Salesforce BLIP (Bootstrapping Language-Image Pre-training).

Internship Project: CodSoft AI/ML Internship
Model: Salesforce/blip-image-captioning-base
"""

import os
import io
import time
from pathlib import Path
from PIL import Image
import streamlit as st

# Import the model loader and generation functions from model.py
from model import load_blip_model, generate_caption, DEFAULT_MODEL_ID

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Image Caption Generator | CodSoft",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #1e1e38 0%, #2b1b4d 50%, #172554 100%);
        border-radius: 18px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        color: #ffffff;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(165, 180, 252, 0.3);
        text-transform: uppercase;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 50%, #93c5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 0.6rem;
        margin-bottom: 0;
        font-weight: 400;
    }

    /* Card Containers */
    .card-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.5rem;
    }
    @media (prefers-color-scheme: dark) {
        .card-box {
            background: #181926;
            border-color: #2e3045;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
        }
    }

    /* Image Preview Container */
    .image-preview-frame {
        border-radius: 12px;
        overflow: hidden;
        border: 2px dashed #cbd5e1;
        padding: 8px;
        background: #f8fafc;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    @media (prefers-color-scheme: dark) {
        .image-preview-frame {
            border-color: #3b4261;
            background: #13141f;
        }
    }

    /* Caption Result Card */
    .caption-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(168, 85, 247, 0.08) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-left: 5px solid #6366f1;
        border-radius: 12px;
        padding: 1.4rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .caption-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 700;
        color: #6366f1;
        margin-bottom: 0.4rem;
    }
    .caption-text {
        font-size: 1.3rem;
        font-weight: 600;
        line-height: 1.5;
        color: #1e293b;
    }
    @media (prefers-color-scheme: dark) {
        .caption-text {
            color: #f1f5f9;
        }
    }

    /* Meta Badges */
    .meta-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #e0e7ff;
        color: #3730a3;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    @media (prefers-color-scheme: dark) {
        .meta-tag {
            background: #232742;
            color: #a5b4fc;
        }
    }

    /* Primary Button Styling */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 1.4rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Model Singleton Loader with Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="⏳ Loading BLIP Vision-Language Model into memory (first time may take ~1-2 min)...")
def get_cached_blip_model():
    """
    Loads BLIP processor and model only once and caches in memory across user sessions.
    """
    return load_blip_model(DEFAULT_MODEL_ID)


# -----------------------------------------------------------------------------
# 3. Sidebar Configuration & Sample Images
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_images"

SAMPLE_IMAGES_DICT = {
    "A Person in Park": SAMPLE_DIR / "person_in_park.jpg",
    "A Dog Playing": SAMPLE_DIR / "dog_playing.jpg",
    "A Group of People": SAMPLE_DIR / "group_people.jpg",
    "A Scenic Landscape": SAMPLE_DIR / "landscape.jpg",
    "A Car on Road": SAMPLE_DIR / "car_on_road.jpg",
}

with st.sidebar:
    logo_path = BASE_DIR / "assets" / "logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=90)
    st.markdown("### 🎓 CodSoft Internship Project")
    st.markdown(
        """
        <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 10px; padding: 12px; font-size: 0.85rem; margin-bottom: 1rem;">
            <strong>Track:</strong> Artificial Intelligence & Machine Learning<br>
            <strong>Project:</strong> AI-Based Image Caption Generator<br>
            <strong>Model:</strong> BLIP-Base (Salesforce)<br>
            <strong>Tech:</strong> PyTorch &bull; Transformers &bull; Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 🖼️ Choose Image Source")
    input_method = st.radio(
        "Select an image to test:",
        options=["Upload My Own Image", "Select a Preloaded Sample Image"],
        index=0
    )

    selected_sample = None
    if input_method == "Select a Preloaded Sample Image":
        selected_sample_label = st.selectbox(
            "Select a test sample:",
            options=list(SAMPLE_IMAGES_DICT.keys())
        )
        selected_sample = SAMPLE_IMAGES_DICT.get(selected_sample_label)

    st.markdown("---")
    st.markdown("### ⚙️ Generation Settings")
    with st.expander("Fine-tune Model Parameters", expanded=False):
        num_beams = st.slider(
            "Beam Search Width",
            min_value=1,
            max_value=8,
            value=4,
            help="Higher values explore more sentence possibilities for higher accuracy, but take slightly longer."
        )
        max_tokens = st.slider(
            "Max Caption Length",
            min_value=15,
            max_value=100,
            value=50,
            step=5,
            help="Maximum number of tokens to generate."
        )
        custom_prompt = st.text_input(
            "Conditional Prompt Prefix (Optional)",
            value="",
            placeholder="e.g., A photograph of, A close up of",
            help="Optional text guide to steer caption generation."
        )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #94a3b8; text-align: center;">
            Built with ❤️ for CodSoft Internship<br>
            Powered by Salesforce BLIP & Hugging Face
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 4. Hero Banner Section
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <span class="hero-badge">Deep Learning &bull; Computer Vision &bull; NLP</span>
        <h1 class="hero-title">AI Image Caption Generator</h1>
        <p class="hero-subtitle">Transform your images into meaningful natural-language descriptions using the Salesforce BLIP Transformer.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 5. Model Loading (Warm-up / Singleton)
# -----------------------------------------------------------------------------
try:
    processor, model, device = get_cached_blip_model()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Failed to initialize BLIP model: {str(e)}")
    model_loaded = False


# -----------------------------------------------------------------------------
# 6. Main Content Area: Image Upload & Preview
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 1], gap="large")

image_to_process = None
image_metadata = {}

with col_left:
    st.markdown("### 1. Provide an Image")

    if input_method == "Upload My Own Image":
        uploaded_file = st.file_uploader(
            "Upload an image in JPG, JPEG, or PNG format:",
            type=["jpg", "jpeg", "png"],
            help="Drag and drop or browse your local file system."
        )

        if uploaded_file is not None:
            try:
                # Validate and open image
                image_to_process = Image.open(uploaded_file)
                image_metadata = {
                    "Name": uploaded_file.name,
                    "Format": image_to_process.format or "Unknown",
                    "Resolution": f"{image_to_process.width} &times; {image_to_process.height} px",
                    "Size": f"{uploaded_file.size / 1024:.1f} KB",
                    "Color Mode": image_to_process.mode,
                }
            except Exception as img_err:
                st.error(f"⚠️ Invalid image file: {img_err}. Please upload a valid JPG or PNG image.")
                image_to_process = None
        else:
            st.info("👆 Please upload an image file (JPG, JPEG, or PNG) above to begin.")

    else:
        # Preloaded sample image selected
        if selected_sample and selected_sample.exists():
            try:
                image_to_process = Image.open(selected_sample)
                image_metadata = {
                    "Name": selected_sample.name,
                    "Format": image_to_process.format or "JPEG",
                    "Resolution": f"{image_to_process.width} &times; {image_to_process.height} px",
                    "Size": f"{os.path.getsize(selected_sample) / 1024:.1f} KB",
                    "Color Mode": image_to_process.mode,
                }
            except Exception as sample_err:
                st.error(f"⚠️ Could not load sample image: {sample_err}")
                image_to_process = None
        else:
            st.warning("Sample image not found on disk. Please upload your own image.")

    # Image Preview & Information Badges
    if image_to_process is not None:
        st.markdown("#### 📷 Image Preview")
        st.image(image_to_process, use_container_width=True, caption=image_metadata.get("Name", "Selected Image"))

        # Render metadata badges
        badges_html = "<div>"
        for k, v in image_metadata.items():
            badges_html += f'<span class="meta-tag"><strong>{k}:</strong> {v}</span>'
        badges_html += f'<span class="meta-tag"><strong>Device:</strong> {str(device).upper()}</span>'
        badges_html += "</div>"
        st.markdown(badges_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. Caption Generation & Output Section
# -----------------------------------------------------------------------------
with col_right:
    st.markdown("### 2. Generate Caption")

    if image_to_process is None:
        st.markdown(
            """
            <div style="padding: 2.5rem; text-align: center; border: 2px dashed #94a3b8; border-radius: 12px; color: #64748b; margin-top: 1rem;">
                <p style="font-size: 2.5rem; margin-bottom: 0.5rem;">🖼️</p>
                <p style="font-size: 1.1rem; font-weight: 600;">No Image Selected</p>
                <p style="font-size: 0.9rem;">Upload an image or pick a preloaded sample from the sidebar to generate an AI caption.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.write("Ready to analyze! Click the button below to have the BLIP model inspect the image.")

        generate_btn = st.button("✨ Generate Caption", type="primary", use_container_width=True)

        # Store caption in session state to persist across interactions
        if "last_caption" not in st.session_state:
            st.session_state["last_caption"] = None
            st.session_state["last_latency"] = None

        if generate_btn:
            if not model_loaded:
                st.error("The BLIP model is not available. Please check model installation.")
            else:
                progress_placeholder = st.empty()
                with progress_placeholder.container():
                    with st.spinner("🧠 BLIP Vision-Language model is analyzing image visual features..."):
                        start_time = time.time()
                        try:
                            caption = generate_caption(
                                image=image_to_process,
                                processor=processor,
                                model=model,
                                device=device,
                                max_new_tokens=max_tokens,
                                num_beams=num_beams,
                                conditional_prefix=custom_prompt if custom_prompt else None,
                            )
                            latency = time.time() - start_time
                            st.session_state["last_caption"] = caption
                            st.session_state["last_latency"] = latency
                        except Exception as gen_err:
                            st.error(f"❌ Caption Generation Error: {gen_err}")
                progress_placeholder.empty()

        # Display generated caption if available
        if st.session_state.get("last_caption"):
            generated_caption = st.session_state["last_caption"]
            latency = st.session_state.get("last_latency", 0.0)

            st.markdown(
                f"""
                <div class="caption-box">
                    <div class="caption-label">🎯 Generated English Caption</div>
                    <div class="caption-text">"{generated_caption}"</div>
                    <div style="margin-top: 0.8rem; font-size: 0.8rem; color: #64748b;">
                        ⚡ Generated in {latency:.2f} seconds using {num_beams}-beam search
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Copy to Clipboard Feature
            st.markdown("#### 📋 Copy Caption")
            st.code(generated_caption, language=None)
            st.caption("Click the copy icon on the top-right corner of the box above to copy instantly.")

            # Web Speech API Audio Feature (Listen to Caption)
            st.markdown("#### 🔊 Listen to Caption")
            speech_js = f"""
            <div style="margin-top: 6px;">
                <button onclick="
                    const synth = window.speechSynthesis;
                    const utter = new SpeechSynthesisUtterance('{generated_caption.replace("'", "")}');
                    utter.rate = 0.95;
                    synth.speak(utter);
                " style="
                    background: #4f46e5;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 8px;
                    font-size: 0.88rem;
                    font-weight: 600;
                    cursor: pointer;
                    display: inline-flex;
                    align-items: center;
                    gap: 6px;
                ">
                    🔊 Speak Caption
                </button>
            </div>
            """
            st.components.v1.html(speech_js, height=50)

            # Technical Explanation Accordion
            with st.expander("🔍 How BLIP generated this caption", expanded=False):
                st.markdown(
                    f"""
                    1. **Image Preprocessing:** The image was resized and normalized into tensor patches by `BlipProcessor`.
                    2. **Vision Encoding:** A Vision Transformer (ViT) encoded the visual patches into multi-dimensional feature embeddings.
                    3. **Multimodal Cross-Attention:** The text decoder attended to the image embeddings through cross-attention layers.
                    4. **Beam Search Decoding:** Beam search with width `{num_beams}` evaluated the highest likelihood sequence of English tokens.
                    5. **Token Decoding:** The generated subword tokens were mapped back into readable vocabulary words.
                    """
                )


# -----------------------------------------------------------------------------
# 8. Architecture & Guide Section
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 💡 Understanding the AI Architecture")

arch_col1, arch_col2, arch_col3 = st.columns(3)

with arch_col1:
    st.markdown(
        """
        <div class="card-box">
            <h4>👁️ Computer Vision</h4>
            <p style="font-size: 0.9rem; color: #64748b;">
                The <strong>Vision Transformer (ViT)</strong> breaks down the image into a grid of 16&times;16 patches,
                projecting them into deep visual embeddings that capture objects, positions, textures, and activities.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with arch_col2:
    st.markdown(
        """
        <div class="card-box">
            <h4>🧠 Cross-Attention</h4>
            <p style="font-size: 0.9rem; color: #64748b;">
                The multimodal encoder uses <strong>cross-attention mechanisms</strong> to align visual features
                with linguistic concepts, creating a shared semantic representation space between vision and text.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with arch_col3:
    st.markdown(
        """
        <div class="card-box">
            <h4>📝 Natural Language Processing</h4>
            <p style="font-size: 0.9rem; color: #64748b;">
                The <strong>Autoregressive Text Decoder</strong> predicts the next word one token at a time,
                guided by beam search to formulate grammatical, contextual, and descriptive English captions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

