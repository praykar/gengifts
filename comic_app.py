import sys
import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
else:
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Disable Streamlit file watcher
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
import streamlit as st
import requests
import io
import base64
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont, ImageFile
import torch
import json
import time
import random
from datetime import datetime
import re
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from ebooklib import epub
import threading
from functools import lru_cache
import concurrent.futures
from typing import List, Tuple, Dict, Any, Optional, Union
import textwrap
import numpy as np
import scipy.ndimage
import dlib
import collections
import bz2
# Create a thread-local storage for models to avoid loading them multiple times
thread_local = threading.local()
# Set page configuration
st.set_page_config(
    page_title="Bedtime Story Maker",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #3a86ff;
    }
    .story-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .character-card {
        background-color: #e2eafc;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin: 10px;
    }
    .comic-panel {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .btn-primary {
        background-color: #3a86ff;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
    }
    .footer {
        margin-top: 30px;
        padding-top: 10px;
        border-top: 1px solid #eee;
        text-align: center;
        font-size: 0.8em;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'character_images' not in st.session_state:
    st.session_state.character_images = []
if 'story' not in st.session_state:
    st.session_state.story = ""
if 'comic_panels' not in st.session_state:
    st.session_state.comic_panels = []
if 'story_theme' not in st.session_state:
    st.session_state.story_theme = ""
if 'character_names' not in st.session_state:
    st.session_state.character_names = []
if 'age_group' not in st.session_state:
    st.session_state.age_group = "3-5 years"
if 'hf_api_key' not in st.session_state:
    st.session_state.hf_api_key = ""
if 'book_id' not in st.session_state:
    st.session_state.book_id = datetime.now().strftime("%Y%m%d%H%M%S")
if 'cover_image' not in st.session_state:
    st.session_state.cover_image = None
if 'book_title' not in st.session_state:
    st.session_state.book_title = ""
if 'author_name' not in st.session_state:
    st.session_state.author_name = ""
if 'dedication' not in st.session_state:
    st.session_state.dedication = ""
if 'comic_style' not in st.session_state:
    st.session_state.comic_style = "colorful cartoon"
if 'saved_books' not in st.session_state:
    st.session_state.saved_books = []

# Hugging Face API endpoints
TEXT_TO_IMAGE_API = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
STORY_GENERATION_API = "https://api-inference.huggingface.co/models/microsoft/Phi-3.5-mini-instruct"
COMICS_PROMPT_API = "https://api-inference.huggingface.co/models/microsoft/Phi-3.5-mini-instruct"

# Helper function for API calls
def query_huggingface_api(api_url, payload, api_key, is_image=False):
    """Generic function to call HuggingFace APIs"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json" if not is_image else "application/octet-stream"
    }
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            if is_image:
                response = requests.post(api_url, headers=headers, data=payload)
            else:
                response = requests.post(api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                return response
            elif response.status_code == 503 and "estimated_time" in response.json():
                # Model is loading, wait and retry
                wait_time = min(response.json().get("estimated_time", retry_delay), 10)
                time.sleep(wait_time)
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Error calling API: {str(e)}")
            time.sleep(retry_delay)
    
    st.error("Failed to get a response after multiple retries")
    return None

@lru_cache(maxsize=4)  # Cache the last 4 loaded models
def get_model(style: str) -> Tuple[Any, Any]:
    """Get or initialize models in a thread-safe way with caching"""
    # Map styles to model types
    style_to_model = {
        "anime": "face_paint_512_v2",
        "cartoon": "paprika",
        "fairy tale": "face_paint_512_v1",
        "superhero": "face_paint_512_v1",
        "animal": "face_paint_512_v1",
    }
    
    model_name = style_to_model.get(style.lower(), "face_paint_512_v1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Check if we have already loaded this model in this thread
    if not hasattr(thread_local, 'models'):
        thread_local.models = {}
    
    if model_name not in thread_local.models:
        # Load the model for this thread
        model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", device=device).eval()
        face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device)
        thread_local.models[model_name] = (model, face2paint)
    
    return thread_local.models[model_name]

def apply_style_effects(image: Image.Image, style: str) -> Image.Image:
    """Apply style-specific post-processing effects"""
    if style.lower() == "fairy tale":
        # Dreamy effect for fairy tales
        image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.2)
    elif style.lower() == "superhero":
        # Bold look for superheroes
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.3)
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.1)
    elif style.lower() == "animal":
        # Softer look for animals
        saturation = ImageEnhance.Color(image)
        image = saturation.enhance(1.15)
    
    return image

def get_dlib_face_detector(predictor_path: str = "shape_predictor_68_face_landmarks.dat"):

    # Download the model if needed
    if not os.path.isfile(predictor_path):
        model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        bz2_path = f"{predictor_path}.bz2"
        
        # 1. Download with requests
        print("Downloading model...")
        response = requests.get(model_url, timeout=10)
        with open(bz2_path, "wb") as f:
            f.write(response.content)
        
        # 2. Extract using bz2 module
        print("Extracting model...")
        with bz2.BZ2File(bz2_path) as fr, open(predictor_path, "wb") as fw:
            fw.write(fr.read())
        
        # Clean up .bz2 file
        os.remove(bz2_path)

    # Initialize detector and predictor
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(predictor_path)

    def detect_face_landmarks(img: Union[Image.Image, np.ndarray]):
        if isinstance(img, Image.Image):
            img = np.array(img)
        faces = []
        dets = detector(img)
        for d in dets:
            shape = shape_predictor(img, d)
            faces.append(np.array([[v.x, v.y] for v in shape.parts()]))
        return faces
    
    return detect_face_landmarks
    
def align_and_crop_face(
    img: Image.Image,
    landmarks: np.ndarray,
    expand: float = 1.0,
    output_size: int = 1024, 
    transform_size: int = 4096,
    enable_padding: bool = True,
):
    # Parse landmarks.
    # pylint: disable=unused-variable
    lm = landmarks
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    x *= expand
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    return img
    
def transform_single_image(args: Tuple[Image.Image, str, int]) -> Tuple[int, Optional[Image.Image]]:
    """Transform a single image with index tracking for preserving order"""
    image, style, index = args
    try:
        # Resize image if too large to save memory and processing time
        max_size = 512
        if max(image.size) > max_size:
            # Calculate new dimensions while preserving aspect ratio
            if image.width > image.height:
                new_width = max_size
                new_height = int(image.height * (max_size / image.width))
            else:
                new_height = max_size
                new_width = int(image.width * (max_size / image.height))
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Get model for this style
        model, face2paint = get_model(style)
        
        # Use GPU if available
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Transform the image
        with torch.no_grad():  # Disable gradient calculation for inference
            face_detector = get_dlib_face_detector()
            landmarks = face_detector(image)
            for landmark in landmarks:
                face = align_and_crop_face(img, landmark, expand=1.3)
            transformed = face2paint(model, face)
        
        # Apply style-specific effects
        transformed = apply_style_effects(transformed, style)
        
        return index, transformed
    
    except Exception as e:
        st.warning(f"Image transformation failed: {str(e)}")
        return index, None

def transform_to_character(images: List[Image.Image], style: str, api_key: str = None) -> List[Image.Image]:
    """Transform multiple photos into cartoon/animated characters in parallel"""
    # Create progress indicator
    progress_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    # Prepare arguments for parallel processing
    args = [(img, style, i) for i, img in enumerate(images)]
    results = [None] * len(images)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(images))) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(transform_single_image, arg): i for i, arg in enumerate(args)}
        
        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_idx)):
            original_idx = future_to_idx[future]
            try:
                idx, transformed_img = future.result()
                if transformed_img:
                    results[idx] = transformed_img
                else:
                    # Fallback to original if transformation failed
                    results[idx] = images[idx]
            except Exception as exc:
                st.warning(f"Image {original_idx} generated an exception: {exc}")
                results[original_idx] = images[original_idx]
            
            # Update progress
            progress = (i + 1) / len(images)
            progress_bar.progress(progress)
            progress_placeholder.text(f"Transforming images: {i+1}/{len(images)} complete")
    
    # Clear progress indicators
    progress_placeholder.empty()
    progress_bar.empty()
    
    return results
    
def generate_story(theme, characters, age_group, elements, morals, api_key):
    """Generate a bedtime story using HF LLM API"""
    # Craft a detailed prompt based on user inputs
    age_appropriate_content = {
        "2-3 years": "very simple language, short sentences, repetitive phrases, bright and happy themes",
        "3-5 years": "simple language, short paragraphs, clear moral lessons, gentle adventures",
        "5-7 years": "more complex sentences, mild challenges for characters, clear resolution",
        "7-10 years": "longer story, more detailed descriptions, character development, nuanced themes"
    }
    
    # Format characters for the prompt
    characters_text = ", ".join([f"{name}" for name in characters])
    elements_text = ", ".join(elements) if elements else "no specific elements"
    morals_text = ", ".join(morals) if morals else "no specific moral lessons"
    
    # Create a well-crafted prompt for the LLM
    prompt = f"""
    Please write a bedtime story for children in the {age_group} age range. The story should:
    - Feature characters named: {characters_text}
    - Be set in a theme of: {theme}
    - Include these elements: {elements_text}
    - Include these moral lessons: {morals_text}
    - Be appropriate for {age_group} with {age_appropriate_content.get(age_group, "age-appropriate content")}
    - Be approximately 250-400 words long
    - Have a clear beginning, middle, and end
    - Be engaging and calming for bedtime
    - End with a gentle, satisfying conclusion that helps children transition to sleep
    - Begin with a title for the story on its own line, followed by the story text
    
    Format your response with the Title on the first line, followed by the story text. Do not include any additional comments or explanations.
    """
    
    # Prepare the payload
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 3072,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    }
    
    # Call the API
    response = query_huggingface_api(STORY_GENERATION_API, payload, api_key)
    
    if response and response.status_code == 200:
        # Extract the generated text
        generated_text = response.json()[0].get("generated_text", "").strip()
        
        # Extract just the story part by removing the prompt
        # Find where the actual story starts after the prompt
        story_start = generated_text.find("Title:") 
        if story_start == -1:
            # Try other common starting patterns
            possible_starts = ["The ", "Once upon", "In the", "Once", "Title", "# "]
            for start_phrase in possible_starts:
                story_start = generated_text.find(start_phrase)
                if story_start != -1:
                    break
        
        if story_start != -1:
            # Extract just the story content
            story_text = generated_text[story_start:].strip()
            
            # Try to separate title and story content
            lines = story_text.split('\n', 1)
            
            if len(lines) >= 2:
                title = lines[0].strip().replace("Title:", "").replace("#", "").strip()
                content = lines[1].strip()
                
                # Store title separately if desired
                st.session_state.story_title = title
                
                # Format the final story with title
                final_story = f"# {title}\n\n{content}"
                return final_story
            else:
                return story_text
        else:
            return generated_text
    else:
        # Fallback story for demo purposes
        st.warning("Story generation API call failed. Using a sample story instead.")
        return f"""
        # The {theme} Adventure
        
        Once upon a time in the magical land of {theme}, there lived {characters_text}. 
        They were the best of friends who loved to explore and have adventures together.
        
        One sunny morning, they decided to go on a special journey to find the Rainbow Crystal.
        Along their path, they encountered friendly woodland creatures who offered guidance.
        
        When they reached the Crystal Cave, they learned that working together was the true magic.
        Each friend contributed their unique talents, and together they solved the puzzle.
        
        As the sun began to set, they returned home with their hearts full of joy and friendship.
        That night, as they drifted off to sleep, they dreamed of their next wonderful adventure together.
        """

# Updated functions for parallel comic panel generation with dialogue inpainting

def generate_comic_panel_with_retry(scene_description, dialogue, style, character_names, api_key, index):
    """Generate a comic panel with embedded dialogue using Text-to-Image API with retries"""
    # Prepare a detailed prompt for the image generation that includes dialogue instructions
    style_descriptions = {
        "colorful cartoon": "bright colors, simple shapes, cartoon style, child-friendly",
        "classic comic": "comic book style, clear lines, primary colors, classic look",
        "watercolor": "soft watercolor style, gentle colors, dreamy appearance",
        "sketch": "hand-drawn sketch style, pencil lines, simple coloring",
        "manga": "manga/anime style, expressive characters, dynamic composition"
    }
    
    style_desc = style_descriptions.get(style.lower(), "cartoon style")
    character_desc = ", ".join(character_names) if character_names else "characters"
    
    # Create a prompt that includes instructions for dialogue integration
    prompt = f"""
    A children's storybook comic panel in {style_desc} showing: {scene_description}
    The panel should include speech bubbles or caption boxes containing this dialogue: "{dialogue}"
    The speech bubbles should be clearly visible with black text on white background.
    Include characters named: {character_desc}.
    The image should be child-appropriate, colorful, and engaging for kids.
    """
    
    # Prepare the payload
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": "blurry text, illegible writing, distorted speech bubbles, adult content",
            "num_inference_steps": 30,
            "guidance_scale": 7.5
        }
    }
    
    # Add retry logic for API calls
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Call the API
            response = query_huggingface_api(TEXT_TO_IMAGE_API, payload, api_key)
            
            if response and response.status_code == 200:
                # Convert response to image
                image_bytes = response.content
                panel_image = Image.open(BytesIO(image_bytes))
                return index, panel_image, scene_description, dialogue
            else:
                # Sleep before retry to avoid rate limits
                time.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"Error generating panel {index}: {str(e)}")
            time.sleep(2 * (attempt + 1))
    
    # Fallback for demo purposes if all retries fail
    print(f"Panel generation failed for panel {index}. Using placeholder image.")
    placeholder = Image.new('RGB', (800, 600), color=(random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)))
    return index, placeholder, scene_description, dialogue

def generate_comic_panels_parallel(panel_prompts, character_names, style, api_key, max_workers=3):
    """Generate comic panels in parallel with controlled concurrency"""
    comic_panels = []
    results = [None] * len(panel_prompts)
    
    # Set up a progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # Process panels in parallel with limited concurrency
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = []
        for i, panel_data in enumerate(panel_prompts):
            scene = panel_data['scene']
            dialogue = panel_data['dialogue']
            
            # Submit the task and store the future
            future = executor.submit(
                generate_comic_panel_with_retry,
                scene_description=scene,
                dialogue=dialogue,
                style=style,
                character_names=character_names,
                api_key=api_key,
                index=i
            )
            futures.append(future)
        
        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                # Get the result
                index, panel_image, scene, dialogue = future.result()
                results[index] = (panel_image, scene, dialogue)
                
                # Update progress
                progress = (i + 1) / len(panel_prompts)
                progress_bar.progress(progress)
                progress_text.write(f"Generated panel {i+1}/{len(panel_prompts)}...")
                
                # Add a small delay between API calls to avoid rate limiting
                time.sleep(1.5)
            except Exception as e:
                st.warning(f"Error processing panel: {str(e)}")
    
    # Clear progress indicators
    progress_text.empty()
    progress_bar.empty()
    
    # Make sure results are in the correct order
    for result in results:
        if result is not None:
            comic_panels.append(result)
    
    return comic_panels

def generate_comic_prompts(story, num_panels, api_key):
    """Break down a story into prompts for comic panels using LLM with improved dialogue extraction"""
    prompt = f"""
    I have a children's bedtime story that I want to turn into a {num_panels}-panel comic book with dialogue bubbles.
    Please break this story into exactly {num_panels} key scenes that would make good comic panels.
    
    For each panel, provide:
    1. 'scene': A detailed visual description of what's happening (setting, characters, actions)
    2. 'dialogue': Short, direct dialogue text that should appear in speech bubbles or caption boxes in the panel (keep this under 20 words)
    
    Format your response as a JSON array of objects with keys 'scene' and 'dialogue'.
    Make sure the dialogue is child-friendly, engaging, and short enough to fit in speech bubbles.
    
    Here's the story:
    {story}
    """
    
    # Prepare the payload
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 3072,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    # Call the API
    response = query_huggingface_api(COMICS_PROMPT_API, payload, api_key)
    
    if response and response.status_code == 200:
        # Extract and parse the generated JSON
        response_text = response.json()[0].get("generated_text", "")
        
        # Find JSON-like content
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
        if json_match:
            try:
                panels_data = json.loads(json_match.group(0))
                return panels_data
            except json.JSONDecodeError:
                pass
    
    # Fallback: create simple panel descriptions
    st.warning("Comic prompt generation failed. Using basic panel breakdown.")
    
    # Split the story into roughly equal parts
    sentences = re.split(r'(?<=[.!?])\s+', story)
    panel_size = max(1, len(sentences) // num_panels)
    
    panels_data = []
    for i in range(num_panels):
        start_idx = i * panel_size
        end_idx = min(start_idx + panel_size, len(sentences))
        panel_text = " ".join(sentences[start_idx:end_idx])
        
        # Extract a potential dialogue (text in quotes if available)
        dialogue_match = re.search(r'"([^"]+)"', panel_text)
        dialogue = dialogue_match.group(1) if dialogue_match else "..."
        
        panels_data.append({
            "scene": f"Panel showing: {panel_text}",
            "dialogue": dialogue
        })
    
    return panels_data

def add_dialogue_to_panel(image, dialogue, style="speech", position="top"):
    """
    Add dialogue overlay to a comic panel image
    
    Args:
        image: PIL Image object
        dialogue: Text to add to the image
        style: 'speech' for speech bubble or 'caption' for caption box
        position: Where to place the dialogue ('top', 'bottom', 'auto')
    
    Returns:
        PIL Image with dialogue added
    """
    if not dialogue or dialogue.strip() == "":
        return image
    
    # Create a copy of the image to avoid modifying the original
    panel_with_dialogue = image.copy()
    draw = ImageDraw.Draw(panel_with_dialogue)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
    
    # Wrap text to fit within the image width
    margin = 20
    max_width = image.width - 2 * margin
    wrapped_text = textwrap.fill(dialogue, width=max_width // 10)  # Approximate character width
    
    # Calculate text size
    try:
        text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    except AttributeError:
        # Fallback for older PIL versions
        text_width, text_height = draw.textsize(wrapped_text, font=font)
    
    # Determine position
    if position == "auto":
        # Use image analysis to find suitable position (simplified version)
        # For now, default to top
        position = "top"
    
    padding = 10
    bubble_width = text_width + 2 * padding
    bubble_height = text_height + 2 * padding
    
    if position == "top":
        bubble_x = (image.width - bubble_width) // 2
        bubble_y = margin
    else:  # bottom
        bubble_x = (image.width - bubble_width) // 2
        bubble_y = image.height - margin - bubble_height
    
    # Draw the bubble or caption box
    if style == "speech":
        # Speech bubble (rounded rectangle)
        draw.rounded_rectangle(
            [bubble_x, bubble_y, bubble_x + bubble_width, bubble_y + bubble_height],
            radius=15,
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=2
        )
        
        # Add a little triangle for speech bubble pointer
        if position == "top":
            # Pointer at bottom of bubble
            pointer_points = [
                (bubble_x + bubble_width // 2, bubble_y + bubble_height),
                (bubble_x + bubble_width // 2 - 10, bubble_y + bubble_height + 15),
                (bubble_x + bubble_width // 2 + 10, bubble_y + bubble_height)
            ]
        else:
            # Pointer at top of bubble
            pointer_points = [
                (bubble_x + bubble_width // 2, bubble_y),
                (bubble_x + bubble_width // 2 - 10, bubble_y - 15),
                (bubble_x + bubble_width // 2 + 10, bubble_y)
            ]
            
        draw.polygon(pointer_points, fill=(255, 255, 255), outline=(0, 0, 0))
    else:
        # Caption box (rectangle)
        draw.rectangle(
            [bubble_x, bubble_y, bubble_x + bubble_width, bubble_y + bubble_height],
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=2
        )
    
    # Draw the text
    text_x = bubble_x + padding
    text_y = bubble_y + padding
    draw.text((text_x, text_y), wrapped_text, font=font, fill=(0, 0, 0))
    
    return panel_with_dialogue

# Modified version of generate_comic_panel_with_retry to use dialogue overlay
def generate_comic_panel_with_overlay(scene_description, dialogue, style, character_names, api_key, index):
    """Generate a comic panel and overlay dialogue with speech bubbles"""
    # First generate the panel without dialogue in the prompt
    style_descriptions = {
        "colorful cartoon": "bright colors, simple shapes, cartoon style, child-friendly",
        "classic comic": "comic book style, clear lines, primary colors, classic look",
        "watercolor": "soft watercolor style, gentle colors, dreamy appearance",
        "sketch": "hand-drawn sketch style, pencil lines, simple coloring",
        "manga": "manga/anime style, expressive characters, dynamic composition"
    }
    
    style_desc = style_descriptions.get(style.lower(), "cartoon style")
    character_desc = ", ".join(character_names) if character_names else "characters"
    
    # Simplify prompt to focus on scene without dialogue instructions
    prompt = f"""
    A children's storybook comic panel in {style_desc} showing: {scene_description}
    Include characters named: {character_desc}.
    The image should be child-appropriate, colorful, and engaging for kids.
    Leave space at the top or bottom for speech bubbles.
    """
    
    # Prepare the payload
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": "text, writing, speech bubbles, words, labels",
            "num_inference_steps": 30,
            "guidance_scale": 7.5
        }
    }
    
    # Add retry logic for API calls
    max_retries = 3
    panel_image = None
    
    for attempt in range(max_retries):
        try:
            # Call the API
            response = query_huggingface_api(TEXT_TO_IMAGE_API, payload, api_key)
            
            if response and response.status_code == 200:
                # Convert response to image
                image_bytes = response.content
                panel_image = Image.open(BytesIO(image_bytes))
                break
            else:
                # Sleep before retry
                time.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"Error generating panel {index}: {str(e)}")
            time.sleep(2 * (attempt + 1))
    
    # Fallback for demo purposes
    if panel_image is None:
        print(f"Panel generation failed for panel {index}. Using placeholder image.")
        panel_image = Image.new('RGB', (800, 600), color=(random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)))
    
    # Add dialogue with speech bubbles
    if dialogue and dialogue.strip():
        # Determine if this is a character speaking or narrator
        is_dialogue = '"' in dialogue or "'" in dialogue
        style_type = "speech" if is_dialogue else "caption"
        panel_image = add_dialogue_to_panel(panel_image, dialogue, style=style_type, position="top" if is_dialogue else "bottom")
    
    return index, panel_image, scene_description, dialogue

def generate_cover_image(title, theme, api_key):
    """Generate a cover image for the storybook"""
    prompt = f"""
    Children's storybook cover for '{title}', {theme} theme, colorful illustration, 
    child-friendly, magical, bedtime story, cute characters, illustrated style
    """
    
    # Prepare the payload
    payload = {
        "inputs": prompt,
        "parameters": {
            #"negative_prompt": "scary, adult content, realistic, photorealistic, detailed faces, inappropriate for children",
            "num_inference_steps": 50,  # Higher quality for cover
            "guidance_scale": 7.5
        }
    }
    
    # Call the API
    response = query_huggingface_api(TEXT_TO_IMAGE_API, payload, api_key)
    
    if response and response.status_code == 200:
        # Convert response to image
        image_bytes = response.content
        cover_image = Image.open(BytesIO(image_bytes))
        return cover_image
    else:
        # Fallback for demo purposes
        st.warning("Cover generation API call failed. Using a placeholder image.")
        placeholder = Image.new('RGB', (800, 1000), color=(random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)))
        return placeholder

def create_pdf_storybook(title, author, dedication, story, cover_image, comic_panels):
    """Create a PDF storybook from the story and panels"""
    try:
        
        
        # Create a BytesIO object to store the PDF
        buffer = BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading_style = styles['Heading1']
        normal_style = styles['Normal']
        
        # Custom styles
        dedication_style = ParagraphStyle(
            'Dedication', 
            parent=styles['Italic'],
            fontSize=12,
            alignment=1,  # Center
            spaceAfter=30
        )
        
        # Create the story elements
        elements = []
        
        # Cover page
        if cover_image:
            # Save cover image to temp file
            temp_cover = BytesIO()
            cover_image.save(temp_cover, format='JPEG')
            temp_cover.seek(0)
            cover = RLImage(temp_cover, width=doc.width, height=doc.width * 0.75)
            elements.append(cover)
        
        # Title
        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Author
        elements.append(Paragraph(f"By: {author}", normal_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Dedication
        if dedication:
            elements.append(Paragraph(dedication, dedication_style))
        
        elements.append(PageBreak())
        
        # Story text
        elements.append(Paragraph("The Story", heading_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Split story into paragraphs
        for paragraph in story.split('\n\n'):
            if paragraph.strip():
                elements.append(Paragraph(paragraph, normal_style))
                elements.append(Spacer(1, 0.2*inch))
        
        elements.append(PageBreak())
        
        # Comic panels
        elements.append(Paragraph("Comic Book", heading_style))
        elements.append(Spacer(1, 0.2*inch))
        
        for i, (image, scene, dialogue) in enumerate(comic_panels):
            # Save panel image to temp file
            temp_panel = BytesIO()
            image.save(temp_panel, format='JPEG')
            temp_panel.seek(0)
            
            panel = RLImage(temp_panel, width=doc.width, height=doc.width * 0.75)
            elements.append(panel)
            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph(dialogue, normal_style))
            elements.append(Spacer(1, 0.3*inch))
            
            # Add page break after every other panel (except the last one)
            if i < len(comic_panels) - 1 and i % 2 == 1:
                elements.append(PageBreak())
        
        # Build the PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer
    except ImportError:
        st.warning("PDF generation requires reportlab library. Using simple text output instead.")
        # Create a simple text version as fallback
        buffer = BytesIO()
        content = f"{title}\nBy: {author}\n\n{dedication}\n\n{story}\n\n"
        buffer.write(content.encode('utf-8'))
        buffer.seek(0)
        return buffer

def create_epub_storybook(title, author, dedication, story, cover_image, comic_panels):
    """Create an EPUB storybook from the story and panels"""
    try:        
        # Create a new EPUB book
        book = epub.EpubBook()
        
        # Set metadata
        book.set_identifier(f"bedtimestory{st.session_state.book_id}")
        book.set_title(title)
        book.set_language('en')
        book.add_author(author)
        
        # Add cover
        if cover_image:
            temp_cover = BytesIO()
            cover_image.save(temp_cover, format='JPEG')
            temp_cover.seek(0)
            book.set_cover("cover.jpg", temp_cover.read())
        
        # Create chapters
        # Title page
        title_content = f'''
        <html>
        <head>
            <title>{title}</title>
        </head>
        <body>
            <h1>{title}</h1>
            <p>By: {author}</p>
            <p><em>{dedication}</em></p>
        </body>
        </html>
        '''
        title_page = epub.EpubHtml(title='Title Page', file_name='title.xhtml', content=title_content)
        book.add_item(title_page)
        
        # Story chapter
        story_content = f'''
        <html>
        <head>
            <title>The Story</title>
        </head>
        <body>
            <h1>The Story</h1>
            <p>{"</p><p>".join(story.split('\n\n'))}</p>
        </body>
        </html>
        '''
        story_chapter = epub.EpubHtml(title='The Story', file_name='story.xhtml', content=story_content)
        book.add_item(story_chapter)
        
        # Comic panels chapters
        comic_chapters = []
        for i, (image, scene, dialogue) in enumerate(comic_panels):
            # Save panel image
            temp_panel = BytesIO()
            image.save(temp_panel, format='JPEG')
            temp_panel.seek(0)
            
            # Create image item
            img_name = f'panel{i+1}.jpg'
            panel_img = epub.EpubItem(
                uid=f'panel{i+1}',
                file_name=f'images/{img_name}',
                media_type='image/jpeg',
                content=temp_panel.read()
            )
            book.add_item(panel_img)
            
            # Create panel chapter
            panel_content = f'''
            <html>
            <head>
                <title>Panel {i+1}</title>
            </head>
            <body>
                <img src="images/{img_name}" alt="Panel {i+1}"/>
                <p>{dialogue}</p>
            </body>
            </html>
            '''
            panel_chapter = epub.EpubHtml(title=f'Panel {i+1}', file_name=f'panel{i+1}.xhtml', content=panel_content)
            book.add_item(panel_chapter)
            comic_chapters.append(panel_chapter)
        
        # Define Table of Contents
        book.toc = ((
            epub.Section('Table of Contents'),
            (title_page, story_chapter, *comic_chapters)
        ))
        
        # Add navigation files
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        
        # Define book spine
        book.spine = ['cover', 'nav', title_page, story_chapter, *comic_chapters]
        
        # Create a BytesIO object to store the EPUB
        buffer = BytesIO()
        epub.write_epub(buffer, book, {})
        buffer.seek(0)
        return buffer
    except ImportError:
        st.warning("EPUB generation requires ebooklib. Using PDF format instead.")
        return create_pdf_storybook(title, author, dedication, story, cover_image, comic_panels)

def image_to_base64(image):
    """Convert PIL Image to base64 string for display"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def validate_api_key(api_key):
    """Validate that the API key works with a simple request"""
    test_url = "https://api-inference.huggingface.co/models/gpt2"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.post(
            test_url, 
            headers=headers, 
            json={"inputs": "Hello, I'm testing my API key"}
        )
        if response.status_code == 200:
            return True
        elif response.status_code == 401:
            return False
        else:
            # Model might be loading, but auth is ok
            return True
    except:
        return False

def save_book_to_collection(book_data):
    """Save the current book to user's collection"""
    st.session_state.saved_books.append(book_data)
    st.success(f"Book '{book_data['title']}' saved to your collection!")

def share_via_email(email, book_data, file_data, file_format):
    """Share the storybook via email (simulated)"""
    # In a real app, this would connect to an email service
    st.success(f"Book '{book_data['title']}' would be sent to {email} as {file_format}!")
    # For demo purposes, we'll just download the file instead
    st.download_button(
        label=f"Download {file_format} instead",
        data=file_data,
        file_name=f"{book_data['title'].replace(' ', '_')}.{file_format.lower()}",
        mime=f"application/{file_format.lower()}"
    )

# Helper function to navigate between steps
def set_step(step_num):
    st.session_state.current_step = step_num

# Sidebar for API key and navigation
with st.sidebar:
    st.image("https://images.seeklogo.com/logo-png/48/1/storybook-logo-png_seeklogo-488504.png", width=150)
    st.title("Bedtime Story Maker")
    st.markdown("Create personalized bedtime stories for your little ones!")
    
    # API Key input
    api_key = st.text_input("Enter Hugging Face API Key", value=st.session_state.hf_api_key, type="password")
    if api_key:
        st.session_state.hf_api_key = api_key
        
        # Validate the API key
        if st.button("Validate API Key"):
            if validate_api_key(api_key):
                st.success("API Key is valid!")
            else:
                st.error("Invalid API Key. Please check and try again.")
    
    st.markdown("### Your Progress")
    step_statuses = ["✅" if i < st.session_state.current_step else "⬜" for i in range(1, 6)]
    st.markdown(f"1. Upload Photos {step_statuses[0]}")
    st.markdown(f"2. Create Characters {step_statuses[1]}")
    st.markdown(f"3. Generate Story {step_statuses[2]}")
    st.markdown(f"4. Create Comic Book {step_statuses[3]}")
    st.markdown(f"5. Save & Share {step_statuses[4]}")
    
    # View saved books
    if st.session_state.saved_books:
        st.markdown("---")
        st.markdown("### My Story Collection")
        for book in st.session_state.saved_books:
            st.markdown(f"📖 {book['title']}")
    
    st.markdown("---")
    st.markdown("Made with ❤️ for bedtime adventures")

# Check if API key is provided before allowing further actions
if not st.session_state.hf_api_key and st.session_state.current_step > 2:
    st.warning("Please enter your Hugging Face API Key in the sidebar to continue.")
    set_step(1)

# Main content based on current step
if st.session_state.current_step == 1:
    # Step 1: Upload Photos
    st.title("Step 1: Upload Photos")
    st.markdown("Upload photos of people to be transformed into story characters.")
    
    # if not st.session_state.hf_api_key:
    #     st.info("Please enter your Hugging Face API Key in the sidebar to enable all features.")
    
    uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        st.session_state.uploaded_images = []
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            # Resize large images to reduce processing time
            if max(image.size) > 1024:
                image.thumbnail((1024, 1024), Image.LANCZOS)
            st.session_state.uploaded_images.append(image)
        
        # Display uploaded images
        st.markdown("### Preview Uploaded Photos")
        cols = st.columns(min(4, len(st.session_state.uploaded_images)))
        for i, col in enumerate(cols):
            if i < len(st.session_state.uploaded_images):
                col.image(st.session_state.uploaded_images[i], use_container_width=True, caption=f"Photo {i+1}")
    
    if st.button("Continue to Character Creation", key="btn_to_step2"):
        if not st.session_state.uploaded_images:
            st.error("Please upload at least one photo to continue.")
        # elif not st.session_state.hf_api_key:
        #     st.error("Please enter your Hugging Face API Key in the sidebar.")
        else:
            set_step(2)

elif st.session_state.current_step == 2:
    # Step 2: Character Creation
    st.title("Step 2: Create Characters")
    st.markdown("Transform your photos into story characters.")
    
    # Display uploaded photos and create character forms
    st.markdown("### Your Photos")
    cols = st.columns(min(4, len(st.session_state.uploaded_images)))
    character_names = []
    
    for i, col in enumerate(cols):
        if i < len(st.session_state.uploaded_images):
            col.image(st.session_state.uploaded_images[i], use_container_width=True)
            name = col.text_input(f"Character {i+1} Name", value=f"Character {i+1}", key=f"name_{i}")
            character_names.append(name)

    st.markdown("### Choose Character Style")
    style_options = ["Cartoon", "Anime", "Fairy Tale", "Superhero", "Animal"]
    selected_style = st.selectbox("Select a style for your characters", style_options)
    
    if st.button("Transform into Characters", key="transform_chars"):
        st.session_state.character_names = character_names
        
        # Show a loading spinner while processing
        with st.spinner("Transforming photos into characters... This may take a minute."):
            # Process all images in parallel
            transformed_images = transform_to_character(
                st.session_state.uploaded_images, 
                style=selected_style.lower()
            )
            st.session_state.character_images = transformed_images
        
        # Display the transformed characters
        st.success("Character transformation complete!")
        st.markdown("### Your Story Characters")
        char_cols = st.columns(min(4, len(st.session_state.character_images)))
        for i, col in enumerate(char_cols):
            if i < len(st.session_state.character_images):
                col.image(st.session_state.character_images[i], use_container_width=True, caption=st.session_state.character_names[i])
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Photo Upload", key="btn_back_to_step1"):
            set_step(1)
    with col2:
        if st.button("Continue to Story Generation ➡️", key="btn_to_step3"):
            if not st.session_state.character_images:
                st.error("Please transform your photos into characters first.")
            else:
                set_step(3)

elif st.session_state.current_step == 3:
    # Step 3: Story Generation
    st.title("Step 3: Generate Your Bedtime Story")
    st.markdown("Customize your story and generate a unique bedtime adventure!")
    
    # Display character thumbnails
    st.markdown("### Story Characters")
    char_cols = st.columns(min(6, len(st.session_state.character_images)))
    for i, col in enumerate(char_cols):
        if i < len(st.session_state.character_images):
            col.image(st.session_state.character_images[i], width=100, caption=st.session_state.character_names[i])
    
    # Story customization options
    st.markdown("### Story Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        story_themes = ["Adventure in the Forest", "Space Exploration", "Underwater Journey", 
                        "Magical Kingdom", "Dinosaur Discovery", "Cloud Castle"]
        st.session_state.story_theme = st.selectbox("Select a story theme", story_themes)
    
    with col2:
        age_groups = ["2-3 years", "3-5 years", "5-7 years", "7-10 years"]
        st.session_state.age_group = st.selectbox("Age group", age_groups)
    
    additional_elements = st.multiselect("Story elements to include",
        ["Talking animals", "Magic spells", "Hidden treasure", "Friendly monster", "Flying", "Music", "Dream"])
    
    moral_lessons = st.multiselect("Moral lessons",
        ["Friendship", "Sharing", "Bravery", "Kindness", "Honesty", "Patience", "Creativity"])
    
    # Generate story button
    if st.button("Generate Story", key="generate_story_btn"):
        with st.spinner("Creating your unique bedtime story... This may take a minute."):
            # Generate the story using LLM
            story = generate_story(
                theme=st.session_state.story_theme,
                characters=st.session_state.character_names,
                age_group=st.session_state.age_group,
                elements=additional_elements,
                morals=moral_lessons,
                api_key=st.session_state.hf_api_key
            )
            st.session_state.story = story
    
    # Display the generated story
    if st.session_state.story:
        st.markdown("### Your Bedtime Story")
        
        with st.container():
            # Use st.write for better HTML/markdown compatibility
            st.write(f"**{st.session_state.story_theme}**")
            # Use an expander to ensure the story is fully visible
            with st.expander("Read the full story", expanded=True):
                # Format the story with proper line breaks
                formatted_story = st.session_state.story.replace("\n", "<br>")
                st.markdown(formatted_story, unsafe_allow_html=True)
            
            st.markdown("#### How do you like your story?")
            satisfaction = st.slider("Story satisfaction", 1, 5, 4)
            
            if satisfaction < 4:
                if st.button("Regenerate Story", key="regenerate_story"):
                    with st.spinner("Creating a new story..."):
                        # Generate a new story
                        story = generate_story(
                            theme=st.session_state.story_theme,
                            characters=st.session_state.character_names,
                            age_group=st.session_state.age_group,
                            elements=additional_elements,
                            morals=moral_lessons,
                            api_key=st.session_state.hf_api_key
                        )
                        st.session_state.story = story
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Character Creation", key="btn_back_to_step2"):
            set_step(2)
    with col2:
        if st.button("Continue to Comic Book Creation ➡️", key="btn_to_step4"):
            if not st.session_state.story:
                st.error("Please generate a story first.")
            else:
                # Get book title an author for next step
                st.session_state.book_title = f"{st.session_state.story_theme} Adventure"
                set_step(4)

elif st.session_state.current_step == 4:
    # Step 4: Comic Book Creation
    st.title("Step 4: Create Your Comic Book")
    st.markdown("Transform your story into a beautiful comic book!")
    
    # Get book details
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.book_title = st.text_input("Book Title", 
                                                    value=st.session_state.book_title or f"{st.session_state.story_theme} Adventure")
    with col2:
        st.session_state.author_name = st.text_input("Author Name", 
                                                     value=st.session_state.author_name or "Your Name")
    
    st.session_state.dedication = st.text_area("Dedication (optional)", 
                                              value=st.session_state.dedication or "To all the dreamers...")
    
    # Comic style selection
    st.markdown("### Comic Style")
    style_options = ["Colorful Cartoon", "Classic Comic", "Watercolor", "Sketch", "Manga"]
    st.session_state.comic_style = st.selectbox("Select comic style", style_options).lower()
    
    # Number of panels
    num_panels = st.slider("Number of Comic Panels", min_value=4, max_value=10, value=6)
    
    # Generate comic panels
    if st.button("Create Comic Panels", key="create_comic_btn"):
        with st.spinner("Creating your comic book panels... This may take a few minutes."):
            # First, generate prompts for each panel
            panel_prompts = generate_comic_prompts(
                story=st.session_state.story, 
                num_panels=num_panels,
                api_key=st.session_state.hf_api_key
            )
            
            # Generate images for each panel using parallel processing
            st.session_state.comic_panels = generate_comic_panels_parallel(
                panel_prompts=panel_prompts,
                character_names=st.session_state.character_names,
                style=st.session_state.comic_style,
                api_key=st.session_state.hf_api_key,
                max_workers=3  # Adjust based on API rate limits
            )
            
            # Generate cover image last
            st.session_state.cover_image = generate_cover_image(
                title=st.session_state.book_title,
                theme=st.session_state.story_theme,
                api_key=st.session_state.hf_api_key
            )
            
            st.success("Comic book created successfully!")
    
    # Display generated comic panels
    if st.session_state.comic_panels:
        st.markdown("### Your Comic Book Preview")
        
        # Show cover image
        if st.session_state.cover_image:
            st.markdown("#### Cover")
            st.image(st.session_state.cover_image, use_container_width=True)
        
        # Show panels
        st.markdown("#### Panels")
        for i, (image, scene, dialogue) in enumerate(st.session_state.comic_panels):
            with st.container():
                st.markdown(f"""
                <div class="comic-panel">
                    <h4>Panel {i+1}</h4>
                </div>
                """, unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                st.caption(dialogue)
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Story Generation", key="btn_back_to_step3"):
            set_step(3)
    with col2:
        if st.button("Continue to Save & Share ➡️", key="btn_to_step5"):
            if not st.session_state.comic_panels:
                st.error("Please create comic panels first.")
            else:
                set_step(5)

elif st.session_state.current_step == 5:
    # Step 5: Save and Share
    st.title("Step 5: Save & Share Your Storybook")
    st.markdown("Your personalized bedtime storybook is ready! Save it or share it with others.")
    
    # Display a final preview
    st.markdown("### Your Completed Storybook")
    
    # Book details
    with st.container():
        st.markdown(f"""
        <div class="story-card">
            <h2>{st.session_state.book_title}</h2>
            <p>By: {st.session_state.author_name}</p>
            <p><em>{st.session_state.dedication}</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Preview tabs
    tab1, tab2 = st.tabs(["📚 Story", "🖼️ Comic"])
    
    with tab1:
        st.markdown(f"""
        <div class="story-card">
            <p>{st.session_state.story}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        # Show a few panels as preview
        preview_panels = st.session_state.comic_panels[:3]
        for i, (image, scene, dialogue) in enumerate(preview_panels):
            with st.container():
                st.image(image, use_container_width=True)
                st.caption(dialogue)
        
        if len(st.session_state.comic_panels) > 3:
            st.info(f"+ {len(st.session_state.comic_panels) - 3} more panels in the full book")
    
    # Download options
    st.markdown("### Download Your Storybook")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # PDF download
        if st.button("Create PDF Storybook", key="create_pdf_btn"):
            with st.spinner("Creating PDF... This may take a moment."):
                pdf_buffer = create_pdf_storybook(
                    title=st.session_state.book_title,
                    author=st.session_state.author_name,
                    dedication=st.session_state.dedication,
                    story=st.session_state.story,
                    cover_image=st.session_state.cover_image,
                    comic_panels=st.session_state.comic_panels
                )
                
                st.download_button(
                    label="Download PDF",
                    data=pdf_buffer,
                    file_name=f"{st.session_state.book_title.replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
    
    with col2:
        # EPUB download
        if st.button("Create EPUB eBook", key="create_epub_btn"):
            with st.spinner("Creating EPUB... This may take a moment."):
                epub_buffer = create_epub_storybook(
                    title=st.session_state.book_title,
                    author=st.session_state.author_name,
                    dedication=st.session_state.dedication,
                    story=st.session_state.story,
                    cover_image=st.session_state.cover_image,
                    comic_panels=st.session_state.comic_panels
                )
                
                st.download_button(
                    label="Download EPUB",
                    data=epub_buffer,
                    file_name=f"{st.session_state.book_title.replace(' ', '_')}.epub",
                    mime="application/epub+zip"
                )
    
    # Share options
    st.markdown("### Share Your Storybook")
    
    # Email sharing (simulated)
    email = st.text_input("Share via Email", placeholder="example@email.com")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Share as PDF", key="share_pdf_btn"):
            if not email or "@" not in email:
                st.error("Please enter a valid email address.")
            else:
                with st.spinner("Preparing to share..."):
                    pdf_buffer = create_pdf_storybook(
                        title=st.session_state.book_title,
                        author=st.session_state.author_name,
                        dedication=st.session_state.dedication,
                        story=st.session_state.story,
                        cover_image=st.session_state.cover_image,
                        comic_panels=st.session_state.comic_panels
                    )
                    share_via_email(email, 
                                    {"title": st.session_state.book_title, 
                                     "author": st.session_state.author_name}, 
                                    pdf_buffer, "PDF")
    
    with col2:
        if st.button("Share as EPUB", key="share_epub_btn"):
            if not email or "@" not in email:
                st.error("Please enter a valid email address.")
            else:
                with st.spinner("Preparing to share..."):
                    epub_buffer = create_epub_storybook(
                        title=st.session_state.book_title,
                        author=st.session_state.author_name,
                        dedication=st.session_state.dedication,
                        story=st.session_state.story,
                        cover_image=st.session_state.cover_image,
                        comic_panels=st.session_state.comic_panels
                    )
                    share_via_email(email, 
                                    {"title": st.session_state.book_title, 
                                     "author": st.session_state.author_name}, 
                                    epub_buffer, "EPUB")
    
    # Save to collection
    st.markdown("### Save to Your Collection")
    
    if st.button("Save to My Stories", key="save_to_collection_btn"):
        # Create a snapshot of the current book
        book_data = {
            "id": st.session_state.book_id,
            "title": st.session_state.book_title,
            "author": st.session_state.author_name,
            "theme": st.session_state.story_theme,
            "date_created": datetime.now().strftime("%Y-%m-%d"),
            "cover_image_base64": image_to_base64(st.session_state.cover_image) if st.session_state.cover_image else None
        }
        save_book_to_collection(book_data)
    
    # Create new story button
    if st.button("Create a New Story", key="create_new_story_btn"):
        # Reset necessary session state variables for a new story
        st.session_state.current_step = 1
        st.session_state.uploaded_images = []
        st.session_state.character_images = []
        st.session_state.story = ""
        st.session_state.comic_panels = []
        st.session_state.character_names = []
        st.session_state.book_id = datetime.now().strftime("%Y%m%d%H%M%S")
        st.session_state.cover_image = None
        st.session_state.book_title = ""
        st.session_state.author_name = ""
        st.session_state.dedication = ""
        st.experimental_rerun()

# Add footer
st.markdown("""
<div class="footer">
    Bedtime Story Maker - Your personal storybook creator<br>
    © 2025 Bedtime Magic Inc.
</div>
""", unsafe_allow_html=True)
