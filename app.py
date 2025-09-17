"""
UniChat - Next-Gen Multi-Model AI Assistant
Created for: Next-Gen Multi-Model AI Chatbot Hackathon (GEA-6)
Author: TEJAS.M.SURVE
Date: 12/09/2025

This is an original implementation created from scratch during the hackathon.
Features: Multi-model chat, image generation, vision analysis, voice input, and more!
"""

import streamlit as st
import os
import time
import json
import base64
import hashlib
import tempfile
import io
from datetime import datetime
from collections import Counter
import re
import httpx

# Environment and API clients
from dotenv import load_dotenv
import google.generativeai as genai
from groq import Groq
from openai import OpenAI

# Image processing
from PIL import Image
import requests

# Audio processing - Made optional for deployment
SPEECH_AVAILABLE = False
PYAUDIO_AVAILABLE = False

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
    # Check for PyAudio separately
    try:
        import pyaudio
        PYAUDIO_AVAILABLE = True
    except ImportError:
        PYAUDIO_AVAILABLE = False
        # Continue with speech_recognition but without microphone
except ImportError:
    SPEECH_AVAILABLE = False

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Data processing
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# 🎨 Professional Theme System
# -----------------------------
def apply_professional_theme():
    """Apply professional, modern theme for competition"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary: #667eea;
        --primary-dark: #5a67d8;
        --secondary: #764ba2;
        --accent: #f093fb;
        --success: #48bb78;
        --warning: #ed8936;
        --error: #f56565;
        --dark: #1a202c;
        --dark-light: #2d3748;
        --light: #f7fafc;
        --text-primary: #2d3748;
        --text-secondary: #718096;
        --border: #e2e8f0;
    }
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    /* Feature Cards */
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border: 1px solid var(--border);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9em;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Professional Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Status Indicators */
    .status-online {
        width: 12px;
        height: 12px;
        background: var(--success);
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-warning {
        width: 12px;
        height: 12px;
        background: var(--warning);
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Loading Animation */
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid var(--border);
        border-top: 4px solid var(--primary);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Deployment status */
    .deployment-notice {
        background: linear-gradient(135deg, #ffd700, #ffed4e);
        color: #744210;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #d69e2e;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# 🔧 Enhanced Configuration
# -----------------------------
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients with better error handling
clients = {}

try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        clients['gemini'] = genai
except Exception as e:
    st.error(f"Failed to initialize Gemini: {e}")

try:
    if GROQ_API_KEY:
        http_client = httpx.Client(timeout=30)
        clients['groq'] = Groq(api_key=GROQ_API_KEY, http_client=http_client)
except Exception as e:
    st.error(f"Failed to initialize Groq: {e}")

try:
    if OPENAI_API_KEY:
        http_client = httpx.Client(timeout=30)
        clients['openai'] = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
except Exception as e:
    st.error(f"Failed to initialize OpenAI: {e}")

# -----------------------------
# 🎤 FIXED Voice Input System
# -----------------------------
class VoiceManager:
    """Handle voice input and text-to-speech"""
    
    @staticmethod
    def speech_to_text():
        """This method is now handled by file upload"""
        return "Use file upload for voice input on cloud deployment"
    
    @staticmethod
    def process_audio_file(audio_file_path):
        """Process uploaded audio file to text"""
        try:
            r = sr.Recognizer()
            with sr.AudioFile(audio_file_path) as source:
                audio = r.record(source)
            text = r.recognize_google(audio)
            return text
        except Exception as e:
            return f"❌ Audio processing error: {str(e)}"
    
    @staticmethod
    def text_to_speech(text):
        """Convert text to speech and return audio file"""
        try:
            if not TTS_AVAILABLE:
                return None
            tts = gTTS(text=text, lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                tts.save(fp.name)
                return fp.name
        except Exception as e:
            st.error(f"Text-to-speech error: {e}")
            return None
# -----------------------------
# 🖼️ Enhanced Image Generation
# -----------------------------
class ImageGenerator:
    """Handle image generation from multiple models"""
    
    @staticmethod
    def generate_with_openai(prompt, size="1024x1024", quality="standard"):
        """Generate image using DALL-E"""
        try:
            if 'openai' not in clients:
                return None, "OpenAI API not configured"
            
            response = clients['openai'].images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality=quality,
                n=1
            )
            
            image_url = response.data[0].url
            image_response = requests.get(image_url)
            image = Image.open(io.BytesIO(image_response.content))
            
            return image, "Success"
        except Exception as e:
            return None, f"OpenAI image generation error: {str(e)}"
    
    @staticmethod
    def enhance_prompt(prompt):
        """Enhance image generation prompts"""
        enhancements = [
            "highly detailed", "professional photography", "4K quality",
            "cinematic lighting", "vibrant colors", "sharp focus"
        ]
        
        if any(word in prompt.lower() for word in ["photo", "realistic", "portrait"]):
            return f"{prompt}, {', '.join(enhancements[:3])}"
        elif any(word in prompt.lower() for word in ["art", "painting", "drawing"]):
            return f"{prompt}, artistic masterpiece, trending on artstation, {enhancements[3]}"
        else:
            return f"{prompt}, {enhancements[0]}, {enhancements[-1]}"

# -----------------------------
# 🧠 Advanced AI Model Manager
# -----------------------------
class AdvancedAIManager:
    """Enhanced AI model management with better error handling"""
    
    model_info = {
        "Gemini": {
            "provider": "google", 
            "model": "gemini-1.5-flash",
            "capabilities": ["text", "vision", "multimodal"],
            "max_tokens": 8192
        },
        "GPT-4o": {
            "provider": "openai",
            "model": "gpt-4o",
            "capabilities": ["text", "vision", "multimodal"],
            "max_tokens": 8192
        },
        "GPT-3.5 Turbo": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "capabilities": ["text"],
            "max_tokens": 4096
        },
        "Groq": {
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
            "capabilities": ["text"],
            "max_tokens": 4096
        }
    }
    
    @staticmethod
    def get_available_models():
        """Get list of available models based on API keys"""
        available = []
        for model_name, info in AdvancedAIManager.model_info.items():
            provider = info["provider"]
            if provider == "google" and GEMINI_API_KEY:
                available.append(model_name)
            elif provider == "openai" and OPENAI_API_KEY:
                available.append(model_name)
            elif provider == "groq" and GROQ_API_KEY:
                available.append(model_name)
        return available
    
    @staticmethod
    def generate_text(prompt, model_name, temperature=0.7, max_tokens=None):
        """Generate text response from specified model"""
        try:
            model_info = AdvancedAIManager.model_info.get(model_name)
            if not model_info:
                return f"Model {model_name} not found"
            
            provider = model_info["provider"]
            model_id = model_info["model"]
            
            if provider == "google" and 'gemini' in clients:
                model = genai.GenerativeModel(model_id)
                generation_config = genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens or model_info["max_tokens"]
                )
                response = model.generate_content(prompt, generation_config=generation_config)
                return response.text
                
            elif provider == "openai" and 'openai' in clients:
                response = clients['openai'].chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens or model_info["max_tokens"]
                )
                return response.choices[0].message.content
                
            elif provider == "groq" and 'groq' in clients:
                response = clients['groq'].chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens or model_info["max_tokens"]
                )
                return response.choices[0].message.content
                
            else:
                return f"{provider} client not available. Check API key."
                
        except Exception as e:
            return f"Error with {model_name}: {str(e)}"
    
    @staticmethod
    def analyze_image(image, prompt, model_name="Gemini"):
        """Analyze image with vision-capable models"""
        try:
            model_info = AdvancedAIManager.model_info.get(model_name)
            if not model_info or "vision" not in model_info["capabilities"]:
                return "Selected model doesn't support image analysis"
            
            if model_info["provider"] == "google" and 'gemini' in clients:
                model = genai.GenerativeModel(model_info["model"])
                response = model.generate_content([prompt, image])
                return response.text
            else:
                return "Image analysis only available with Gemini models"
                
        except Exception as e:
            return f"Image analysis error: {str(e)}"

# -----------------------------
# 📊 Advanced Analytics System
# -----------------------------
class AdvancedAnalytics:
    """Comprehensive conversation and usage analytics"""
    
    @staticmethod
    def analyze_conversation(messages):
        """Analyze conversation patterns and metrics"""
        if not messages:
            return {}
        
        # Basic metrics
        total_messages = len(messages)
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        ai_messages = [msg for msg in messages if msg["role"] == "assistant"]
        
        # Text analysis
        all_text = " ".join([msg["content"] for msg in messages])
        words = re.findall(r'\w+', all_text.lower())
        
        # Sentiment analysis (simple)
        positive_words = ["good", "great", "excellent", "amazing", "helpful", "thanks", "love", "perfect", "wonderful"]
        negative_words = ["bad", "wrong", "error", "problem", "hate", "terrible", "awful", "worse"]
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total_sentiment_words = positive_count + negative_count
        
        sentiment_score = 0
        if total_sentiment_words > 0:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words * 100
        
        # Topic extraction
        common_words = [word for word, count in Counter(words).most_common(10) if len(word) > 3]
        
        # Complexity analysis
        avg_message_length = sum(len(msg["content"]) for msg in messages) / len(messages)
        complexity_score = min(100, (avg_message_length / 100) * 100)
        
        return {
            "total_messages": total_messages,
            "user_messages": len(user_messages),
            "ai_messages": len(ai_messages),
            "avg_message_length": avg_message_length,
            "sentiment_score": sentiment_score,
            "complexity_score": complexity_score,
            "common_topics": common_words[:5],
            "total_words": len(words)
        }
    
    @staticmethod
    def create_usage_chart(model_usage):
        """Create usage visualization"""
        if not model_usage:
            return None
        
        fig = px.pie(
            values=list(model_usage.values()),
            names=list(model_usage.keys()),
            title="Model Usage Distribution"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig

# -----------------------------
# 💾 Session Management
# -----------------------------
def initialize_advanced_session():
    """Initialize all session state variables"""
    defaults = {
        "messages": [],
        "model_usage": {},
        "response_times": [],
        "favorite_responses": [],
        "conversation_id": hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8],
        "total_tokens_used": 0,
        "images_generated": 0,
        "voice_inputs": 0,
        "current_model": "Gemini",
        "theme": "professional",
        "input_key": 0,
        "captured_voice_text": "",
        "processing_voice": False,
        "deployment_warnings_shown": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# -----------------------------
# Message Display System
# -----------------------------
def display_chat_messages():
    """Display chat messages using native Streamlit components"""
    for i, message_data in enumerate(st.session_state.messages):
        role = message_data["role"]
        content = message_data["content"]
        model = message_data.get("model", "Unknown")
        timestamp = message_data.get("timestamp", "")
        
        if role == "user":
            col1, col2, col3 = st.columns([1, 4, 0.1])
            with col2:
                st.info(f"**You** ({timestamp})\n\n{content}")
        else:
            col1, col2, col3 = st.columns([0.1, 4, 1])
            with col2:
                st.success(f"**{model}** ({timestamp})\n\n{content}")

# -----------------------------
# Cloud-Compatible Voice Input
# -----------------------------
def handle_voice_input():
    """Handle voice input with cloud-compatible file upload"""
    st.markdown("### 🎤 Voice Input")
    
    # Two options for voice input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(" Upload Audio File**")
        uploaded_audio = st.file_uploader(
         
        )
        
        if uploaded_audio is not None:
            st.audio(uploaded_audio, format='audio/wav')
            
            if st.button("🔍 Convert Audio to Text"):
                with st.spinner("Processing audio..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(uploaded_audio.getvalue())
                            temp_path = tmp_file.name
                        
                        # Use speech recognition on uploaded file
                        r = sr.Recognizer()
                        with sr.AudioFile(temp_path) as source:
                            audio = r.record(source)
                        
                        text = r.recognize_google(audio)
                        
                        if text:
                            st.session_state.captured_voice_text = text
                            st.session_state.voice_inputs += 1
                            st.success(f"✅ Audio converted: {text}")
                            st.rerun()
                        
                        # Clean up temp file
                        os.unlink(temp_path)
                        
                    except Exception as e:
                        st.error(f"❌ Audio processing error: {str(e)}")
                        st.info("💡 Try uploading a clear WAV or MP3 file")
    

        
# -----------------------------
# Deployment Status Display
# -----------------------------

# -----------------------------
# 🎯 Main Application
# -----------------------------
def main():
    st.set_page_config(
        page_title="UniChat - Next-Gen AI Assistant",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_professional_theme()
    initialize_advanced_session()
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: white; font-size: 3.5em; margin: 0; font-weight: 700;">
            🚀 UniChat
        </h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.3em; margin: 0.5rem 0;">
            Next-Generation Multi-Model AI Assistant
        </p>
        <div style="margin: 1rem 0;">
            <span class="status-online"></span>
            <span style="color: rgba(255,255,255,0.8);">Cloud Deployment Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Center")
        
        # Model Selection
        available_models = AdvancedAIManager.get_available_models()
        if not available_models:
            st.error("⚠️ No AI models available. Please configure API keys.")
            st.info("💡 Add API keys to environment variables or .env file")
            return
        
        selected_model = st.selectbox(
            "🤖 AI Model",
            available_models,
            index=available_models.index(st.session_state.current_model) if st.session_state.current_model in available_models else 0
        )
        st.session_state.current_model = selected_model
        
        # Model info
        model_info = AdvancedAIManager.model_info.get(selected_model, {})
        st.info(f"**Provider:** {model_info.get('provider', 'Unknown').title()}")
        
        # Advanced Settings
        with st.expander("⚙️ Advanced Settings"):
            temperature = st.slider("🌡️ Creativity", 0.0, 1.0, 0.7, 0.1)
            max_tokens = st.number_input("📝 Max Response Length", 100, 4000, 1000)
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            handle_voice_input()
        
        with col2:
            if st.button("💾 Export Chat"):
                export_conversation()
        
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.messages = []
            st.session_state.captured_voice_text = ""
            st.session_state.input_key += 1
            st.rerun()
        
       
        # Usage Statistics
        st.markdown("### 📊 Session Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.messages))
        with col2:
            st.metric("Voice Inputs", st.session_state.voice_inputs)
        
        if st.session_state.response_times:
            avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
    
    # Main Interface Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Chat", "🖼️ Image Generation", "🔍 Vision Analysis", 
        "📊 Analytics", "⭐ Favorites"
    ])
    
    with tab1:
        # Chat Interface
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        # Display conversation
        display_chat_messages()
        
        # Input section
        st.markdown("---")
        
        col1, col2, col3 = st.columns([6, 1, 1])
        
        with col1:
            current_input = ""
            if st.session_state.captured_voice_text:
                current_input = st.session_state.captured_voice_text
            
            user_input = st.text_area(
                "💭 Your message...",
                value=current_input,
                height=100,
                placeholder="Type your message or use voice input...",
                key=f"main_chat_input_{st.session_state.input_key}"
            )
        
        with col2:
            send_button = st.button("Send 🚀", use_container_width=True)
        
        with col3:
            speak_button = st.button("🔊 TTS", use_container_width=True)
        
        # Send message logic
        if send_button and user_input and user_input.strip():
            message_content = user_input.strip()
            
            # Add user message
            user_message = {
                "role": "user",
                "content": message_content,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "model": None
            }
            st.session_state.messages.append(user_message)
            
            # Generate AI response
            with st.spinner(f"🤔 {selected_model} is thinking..."):
                start_time = time.time()
                
                response = AdvancedAIManager.generate_text(
                    message_content,
                    selected_model,
                    temperature,
                    max_tokens
                )
                
                response_time = time.time() - start_time
                st.session_state.response_times.append(response_time)
                
                # Add AI response
                ai_message = {
                    "role": "assistant",
                    "content": response,
                    "model": selected_model,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "response_time": response_time
                }
                st.session_state.messages.append(ai_message)
                
                # Update usage stats
                if selected_model in st.session_state.model_usage:
                    st.session_state.model_usage[selected_model] += 1
                else:
                    st.session_state.model_usage[selected_model] = 1
            
            # Clear input and voice text
            st.session_state.captured_voice_text = ""
            st.session_state.input_key += 1
            st.rerun()
        
        # Text-to-speech for last AI response
        if speak_button and st.session_state.messages:
            last_message = st.session_state.messages[-1]
            if last_message["role"] == "assistant":
                with st.spinner("🔊 Generating speech..."):
                    audio_file = VoiceManager.text_to_speech(last_message["content"])
                    if audio_file:
                        with open(audio_file, 'rb') as f:
                            st.audio(f.read(), format='audio/mp3')
                        os.unlink(audio_file)
                    else:
                        st.warning("Text-to-speech not available")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        # Image Generation Interface
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown("## 🎨 AI Image Generation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            image_prompt = st.text_area(
                "🖼️ Describe the image you want to generate",
                height=150,
                placeholder="A futuristic cityscape at sunset with flying cars..."
            )
            
            enhance_prompt = st.checkbox("✨ Auto-enhance prompt", value=True)
            
        with col2:
            image_size = st.selectbox("📐 Image Size", ["1024x1024", "1024x1792", "1792x1024"])
            image_quality = st.selectbox("💎 Quality", ["standard", "hd"])
            
            if st.button("🎨 Generate Image", use_container_width=True):
                if not OPENAI_API_KEY:
                    st.error("🔑 OpenAI API key required for image generation")
                    st.info("💡 Add OPENAI_API_KEY to your environment variables")
                elif image_prompt.strip():
                    with st.spinner("🎨 Creating your masterpiece..."):
                        final_prompt = image_prompt.strip()
                        if enhance_prompt:
                            final_prompt = ImageGenerator.enhance_prompt(final_prompt)
                            st.info(f"Enhanced prompt: {final_prompt}")
                        
                        image, error_msg = ImageGenerator.generate_with_openai(
                            final_prompt, image_size, image_quality
                        )
                        
                        if image:
                            st.success("🎉 Image generated successfully!")
                            st.image(image, caption=final_prompt, use_column_width=True)
                            
                            st.session_state.images_generated += 1
                            
                            # Download button
                            img_buffer = io.BytesIO()
                            image.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            
                            st.download_button(
                                "📥 Download Image",
                                data=img_buffer.getvalue(),
                                file_name=f"unichat_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png"
                            )
                        else:
                            st.error(f"❌ {error_msg}")
                else:
                    st.warning("Please enter an image description")
        
        if st.session_state.images_generated > 0:
            st.markdown("### 🖼️ Generation Stats")
            st.metric("Images Created", st.session_state.images_generated)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        # Vision Analysis Interface
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown("## 👁️ AI Vision Analysis")
        
        uploaded_file = st.file_uploader(
            "📤 Upload an image for AI analysis",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Supported formats: PNG, JPG, JPEG, GIF, BMP"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                st.info(f"""
                **File:** {uploaded_file.name}
                **Size:** {image.size[0]} x {image.size[1]} pixels
                **Format:** {image.format}
                """)
            
            with col2:
                st.markdown("### 🔍 Analysis Options")
                
                analysis_type = st.selectbox(
                    "Analysis Type",
                    [
                        "General Description",
                        "Object Detection", 
                        "Text Extraction (OCR)",
                        "Scene Analysis",
                        "Color Palette",
                        "Custom Question"
                    ]
                )
                
                if analysis_type == "Custom Question":
                    custom_question = st.text_area(
                        "Ask about the image:",
                        placeholder="What emotions does this image convey?"
                    )
                else:
                    prompts = {
                        "General Description": "Describe this image in detail, including objects, people, setting, and overall composition.",
                        "Object Detection": "List all the objects you can identify in this image with their locations.",
                        "Text Extraction (OCR)": "Extract and transcribe any text visible in this image.",
                        "Scene Analysis": "Analyze the scene: location, time of day, weather, mood, and atmosphere.",
                        "Color Palette": "Describe the color scheme and palette used in this image."
                    }
                    custom_question = prompts.get(analysis_type, "Describe this image.")
                
                vision_models = [model for model in AdvancedAIManager.get_available_models() 
                               if "Gemini" in model]
                
                if vision_models:
                    vision_model = st.selectbox("🤖 Vision Model", vision_models)
                    
                    if st.button("🔍 Analyze Image", use_container_width=True):
                        with st.spinner(f"👁️ {vision_model} is analyzing..."):
                            analysis_result = AdvancedAIManager.analyze_image(
                                image, custom_question, vision_model
                            )
                            
                            st.success("✅ Analysis complete!")
                            st.markdown("### 📋 Analysis Results")
                            st.write(analysis_result)
                            
                            # Save to chat history
                            st.session_state.messages.extend([
                                {
                                    "role": "user",
                                    "content": f"[Image Analysis] {custom_question}",
                                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                                    "model": None
                                },
                                {
                                    "role": "assistant", 
                                    "content": analysis_result,
                                    "model": vision_model,
                                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                                    "response_time": 0
                                }
                            ])
                else:
                    st.warning("🔑 Vision analysis requires Gemini API key")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        # Analytics Dashboard
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown("## 📊 Advanced Analytics")
        
        if st.session_state.messages:
            analytics = AdvancedAnalytics.analyze_conversation(st.session_state.messages)
            
            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{analytics['total_messages']}</div>
                    <div class="metric-label">Total Messages</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{analytics['sentiment_score']:.1f}</div>
                    <div class="metric-label">Sentiment Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{analytics['complexity_score']:.0f}</div>
                    <div class="metric-label">Complexity Level</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{analytics['total_words']}</div>
                    <div class="metric-label">Total Words</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Charts
            if st.session_state.model_usage:
                fig = AdvancedAnalytics.create_usage_chart(st.session_state.model_usage)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Analysis
            st.markdown("### 🔍 Conversation Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📝 Message Distribution**")
                st.write(f"- User messages: {analytics['user_messages']}")
                st.write(f"- AI responses: {analytics['ai_messages']}")
                st.write(f"- Average length: {analytics['avg_message_length']:.0f} chars")
                
                if st.session_state.response_times:
                    avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
                    fastest = min(st.session_state.response_times)
                    slowest = max(st.session_state.response_times)
                    
                    st.markdown("**⚡ Response Times**")
                    st.write(f"- Average: {avg_time:.2f}s")
                    st.write(f"- Fastest: {fastest:.2f}s") 
                    st.write(f"- Slowest: {slowest:.2f}s")
            
            with col2:
                st.markdown("**🏷️ Common Topics**")
                for topic in analytics['common_topics']:
                    st.write(f"- {topic}")
                
                st.markdown("**📊 Session Statistics**")
                st.write(f"- Images generated: {st.session_state.images_generated}")
                st.write(f"- Voice inputs: {st.session_state.voice_inputs}")
                st.write(f"- Conversation ID: {st.session_state.conversation_id}")
        
        else:
            st.info("📈 Start chatting to see analytics!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        # Favorites Interface
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown("## ⭐ Favorite Responses")
        
        if st.session_state.messages and len(st.session_state.messages) >= 2:
            if st.button("⭐ Save Last Exchange", use_container_width=True):
                last_user = None
                last_ai = None
                
                # Find last user-AI pair
                for i in range(len(st.session_state.messages) - 1, -1, -1):
                    msg = st.session_state.messages[i]
                    if msg["role"] == "assistant" and not last_ai:
                        last_ai = msg
                    elif msg["role"] == "user" and not last_user and last_ai:
                        last_user = msg
                        break
                
                if last_user and last_ai:
                    favorite = {
                        "id": len(st.session_state.favorite_responses),
                        "user_message": last_user["content"],
                        "ai_response": last_ai["content"],
                        "model": last_ai.get("model", "Unknown"),
                        "timestamp": datetime.now().isoformat(),
                        "rating": 5
                    }
                    st.session_state.favorite_responses.append(favorite)
                    st.success("💫 Added to favorites!")
                    st.rerun()
        
        # Display favorites
        if st.session_state.favorite_responses:
            for i, fav in enumerate(st.session_state.favorite_responses):
                with st.expander(f"⭐ Favorite #{i+1} - {fav['model']} ({fav['timestamp'][:10]})"):
                    st.markdown(f"**👤 You:** {fav['user_message']}")
                    st.markdown(f"**🤖 {fav['model']}:** {fav['ai_response']}")
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        rating = st.slider(f"Rating", 1, 5, fav.get('rating', 5), key=f"rating_{i}")
                        fav['rating'] = rating
                    
                    with col2:
                        if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                            st.session_state.favorite_responses.pop(i)
                            st.rerun()
        else:
            st.info("🌟 Your favorite exchanges will appear here!")
        
        # Export favorites
        if st.session_state.favorite_responses:
            favorites_json = json.dumps(st.session_state.favorite_responses, indent=2)
            st.download_button(
                "📄 Download Favorites as JSON",
                data=favorites_json,
                file_name=f"unichat_favorites_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

def export_conversation():
    """Export conversation with analytics"""
    if not st.session_state.messages:
        st.warning("No conversation to export")
        return
    
    analytics = AdvancedAnalytics.analyze_conversation(st.session_state.messages)
    
    export_data = {
        "metadata": {
            "conversation_id": st.session_state.conversation_id,
            "export_timestamp": datetime.now().isoformat(),
            "total_messages": len(st.session_state.messages),
            "models_used": list(st.session_state.model_usage.keys()),
            "analytics": analytics
        },
        "messages": st.session_state.messages,
        "statistics": {
            "model_usage": st.session_state.model_usage,
            "response_times": st.session_state.response_times,
            "images_generated": st.session_state.images_generated,
            "voice_inputs": st.session_state.voice_inputs
        },
        "favorites": st.session_state.favorite_responses
    }
    
    json_str = json.dumps(export_data, indent=2, default=str)
    
    st.download_button(
        "📥 Download Complete Export",
        data=json_str,
        file_name=f"unichat_export_{st.session_state.conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    
    st.success("📊 Export ready for download!")

# -----------------------------
# 🎯 Entry Point
# -----------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Application Error: {str(e)}")
        st.info("Please refresh the page or contact support if the issue persists.")
        
        # Show deployment help
        st.markdown("""
        ### 🛠️ Troubleshooting Tips:
        1. **API Keys**: Make sure your API keys are properly configured
        2. **Voice Features**: Work best in local environment
        3. **Image Generation**: Requires OpenAI API key
        4. **Refresh**: Try refreshing the page if issues persist
        """)
        
        if st.checkbox("🔍 Show Debug Info"):
            st.exception(e)



