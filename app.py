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

# Audio processing
import speech_recognition as sr
from gtts import gTTS

# Data processing
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Streamlit extensions (install if available, skip if not)
try:
    from streamlit_chat import message
    STREAMLIT_CHAT_AVAILABLE = True
except ImportError:
    STREAMLIT_CHAT_AVAILABLE = False
    
try:
    from streamlit_extras.colored_header import colored_header
    from streamlit_extras.add_vertical_space import add_vertical_space
    STREAMLIT_EXTRAS_AVAILABLE = True
except ImportError:
    STREAMLIT_EXTRAS_AVAILABLE = False


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
    
    /* Professional Chat Bubbles */
    .user-message {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        margin-left: 20%;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        animation: slideInRight 0.3s ease;
        position: relative;
    }
    
    .ai-message {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        color: var(--text-primary);
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        margin-right: 20%;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        animation: slideInLeft 0.3s ease;
        border-left: 4px solid var(--primary);
        position: relative;
    }
    
    .model-badge {
        position: absolute;
        top: -8px;
        right: 10px;
        background: var(--accent);
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.7em;
        font-weight: 600;
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
    
    .status-thinking {
        width: 12px;
        height: 12px;
        background: var(--warning);
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        animation: pulse 1s infinite;
    }
    
    .status-error {
        width: 12px;
        height: 12px;
        background: var(--error);
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    /* Animations */
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
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
  /* header {visibility: hidden;} */
    
    /* Loading Animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
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

# Initialize clients
clients = {}
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    clients['gemini'] = genai
import httpx

if GROQ_API_KEY:
    try:
        # If you need proxies, configure here
        # proxies = {
        #     "http://": "http://127.0.0.1:8080",
        #     "https://": "http://127.0.0.1:8080",
        # }
        # http_client = httpx.Client(proxies=proxies, timeout=30)

        # ✅ Use plain client (no proxies) for smooth run
        http_client = httpx.Client(timeout=30)

        clients['groq'] = Groq(api_key=GROQ_API_KEY, http_client=http_client)
    except Exception as e:
        st.error(f"⚠️ Failed to initialize Groq client: {e}")

import httpx

if OPENAI_API_KEY:
    try:
        # If you need proxies, configure here
        # proxies = {
        #     "http://": "http://127.0.0.1:8080",
        #     "https://": "http://127.0.0.1:8080",
        # }
        # http_client = httpx.Client(proxies=proxies, timeout=30)

        # ✅ Use plain client (no proxies) for smooth run
        http_client = httpx.Client(timeout=30)

        clients['openai'] = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
    except Exception as e:
        st.error(f"⚠️ Failed to initialize OpenAI client: {e}")


# -----------------------------
# 🎤 Voice Input System
# -----------------------------
class VoiceManager:
    """Handle voice input and text-to-speech"""
    
    @staticmethod
    def speech_to_text():
        """Convert speech to text using speech_recognition"""
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("🎤 Listening... Speak now!")
                r.adjust_for_ambient_noise(source, duration=1)
                audio = r.listen(source, timeout=10, phrase_time_limit=10)
                
            text = r.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            return "⏱️ No speech detected. Please try again."
        except sr.UnknownValueError:
            return "❌ Could not understand audio. Please try again."
        except sr.RequestError as e:
            return f"❌ Error with speech recognition service: {e}"
        except Exception as e:
            return f"❌ Voice input error: {str(e)}"
    
    @staticmethod
    def text_to_speech(text, language='en'):
        """Convert text to speech and return audio file"""
        try:
            tts = gTTS(text=text, lang=language, slow=False)
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
    def generate_with_stability(prompt):
        """Placeholder for Stability AI integration"""
        return None, "Stability AI integration coming soon"
    
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
            "model": "gemini-2.5-flash",
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
            "capabilities": ["text" "vision", "multimodal"],
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
    def analyze_image(image, prompt, model_name="Gemini Pro"):
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
    
    @staticmethod
    def create_sentiment_chart(sentiment_history):
        """Create sentiment over time chart"""
        if not sentiment_history:
            return None
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=sentiment_history,
            mode='lines+markers',
            name='Sentiment Score',
            line=dict(color='#667eea', width=3)
        ))
        fig.update_layout(
            title="Conversation Sentiment Over Time",
            yaxis_title="Sentiment Score",
            xaxis_title="Message Number"
        )
        return fig

# -----------------------------
# 💾 Enhanced Session Management
# -----------------------------
def initialize_advanced_session():
    """Initialize all session state variables"""
    defaults = {
        "messages": [],
        "model_usage": {},
        "response_times": [],
        "sentiment_history": [],
        "favorite_responses": [],
        "conversation_id": hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8],
        "total_tokens_used": 0,
        "images_generated": 0,
        "voice_inputs": 0,
        "current_model": "Gemini Pro",
        "theme": "professional",
        "language": "en"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

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
            <span style="color: rgba(255,255,255,0.8);">All Systems Online</span>
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
            language = st.selectbox("🌍 Language", ["en", "es", "fr", "de", "it", "pt"])
            st.session_state.language = language
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎤 Voice Input"):
                with st.spinner("Listening..."):
                    voice_text = VoiceManager.speech_to_text()
                    if voice_text and not voice_text.startswith("❌") and not voice_text.startswith("⏱️"):
                        st.session_state.voice_input = voice_text
                        st.session_state.voice_inputs += 1
                        st.success(f"Voice captured: {voice_text[:50]}...")
        
        with col2:
            if st.button("💾 Export Chat"):
                export_conversation()
        
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.messages = []
            st.rerun()
        
        # Usage Statistics
        st.markdown("### 📊 Session Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.messages))
        with col2:
            st.metric("Models Used", len(st.session_state.model_usage))
        
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
        for i, message_data in enumerate(st.session_state.messages):
            role = message_data["role"]
            content = message_data["content"]
            model = message_data.get("model", "Unknown")
            timestamp = message_data.get("timestamp", "")
            
            if role == "user":
                st.markdown(f"""
                <div class="user-message">
                    {content}
                    <div style="font-size: 0.8em; opacity: 0.7; margin-top: 0.5rem;">
                        {timestamp}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ai-message">
                    <div class="model-badge">{model}</div>
                    {content}
                    <div style="font-size: 0.8em; opacity: 0.7; margin-top: 0.5rem;">
                        {timestamp}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Input section
        col1, col2, col3 = st.columns([6, 1, 1])
        
        with col1:
            # Check for voice input
            default_text = ""
            if "voice_input" in st.session_state:
                default_text = st.session_state.voice_input
                del st.session_state.voice_input
            
            user_input = st.text_area(
                "💭 Your message...",
                value=default_text,
                height=100,
                placeholder="Type your message or use voice input..."
            )
        
        with col2:
            send_button = st.button("Send 🚀", use_container_width=True)
        
        with col3:
            speak_button = st.button("🔊 TTS", use_container_width=True)
        
        if send_button and user_input.strip():
            # Add user message
            user_message = {
                "role": "user",
                "content": user_input.strip(),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "model": None
            }
            st.session_state.messages.append(user_message)
            
            # Generate AI response
            with st.spinner(f"🤔 {selected_model} is thinking..."):
                start_time = time.time()
                
                response = AdvancedAIManager.generate_text(
                    user_input.strip(),
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
            
            st.rerun()
        
        # Text-to-speech for last AI response
        if speak_button and st.session_state.messages:
            last_message = st.session_state.messages[-1]
            if last_message["role"] == "assistant":
                with st.spinner("🔊 Generating speech..."):
                    audio_file = VoiceManager.text_to_speech(
                        last_message["content"], 
                        st.session_state.language
                    )
                    if audio_file:
                        with open(audio_file, 'rb') as f:
                            st.audio(f.read(), format='audio/mp3')
                        os.unlink(audio_file)  # Clean up
        
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
                if image_prompt.strip():
                    with st.spinner("🎨 Creating your masterpiece..."):
                        # Enhance prompt if requested
                        final_prompt = image_prompt.strip()
                        if enhance_prompt:
                            final_prompt = ImageGenerator.enhance_prompt(final_prompt)
                            st.info(f"Enhanced prompt: {final_prompt}")
                        
                        # Generate image
                        image, error_msg = ImageGenerator.generate_with_openai(
                            final_prompt, image_size, image_quality
                        )
                        
                        if image:
                            st.success("🎉 Image generated successfully!")
                            st.image(image, caption=final_prompt, use_column_width=True)
                            
                            # Save to session
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
        
        # Recent generations
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
                
                # Image info
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
                    # Pre-defined prompts
                    prompts = {
                        "General Description": "Describe this image in detail, including objects, people, setting, and overall composition.",
                        "Object Detection": "List all the objects you can identify in this image with their locations.",
                        "Text Extraction (OCR)": "Extract and transcribe any text visible in this image.",
                        "Scene Analysis": "Analyze the scene: location, time of day, weather, mood, and atmosphere.",
                        "Color Palette": "Describe the color scheme and palette used in this image."
                    }
                    custom_question = prompts.get(analysis_type, "Describe this image.")
                
                # Vision model selection
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
            col1, col2 = st.columns(2)
            
            with col1:
                if st.session_state.model_usage:
                    fig = AdvancedAnalytics.create_usage_chart(st.session_state.model_usage)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if len(st.session_state.messages) > 2:
                    # Simple sentiment over time
                    sentiment_history = []
                    for i in range(0, len(st.session_state.messages), 2):
                        subset = st.session_state.messages[max(0, i-5):i+1]
                        if subset:
                            temp_analytics = AdvancedAnalytics.analyze_conversation(subset)
                            sentiment_history.append(temp_analytics.get('sentiment_score', 0))
                    
                    if sentiment_history:
                        fig = AdvancedAnalytics.create_sentiment_chart(sentiment_history)
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
        
        # Add current exchange to favorites
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
            if st.button("📥 Export Favorites"):
                favorites_json = json.dumps(st.session_state.favorite_responses, indent=2)
                st.download_button(
                    "📄 Download as JSON",
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
    b64 = base64.b64encode(json_str.encode()).decode()
    
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
        
        # Debug info for development
        if st.checkbox("🔍 Show Debug Info"):

            st.exception(e)


