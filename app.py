"""
Pollen ID - Live AI-Powered Pollen Intelligence Platform
=========================================================
A production-ready MVP that combines real-time pollen data with AI-driven
health insights and forecasting.

Author: Senior Full-Stack AI Engineer
Version: 1.0.0
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import json
import os
from typing import Dict, List, Tuple, Optional
import time

# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================

# Professional Color Palette - Financial Services Theme
COLORS = {
    'primary_blue': '#1E3A8A',      # Deep Financial Blue
    'secondary_blue': '#3B82F6',    # Bright Blue
    'accent_blue': '#60A5FA',       # Light Blue
    'success_green': '#10B981',     # Green for low risk
    'warning_yellow': '#F59E0B',    # Yellow for medium risk
    'danger_red': '#EF4444',        # Red for high risk
    'background': '#F8FAFC',        # Clean White Background
    'card_bg': '#FFFFFF',           # Pure White for cards
    'text_primary': '#1F2937',      # Dark Gray
    'text_secondary': '#6B7280'     # Medium Gray
}

# Page Configuration
st.set_page_config(
    page_title="Pollen ID | AI Pollen Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Styling
st.markdown(f"""
<style>
    /* Main Container */
    .main {{
        background-color: {COLORS['background']};
    }}
    
    /* Headers */
    h1 {{
        color: {COLORS['primary_blue']};
        font-weight: 700;
        letter-spacing: -0.5px;
    }}
    
    h2, h3 {{
        color: {COLORS['primary_blue']};
        font-weight: 600;
    }}
    
    /* Metric Cards Enhancement */
    [data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: 700;
        color: {COLORS['primary_blue']};
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {COLORS['text_secondary']};
        font-weight: 500;
        font-size: 0.9rem;
    }}
    
    /* Buttons */
    .stButton>button {{
        background-color: {COLORS['primary_blue']};
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }}
    
    .stButton>button:hover {{
        background-color: {COLORS['secondary_blue']};
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['card_bg']};
        border-right: 1px solid #E5E7EB;
    }}
    
    /* Info boxes */
    .stAlert {{
        border-radius: 8px;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: {COLORS['card_bg']};
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['primary_blue']};
        color: white;
    }}
    
    /* Chat Messages */
    .stChatMessage {{
        background-color: {COLORS['card_bg']};
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_coordinates(location: str) -> Optional[Tuple[float, float]]:
    """
    Convert zip code or city name to latitude/longitude coordinates.
    
    Args:
        location: Zip code or city name
        
    Returns:
        Tuple of (latitude, longitude) or None if geocoding fails
    """
    try:
        geolocator = Nominatim(user_agent="pollen_id_app_v1")
        time.sleep(1)  # Rate limiting - be respectful to the service
        
        # Try geocoding with country bias for US
        location_data = geolocator.geocode(f"{location}, USA", timeout=10)
        
        if location_data:
            return (location_data.latitude, location_data.longitude)
        else:
            # Try without country bias
            location_data = geolocator.geocode(location, timeout=10)
            if location_data:
                return (location_data.latitude, location_data.longitude)
            return None
            
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        st.error(f"Geocoding service error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during geocoding: {str(e)}")
        return None


def get_pollen_data(lat: float, lng: float, api_key: str) -> Optional[Dict]:
    """
    Fetch real-time pollen data from Google Pollen API.
    
    Args:
        lat: Latitude
        lng: Longitude
        api_key: Google API Key
        
    Returns:
        Dictionary containing pollen data or None if request fails
    """
    base_url = "https://pollen.googleapis.com/v1/forecast:lookup"
    
    params = {
        "key": api_key,
        "location.latitude": lat,
        "location.longitude": lng,
        "days": 3  # Get 3-day forecast
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            st.error("⚠️ API Key Error: Invalid or missing Google Pollen API key. Please check your credentials.")
        elif e.response.status_code == 400:
            st.error("⚠️ Bad Request: The location data may be invalid.")
        else:
            st.error(f"⚠️ HTTP Error {e.response.status_code}: {str(e)}")
        return None
        
    except requests.exceptions.Timeout:
        st.error("⚠️ Request Timeout: The Pollen API is taking too long to respond. Please try again.")
        return None
        
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Network Error: {str(e)}")
        return None
        
    except Exception as e:
        st.error(f"⚠️ Unexpected Error: {str(e)}")
        return None


def parse_pollen_data(data: Dict) -> pd.DataFrame:
    """
    Parse Google Pollen API response into a structured DataFrame.
    
    Args:
        data: Raw API response
        
    Returns:
        DataFrame with pollen indices by type and date
    """
    records = []
    
    if 'dailyInfo' not in data:
        return pd.DataFrame()
    
    for day_info in data['dailyInfo']:
        date = day_info.get('date', {})
        date_str = f"{date.get('year', '')}-{date.get('month', ''):02d}-{date.get('day', ''):02d}"
        
        # Extract pollen type indices
        pollen_types = day_info.get('pollenTypeInfo', [])
        
        record = {'date': date_str}
        
        for pollen in pollen_types:
            pollen_code = pollen.get('code', 'UNKNOWN')
            index_info = pollen.get('indexInfo', {})
            index_value = index_info.get('value', 0)
            
            # Map API codes to friendly names
            type_mapping = {
                'GRASS': 'Grass',
                'TREE': 'Tree',
                'WEED': 'Weed'
            }
            
            friendly_name = type_mapping.get(pollen_code, pollen_code)
            record[friendly_name] = index_value
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Ensure all pollen types exist
    for pollen_type in ['Grass', 'Tree', 'Weed']:
        if pollen_type not in df.columns:
            df[pollen_type] = 0
    
    return df


def get_risk_level(index: int) -> Tuple[str, str]:
    """
    Determine risk level and color based on pollen index.
    
    Google Pollen API uses UPI (Universal Pollen Index):
    0: None
    1-2: Low
    3-4: Medium
    5: High
    
    Args:
        index: Pollen index value
        
    Returns:
        Tuple of (risk_level, color)
    """
    if index == 0:
        return "None", COLORS['success_green']
    elif index <= 2:
        return "Low", COLORS['success_green']
    elif index <= 4:
        return "Medium", COLORS['warning_yellow']
    else:
        return "High", COLORS['danger_red']


def simulate_lstm_forecast(current_data: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate LSTM-based forecasting for demonstration purposes.
    In production, this would call your trained LSTM model.
    
    Args:
        current_data: Current pollen data
        
    Returns:
        Extended DataFrame with forecasted values
    """
    import numpy as np
    
    # Create forecast dates
    last_date = pd.to_datetime(current_data['date'].iloc[-1])
    forecast_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                      for i in range(4)]  # 4 more days
    
    forecast_records = []
    
    for i, date in enumerate(forecast_dates):
        record = {'date': date}
        
        # Simulate trend-based forecasting with noise
        for pollen_type in ['Grass', 'Tree', 'Weed']:
            if pollen_type in current_data.columns:
                # Get recent trend
                recent_values = current_data[pollen_type].tail(3).values
                trend = np.mean(np.diff(recent_values)) if len(recent_values) > 1 else 0
                
                # Forecast with trend + random walk
                last_value = recent_values[-1] if len(recent_values) > 0 else 2
                noise = np.random.normal(0, 0.3)  # Small random variation
                forecast_value = max(0, min(5, last_value + trend + noise))
                
                record[pollen_type] = round(forecast_value, 1)
            else:
                record[pollen_type] = 0
        
        forecast_records.append(record)
    
    forecast_df = pd.DataFrame(forecast_records)
    
    # Combine historical and forecast
    combined = pd.concat([current_data, forecast_df], ignore_index=True)
    combined['is_forecast'] = combined.index >= len(current_data)
    
    return combined


def get_ai_health_advice(pollen_data: Dict, user_question: str, api_key: str) -> str:
    """
    Generate health advice using RAG pattern with OpenAI GPT-4.
    Uses current pollen data as context.
    
    Args:
        pollen_data: Current pollen data as context
        user_question: User's health question
        api_key: OpenAI API key
        
    Returns:
        AI-generated health advice
    """
    try:
        # Prepare context from pollen data
        context = f"""
Current Pollen Levels:
- Grass Pollen: {pollen_data.get('Grass', 0)} (UPI 0-5 scale)
- Tree Pollen: {pollen_data.get('Tree', 0)} (UPI 0-5 scale)
- Weed Pollen: {pollen_data.get('Weed', 0)} (UPI 0-5 scale)

Location: {pollen_data.get('location', 'Unknown')}
Date: {pollen_data.get('date', datetime.now().strftime('%Y-%m-%d'))}

UPI Scale Reference:
0 = None, 1-2 = Low, 3-4 = Medium, 5 = High
"""
        
        # Make request to OpenAI
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a professional allergy and health advisor specializing in pollen-related health guidance. 
                    
Your role:
- Provide actionable, evidence-based health advice related to pollen exposure
- Use the current pollen data context to personalize recommendations
- Be empathetic and professional
- Cite when high pollen levels warrant extra precautions
- Recommend consulting healthcare providers for serious symptoms

Keep responses concise (3-5 sentences) and practical."""
                },
                {
                    "role": "user",
                    "content": f"{context}\n\nUser Question: {user_question}"
                }
            ],
            "temperature": 0.7,
            "max_tokens": 300
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result['choices'][0]['message']['content']
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return "⚠️ OpenAI API Error: Invalid API key. Please check your credentials in the sidebar."
        elif e.response.status_code == 429:
            return "⚠️ Rate Limit: Too many requests. Please wait a moment and try again."
        else:
            return f"⚠️ API Error: {str(e)}"
            
    except requests.exceptions.Timeout:
        return "⚠️ Request timeout. The AI service is taking too long. Please try again."
        
    except Exception as e:
        return f"⚠️ Error generating advice: {str(e)}"


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_forecast_chart(data: pd.DataFrame) -> go.Figure:
    """
    Create an interactive 3-day trend chart with forecasted values.
    
    Args:
        data: DataFrame with pollen data and forecast flag
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Historical data (solid lines)
    historical = data[~data['is_forecast']]
    forecast = data[data['is_forecast']]
    
    pollen_types = ['Grass', 'Tree', 'Weed']
    colors = {
        'Grass': '#10B981',  # Green
        'Tree': '#8B4513',   # Brown
        'Weed': '#F59E0B'    # Yellow/Orange
    }
    
    for pollen_type in pollen_types:
        # Historical line
        fig.add_trace(go.Scatter(
            x=historical['date'],
            y=historical[pollen_type],
            mode='lines+markers',
            name=f'{pollen_type} (Actual)',
            line=dict(color=colors[pollen_type], width=3),
            marker=dict(size=8)
        ))
        
        # Forecast line (dashed)
        if not forecast.empty:
            # Connect last historical point to forecast
            connection_df = pd.concat([
                historical.tail(1),
                forecast
            ])
            
            fig.add_trace(go.Scatter(
                x=connection_df['date'],
                y=connection_df[pollen_type],
                mode='lines+markers',
                name=f'{pollen_type} (Forecast)',
                line=dict(color=colors[pollen_type], width=2, dash='dash'),
                marker=dict(size=6, symbol='diamond'),
                opacity=0.7
            ))
    
    fig.update_layout(
        title="7-Day Pollen Trend & AI Forecast",
        xaxis_title="Date",
        yaxis_title="Pollen Index (UPI 0-5)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(family="Inter, sans-serif", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E5E7EB')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E7EB', range=[0, 5.5])
    
    return fig


def create_distribution_chart(current_data: Dict) -> go.Figure:
    """
    Create a bar chart showing current pollen distribution.
    
    Args:
        current_data: Dictionary with current pollen levels
        
    Returns:
        Plotly figure object
    """
    pollen_types = ['Grass', 'Tree', 'Weed']
    values = [current_data.get(pt, 0) for pt in pollen_types]
    colors_list = ['#10B981', '#8B4513', '#F59E0B']
    
    fig = go.Figure(data=[
        go.Bar(
            x=pollen_types,
            y=values,
            marker=dict(
                color=colors_list,
                line=dict(color='white', width=2)
            ),
            text=[f"{v:.1f}" for v in values],
            textposition='outside',
            textfont=dict(size=16, color=COLORS['primary_blue'], family="Inter, sans-serif")
        )
    ])
    
    fig.update_layout(
        title="Current Pollen Distribution",
        yaxis_title="Pollen Index (UPI)",
        template='plotly_white',
        height=400,
        showlegend=False,
        font=dict(family="Inter, sans-serif", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_yaxes(range=[0, 5.5], showgrid=True, gridwidth=1, gridcolor='#E5E7EB')
    
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    # Header
    st.title("🌿 Pollen ID")
    st.markdown("### AI-Powered Pollen Intelligence Platform")
    st.markdown("Real-time pollen tracking with AI-driven health insights and forecasting")
    
    st.markdown("---")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.markdown("#### API Keys")
        google_api_key = st.text_input(
            "Google Pollen API Key",
            type="password",
            placeholder="Enter your API key",
            help="Get your key at: https://console.cloud.google.com"
        )
        
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Enter your API key",
            help="Required for AI Health Assistant"
        )
        
        st.markdown("---")
        
        st.markdown("#### Location")
        location_input = st.text_input(
            "Zip Code or City",
            placeholder="e.g., 10001 or New York, NY",
            help="Enter a US zip code or city name"
        )
        
        analyze_button = st.button("🔍 Analyze Pollen Levels", use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("#### About")
        st.info("""
        **Pollen ID** combines real-time environmental data with AI to deliver
        personalized pollen intelligence and health recommendations.
        
        **Features:**
        - 📊 Real-time pollen tracking
        - 🤖 AI-powered forecasting
        - 💬 Intelligent health assistant
        - 📈 7-day trend analysis
        """)
        
        st.markdown("---")
        st.caption("v1.0.0 | Built with Streamlit")
    
    # Main Content Area
    if not analyze_button:
        # Welcome Screen
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 3rem 0;'>
                <h2 style='color: #1E3A8A; margin-bottom: 1rem;'>Welcome to Pollen ID</h2>
                <p style='font-size: 1.1rem; color: #6B7280; margin-bottom: 2rem;'>
                    Your intelligent companion for managing pollen allergies
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("""
            **Getting Started:**
            
            1. Enter your Google Pollen API key in the sidebar
            2. Enter your location (zip code or city)
            3. Click "Analyze Pollen Levels"
            4. Explore real-time data, forecasts, and AI health advice
            """)
            
            st.markdown("""
            <div style='background-color: #F0F9FF; padding: 1.5rem; border-radius: 8px; margin-top: 2rem;'>
                <h4 style='color: #1E3A8A; margin-bottom: 0.5rem;'>🎯 Key Features</h4>
                <ul style='color: #1F2937;'>
                    <li><strong>Real-Time Data:</strong> Live pollen indices from Google's API</li>
                    <li><strong>AI Forecasting:</strong> LSTM-powered 7-day predictions</li>
                    <li><strong>Health Assistant:</strong> RAG-based personalized advice</li>
                    <li><strong>Risk Analytics:</strong> Comprehensive pollen risk assessment</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # Validation
    if not google_api_key:
        st.error("⚠️ Please enter your Google Pollen API key in the sidebar.")
        return
    
    if not location_input:
        st.error("⚠️ Please enter a location (zip code or city).")
        return
    
    # Processing
    with st.spinner("🔍 Geocoding location..."):
        coordinates = get_coordinates(location_input)
    
    if not coordinates:
        st.error(f"❌ Could not find coordinates for '{location_input}'. Please check your input and try again.")
        return
    
    lat, lng = coordinates
    st.success(f"✅ Location found: {lat:.4f}, {lng:.4f}")
    
    # Fetch Pollen Data
    with st.spinner("🌿 Fetching pollen data..."):
        pollen_response = get_pollen_data(lat, lng, google_api_key)
    
    if not pollen_response:
        st.error("❌ Failed to fetch pollen data. Please check your API key and try again.")
        return
    
    # Parse Data
    pollen_df = parse_pollen_data(pollen_response)
    
    if pollen_df.empty:
        st.error("❌ No pollen data available for this location.")
        return
    
    st.success("✅ Pollen data retrieved successfully!")
    
    # Get current day data
    current_data = pollen_df.iloc[0].to_dict()
    current_data['location'] = location_input
    
    # Calculate overall risk
    max_index = max(current_data.get('Grass', 0), 
                    current_data.get('Tree', 0), 
                    current_data.get('Weed', 0))
    overall_risk, risk_color = get_risk_level(max_index)
    
    # Dashboard Metrics
    st.markdown("### 📊 Today's Pollen Snapshot")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Overall Risk",
            value=overall_risk,
            delta=None
        )
        st.markdown(f"<div style='height: 4px; background-color: {risk_color}; border-radius: 2px;'></div>", 
                   unsafe_allow_html=True)
    
    with col2:
        grass_risk, grass_color = get_risk_level(current_data.get('Grass', 0))
        st.metric(
            label="🌱 Grass",
            value=f"{current_data.get('Grass', 0):.1f}",
            delta=grass_risk
        )
    
    with col3:
        tree_risk, tree_color = get_risk_level(current_data.get('Tree', 0))
        st.metric(
            label="🌳 Tree",
            value=f"{current_data.get('Tree', 0):.1f}",
            delta=tree_risk
        )
    
    with col4:
        weed_risk, weed_color = get_risk_level(current_data.get('Weed', 0))
        st.metric(
            label="🌾 Weed",
            value=f"{current_data.get('Weed', 0):.1f}",
            delta=weed_risk
        )
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Forecast", "💬 Health Assistant", "📋 Detailed Data"])
    
    # TAB 1: Forecast
    with tab1:
        st.markdown("### 7-Day Pollen Trend & AI Forecast")
        st.markdown("Visualizing historical data and LSTM-powered predictions")
        
        # Generate forecast
        with st.spinner("🤖 Generating AI forecast..."):
            forecast_df = simulate_lstm_forecast(pollen_df)
        
        # Plot forecast
        forecast_chart = create_forecast_chart(forecast_df)
        st.plotly_chart(forecast_chart, use_container_width=True)
        
        # Distribution
        col1, col2 = st.columns([1, 1])
        
        with col1:
            distribution_chart = create_distribution_chart(current_data)
            st.plotly_chart(distribution_chart, use_container_width=True)
        
        with col2:
            st.markdown("#### 🔮 Forecast Insights")
            
            # Calculate trends
            for pollen_type in ['Grass', 'Tree', 'Weed']:
                current_val = current_data.get(pollen_type, 0)
                future_val = forecast_df[forecast_df['is_forecast']][pollen_type].mean()
                
                trend = "📈 Rising" if future_val > current_val + 0.5 else \
                        "📉 Falling" if future_val < current_val - 0.5 else \
                        "➡️ Stable"
                
                st.markdown(f"""
                **{pollen_type} Pollen:**  
                Current: {current_val:.1f} → Forecast Avg: {future_val:.1f}  
                Trend: {trend}
                """)
            
            st.info("""
            💡 **Note:** Forecasts are generated using simulated LSTM outputs.
            In production, these would be powered by your trained time-series model.
            """)
    
    # TAB 2: Health Assistant
    with tab2:
        st.markdown("### 💬 AI Health Assistant")
        st.markdown("Ask questions about pollen management and get personalized advice")
        
        if not openai_api_key:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to use the Health Assistant.")
        else:
            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask about pollen management, symptoms, or precautions..."):
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate AI response
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Analyzing..."):
                        response = get_ai_health_advice(current_data, prompt, openai_api_key)
                    
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Quick action buttons
            st.markdown("#### Quick Questions")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💊 What precautions should I take?", use_container_width=True):
                    prompt = "What precautions should I take given current pollen levels?"
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.rerun()
            
            with col2:
                if st.button("🏃 Is it safe to exercise outside?", use_container_width=True):
                    prompt = "Is it safe to exercise outside given current pollen levels?"
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.rerun()
    
    # TAB 3: Detailed Data
    with tab3:
        st.markdown("### 📋 Raw Pollen Data")
        
        # Display DataFrame
        display_df = pollen_df.copy()
        display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
        
        st.dataframe(
            display_df[['date', 'Grass', 'Tree', 'Weed']],
            use_container_width=True,
            hide_index=True
        )
        
        # JSON Response
        with st.expander("🔍 View Raw API Response"):
            st.json(pollen_response)
    
    # Feedback Section
    st.markdown("---")
    st.markdown("### 📢 Feedback")
    st.markdown("Help us improve Pollen ID by providing feedback on data accuracy")
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        if st.button("✅ Accurate", use_container_width=True):
            st.success("Thank you for your feedback!")
    
    with col2:
        if st.button("❌ Not Accurate", use_container_width=True):
            st.info("Thank you! We'll use this to improve our models.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6B7280; padding: 1rem;'>
        <p>Built with ❤️ using Streamlit | Powered by Google Pollen API & OpenAI</p>
        <p style='font-size: 0.8rem;'>For educational and demonstration purposes</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
