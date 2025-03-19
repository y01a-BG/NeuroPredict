import streamlit as st
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="Hello World App",
    page_icon="👋",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        color: #4682B4;
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #708090;
    }
    .api-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Main app
st.markdown('<h1 class="main-header">Hello, World! 👋</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Welcome to my Streamlit Frontend</p>', unsafe_allow_html=True)

# Add a divider
st.divider()

# Interactive elements
name = st.text_input("What's your name?", "World")
if st.button("Say Hello 👋"):
    st.success(f"Hello, {name}! Welcome to Streamlit!")
    st.balloons()

# API Connection section
st.markdown('<div class="api-section">', unsafe_allow_html=True)
st.subheader("Connect to FastAPI Backend")

api_url = st.text_input("FastAPI URL", "https://hello-world-service-zsuoxbhheq-uc.a.run.app")

col1, col2 = st.columns(2)
with col1:
    if st.button("Test Root Endpoint"):
        try:
            response = requests.get(f"{api_url}/")
            st.json(response.json())
        except Exception as e:
            st.error(f"Error connecting to API: {e}")

with col2:
    if st.button(f"Say Hello to {name}"):
        try:
            response = requests.get(f"{api_url}/hello/{name}")
            st.json(response.json())
        except Exception as e:
            st.error(f"Error connecting to API: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# About this app
with st.expander("About this app"):
    st.write("""
    This is a simple Hello World application built with Streamlit.
    
    It demonstrates:
    - Basic Streamlit UI components
    - User input handling
    - Connecting to a FastAPI backend
    - Styling with custom CSS
    """)

# Footer
st.divider()
st.caption("Built with Streamlit ❤️") 