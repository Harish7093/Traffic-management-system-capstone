import streamlit as st
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import page modules
from pages import adaptive_traffic, violation_detection, vehicle_classification, pedestrian_monitoring

# Configure Streamlit page
st.set_page_config(
    page_title="Traffic Management System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide default Streamlit navigation
hide_streamlit_style = """
    <style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def main():
    """Main application function with sidebar navigation"""
    
    # Sidebar navigation - clean design without extra headers
    pages = {
        "🚥 Adaptive Traffic Signals": adaptive_traffic,
        "⚠️ Traffic Violation Detection": violation_detection,
        "🚗 Vehicle Classification": vehicle_classification,
        "🚶 Pedestrian Monitoring": pedestrian_monitoring
    }
    
    # Initialize session state for selected page
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "🚥 Adaptive Traffic Signals"
    
    # Display navigation buttons
    for page_name in pages.keys():
        if st.sidebar.button(
            page_name, 
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if st.session_state.selected_page == page_name else "secondary"
        ):
            st.session_state.selected_page = page_name
            st.rerun()
    
    # Display selected page
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This Traffic Management System uses YOLOv8 for real-time object detection "
        "to analyze traffic patterns, detect violations, classify vehicles, and monitor pedestrians."
    )
    
    # Run the selected page
    pages[st.session_state.selected_page].show()

if __name__ == "__main__":
    main()
