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

def main():
    """Main application function with sidebar navigation"""
    
    # Sidebar navigation
    st.sidebar.title("🚦 Traffic Management System")
    st.sidebar.markdown("---")
    
    # Navigation menu
    pages = {
        "🚥 Addddaptive Traffic Signals": adaptive_traffic,
        "⚠️ Traffic Violation Detection": violation_detection,
        "🚗 Vehicle Classification": vehicle_classification,
        "🚶 Pedestrian Monitoring": pedestrian_monitoring
    }
    
    selected_page = st.sidebar.selectbox(
        "Select a Feature:",
        list(pages.keys()),
        index=0
    )
    
    # Display selected page
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This Traffic Management System uses YOLOv8 for real-time object detection "
        "to analyze traffic patterns, detect violations, classify vehicles, and monitor pedestrians."
    )
    
    # Run the selected page
    pages[selected_page].show()

if __name__ == "__main__":
    main()
