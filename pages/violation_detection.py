import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from utils.detection_utils import VideoProcessor, TrafficAnalyzer, CSVLogger, WebcamCapture

def show():
    """Traffic Violation Detection page"""
    st.title("⚠️ Traffic Violation Detection")
    st.markdown("Real-time detection and logging of traffic violations")
    
    # Initialize session state
    if 'violation_detector' not in st.session_state:
        st.session_state.violation_detector = TrafficAnalyzer()
        st.session_state.csv_logger = CSVLogger()
        st.session_state.video_processor = VideoProcessor()
        st.session_state.webcam_capture = WebcamCapture()
        st.session_state.violations_log = []
        st.session_state.violation_stats = {}
    
    # Sidebar controls
    st.sidebar.header("Violation Detection Settings")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Select Input Method:",
        ["Upload Video", "Upload Image", "Live Webcam"]
    )
    
    # Traffic light state
    traffic_light_state = st.sidebar.selectbox(
        "Current Traffic Light State:",
        ["red", "yellow", "green"]
    )
    
    # Violation types to detect
    st.sidebar.subheader("Violation Types")
    detect_red_light = st.sidebar.checkbox("Red Light Violations", value=True)
    detect_speed = st.sidebar.checkbox("Speed Violations", value=False)
    detect_wrong_lane = st.sidebar.checkbox("Wrong Lane Usage", value=False)
    detect_illegal_turn = st.sidebar.checkbox("Illegal Turns", value=False)
    
    # Detection sensitivity
    confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Violation Detection")
        
        if input_method == "Upload Video":
            uploaded_file = st.file_uploader(
                "Choose a video file", 
                type=['mp4', 'avi', 'mov', 'mkv']
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                # Process video
                st.info("Processing video for violations... This may take a moment.")
                progress_bar = st.progress(0)
                
                def update_progress(progress):
                    progress_bar.progress(progress)
                
                try:
                    frames_data = st.session_state.video_processor.process_video_file(
                        temp_path, 
                        progress_callback=update_progress
                    )
                    
                    if frames_data:
                        # Display processed video frames
                        frame_idx = st.slider(
                            "Frame", 
                            0, 
                            len(frames_data) - 1, 
                            0
                        )
                        
                        current_frame_data = frames_data[frame_idx]
                        frame = current_frame_data['frame']
                        vehicles = current_frame_data['vehicles']
                        
                        # Detect violations in current frame
                        violations = []
                        if detect_red_light:
                            red_light_violations = st.session_state.violation_detector.detect_violations(
                                vehicles, traffic_light_state
                            )
                            violations.extend(red_light_violations)
                        
                        # Draw violation markers on frame
                        violation_frame = frame.copy()
                        for violation in violations:
                            x1, y1, x2, y2 = violation['bbox']
                            # Draw red box for violations
                            cv2.rectangle(violation_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(violation_frame, f"VIOLATION: {violation['type']}", 
                                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Convert BGR to RGB for display
                        frame_rgb = cv2.cvtColor(violation_frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption=f"Frame {frame_idx + 1} - Violations: {len(violations)}")
                        
                        # Log violations
                        if violations:
                            st.session_state.violations_log.extend(violations)
                            st.session_state.csv_logger.log_violations(violations)
                            
                            # Show violation details
                            st.warning(f"🚨 {len(violations)} violation(s) detected in this frame!")
                            for i, violation in enumerate(violations):
                                with st.expander(f"Violation {i+1}: {violation['type']}"):
                                    col_v1, col_v2 = st.columns(2)
                                    with col_v1:
                                        st.write(f"**Vehicle Type:** {violation['vehicle_class']}")
                                        st.write(f"**Confidence:** {violation['confidence']:.2f}")
                                    with col_v2:
                                        st.write(f"**Timestamp:** {violation['timestamp']}")
                                        st.write(f"**Location:** {violation['bbox']}")
                
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                finally:
                    # Clean up temp file
                    import os
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        elif input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file", 
                type=['jpg', 'jpeg', 'png', 'bmp']
            )
            
            if uploaded_file is not None:
                # Read and process image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                
                # Process image for detection
                frame_data = st.session_state.video_processor.process_webcam_frame(image)
                vehicles = frame_data['vehicles']
                
                # Detect violations
                violations = []
                if detect_red_light:
                    red_light_violations = st.session_state.violation_detector.detect_violations(
                        vehicles, traffic_light_state
                    )
                    violations.extend(red_light_violations)
                
                # Draw violation markers
                violation_frame = frame_data['frame'].copy()
                for violation in violations:
                    x1, y1, x2, y2 = violation['bbox']
                    cv2.rectangle(violation_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(violation_frame, f"VIOLATION: {violation['type']}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display image
                frame_rgb = cv2.cvtColor(violation_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Uploaded Image - Violations: {len(violations)}")
                
                # Log violations
                if violations:
                    st.session_state.violations_log.extend(violations)
                    st.session_state.csv_logger.log_violations(violations)
                    st.warning(f"🚨 {len(violations)} violation(s) detected!")
                else:
                    st.success("✅ No violations detected in this image.")
        
        elif input_method == "Live Webcam":
            st.info("📹 Live webcam detection - Click 'Capture Frame' to analyze current view")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                capture_button = st.button("📸 Capture Frame", type="primary")
            with col_btn2:
                if st.button("🔄 Clear Display"):
                    if 'webcam_frame_data' in st.session_state:
                        del st.session_state.webcam_frame_data
                    st.rerun()
            
            if capture_button:
                with st.spinner("Capturing and processing frame..."):
                    # Capture frame from webcam
                    frame_data, error = st.session_state.webcam_capture.capture_and_process()
                    
                    if error:
                        st.error(f"❌ {error}")
                        st.info("**Troubleshooting tips:**\n"
                               "- Ensure your webcam is connected\n"
                               "- Check if another application is using the camera\n"
                               "- Grant camera permissions to your browser/application\n"
                               "- Try restarting the application")
                    else:
                        st.session_state.webcam_frame_data = frame_data
            
            # Display captured frame if available
            if 'webcam_frame_data' in st.session_state:
                frame_data = st.session_state.webcam_frame_data
                vehicles = frame_data['vehicles']
                
                # Detect violations
                violations = []
                if detect_red_light:
                    red_light_violations = st.session_state.violation_detector.detect_violations(
                        vehicles, traffic_light_state
                    )
                    violations.extend(red_light_violations)
                
                # Draw violation markers
                violation_frame = frame_data['frame'].copy()
                for violation in violations:
                    x1, y1, x2, y2 = violation['bbox']
                    cv2.rectangle(violation_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(violation_frame, f"VIOLATION: {violation['type']}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(violation_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Live Webcam Feed - Violations: {len(violations)}", use_container_width=True)
                
                # Show detection stats
                st.info(f"🚗 Detected: {len(vehicles)} vehicles | 🚨 Violations: {len(violations)}")
                
                # Log violations
                if violations:
                    st.session_state.violations_log.extend(violations)
                    st.session_state.csv_logger.log_violations(violations)
                    st.error(f"🚨 {len(violations)} violation(s) detected!")
                    
                    # Show violation details
                    for i, violation in enumerate(violations):
                        with st.expander(f"Violation {i+1}: {violation['type']}"):
                            col_v1, col_v2 = st.columns(2)
                            with col_v1:
                                st.write(f"**Vehicle Type:** {violation['vehicle_class']}")
                                st.write(f"**Confidence:** {violation['confidence']:.2f}")
                            with col_v2:
                                st.write(f"**Timestamp:** {violation['timestamp']}")
                                st.write(f"**Location:** {violation['bbox']}")
                else:
                    st.success("✅ No violations detected in current frame")
    
    with col2:
        st.subheader("Violation Statistics")
        
        if st.session_state.violations_log:
            # Current session stats
            total_violations = len(st.session_state.violations_log)
            st.metric("Total Violations", total_violations)
            
            # Violation types breakdown
            violation_types = {}
            vehicle_types = {}
            
            for violation in st.session_state.violations_log:
                v_type = violation['type']
                vehicle_type = violation['vehicle_class']
                
                violation_types[v_type] = violation_types.get(v_type, 0) + 1
                vehicle_types[vehicle_type] = vehicle_types.get(vehicle_type, 0) + 1
            
            # Display breakdown
            st.subheader("Violation Types")
            for v_type, count in violation_types.items():
                st.metric(v_type.replace('_', ' ').title(), count)
            
            st.subheader("Vehicle Types")
            for vehicle_type, count in vehicle_types.items():
                st.metric(vehicle_type.title(), count)
            
            # Recent violations
            st.subheader("Recent Violations")
            recent_violations = st.session_state.violations_log[-5:]  # Last 5 violations
            
            for i, violation in enumerate(reversed(recent_violations)):
                with st.expander(f"Violation {len(recent_violations)-i}"):
                    st.write(f"**Type:** {violation['type'].replace('_', ' ').title()}")
                    st.write(f"**Vehicle:** {violation['vehicle_class'].title()}")
                    st.write(f"**Time:** {violation['timestamp'].strftime('%H:%M:%S')}")
                    st.write(f"**Confidence:** {violation['confidence']:.2f}")
        else:
            st.info("No violations detected yet.")
    
    # Violations log table
    st.markdown("---")
    st.subheader("📋 Violations Log")
    
    if st.session_state.violations_log:
        # Create dataframe for display
        violations_df = pd.DataFrame([
            {
                'Timestamp': violation['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'Type': violation['type'].replace('_', ' ').title(),
                'Vehicle': violation['vehicle_class'].title(),
                'Confidence': f"{violation['confidence']:.2f}",
                'Location': f"({violation['bbox'][0]}, {violation['bbox'][1]})"
            }
            for violation in st.session_state.violations_log
        ])
        
        # Display with pagination
        page_size = 10
        total_pages = len(violations_df) // page_size + (1 if len(violations_df) % page_size > 0 else 0)
        
        if total_pages > 1:
            page = st.selectbox("Page", range(1, total_pages + 1))
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            display_df = violations_df.iloc[start_idx:end_idx]
        else:
            display_df = violations_df
        
        st.dataframe(display_df, use_container_width=True)
        
        # Analytics charts
        if len(violations_df) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Violations by type
                type_counts = violations_df['Type'].value_counts()
                fig_types = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Violations by Type"
                )
                st.plotly_chart(fig_types, use_container_width=True)
            
            with col2:
                # Violations by vehicle type
                vehicle_counts = violations_df['Vehicle'].value_counts()
                fig_vehicles = px.bar(
                    x=vehicle_counts.index,
                    y=vehicle_counts.values,
                    title="Violations by Vehicle Type"
                )
                fig_vehicles.update_layout(
                    xaxis_title="Vehicle Type",
                    yaxis_title="Number of Violations"
                )
                st.plotly_chart(fig_vehicles, use_container_width=True)
    else:
        st.info("No violations logged yet. Upload a video or image to start detection.")
    
    # Export and management options
    st.markdown("---")
    st.subheader("📥 Export & Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Download Violations Report"):
            csv_data = st.session_state.csv_logger.get_csv_download_data("violations.csv")
            if csv_data is not None:
                st.download_button(
                    label="Download CSV",
                    data=csv_data.to_csv(index=False),
                    file_name=f"violations_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No violation data available for download.")
    
    with col2:
        if st.button("Clear Violations Log"):
            st.session_state.violations_log = []
            st.success("Violations log cleared successfully!")
            st.rerun()
    
    with col3:
        if st.button("Generate Summary Report"):
            if st.session_state.violations_log:
                # Generate summary statistics
                summary_stats = {
                    'total_violations': len(st.session_state.violations_log),
                    'violation_types': len(set(v['type'] for v in st.session_state.violations_log)),
                    'most_common_violation': max(set(v['type'] for v in st.session_state.violations_log), 
                                               key=lambda x: sum(1 for v in st.session_state.violations_log if v['type'] == x)),
                    'most_common_vehicle': max(set(v['vehicle_class'] for v in st.session_state.violations_log), 
                                             key=lambda x: sum(1 for v in st.session_state.violations_log if v['vehicle_class'] == x)),
                    'average_confidence': np.mean([v['confidence'] for v in st.session_state.violations_log])
                }
                
                # Display summary
                st.success("📊 Summary Report Generated")
                st.json(summary_stats)
            else:
                st.warning("No violations to summarize.")
