import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from collections import Counter
from utils.detection_utils import VideoProcessor, CSVLogger, WebcamCapture

def show():
    """Vehicle Category Classification page"""
    st.title("🚗 Vehicle Category Classification")
    st.markdown("Classify and analyze different types of vehicles using YOLOv8")
    
    # Initialize session state
    if 'vehicle_classifier' not in st.session_state:
        st.session_state.video_processor = VideoProcessor()
        st.session_state.csv_logger = CSVLogger()
        st.session_state.webcam_capture = WebcamCapture()
        st.session_state.classification_history = []
        st.session_state.vehicle_counts = {}
    
    # Sidebar controls
    # Input method selection
    input_method = st.sidebar.radio(
        "Select Input Method:",
        ["Upload Video", "Upload Image", "Live Webcam"]
    )
    
    # Classification parameters
    confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5)
    
    # Vehicle categories to track
    st.sidebar.subheader("Vehicle Categories")
    track_cars = st.sidebar.checkbox("Cars", value=True)
    track_buses = st.sidebar.checkbox("Buses", value=True)
    track_trucks = st.sidebar.checkbox("Trucks", value=True)
    track_motorcycles = st.sidebar.checkbox("Motorcycles", value=True)
    track_bicycles = st.sidebar.checkbox("Bicycles", value=True)
    
    # Display options
    st.sidebar.subheader("Display Options")
    show_bboxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
    show_labels = st.sidebar.checkbox("Show Labels", value=True)
    show_confidence = st.sidebar.checkbox("Show Confidence", value=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Vehicle Detection & Classification")
        
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
                st.info("Classifying vehicles in video... This may take a moment.")
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
                        
                        # Filter vehicles based on user selection
                        filtered_vehicles = []
                        for vehicle in vehicles:
                            vehicle_class = vehicle['class'].lower()
                            if ((track_cars and vehicle_class == 'car') or
                                (track_buses and vehicle_class == 'bus') or
                                (track_trucks and vehicle_class == 'truck') or
                                (track_motorcycles and vehicle_class == 'motorcycle') or
                                (track_bicycles and vehicle_class == 'bicycle')):
                                filtered_vehicles.append(vehicle)
                        
                        # Create annotated frame
                        display_frame = frame.copy()
                        if show_bboxes:
                            for vehicle in filtered_vehicles:
                                x1, y1, x2, y2 = vehicle['bbox']
                                
                                # Color coding by vehicle type
                                colors = {
                                    'car': (0, 255, 0),      # Green
                                    'bus': (255, 0, 0),      # Blue
                                    'truck': (0, 0, 255),    # Red
                                    'motorcycle': (255, 255, 0),  # Cyan
                                    'bicycle': (255, 0, 255)      # Magenta
                                }
                                color = colors.get(vehicle['class'].lower(), (255, 255, 255))
                                
                                # Draw bounding box
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Draw label
                                if show_labels:
                                    label = vehicle['class'].title()
                                    if show_confidence:
                                        label += f" ({vehicle['confidence']:.2f})"
                                    
                                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                    cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                                                 (x1 + label_size[0], y1), color, -1)
                                    cv2.putText(display_frame, label, (x1, y1 - 5), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Convert BGR to RGB for display
                        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption=f"Frame {frame_idx + 1} - Vehicles: {len(filtered_vehicles)}")
                        
                        # Count vehicles by category
                        vehicle_counts = Counter(vehicle['class'] for vehicle in filtered_vehicles)
                        
                        # Store classification data
                        classification_data = {
                            'timestamp': datetime.now(),
                            'frame_number': frame_idx,
                            'vehicle_counts': dict(vehicle_counts),
                            'total_vehicles': len(filtered_vehicles)
                        }
                        
                        # Update history
                        if classification_data not in st.session_state.classification_history:
                            st.session_state.classification_history.append(classification_data)
                        
                        # Log to CSV
                        st.session_state.csv_logger.log_vehicle_counts(
                            dict(vehicle_counts), 
                            classification_data['timestamp']
                        )
                
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
                
                # Filter vehicles based on user selection
                filtered_vehicles = []
                for vehicle in vehicles:
                    vehicle_class = vehicle['class'].lower()
                    if ((track_cars and vehicle_class == 'car') or
                        (track_buses and vehicle_class == 'bus') or
                        (track_trucks and vehicle_class == 'truck') or
                        (track_motorcycles and vehicle_class == 'motorcycle') or
                        (track_bicycles and vehicle_class == 'bicycle')):
                        filtered_vehicles.append(vehicle)
                
                # Create annotated frame
                display_frame = frame_data['frame'].copy()
                if show_bboxes:
                    for vehicle in filtered_vehicles:
                        x1, y1, x2, y2 = vehicle['bbox']
                        
                        # Color coding by vehicle type
                        colors = {
                            'car': (0, 255, 0),
                            'bus': (255, 0, 0),
                            'truck': (0, 0, 255),
                            'motorcycle': (255, 255, 0),
                            'bicycle': (255, 0, 255)
                        }
                        color = colors.get(vehicle['class'].lower(), (255, 255, 255))
                        
                        # Draw bounding box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        if show_labels:
                            label = vehicle['class'].title()
                            if show_confidence:
                                label += f" ({vehicle['confidence']:.2f})"
                            
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                                         (x1 + label_size[0], y1), color, -1)
                            cv2.putText(display_frame, label, (x1, y1 - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display image
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Classified Vehicles: {len(filtered_vehicles)}")
                
                # Count vehicles by category
                vehicle_counts = Counter(vehicle['class'] for vehicle in filtered_vehicles)
                
                # Store and log data
                classification_data = {
                    'timestamp': datetime.now(),
                    'frame_number': 0,
                    'vehicle_counts': dict(vehicle_counts),
                    'total_vehicles': len(filtered_vehicles)
                }
                
                st.session_state.classification_history.append(classification_data)
                st.session_state.csv_logger.log_vehicle_counts(
                    dict(vehicle_counts), 
                    classification_data['timestamp']
                )
        
        elif input_method == "Live Webcam":
            st.info("📹 Live webcam classification - Click 'Capture Frame' to classify vehicles")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                capture_button = st.button("📸 Capture Frame", type="primary")
            with col_btn2:
                if st.button("🔄 Clear Display"):
                    if 'classification_webcam_frame' in st.session_state:
                        del st.session_state.classification_webcam_frame
                    st.rerun()
            
            if capture_button:
                with st.spinner("Capturing and classifying vehicles..."):
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
                        st.session_state.classification_webcam_frame = frame_data
            
            # Display captured frame if available
            if 'classification_webcam_frame' in st.session_state:
                frame_data = st.session_state.classification_webcam_frame
                vehicles = frame_data['vehicles']
                frame = frame_data['frame']
                
                # Define colors for different vehicle types
                colors = {
                    'car': (0, 255, 0),
                    'bus': (255, 0, 0),
                    'truck': (0, 0, 255),
                    'motorcycle': (255, 255, 0),
                    'bicycle': (255, 0, 255)
                }
                
                # Draw colored boxes based on vehicle type
                display_frame = frame.copy()
                for vehicle in vehicles:
                    x1, y1, x2, y2 = vehicle['bbox']
                    vehicle_class = vehicle['class']
                    color = colors.get(vehicle_class, (255, 255, 255))
                    
                    # Draw thicker colored box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw label with background
                    label = f"{vehicle_class.title()}: {vehicle['confidence']:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0], y1), color, -1)
                    cv2.putText(display_frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Live Vehicle Classification - Total: {len(vehicles)}", use_container_width=True)
                
                # Count vehicles by type
                vehicle_counts = Counter(vehicle['class'] for vehicle in vehicles)
                
                # Show detection stats
                if len(vehicles) > 0:
                    st.success(f"🚗 Total Vehicles Detected: {len(vehicles)}")
                    
                    # Display breakdown
                    cols = st.columns(len(vehicle_counts) if len(vehicle_counts) > 0 else 1)
                    for idx, (vehicle_type, count) in enumerate(vehicle_counts.items()):
                        with cols[idx % len(cols)]:
                            st.metric(vehicle_type.title(), count)
                else:
                    st.info("No vehicles detected in current frame")
                
                # Log classification data
                classification_data = {
                    'timestamp': datetime.now(),
                    'frame_number': 0,
                    'vehicle_counts': dict(vehicle_counts),
                    'total_vehicles': len(vehicles)
                }
                
                st.session_state.classification_history.append(classification_data)
                st.session_state.csv_logger.log_vehicle_counts(
                    dict(vehicle_counts), 
                    classification_data['timestamp']
                )
    
    with col2:
        st.subheader("Classification Statistics")
        
        if st.session_state.classification_history:
            # Get latest classification data
            latest_data = st.session_state.classification_history[-1]
            vehicle_counts = latest_data['vehicle_counts']
            total_vehicles = latest_data['total_vehicles']
            
            # Display current counts
            st.metric("Total Vehicles", total_vehicles)
            
            # Individual vehicle counts
            st.subheader("Vehicle Breakdown")
            for vehicle_type, count in vehicle_counts.items():
                percentage = (count / total_vehicles * 100) if total_vehicles > 0 else 0
                st.metric(
                    vehicle_type.title(), 
                    f"{count} ({percentage:.1f}%)"
                )
            
            # Vehicle type distribution pie chart
            if vehicle_counts:
                st.subheader("Distribution")
                fig_pie = px.pie(
                    values=list(vehicle_counts.values()),
                    names=[name.title() for name in vehicle_counts.keys()],
                    title="Vehicle Type Distribution"
                )
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No classification data available yet.")
    
    # Analytics section
    st.markdown("---")
    st.subheader("📊 Classification Analytics")
    
    if len(st.session_state.classification_history) > 1:
        # Create comprehensive analytics
        
        # Aggregate all vehicle counts
        all_vehicle_types = set()
        for data in st.session_state.classification_history:
            all_vehicle_types.update(data['vehicle_counts'].keys())
        
        # Time series data
        timestamps = [data['timestamp'] for data in st.session_state.classification_history]
        
        # Create time series plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Vehicle counts over time
            fig_timeline = go.Figure()
            
            for vehicle_type in all_vehicle_types:
                counts = []
                for data in st.session_state.classification_history:
                    counts.append(data['vehicle_counts'].get(vehicle_type, 0))
                
                fig_timeline.add_trace(go.Scatter(
                    x=timestamps,
                    y=counts,
                    mode='lines+markers',
                    name=vehicle_type.title(),
                    line=dict(width=2)
                ))
            
            fig_timeline.update_layout(
                title="Vehicle Counts Over Time",
                xaxis_title="Time",
                yaxis_title="Number of Vehicles",
                height=400
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        with col2:
            # Total vehicles over time
            total_counts = [data['total_vehicles'] for data in st.session_state.classification_history]
            
            fig_total = go.Figure()
            fig_total.add_trace(go.Scatter(
                x=timestamps,
                y=total_counts,
                mode='lines+markers',
                name='Total Vehicles',
                line=dict(color='blue', width=3)
            ))
            
            fig_total.update_layout(
                title="Total Vehicle Count Over Time",
                xaxis_title="Time",
                yaxis_title="Total Vehicles",
                height=400
            )
            st.plotly_chart(fig_total, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        
        # Calculate aggregated statistics
        total_detections = sum(data['total_vehicles'] for data in st.session_state.classification_history)
        avg_vehicles_per_frame = total_detections / len(st.session_state.classification_history)
        
        # Most common vehicle type
        all_counts = {}
        for data in st.session_state.classification_history:
            for vehicle_type, count in data['vehicle_counts'].items():
                all_counts[vehicle_type] = all_counts.get(vehicle_type, 0) + count
        
        most_common_vehicle = max(all_counts.items(), key=lambda x: x[1]) if all_counts else ("None", 0)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Detections", total_detections)
        
        with col2:
            st.metric("Avg Vehicles/Frame", f"{avg_vehicles_per_frame:.1f}")
        
        with col3:
            st.metric("Most Common Type", most_common_vehicle[0].title())
        
        with col4:
            st.metric("Vehicle Types", len(all_vehicle_types))
        
        # Detailed breakdown table
        st.subheader("Detailed Breakdown")
        
        breakdown_data = []
        for vehicle_type in all_vehicle_types:
            total_count = all_counts.get(vehicle_type, 0)
            percentage = (total_count / total_detections * 100) if total_detections > 0 else 0
            avg_per_frame = total_count / len(st.session_state.classification_history)
            
            breakdown_data.append({
                'Vehicle Type': vehicle_type.title(),
                'Total Count': total_count,
                'Percentage': f"{percentage:.1f}%",
                'Avg per Frame': f"{avg_per_frame:.1f}"
            })
        
        breakdown_df = pd.DataFrame(breakdown_data)
        breakdown_df = breakdown_df.sort_values('Total Count', ascending=False)
        st.dataframe(breakdown_df, use_container_width=True)
    
    # Export options
    st.markdown("---")
    st.subheader("📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Download Classification Report"):
            csv_data = st.session_state.csv_logger.get_csv_download_data("vehicle_counts.csv")
            if csv_data is not None:
                st.download_button(
                    label="Download CSV",
                    data=csv_data.to_csv(index=False),
                    file_name=f"vehicle_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No classification data available for download.")
    
    with col2:
        if st.button("Clear Classification History"):
            st.session_state.classification_history = []
            st.success("Classification history cleared successfully!")
            st.rerun()
    
    with col3:
        if st.button("Export Summary Statistics"):
            if st.session_state.classification_history:
                # Generate summary report
                summary_data = {
                    'total_frames_analyzed': len(st.session_state.classification_history),
                    'total_vehicles_detected': sum(data['total_vehicles'] for data in st.session_state.classification_history),
                    'vehicle_type_breakdown': dict(all_counts) if 'all_counts' in locals() else {},
                    'analysis_period': {
                        'start': st.session_state.classification_history[0]['timestamp'].isoformat(),
                        'end': st.session_state.classification_history[-1]['timestamp'].isoformat()
                    }
                }
                
                st.download_button(
                    label="Download Summary JSON",
                    data=pd.Series(summary_data).to_json(indent=2),
                    file_name=f"classification_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No data to export.")
