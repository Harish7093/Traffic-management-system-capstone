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
    """Adaptive Traffic Signals page"""
    st.title("🚥 Adaptive Traffic Signals")
    st.markdown("Real-time vehicle density detection with adaptive traffic light timing")
    
    # Initialize session state
    if 'traffic_analyzer' not in st.session_state:
        st.session_state.traffic_analyzer = TrafficAnalyzer()
        st.session_state.csv_logger = CSVLogger()
        st.session_state.video_processor = VideoProcessor()
        st.session_state.webcam_capture = WebcamCapture()
        st.session_state.density_history = []
        st.session_state.timing_history = []
    
    # Sidebar controls
    st.sidebar.header("Traffic Signal Controls")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Select Input Method:",
        ["Upload Video", "Upload Image", "Live Webcam"]
    )
    
    # Traffic signal parameters
    st.sidebar.subheader("Signal Parameters")
    num_lanes = st.sidebar.slider("Number of Lanes", 2, 6, 4)
    base_time = st.sidebar.slider("Base Signal Time (seconds)", 15, 60, 30)
    max_time = st.sidebar.slider("Maximum Signal Time (seconds)", 30, 120, 60)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Traffic Analysis")
        
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
                st.info("Processing video... This may take a moment.")
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
                        
                        # Convert BGR to RGB for display
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption=f"Frame {frame_idx + 1}")
                        
                        # Calculate density and timing
                        frame_height, frame_width = frame.shape[:2]
                        lane_densities = st.session_state.traffic_analyzer.calculate_vehicle_density(
                            vehicles, frame_width, frame_height, num_lanes
                        )
                        
                        adaptive_timings = st.session_state.traffic_analyzer.calculate_adaptive_timing(
                            lane_densities, base_time, max_time
                        )
                        
                        # Store data for analysis
                        timestamp = datetime.now()
                        density_data = {
                            'timestamp': timestamp,
                            'lane_densities': lane_densities,
                            'adaptive_timings': adaptive_timings,
                            'total_vehicles': sum(lane_densities)
                        }
                        
                        # Update history
                        st.session_state.density_history.append(density_data)
                        
                        # Log to CSV
                        st.session_state.csv_logger.log_density_stats(lane_densities, timestamp)
                        
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
                frame = frame_data['frame']
                
                # Get frame dimensions
                frame_height, frame_width = frame.shape[:2]
                
                # Calculate density
                lane_densities = st.session_state.traffic_analyzer.calculate_vehicle_density(
                    vehicles, frame_width, frame_height, num_lanes
                )
                
                adaptive_timings = st.session_state.traffic_analyzer.calculate_adaptive_timing(
                    lane_densities, base_time, max_time
                )
                
                # Draw lane dividers on frame
                display_frame = frame.copy()
                lane_width = frame_width // num_lanes
                for i in range(1, num_lanes):
                    x = i * lane_width
                    cv2.line(display_frame, (x, 0), (x, frame_height), (0, 255, 255), 2)
                
                # Add lane labels and vehicle counts
                for i in range(num_lanes):
                    x_center = (i * lane_width) + (lane_width // 2)
                    y_pos = 30
                    # Draw background for text
                    text = f"Lane {i+1}: {lane_densities[i]} vehicles"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(display_frame, 
                                 (x_center - text_size[0]//2 - 5, y_pos - text_size[1] - 5),
                                 (x_center + text_size[0]//2 + 5, y_pos + 5),
                                 (0, 0, 0), -1)
                    cv2.putText(display_frame, text, 
                               (x_center - text_size[0]//2, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Traffic Analysis - Total Vehicles: {len(vehicles)}", use_container_width=True)
                
                # Show detection stats
                st.success(f"🚗 Total Vehicles Detected: {len(vehicles)}")
                
                # Display lane-wise breakdown
                st.subheader("Lane-wise Vehicle Distribution")
                cols = st.columns(num_lanes)
                for i in range(num_lanes):
                    with cols[i]:
                        st.metric(f"Lane {i+1}", f"{lane_densities[i]} vehicles")
                        st.caption(f"Signal Time: {adaptive_timings[i]}s")
                
                # Store data
                timestamp = datetime.now()
                density_data = {
                    'timestamp': timestamp,
                    'lane_densities': lane_densities,
                    'adaptive_timings': adaptive_timings,
                    'total_vehicles': sum(lane_densities)
                }
                
                st.session_state.density_history.append(density_data)
                st.session_state.csv_logger.log_density_stats(lane_densities, timestamp)
                
                # Show adaptive timing recommendation
                st.info("💡 **Adaptive Signal Timing Recommendation**")
                timing_df = pd.DataFrame({
                    'Lane': [f"Lane {i+1}" for i in range(num_lanes)],
                    'Vehicles': lane_densities,
                    'Recommended Time (s)': adaptive_timings
                })
                st.dataframe(timing_df, use_container_width=True)
        
        elif input_method == "Live Webcam":
            st.info("📹 Live webcam detection - Click 'Capture Frame' to analyze traffic density")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                capture_button = st.button("📸 Capture Frame", type="primary")
            with col_btn2:
                if st.button("🔄 Clear Display"):
                    if 'adaptive_webcam_frame' in st.session_state:
                        del st.session_state.adaptive_webcam_frame
                    st.rerun()
            
            if capture_button:
                with st.spinner("Capturing and analyzing traffic..."):
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
                        st.session_state.adaptive_webcam_frame = frame_data
            
            # Display captured frame if available
            if 'adaptive_webcam_frame' in st.session_state:
                frame_data = st.session_state.adaptive_webcam_frame
                vehicles = frame_data['vehicles']
                frame = frame_data['frame']
                
                # Get frame dimensions
                frame_height, frame_width = frame.shape[:2]
                
                # Calculate density
                lane_densities = st.session_state.traffic_analyzer.calculate_vehicle_density(
                    vehicles, frame_width, frame_height, num_lanes
                )
                
                adaptive_timings = st.session_state.traffic_analyzer.calculate_adaptive_timing(
                    lane_densities, base_time, max_time
                )
                
                # Draw lane dividers on frame
                display_frame = frame.copy()
                lane_width = frame_width // num_lanes
                for i in range(1, num_lanes):
                    x = i * lane_width
                    cv2.line(display_frame, (x, 0), (x, frame_height), (0, 255, 255), 2)
                
                # Add lane labels
                for i in range(num_lanes):
                    x_center = (i * lane_width) + (lane_width // 2)
                    cv2.putText(display_frame, f"Lane {i+1}: {lane_densities[i]} vehicles", 
                               (x_center - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Live Traffic Feed - Total Vehicles: {len(vehicles)}", use_container_width=True)
                
                # Show detection stats
                st.success(f"🚗 Total Vehicles Detected: {len(vehicles)}")
                
                # Store data
                timestamp = datetime.now()
                density_data = {
                    'timestamp': timestamp,
                    'lane_densities': lane_densities,
                    'adaptive_timings': adaptive_timings,
                    'total_vehicles': sum(lane_densities)
                }
                
                st.session_state.density_history.append(density_data)
                st.session_state.csv_logger.log_density_stats(lane_densities, timestamp)
    
    with col2:
        st.subheader("Real-time Statistics")
        
        if st.session_state.density_history:
            latest_data = st.session_state.density_history[-1]
            lane_densities = latest_data['lane_densities']
            adaptive_timings = latest_data['adaptive_timings']
            
            # Display current metrics
            st.metric("Total Vehicles", latest_data['total_vehicles'])
            
            # Lane density display
            st.subheader("Lane Density")
            for i, (density, timing) in enumerate(zip(lane_densities, adaptive_timings)):
                col_lane, col_timing = st.columns(2)
                with col_lane:
                    st.metric(f"Lane {i+1}", f"{density} vehicles")
                with col_timing:
                    st.metric(f"Signal Time", f"{timing}s")
            
            # Traffic light simulation
            st.subheader("Traffic Light Status")
            max_density_lane = np.argmax(lane_densities) if any(lane_densities) else 0
            
            for i in range(num_lanes):
                if i == max_density_lane and latest_data['total_vehicles'] > 0:
                    st.success(f"🟢 Lane {i+1}: GREEN ({adaptive_timings[i]}s)")
                else:
                    st.error(f"🔴 Lane {i+1}: RED")
    
    # Analytics section
    st.markdown("---")
    st.subheader("📊 Traffic Analytics")
    
    if len(st.session_state.density_history) > 1:
        # Create dataframe for plotting
        df_history = pd.DataFrame([
            {
                'timestamp': data['timestamp'],
                'total_vehicles': data['total_vehicles'],
                **{f'lane_{i+1}': density for i, density in enumerate(data['lane_densities'])},
                **{f'timing_{i+1}': timing for i, timing in enumerate(data['adaptive_timings'])}
            }
            for data in st.session_state.density_history[-20:]  # Last 20 data points
        ])
        
        # Time series plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Vehicle density over time
            fig_density = go.Figure()
            
            for i in range(num_lanes):
                if f'lane_{i+1}' in df_history.columns:
                    fig_density.add_trace(go.Scatter(
                        x=df_history['timestamp'],
                        y=df_history[f'lane_{i+1}'],
                        mode='lines+markers',
                        name=f'Lane {i+1}',
                        line=dict(width=2)
                    ))
            
            fig_density.update_layout(
                title="Vehicle Density Over Time",
                xaxis_title="Time",
                yaxis_title="Number of Vehicles",
                height=400
            )
            st.plotly_chart(fig_density, use_container_width=True)
        
        with col2:
            # Adaptive timing over time
            fig_timing = go.Figure()
            
            for i in range(num_lanes):
                if f'timing_{i+1}' in df_history.columns:
                    fig_timing.add_trace(go.Scatter(
                        x=df_history['timestamp'],
                        y=df_history[f'timing_{i+1}'],
                        mode='lines+markers',
                        name=f'Lane {i+1}',
                        line=dict(width=2)
                    ))
            
            fig_timing.update_layout(
                title="Adaptive Signal Timing",
                xaxis_title="Time",
                yaxis_title="Signal Duration (seconds)",
                height=400
            )
            st.plotly_chart(fig_timing, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_vehicles = df_history['total_vehicles'].mean()
            st.metric("Average Vehicles", f"{avg_vehicles:.1f}")
        
        with col2:
            max_vehicles = df_history['total_vehicles'].max()
            st.metric("Peak Traffic", f"{max_vehicles}")
        
        with col3:
            busiest_lane = max(range(num_lanes), 
                             key=lambda i: df_history[f'lane_{i+1}'].sum() if f'lane_{i+1}' in df_history.columns else 0)
            st.metric("Busiest Lane", f"Lane {busiest_lane + 1}")
        
        with col4:
            efficiency_gain = ((max_time - df_history[[f'timing_{i+1}' for i in range(num_lanes) 
                                                     if f'timing_{i+1}' in df_history.columns]].mean().mean()) / max_time) * 100
            st.metric("Efficiency Gain", f"{efficiency_gain:.1f}%")
    
    # Export options
    st.markdown("---")
    st.subheader("📥 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Density Report"):
            csv_data = st.session_state.csv_logger.get_csv_download_data("density_stats.csv")
            if csv_data is not None:
                st.download_button(
                    label="Download CSV",
                    data=csv_data.to_csv(index=False),
                    file_name=f"density_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No density data available for download.")
    
    with col2:
        if st.button("Clear History"):
            st.session_state.density_history = []
            st.success("History cleared successfully!")
            st.rerun()
