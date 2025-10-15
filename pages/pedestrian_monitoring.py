import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from utils.detection_utils import VideoProcessor, CSVLogger, WebcamCapture

def show():
    """Pedestrian Monitoring page"""
    st.title("🚶 Pedestrian Monitoring")
    st.markdown("Monitor pedestrians in crosswalk zones with safety alerts")
    
    # Initialize session state
    if 'pedestrian_monitor' not in st.session_state:
        st.session_state.video_processor = VideoProcessor()
        st.session_state.csv_logger = CSVLogger()
        st.session_state.webcam_capture = WebcamCapture()
        st.session_state.pedestrian_history = []
        st.session_state.crossing_alerts = []
        st.session_state.crosswalk_zones = []
    
    # Sidebar controls
    # Input method selection
    input_method = st.sidebar.radio(
        "Select Input Method:",
        ["Upload Video", "Upload Image", "CC -Live Webcam"]
    )
    
    # Traffic light state for crosswalk safety
    traffic_light_state = st.sidebar.selectbox(
        "Traffic Light State:",
        ["red", "yellow", "green"]
    )
    
    # Crosswalk zone definition
    st.sidebar.subheader("Crosswalk Zone")
    zone_x1 = st.sidebar.slider("Zone X1", 0, 640, 200)
    zone_y1 = st.sidebar.slider("Zone Y1", 0, 480, 300)
    zone_x2 = st.sidebar.slider("Zone X2", 0, 640, 440)
    zone_y2 = st.sidebar.slider("Zone Y2", 0, 480, 400)
    
    crosswalk_zone = (zone_x1, zone_y1, zone_x2, zone_y2)
    
    # Alert settings
    st.sidebar.subheader("Alert Settings")
    enable_crossing_alerts = st.sidebar.checkbox("Enable Crossing Alerts", value=True)
    alert_on_green = st.sidebar.checkbox("Alert on Green Light Crossing", value=True)
    min_pedestrians_alert = st.sidebar.slider("Min Pedestrians for Alert", 1, 10, 3)
    
    # Detection parameters
    confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Pedestrian Detection")
        
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
                st.info("Processing video for pedestrian detection... This may take a moment.")
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
                        persons = current_frame_data['persons']
                        
                        # Analyze pedestrians in crosswalk zone
                        pedestrians_in_zone = []
                        for person in persons:
                            x1, y1, x2, y2 = person['bbox']
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # Check if pedestrian is in crosswalk zone
                            if (crosswalk_zone[0] <= center_x <= crosswalk_zone[2] and 
                                crosswalk_zone[1] <= center_y <= crosswalk_zone[3]):
                                pedestrians_in_zone.append(person)
                        
                        # Create annotated frame
                        display_frame = frame.copy()
                        
                        # Draw crosswalk zone
                        cv2.rectangle(display_frame, 
                                     (crosswalk_zone[0], crosswalk_zone[1]), 
                                     (crosswalk_zone[2], crosswalk_zone[3]), 
                                     (255, 255, 0), 2)  # Yellow crosswalk zone
                        cv2.putText(display_frame, "CROSSWALK ZONE", 
                                   (crosswalk_zone[0], crosswalk_zone[1] - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        # Draw all pedestrians
                        for person in persons:
                            x1, y1, x2, y2 = person['bbox']
                            color = (0, 255, 0)  # Green for safe pedestrians
                            
                            # Check if in crosswalk zone
                            if person in pedestrians_in_zone:
                                if traffic_light_state == "green" and alert_on_green:
                                    color = (0, 0, 255)  # Red for unsafe crossing
                                else:
                                    color = (255, 165, 0)  # Orange for crosswalk pedestrians
                            
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(display_frame, f"Person ({person['confidence']:.2f})", 
                                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Generate alerts
                        alerts = []
                        if enable_crossing_alerts:
                            if len(pedestrians_in_zone) >= min_pedestrians_alert:
                                alerts.append(f"High pedestrian density: {len(pedestrians_in_zone)} people in crosswalk")
                            
                            if traffic_light_state == "green" and pedestrians_in_zone and alert_on_green:
                                alerts.append("DANGER: Pedestrians crossing during green light!")
                        
                        # Convert BGR to RGB for display
                        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption=f"Frame {frame_idx + 1} - Pedestrians: {len(persons)} (In Zone: {len(pedestrians_in_zone)})")
                        
                        # Display alerts
                        if alerts:
                            for alert in alerts:
                                st.error(f"🚨 {alert}")
                        
                        # Store monitoring data
                        monitoring_data = {
                            'timestamp': datetime.now(),
                            'frame_number': frame_idx,
                            'total_pedestrians': len(persons),
                            'pedestrians_in_zone': len(pedestrians_in_zone),
                            'traffic_light_state': traffic_light_state,
                            'alerts': alerts
                        }
                        
                        # Update history
                        if monitoring_data not in st.session_state.pedestrian_history:
                            st.session_state.pedestrian_history.append(monitoring_data)
                        
                        # Log to CSV
                        st.session_state.csv_logger.log_pedestrian_data(
                            len(persons), 
                            alerts, 
                            monitoring_data['timestamp']
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
                persons = frame_data['persons']
                
                # Analyze pedestrians in crosswalk zone
                pedestrians_in_zone = []
                for person in persons:
                    x1, y1, x2, y2 = person['bbox']
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    if (crosswalk_zone[0] <= center_x <= crosswalk_zone[2] and 
                        crosswalk_zone[1] <= center_y <= crosswalk_zone[3]):
                        pedestrians_in_zone.append(person)
                
                # Create annotated frame
                display_frame = frame_data['frame'].copy()
                
                # Draw crosswalk zone
                cv2.rectangle(display_frame, 
                             (crosswalk_zone[0], crosswalk_zone[1]), 
                             (crosswalk_zone[2], crosswalk_zone[3]), 
                             (255, 255, 0), 2)
                cv2.putText(display_frame, "CROSSWALK ZONE", 
                           (crosswalk_zone[0], crosswalk_zone[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Draw pedestrians with color coding
                for person in persons:
                    x1, y1, x2, y2 = person['bbox']
                    color = (0, 255, 0)
                    
                    if person in pedestrians_in_zone:
                        if traffic_light_state == "green" and alert_on_green:
                            color = (0, 0, 255)
                        else:
                            color = (255, 165, 0)
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"Person ({person['confidence']:.2f})", 
                               (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Generate alerts
                alerts = []
                if enable_crossing_alerts:
                    if len(pedestrians_in_zone) >= min_pedestrians_alert:
                        alerts.append(f"High pedestrian density: {len(pedestrians_in_zone)} people in crosswalk")
                    
                    if traffic_light_state == "green" and pedestrians_in_zone and alert_on_green:
                        alerts.append("DANGER: Pedestrians crossing during green light!")
                
                # Display image
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Pedestrians: {len(persons)} (In Zone: {len(pedestrians_in_zone)})")
                
                # Display alerts
                if alerts:
                    for alert in alerts:
                        st.error(f"🚨 {alert}")
                else:
                    st.success("✅ No safety alerts detected.")
                
                # Store and log data
                monitoring_data = {
                    'timestamp': datetime.now(),
                    'frame_number': 0,
                    'total_pedestrians': len(persons),
                    'pedestrians_in_zone': len(pedestrians_in_zone),
                    'traffic_light_state': traffic_light_state,
                    'alerts': alerts
                }
                
                st.session_state.pedestrian_history.append(monitoring_data)
                st.session_state.csv_logger.log_pedestrian_data(
                    len(persons), 
                    alerts, 
                    monitoring_data['timestamp']
                )
        
        elif input_method == "CC -Live Webcam":
            st.info("📹 Live webcam monitoring - Click 'Capture Frame' to monitor pedestrians")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                capture_button = st.button("📸 Capture Frame", type="primary")
            with col_btn2:
                if st.button("🔄 Clear Display"):
                    if 'pedestrian_webcam_frame' in st.session_state:
                        del st.session_state.pedestrian_webcam_frame
                    st.rerun()
            
            if capture_button:
                with st.spinner("Capturing and monitoring pedestrians..."):
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
                        st.session_state.pedestrian_webcam_frame = frame_data
            
            # Display captured frame if available
            if 'pedestrian_webcam_frame' in st.session_state:
                frame_data = st.session_state.pedestrian_webcam_frame
                pedestrians = frame_data['persons']
                frame = frame_data['frame']
                
                # Check which pedestrians are in crosswalk zone
                pedestrians_in_zone = []
                for person in pedestrians:
                    x1, y1, x2, y2 = person['bbox']
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    if (crosswalk_zone[0] <= center_x <= crosswalk_zone[2] and 
                        crosswalk_zone[1] <= center_y <= crosswalk_zone[3]):
                        pedestrians_in_zone.append(person)
                
                # Draw on frame
                display_frame = frame.copy()
                
                # Draw crosswalk zone
                cv2.rectangle(display_frame, 
                             (crosswalk_zone[0], crosswalk_zone[1]), 
                             (crosswalk_zone[2], crosswalk_zone[3]), 
                             (255, 255, 0), 3)
                cv2.putText(display_frame, "CROSSWALK ZONE", 
                           (crosswalk_zone[0] + 10, crosswalk_zone[1] + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Draw pedestrians with color coding
                for person in pedestrians:
                    x1, y1, x2, y2 = person['bbox']
                    
                    # Determine color based on location and traffic light
                    if person in pedestrians_in_zone:
                        if traffic_light_state == "green" and alert_on_green:
                            color = (0, 0, 255)  # Red for danger
                            label = "DANGER"
                        else:
                            color = (255, 165, 0)  # Orange for in zone
                            label = "In Zone"
                    else:
                        color = (0, 255, 0)  # Green for safe
                        label = "Safe"
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(display_frame, f"{label} ({person['confidence']:.2f})", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Live Pedestrian Monitoring - Total: {len(pedestrians)}", use_container_width=True)
                
                # Show detection stats
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("Total Pedestrians", len(pedestrians))
                with col_stat2:
                    st.metric("In Crosswalk", len(pedestrians_in_zone))
                
                # Generate alerts
                alerts = []
                if enable_crossing_alerts:
                    if len(pedestrians_in_zone) >= min_pedestrians_alert:
                        alerts.append(f"High pedestrian density: {len(pedestrians_in_zone)} people in crosswalk")
                    
                    if traffic_light_state == "green" and pedestrians_in_zone and alert_on_green:
                        alerts.append("DANGER: Pedestrians crossing during green light!")
                
                # Display alerts
                if alerts:
                    for alert in alerts:
                        st.error(f"🚨 {alert}")
                else:
                    st.success("✅ No safety alerts")
                
                # Store monitoring data
                monitoring_data = {
                    'timestamp': datetime.now(),
                    'frame_number': 0,
                    'total_pedestrians': len(pedestrians),
                    'pedestrians_in_zone': len(pedestrians_in_zone),
                    'traffic_light_state': traffic_light_state,
                    'alerts': alerts
                }
                
                st.session_state.pedestrian_history.append(monitoring_data)
                st.session_state.csv_logger.log_pedestrian_data(
                    len(pedestrians), 
                    alerts, 
                    monitoring_data['timestamp']
                )
    
    with col2:
        st.subheader("Monitoring Statistics")
        
        if st.session_state.pedestrian_history:
            # Get latest monitoring data
            latest_data = st.session_state.pedestrian_history[-1]
            
            # Display current metrics
            st.metric("Total Pedestrians", latest_data['total_pedestrians'])
            st.metric("In Crosswalk Zone", latest_data['pedestrians_in_zone'])
            st.metric("Traffic Light", latest_data['traffic_light_state'].upper())
            
            # Safety status
            st.subheader("Safety Status")
            if latest_data['alerts']:
                st.error(f"⚠️ {len(latest_data['alerts'])} Alert(s)")
                for alert in latest_data['alerts']:
                    st.warning(f"• {alert}")
            else:
                st.success("✅ All Clear")
            
            # Crosswalk activity indicator
            st.subheader("Crosswalk Activity")
            activity_level = "Low"
            activity_color = "🟢"
            
            if latest_data['pedestrians_in_zone'] >= min_pedestrians_alert:
                activity_level = "High"
                activity_color = "🔴"
            elif latest_data['pedestrians_in_zone'] > 0:
                activity_level = "Medium"
                activity_color = "🟡"
            
            st.info(f"{activity_color} Activity Level: {activity_level}")
            
            # Recent activity summary
            st.subheader("Recent Activity")
            recent_data = st.session_state.pedestrian_history[-5:]  # Last 5 entries
            
            for i, data in enumerate(reversed(recent_data)):
                time_str = data['timestamp'].strftime('%H:%M:%S')
                alert_count = len(data['alerts'])
                
                if alert_count > 0:
                    st.error(f"{time_str}: {data['pedestrians_in_zone']} in zone, {alert_count} alerts")
                else:
                    st.success(f"{time_str}: {data['pedestrians_in_zone']} in zone, no alerts")
        else:
            st.info("No monitoring data available yet.")
    
    # Analytics section
    st.markdown("---")
    st.subheader("📊 Pedestrian Analytics")
    
    if len(st.session_state.pedestrian_history) > 1:
        # Create dataframe for analysis
        df_history = pd.DataFrame(st.session_state.pedestrian_history)
        
        # Time series plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Pedestrian count over time
            fig_pedestrians = go.Figure()
            
            fig_pedestrians.add_trace(go.Scatter(
                x=df_history['timestamp'],
                y=df_history['total_pedestrians'],
                mode='lines+markers',
                name='Total Pedestrians',
                line=dict(color='blue', width=2)
            ))
            
            fig_pedestrians.add_trace(go.Scatter(
                x=df_history['timestamp'],
                y=df_history['pedestrians_in_zone'],
                mode='lines+markers',
                name='In Crosswalk Zone',
                line=dict(color='orange', width=2)
            ))
            
            fig_pedestrians.update_layout(
                title="Pedestrian Count Over Time",
                xaxis_title="Time",
                yaxis_title="Number of Pedestrians",
                height=400
            )
            st.plotly_chart(fig_pedestrians, use_container_width=True)
        
        with col2:
            # Alert frequency over time
            alert_counts = [len(data['alerts']) for data in st.session_state.pedestrian_history]
            
            fig_alerts = go.Figure()
            fig_alerts.add_trace(go.Scatter(
                x=df_history['timestamp'],
                y=alert_counts,
                mode='lines+markers',
                name='Alerts',
                line=dict(color='red', width=2),
                fill='tozeroy'
            ))
            
            fig_alerts.update_layout(
                title="Safety Alerts Over Time",
                xaxis_title="Time",
                yaxis_title="Number of Alerts",
                height=400
            )
            st.plotly_chart(fig_alerts, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_pedestrians = df_history['total_pedestrians'].mean()
            st.metric("Avg Pedestrians", f"{avg_pedestrians:.1f}")
        
        with col2:
            max_in_zone = df_history['pedestrians_in_zone'].max()
            st.metric("Peak Crosswalk Usage", max_in_zone)
        
        with col3:
            total_alerts = sum(alert_counts)
            st.metric("Total Alerts", total_alerts)
        
        with col4:
            crosswalk_usage = (df_history['pedestrians_in_zone'] > 0).sum()
            usage_percentage = (crosswalk_usage / len(df_history)) * 100
            st.metric("Crosswalk Usage", f"{usage_percentage:.1f}%")
        
        # Traffic light state analysis
        if 'traffic_light_state' in df_history.columns:
            st.subheader("Traffic Light Analysis")
            
            light_state_counts = df_history['traffic_light_state'].value_counts()
            
            fig_lights = px.pie(
                values=light_state_counts.values,
                names=light_state_counts.index,
                title="Monitoring Time by Traffic Light State",
                color_discrete_map={
                    'red': '#ff0000',
                    'yellow': '#ffff00', 
                    'green': '#00ff00'
                }
            )
            st.plotly_chart(fig_lights, use_container_width=True)
    
    # Export options
    st.markdown("---")
    st.subheader("📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Download Monitoring Report"):
            csv_data = st.session_state.csv_logger.get_csv_download_data("pedestrian_data.csv")
            if csv_data is not None:
                st.download_button(
                    label="Download CSV",
                    data=csv_data.to_csv(index=False),
                    file_name=f"pedestrian_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No monitoring data available for download.")
    
    with col2:
        if st.button("Clear Monitoring History"):
            st.session_state.pedestrian_history = []
            st.session_state.crossing_alerts = []
            st.success("Monitoring history cleared successfully!")
            st.rerun()
    
    with col3:
        if st.button("Generate Safety Report"):
            if st.session_state.pedestrian_history:
                # Generate comprehensive safety report
                total_observations = len(st.session_state.pedestrian_history)
                total_alerts = sum(len(data['alerts']) for data in st.session_state.pedestrian_history)
                
                safety_score = max(0, 100 - (total_alerts / total_observations * 100)) if total_observations > 0 else 100
                
                safety_report = {
                    'monitoring_period': {
                        'start': st.session_state.pedestrian_history[0]['timestamp'].isoformat(),
                        'end': st.session_state.pedestrian_history[-1]['timestamp'].isoformat(),
                        'total_observations': total_observations
                    },
                    'pedestrian_statistics': {
                        'total_pedestrians_detected': sum(data['total_pedestrians'] for data in st.session_state.pedestrian_history),
                        'crosswalk_interactions': sum(data['pedestrians_in_zone'] for data in st.session_state.pedestrian_history),
                        'average_pedestrians_per_observation': np.mean([data['total_pedestrians'] for data in st.session_state.pedestrian_history])
                    },
                    'safety_metrics': {
                        'total_alerts': total_alerts,
                        'safety_score': f"{safety_score:.1f}%",
                        'alert_frequency': f"{total_alerts / total_observations:.2f}" if total_observations > 0 else "0"
                    }
                }
                
                st.download_button(
                    label="Download Safety Report JSON",
                    data=pd.Series(safety_report).to_json(indent=2),
                    file_name=f"safety_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                # Display safety score
                st.success(f"🛡️ Safety Score: {safety_score:.1f}%")
            else:
                st.warning("No data available for safety report.")
