import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os
from pathlib import Path

class YOLODetector:
    """YOLOv8 detection utility class for traffic management"""
    
    def __init__(self, model_path="yolov8n.pt"):
        """Initialize YOLO detector with model"""
        self.model = YOLO(model_path)
        self.vehicle_classes = ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']
        self.person_class = 'person'
        
    def detect_objects(self, frame, conf_threshold=0.5):
        """Detect objects in frame and return results"""
        results = self.model(frame, conf=conf_threshold)
        return results[0] if results else None
    
    def get_vehicle_detections(self, results):
        """Filter detections for vehicles only"""
        if not results or not results.boxes:
            return []
        
        vehicle_detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            
            if class_name in self.vehicle_classes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                vehicle_detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class': class_name,
                    'class_id': class_id
                })
        
        return vehicle_detections
    
    def get_person_detections(self, results):
        """Filter detections for persons only"""
        if not results or not results.boxes:
            return []
        
        person_detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            
            if class_name == self.person_class:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                person_detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class': class_name,
                    'class_id': class_id
                })
        
        return person_detections
    
    def draw_detections(self, frame, detections, color=(0, 255, 0)):
        """Draw bounding boxes and labels on frame"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            class_name = detection['class']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame

class VideoProcessor:
    """Video processing utility for traffic analysis"""
    
    def __init__(self):
        self.detector = YOLODetector()
    
    def process_video_file(self, video_path, progress_callback=None):
        """Process uploaded video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = self.detector.detect_objects(frame)
            vehicle_detections = self.detector.get_vehicle_detections(results)
            person_detections = self.detector.get_person_detections(results)
            
            # Annotate frame
            annotated_frame = self.detector.draw_detections(frame, vehicle_detections)
            annotated_frame = self.detector.draw_detections(annotated_frame, person_detections, color=(255, 0, 0))
            
            frames.append({
                'frame': annotated_frame,
                'vehicles': vehicle_detections,
                'persons': person_detections,
                'frame_number': frame_count
            })
            
            frame_count += 1
            
            # Update progress
            if progress_callback:
                progress_callback(frame_count / total_frames)
        
        cap.release()
        return frames
    
    def process_webcam_frame(self, frame):
        """Process single webcam frame"""
        results = self.detector.detect_objects(frame)
        vehicle_detections = self.detector.get_vehicle_detections(results)
        person_detections = self.detector.get_person_detections(results)
        
        # Annotate frame
        annotated_frame = self.detector.draw_detections(frame, vehicle_detections)
        annotated_frame = self.detector.draw_detections(annotated_frame, person_detections, color=(255, 0, 0))
        
        return {
            'frame': annotated_frame,
            'vehicles': vehicle_detections,
            'persons': person_detections
        }

class TrafficAnalyzer:
    """Traffic analysis utilities"""
    
    @staticmethod
    def auto_detect_lanes(detections, frame_width, frame_height):
        """Automatically detect number of lanes based on vehicle distribution"""
        if not detections or len(detections) == 0:
            return 2  # Default to 2 lanes if no vehicles detected
        
        # Get x-coordinates of vehicle centers
        vehicle_x_positions = []
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x = (x1 + x2) // 2
            vehicle_x_positions.append(center_x)
        
        if len(vehicle_x_positions) < 2:
            return 2  # Default to 2 lanes if too few vehicles
        
        # Sort positions
        vehicle_x_positions.sort()
        
        # Calculate gaps between vehicles
        gaps = []
        for i in range(1, len(vehicle_x_positions)):
            gap = vehicle_x_positions[i] - vehicle_x_positions[i-1]
            gaps.append(gap)
        
        if not gaps:
            return 2
        
        # Use clustering approach - find significant gaps
        avg_gap = sum(gaps) / len(gaps)
        significant_gaps = [g for g in gaps if g > avg_gap * 1.5]
        
        # Number of lanes = number of significant gaps + 1
        detected_lanes = len(significant_gaps) + 1
        
        # Constrain between 2 and 6 lanes
        detected_lanes = max(2, min(6, detected_lanes))
        
        # If we have many vehicles spread across width, estimate based on density
        if len(vehicle_x_positions) >= 4:
            # Calculate approximate lane width based on typical vehicle distribution
            spread = max(vehicle_x_positions) - min(vehicle_x_positions)
            if spread > frame_width * 0.7:  # Vehicles spread across most of width
                # Estimate lanes based on vehicle density
                estimated_lanes = max(2, min(6, len(vehicle_x_positions) // 2))
                detected_lanes = max(detected_lanes, estimated_lanes)
        
        return detected_lanes
    
    @staticmethod
    def calculate_vehicle_density(detections, frame_width, frame_height, lanes=4):
        """Calculate vehicle density per lane"""
        if not detections:
            return [0] * lanes
        
        lane_width = frame_width // lanes
        lane_counts = [0] * lanes
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x = (x1 + x2) // 2
            lane_idx = min(center_x // lane_width, lanes - 1)
            lane_counts[lane_idx] += 1
        
        return lane_counts
    
    @staticmethod
    def calculate_adaptive_timing(lane_densities, base_time=30, max_time=60):
        """Calculate adaptive traffic light timing based on density"""
        total_vehicles = sum(lane_densities)
        if total_vehicles == 0:
            return [base_time] * len(lane_densities)
        
        timings = []
        for density in lane_densities:
            ratio = density / total_vehicles
            timing = base_time + (max_time - base_time) * ratio
            timings.append(int(timing))
        
        return timings
    
    @staticmethod
    def detect_violations(detections, traffic_light_state="red", crosswalk_zone=None):
        """Detect traffic violations"""
        violations = []
        
        if traffic_light_state == "red":
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                center_y = (y1 + y2) // 2
                
                # Simple violation detection (vehicles crossing stop line during red)
                if center_y > 300:  # Assuming stop line at y=300
                    violations.append({
                        'type': 'red_light_violation',
                        'vehicle_class': detection['class'],
                        'bbox': detection['bbox'],
                        'timestamp': datetime.now(),
                        'confidence': detection['confidence']
                    })
        
        return violations

class WebcamCapture:
    """Webcam capture utility for live video feed"""
    
    def __init__(self, camera_index=0):
        """Initialize webcam capture"""
        self.camera_index = camera_index
        self.detector = YOLODetector()
    
    def capture_frame(self):
        """Capture a single frame from webcam"""
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            return None, "Unable to access webcam. Please check camera permissions."
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None, "Failed to capture frame from webcam."
        
        return frame, None
    
    def process_frame(self, frame):
        """Process captured frame with YOLO detection"""
        results = self.detector.detect_objects(frame)
        vehicle_detections = self.detector.get_vehicle_detections(results)
        person_detections = self.detector.get_person_detections(results)
        
        # Annotate frame
        annotated_frame = self.detector.draw_detections(frame, vehicle_detections)
        annotated_frame = self.detector.draw_detections(annotated_frame, person_detections, color=(255, 0, 0))
        
        return {
            'frame': annotated_frame,
            'vehicles': vehicle_detections,
            'persons': person_detections
        }
    
    def capture_and_process(self):
        """Capture and process a single frame"""
        frame, error = self.capture_frame()
        if error:
            return None, error
        
        processed_data = self.process_frame(frame)
        return processed_data, None

class CSVLogger:
    """CSV logging utility for reports and data export"""
    
    def __init__(self, reports_dir="reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
    
    def log_vehicle_counts(self, vehicle_counts, timestamp=None):
        """Log vehicle count data to CSV"""
        if timestamp is None:
            timestamp = datetime.now()
        
        df = pd.DataFrame([{
            'timestamp': timestamp,
            **vehicle_counts
        }])
        
        csv_path = self.reports_dir / "vehicle_counts.csv"
        
        if csv_path.exists():
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
    
    def log_violations(self, violations):
        """Log traffic violations to CSV"""
        if not violations:
            return
        
        violation_data = []
        for violation in violations:
            violation_data.append({
                'timestamp': violation['timestamp'],
                'violation_type': violation['type'],
                'vehicle_class': violation['vehicle_class'],
                'confidence': violation['confidence'],
                'bbox_x1': violation['bbox'][0],
                'bbox_y1': violation['bbox'][1],
                'bbox_x2': violation['bbox'][2],
                'bbox_y2': violation['bbox'][3]
            })
        
        df = pd.DataFrame(violation_data)
        csv_path = self.reports_dir / "violations.csv"
        
        if csv_path.exists():
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
    
    def log_density_stats(self, lane_densities, timestamp=None):
        """Log traffic density statistics"""
        if timestamp is None:
            timestamp = datetime.now()
        
        df = pd.DataFrame([{
            'timestamp': timestamp,
            'lane_1_density': lane_densities[0] if len(lane_densities) > 0 else 0,
            'lane_2_density': lane_densities[1] if len(lane_densities) > 1 else 0,
            'lane_3_density': lane_densities[2] if len(lane_densities) > 2 else 0,
            'lane_4_density': lane_densities[3] if len(lane_densities) > 3 else 0,
            'total_vehicles': sum(lane_densities)
        }])
        
        csv_path = self.reports_dir / "density_stats.csv"
        
        if csv_path.exists():
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
    
    def log_pedestrian_data(self, pedestrian_count, crossing_alerts, timestamp=None):
        """Log pedestrian monitoring data"""
        if timestamp is None:
            timestamp = datetime.now()
        
        df = pd.DataFrame([{
            'timestamp': timestamp,
            'pedestrian_count': pedestrian_count,
            'crossing_alerts': len(crossing_alerts),
            'alert_details': str(crossing_alerts) if crossing_alerts else ""
        }])
        
        csv_path = self.reports_dir / "pedestrian_data.csv"
        
        if csv_path.exists():
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
    
    def get_csv_download_data(self, filename):
        """Get CSV data for download"""
        csv_path = self.reports_dir / filename
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return None
