# Traffic Management System 🚦

A comprehensive web application built with Streamlit and YOLOv8 for intelligent traffic monitoring and analysis. This system provides real-time detection and analysis of vehicles, pedestrians, and traffic violations with adaptive traffic signal management.

## Features

### 🚥 Adaptive Traffic Signals
- Real-time vehicle density detection per lane
- Adaptive traffic light timing based on traffic density
- Visual traffic flow analysis with charts and statistics
- CSV export of density data for further analysis

### ⚠️ Traffic Violation Detection
- Red light violation detection
- Real-time violation logging with timestamps
- Comprehensive violation reports and analytics
- Downloadable CSV reports for compliance

### 🚗 Vehicle Category Classification
- Multi-class vehicle detection (cars, buses, trucks, motorcycles, bicycles)
- Real-time vehicle counting and categorization
- Statistical analysis with interactive charts
- Vehicle distribution analytics

### 🚶 Pedestrian Monitoring
- Crosswalk zone monitoring with configurable boundaries
- Safety alerts for unsafe crossing conditions
- Pedestrian count tracking and analytics
- Traffic light state integration for safety analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (optional, for live detection)
- GPU (recommended for faster processing)

### Setup Instructions

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd traffic-management-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the application**
   - Open your web browser
   - Navigate to `http://localhost:8501`
   - The application will automatically download the YOLOv8 model on first run

## Project Structure

```
traffic-management-system/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── pages/                # Individual feature pages
│   ├── __init__.py
│   ├── adaptive_traffic.py      # Adaptive traffic signals
│   ├── violation_detection.py   # Traffic violation detection
│   ├── vehicle_classification.py # Vehicle classification
│   └── pedestrian_monitoring.py # Pedestrian monitoring
├── utils/                # Utility functions
│   ├── __init__.py
│   └── detection_utils.py       # YOLOv8 detection utilities
├── models/               # YOLO model storage (auto-created)
├── data/                 # Sample videos/images for testing
├── └── sample_info.txt   # Information about sample data
└── reports/              # Generated CSV reports (auto-created)
```

## Usage Guide

### Getting Started

1. **Launch the application** using `streamlit run app.py`
2. **Select a feature** from the sidebar navigation
3. **Choose input method**: Upload video/image or use live webcam
4. **Configure settings** using the sidebar controls
5. **View results** in real-time with interactive visualizations

### Input Methods

#### Upload Video
- Supported formats: MP4, AVI, MOV, MKV
- Process frame by frame with navigation controls
- Automatic detection and analysis

#### Upload Image
- Supported formats: JPG, JPEG, PNG, BMP
- Instant detection and classification
- Static analysis with detailed results

#### Live Webcam (Simulated)
- Real-time processing capability
- Continuous monitoring and alerts
- Live statistics and reporting

### Features Overview

#### 1. Adaptive Traffic Signals
- **Purpose**: Optimize traffic flow using AI-driven density analysis
- **Key Features**:
  - Lane-by-lane vehicle counting
  - Dynamic signal timing calculation
  - Real-time traffic flow visualization
  - Historical trend analysis
- **Use Cases**: Smart city traffic management, intersection optimization

#### 2. Traffic Violation Detection
- **Purpose**: Automated violation detection and logging
- **Key Features**:
  - Red light violation detection
  - Violation timestamp and vehicle type logging
  - Comprehensive violation analytics
  - Export capabilities for law enforcement
- **Use Cases**: Traffic law enforcement, safety monitoring

#### 3. Vehicle Classification
- **Purpose**: Detailed vehicle type analysis and statistics
- **Key Features**:
  - Multi-class vehicle detection
  - Real-time counting and categorization
  - Statistical distribution analysis
  - Traffic composition insights
- **Use Cases**: Traffic planning, road capacity analysis

#### 4. Pedestrian Monitoring
- **Purpose**: Pedestrian safety and crosswalk monitoring
- **Key Features**:
  - Configurable crosswalk zone detection
  - Safety alert system
  - Traffic light integration
  - Pedestrian flow analysis
- **Use Cases**: Pedestrian safety, crosswalk optimization

## Configuration

### Detection Parameters
- **Confidence Threshold**: Adjust detection sensitivity (0.1-1.0)
- **Traffic Light State**: Set current traffic light status
- **Crosswalk Zones**: Define monitoring areas for pedestrian safety
- **Alert Settings**: Configure safety alerts and thresholds

### Export Options
All features support data export:
- **CSV Reports**: Detailed data for analysis
- **JSON Summaries**: Comprehensive statistics
- **Real-time Downloads**: Instant report generation

## Technical Details

### AI Model
- **YOLOv8**: State-of-the-art object detection
- **Classes Supported**: Vehicles (car, bus, truck, motorcycle, bicycle) and persons
- **Performance**: Real-time detection with high accuracy
- **Auto-download**: Model automatically downloaded on first run

### Data Processing
- **OpenCV**: Video and image processing
- **Pandas**: Data analysis and CSV generation
- **Plotly**: Interactive visualizations and charts
- **Streamlit**: Web interface and real-time updates

### Performance Optimization
- **GPU Support**: Automatic GPU acceleration when available
- **Batch Processing**: Efficient video frame processing
- **Memory Management**: Optimized for large video files
- **Caching**: Streamlit caching for improved performance

## Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Ensure internet connection
   - Check firewall settings
   - Manually download YOLOv8 model if needed

2. **Video Processing Slow**
   - Use GPU acceleration if available
   - Reduce video resolution
   - Process shorter video segments

3. **Webcam Not Working**
   - Check camera permissions
   - Ensure camera is not used by other applications
   - Try different camera indices

4. **Memory Issues**
   - Close other applications
   - Process smaller video files
   - Reduce batch size in processing

### Performance Tips

- **Use GPU**: Install CUDA-compatible PyTorch for faster processing
- **Optimize Videos**: Use compressed formats and reasonable resolutions
- **Batch Processing**: Process multiple files in sequence rather than simultaneously
- **Clear Cache**: Regularly clear Streamlit cache for optimal performance

## Sample Data

Add your own sample videos and images to the `data/` directory:
- Traffic intersection videos
- Pedestrian crossing footage
- Vehicle classification samples
- Violation detection examples

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or contributions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the configuration options

## Acknowledgments

- **Ultralytics**: YOLOv8 object detection model
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision library
- **Plotly**: Interactive visualization library

---

**Note**: This application is designed for educational and research purposes. For production deployment in traffic management systems, additional testing, validation, and compliance with local regulations may be required.
