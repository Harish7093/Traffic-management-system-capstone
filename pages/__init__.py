"""Pages module for Traffic Management System
Exports all page modules for easy importing
"""

from . import adaptive_traffic
from . import violation_detection
from . import vehicle_classification
from . import pedestrian_monitoring

__all__ = [
    'adaptive_traffic',
    'violation_detection',
    'vehicle_classification',
    'pedestrian_monitoring',
    
]
