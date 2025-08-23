"""Utility functions for the Flow Prediction Lab"""
import re
import logging

def sanitize_for_logging(text):
    """Sanitize text for safe logging"""
    if not isinstance(text, str):
        text = str(text)
    # Remove newlines and control characters
    return re.sub(r'[\r\n\t]', ' ', text).strip()

def validate_coordinates(lat, lon):
    """Validate latitude and longitude coordinates"""
    try:
        lat, lon = float(lat), float(lon)
        if not (-90 <= lat <= 90):
            raise ValueError(f"Invalid latitude: {lat}")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Invalid longitude: {lon}")
        return lat, lon
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid coordinates: {e}")

def parse_coordinates_safe(coord_str):
    """Safely parse coordinate string"""
    try:
        clean_str = coord_str.strip('()"\'')
        lat_str, lon_str = clean_str.split()
        return validate_coordinates(lat_str, lon_str)
    except (ValueError, AttributeError, TypeError):
        return None, None

def get_confidence_class(confidence, high_threshold=0.7, medium_threshold=0.4):
    """Get confidence class based on thresholds"""
    if confidence > high_threshold:
        return 'confidence-high'
    elif confidence > medium_threshold:
        return 'confidence-medium'
    else:
        return 'confidence-low'