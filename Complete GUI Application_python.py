# main.py
import sys
import argparse
import yaml
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from src.detection.detector import VehicleDetector
from src.tracking.byte_tracker import EnhancedByteTracker
from src.gui.main_window import TrafficMonitoringGUI

class TrafficMonitoringApp:
    """Main application class"""
    
    def __init__(self, config_path="config/config.yaml"):
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize components
        self.detector = VehicleDetector(self.config)
        self.tracker = EnhancedByteTracker(self.config.get('tracking', {}))
        
        # Initialize GUI
        self.app = QApplication(sys.argv)
        self.gui = TrafficMonitoringGUI(self.detector, self.tracker, self.config)
        
        # Set application style
        self.set_application_style()
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set default values if not present
        defaults = {
            'model': {
                'path': 'models/yolov8n.pt',
                'input_size': [640, 640],
                'confidence_threshold': 0.5,
                'iou_threshold': 0.45
            },
            'tracking': {
                'track_thresh': 0.5,
                'match_thresh': 0.8,
                'max_age': 30,
                'min_hits': 3
            },
            'gui': {
                'window_size': [1600, 900],
                'update_interval': 30,
                'theme': 'dark'
            }
        }
        
        # Merge defaults
        for section, values in defaults.items():
            if section not in config:
                config[section] = values
            else:
                for key, value in values.items():
                    if key not in config[section]:
                        config[section][key] = value
        
        return config
    
    def set_application_style(self):
        """Set application-wide styling"""
        # Set font
        font = QFont("Segoe UI", 10)
        self.app.setFont(font)
        
        # Set style sheet
        if self.config['gui'].get('theme', 'dark') == 'dark':
            dark_stylesheet = """
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QPushButton {
                background-color: #3c3c3c;
                border: 1px solid #555;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
            QLabel {
                color: #ffffff;
            }
            QGroupBox {
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #555;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0078d7;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 4px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
            }
            """
            self.app.setStyleSheet(dark_stylesheet)
    
    def run(self, video_source=None):
        """Run the application"""
        self.gui.show()
        
        # If video source provided, load it
        if video_source:
            self.gui.load_video(video_source)
        
        return self.app.exec()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Traffic Monitoring System with AI Object Detection and Tracking"
    )
    
    parser.add_argument(
        "--video", 
        type=str, 
        help="Video file path or camera index (0 for webcam)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        help="Model file path"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="output.mp4",
        help="Output video file path"
    )
    
    args = parser.parse_args()
    
    # Create application
    app = TrafficMonitoringApp(args.config)
    
    # Override model path if provided
    if args.model:
        app.config['model']['path'] = args.model
    
    # Run application
    sys.exit(app.run(args.video))

if __name__ == "__main__":
    main()