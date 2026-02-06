# scripts/evaluate_performance.py
import cv2
import time
import numpy as np
import pandas as pd
from datetime import datetime
from src.detection.detector import VehicleDetector
from src.tracking.byte_tracker import EnhancedByteTracker

class PerformanceEvaluator:
    """Comprehensive performance evaluation for traffic monitoring system"""
    
    def __init__(self, config_path="config/config.yaml"):
        self.config = self.load_config(config_path)
        self.detector = VehicleDetector(self.config)
        self.tracker = EnhancedByteTracker(self.config.get('tracking', {}))
        self.metrics = {
            'fps': [],
            'inference_time': [],
            'tracking_time': [],
            'detection_count': [],
            'track_count': [],
            'memory_usage': [],
            'cpu_usage': []
        }
        
    def evaluate_video(self, video_path, duration=60):
        """
        Evaluate system performance on a video
        
        Args:
            video_path: Path to video file
            duration: Evaluation duration in seconds
        """
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
        
        print(f"Evaluating performance on {video_path}")
        print("=" * 60)
        
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened() and (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            # Run detection
            detections = self.detector.detect(frame)
            detection_time = time.time() - frame_start
            
            # Run tracking
            track_start = time.time()
            tracks = self.tracker.update(detections, frame)
            tracking_time = time.time() - track_start
            
            # Calculate FPS
            total_time = time.time() - frame_start
            fps = 1.0 / total_time if total_time > 0 else 0
            
            # Record metrics
            self.metrics['fps'].append(fps)
            self.metrics['inference_time'].append(detection_time * 1000)  # ms
            self.metrics['tracking_time'].append(tracking_time * 1000)  # ms
            self.metrics['detection_count'].append(len(detections))
            self.metrics['track_count'].append(len(tracks))
            
            frame_count += 1
            
            # Print progress
            if frame_count % 30 == 0:
                avg_fps = np.mean(self.metrics['fps'][-30:])
                print(f"Processed {frame_count} frames | Avg FPS: {avg_fps:.1f}")
        
        cap.release()
        
        # Calculate and print summary
        self.print_summary()
        
        # Save results
        self.save_results(video_path)
    
    def print_summary(self):
        """Print performance summary"""
        
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        
        if not self.metrics['fps']:
            print("No metrics collected")
            return
        
        print(f"Total frames processed: {len(self.metrics['fps'])}")
        print(f"Average FPS: {np.mean(self.metrics['fps']):.2f}")
        print(f"Min FPS: {np.min(self.metrics['fps']):.2f}")
        print(f"Max FPS: {np.max(self.metrics['fps']):.2f}")
        print(f"Std FPS: {np.std(self.metrics['fps']):.2f}")
        print()
        print(f"Average inference time: {np.mean(self.metrics['inference_time']):.2f} ms")
        print(f"Average tracking time: {np.mean(self.metrics['tracking_time']):.2f} ms")
        print()
        print(f"Average detections per frame: {np.mean(self.metrics['detection_count']):.1f}")
        print(f"Average tracks per frame: {np.mean(self.metrics['track_count']):.1f}")
        print("=" * 60)
    
    def save_results(self, video_path):
        """Save evaluation results to CSV"""
        
        # Create DataFrame
        df = pd.DataFrame(self.metrics)
        
        # Add summary statistics
        summary = {
            'metric': [
                'total_frames', 'avg_fps', 'min_fps', 'max_fps', 'std_fps',
                'avg_inference_ms', 'avg_tracking_ms',
                'avg_detections', 'avg_tracks'
            ],
            'value': [
                len(self.metrics['fps']),
                np.mean(self.metrics['fps']),
                np.min(self.metrics['fps']),
                np.max(self.metrics['fps']),
                np.std(self.metrics['fps']),
                np.mean(self.metrics['inference_time']),
                np.mean(self.metrics['tracking_time']),
                np.mean(self.metrics['detection_count']),
                np.mean(self.metrics['track_count'])
            ]
        }
        
        summary_df = pd.DataFrame(summary)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/performance_{timestamp}.csv"
        summary_file = f"results/summary_{timestamp}.csv"
        
        os.makedirs("results", exist_ok=True)
        
        df.to_csv(results_file, index=False)
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Detailed results saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")
    
    def compare_models(self, model_paths, test_video):
        """
        Compare performance of multiple models
        
        Args:
            model_paths: List of model paths to compare
            test_video: Test video path
        """
        
        print("=" * 60)
        print("MODEL COMPARISON TEST")
        print("=" * 60)
        
        results = {}
        
        for model_path in model_paths:
            print(f"\nTesting model: {model_path}")
            
            # Update model path in config
            self.config['model']['path'] = model_path
            
            # Reinitialize detector
            self.detector = VehicleDetector(self.config)
            
            # Run evaluation
            self.evaluate_video(test_video, duration=30)
            
            # Store results
            results[model_path] = {
                'avg_fps': np.mean(self.metrics['fps']),
                'avg_inference_ms': np.mean(self.metrics['inference_time']),
                'avg_detections': np.mean(self.metrics['detection_count'])
            }
        
        # Print comparison
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        
        for model_path, metrics in results.items():
            model_name = os.path.basename(model_path)
            print(f"\n{model_name}:")
            print(f"  Average FPS: {metrics['avg_fps']:.2f}")
            print(f"  Average inference: {metrics['avg_inference_ms']:.2f} ms")
            print(f"  Average detections: {metrics['avg_detections']:.1f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate traffic monitoring system performance")
    parser.add_argument("--video", type=str, required=True, help="Test video path")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--compare", nargs='+', help="Compare multiple models")
    
    args = parser.parse_args()
    
    evaluator = PerformanceEvaluator()
    
    if args.compare:
        evaluator.compare_models(args.compare, args.video)
    else:
        evaluator.evaluate_video(args.video, args.duration)