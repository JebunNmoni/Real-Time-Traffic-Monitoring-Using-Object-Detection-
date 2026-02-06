# src/tracking/byte_tracker.py
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional
import cv2

class EnhancedByteTracker:
    """
    Enhanced ByteTrack with feature matching for vehicle tracking
    Includes Re-ID capabilities and occlusion handling
    """
    
    def __init__(self, config):
        self.config = config
        self.trackers = []
        self.next_id = 1
        self.frame_count = 0
        
        # Feature database for Re-ID
        self.feature_db = {}  # track_id -> list of feature vectors
        self.max_features_per_track = 50
        
        # Tracking parameters
        self.track_thresh = config.get('track_thresh', 0.5)
        self.match_thresh = config.get('match_thresh', 0.8)
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 3)
        
        # Feature extractor for Re-ID
        self.feature_extractor = self._init_feature_extractor()
        
    def _init_feature_extractor(self):
        """Initialize feature extractor for Re-ID"""
        try:
            # Use OpenCV's ORB for feature extraction
            return cv2.ORB_create(nfeatures=100)
        except:
            # Fallback to simple color histogram
            return None
    
    def extract_features(self, frame, bbox):
        """Extract features from bounding box for Re-ID"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure bbox is within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None
        
        if self.feature_extractor is not None and hasattr(self.feature_extractor, 'detectAndCompute'):
            # Use ORB features
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, descriptors = self.feature_extractor.detectAndCompute(gray, None)
            if descriptors is not None:
                return descriptors.flatten()[:128]  # Limit feature size
        
        # Fallback: Color histogram
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    
    def match_features(self, features1, features2):
        """Match two feature vectors"""
        if features1 is None or features2 is None:
            return 0.0
        
        # Use cosine similarity or histogram intersection
        if len(features1) == len(features2):
            # Histogram intersection
            intersection = np.minimum(features1, features2).sum()
            union = np.maximum(features1, features2).sum()
            if union > 0:
                return intersection / union
        
        return 0.0
    
    def update(self, detections, frame):
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dictionaries
            frame: Current video frame
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Separate high and low confidence detections
        high_conf_dets = []
        low_conf_dets = []
        
        for det in detections:
            if det['confidence'] >= self.track_thresh:
                high_conf_dets.append(det)
            else:
                low_conf_dets.append(det)
        
        # Predict new locations for existing trackers
        for tracker in self.trackers:
            tracker.predict()
        
        # Match high confidence detections
        matched_pairs, unmatched_dets, unmatched_trks = self._match_detections_to_tracks(
            high_conf_dets, self.trackers, frame
        )
        
        # Update matched trackers
        for det_idx, trk_idx in matched_pairs:
            det = high_conf_dets[det_idx]
            tracker = self.trackers[trk_idx]
            
            # Update tracker state
            tracker.update(det['bbox'])
            
            # Update features for Re-ID
            features = self.extract_features(frame, det['bbox'])
            if features is not None:
                tracker_id = tracker.id
                if tracker_id not in self.feature_db:
                    self.feature_db[tracker_id] = []
                self.feature_db[tracker_id].append(features)
                # Keep only recent features
                if len(self.feature_db[tracker_id]) > self.max_features_per_track:
                    self.feature_db[tracker_id].pop(0)
        
        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            det = high_conf_dets[det_idx]
            new_tracker = KalmanBoxTracker(
                det['bbox'], 
                self.next_id, 
                det['class_id']
            )
            self.trackers.append(new_tracker)
            
            # Store initial features
            features = self.extract_features(frame, det['bbox'])
            if features is not None:
                self.feature_db[self.next_id] = [features]
            
            self.next_id += 1
        
        # Try to match low confidence detections with unmatched trackers
        if low_conf_dets and unmatched_trks:
            low_matched, _, _ = self._match_detections_to_tracks(
                low_conf_dets, 
                [self.trackers[i] for i in unmatched_trks],
                frame,
                use_features=True
            )
            
            for det_idx, trk_idx in low_matched:
                original_trk_idx = unmatched_trks[trk_idx]
                det = low_conf_dets[det_idx]
                tracker = self.trackers[original_trk_idx]
                
                tracker.update(det['bbox'])
        
        # Remove dead trackers
        active_tracks = []
        for tracker in self.trackers:
            if tracker.time_since_update < self.max_age:
                active_tracks.append(tracker)
            else:
                # Remove from feature database
                if tracker.id in self.feature_db:
                    del self.feature_db[tracker.id]
        
        self.trackers = active_tracks
        
        # Prepare output
        output_tracks = []
        for tracker in self.trackers:
            if tracker.time_since_update == 0 and tracker.hits >= self.min_hits:
                state = tracker.get_state()
                output_tracks.append({
                    'id': tracker.id,
                    'bbox': state.flatten().tolist(),
                    'class_id': tracker.class_id,
                    'confidence': tracker.confidence if hasattr(tracker, 'confidence') else 1.0,
                    'age': tracker.age,
                    'hits': tracker.hits
                })
        
        return output_tracks
    
    def _match_detections_to_tracks(self, detections, trackers, frame, use_features=False):
        """
        Match detections to tracks using IoU and optionally features
        
        Returns:
            matched_pairs, unmatched_dets, unmatched_trks
        """
        if len(detections) == 0 or len(trackers) == 0:
            return [], list(range(len(detections))), list(range(len(trackers)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        for i, det in enumerate(detections):
            for j, trk in enumerate(trackers):
                iou_matrix[i, j] = self._iou(det['bbox'], trk.get_state().flatten())
        
        # Calculate feature similarity matrix if needed
        feature_matrix = None
        if use_features:
            feature_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
            for i, det in enumerate(detections):
                det_features = self.extract_features(frame, det['bbox'])
                for j, trk in enumerate(trackers):
                    trk_id = trk.id
                    if trk_id in self.feature_db and det_features is not None:
                        # Average similarity with stored features
                        similarities = []
                        for stored_features in self.feature_db[trk_id]:
                            sim = self.match_features(det_features, stored_features)
                            similarities.append(sim)
                        if similarities:
                            feature_matrix[i, j] = np.mean(similarities)
        
        # Combine IoU and feature similarity
        if feature_matrix is not None:
            # Weighted combination
            combined_matrix = 0.7 * iou_matrix + 0.3 * feature_matrix
        else:
            combined_matrix = iou_matrix
        
        # Hungarian algorithm for matching
        det_indices, trk_indices = linear_sum_assignment(-combined_matrix)
        
        # Filter matches
        matched_pairs = []
        unmatched_dets = []
        unmatched_trks = []
        
        for d in range(len(detections)):
            if d not in det_indices:
                unmatched_dets.append(d)
        
        for t in range(len(trackers)):
            if t not in trk_indices:
                unmatched_trks.append(t)
        
        for d_idx, t_idx in zip(det_indices, trk_indices):
            if combined_matrix[d_idx, t_idx] < self.match_thresh:
                unmatched_dets.append(d_idx)
                unmatched_trks.append(t_idx)
            else:
                matched_pairs.append((d_idx, t_idx))
        
        return matched_pairs, unmatched_dets, unmatched_trks
    
    @staticmethod
    def _iou(bbox1, bbox2):
        """Calculate Intersection over Union"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class KalmanBoxTracker:
    """Kalman Filter tracker for bounding boxes"""
    
    def __init__(self, bbox, track_id, class_id):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        
        # Initialize with first detection
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        
        # Tracking variables
        self.id = track_id
        self.class_id = class_id
        self.time_since_update = 0
        self.hits = 1
        self.age = 0
        self.history = []
        self.confidence = 1.0
        
    def convert_bbox_to_z(self, bbox):
        """Convert bounding box to measurement vector"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2
        y = bbox[1] + h/2
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def convert_x_to_bbox(self):
        """Convert state vector to bounding box"""
        w = np.sqrt(self.kf.x[2] * self.kf.x[3])
        h = self.kf.x[2] / w
        x1 = self.kf.x[0] - w/2
        y1 = self.kf.x[1] - h/2
        x2 = self.kf.x[0] + w/2
        y2 = self.kf.x[1] + h/2
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    
    def predict(self):
        """Predict next state"""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hits = 0
        
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox())
        
        return self.history[-1]
    
    def update(self, bbox):
        """Update with new measurement"""
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(self.convert_bbox_to_z(bbox))
    
    def get_state(self):
        """Get current state"""
        return self.convert_x_to_bbox()