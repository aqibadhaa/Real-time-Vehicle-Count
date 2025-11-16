"""
Vehicle Counter with YOLOv8 - ULTRA FIXED VERSION
Fix: Counting lebih sensitif & tracking lebih stabil
"""

from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import time

class VehicleCounter:
    def __init__(self, model_path='yolov8m.pt', line_position=0.3, device='cuda'):
        """
        Initialize Vehicle Counter
        
        Args:
            model_path: Path ke model YOLO (yolov8n.pt, yolov8s.pt, dll)
            line_position: Posisi garis counting (0.0 - 1.0, dari atas ke bawah)
            device: 'cuda' untuk GPU, 'cpu' untuk CPU
        """
        print(f"Loading YOLO model on {device}...")
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device
        self.line_position = line_position
        
        # Tracking data
        self.tracked_objects = {}
        self.counted_ids = set()
        self.count = 0
        
        # Vehicle classes dari COCO dataset
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
        # Untuk FPS calculation
        self.prev_time = 0
        
        # Statistik per class
        self.class_counts = defaultdict(int)
        
    def check_line_crossing(self, bbox, track_id, frame_height, cls):
        """
        Cek apakah kendaraan melewati garis (ULTRA SENSITIVE VERSION)
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            track_id: ID tracking objek
            frame_height: Tinggi frame
            cls: Class ID kendaraan
        
        Returns:
            bool: True jika melewati garis
        """
        x1, y1, x2, y2 = bbox
        
        # Gunakan CENTER point (lebih stabil dari bottom)
        center_y = (y1 + y2) / 2
        line_y = self.line_position * frame_height
        
        # Margin untuk line crossing (LEBIH BESAR = LEBIH SENSITIF)
        margin = frame_height * 0.03  # Naik dari 0.02 ke 0.03
        
        # Simpan posisi sebelumnya
        if track_id in self.tracked_objects:
            prev_y, prev_cls = self.tracked_objects[track_id]
            
            # Debug print untuk monitoring
            distance_to_line = abs(center_y - line_y)
            if distance_to_line < 80:  # Kalau deket line
                print(f"\n[DEBUG] ID#{track_id:3d} | prev:{prev_y:5.0f} -> now:{center_y:5.0f} | line:{line_y:5.0f} | dist:{distance_to_line:4.0f}")
            
            # CROSSING DETECTION - SIMPLIFIED & MORE SENSITIVE
            # Cek apakah object melewati line (dari atas ke bawah ATAU bawah ke atas)
            crossed_down = prev_y < (line_y - margin) and center_y > (line_y + margin)
            crossed_up = prev_y > (line_y + margin) and center_y < (line_y - margin)
            
            if (crossed_down or crossed_up) and track_id not in self.counted_ids:
                self.counted_ids.add(track_id)
                direction = "DOWN ⬇" if crossed_down else "UP ⬆"
                print(f"\n✅ COUNTED! ID#{track_id} | {direction} | prev:{prev_y:.0f} -> now:{center_y:.0f}")
                return True
        
        # Update posisi terakhir
        self.tracked_objects[track_id] = (center_y, cls)
        return False
    
    def draw_info(self, frame, fps):
        """Draw informasi di frame"""
        h, w = frame.shape[:2]
        
        # Draw counting line dengan area margin
        line_y = int(self.line_position * h)
        margin = int(h * 0.03)
        
        # Area margin (semi-transparent) - LEBIH TEBAL
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, line_y - margin), (w, line_y + margin), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Counting line utama - LEBIH TEBAL
        cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 4)
        cv2.putText(frame, 'COUNTING LINE', (10, line_y - margin - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Background untuk info - LEBIH BESAR
        info_overlay = frame.copy()
        cv2.rectangle(info_overlay, (0, 0), (500, 220), (0, 0, 0), -1)
        cv2.addWeighted(info_overlay, 0.7, frame, 0.3, 0, frame)
        
        # Total count (BIGGER & BOLDER)
        cv2.putText(frame, f'TOTAL: {len(self.tracked_objects)}', (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 5)
        
        # FPS
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        
        # Tracked objects & Counted IDs
        cv2.putText(frame, f'Active: {len(self.tracked_objects)} | Counted IDs: {len(self.counted_ids)}', 
                   (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Class breakdown
        y_offset = 180
        for cls_id, cls_name in self.vehicle_classes.items():
            count = self.class_counts[cls_id]
            if count > 0:
                cv2.putText(frame, f'{cls_name}: {count}', (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                y_offset += 35
        
        return frame
    
    def process_video(self, video_path, output_path=None, show_video=True, resize_to=(960, 1280)):
        """
        Process video dan hitung kendaraan
        
        Args:
            video_path: Path ke video input
            output_path: Path untuk save video output (optional)
            show_video: Tampilkan video real-time
            resize_to: Tuple (width, height) untuk resize frame, None = no resize
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Tidak bisa buka video {video_path}")
            return
        
        # Get video properties
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set width & height (resize or original)
        if resize_to:
            width, height = resize_to
            print(f"⚠️  RESIZE ENABLED: {orig_width}x{orig_height} → {width}x{height}")
        else:
            width = orig_width
            height = orig_height
        
        print(f"\nProcessing video: {video_path}")
        print(f"Original Resolution: {orig_width}x{orig_height}")
        if resize_to:
            print(f"Output Resolution  : {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        print(f"Counting line position: {self.line_position * 100:.0f}% dari atas")
        print(f"⚠️  LINE Y-COORDINATE: {int(self.line_position * height)} pixels")
        print("\nControls:")
        print("  'q' = quit")
        print("  's' = screenshot")
        print("  'r' = reset counter")
        print("="*60 + "\n")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame kalau perlu
            if resize_to:
                frame = cv2.resize(frame, (width, height))
            
            frame_count += 1
            
            # Calculate FPS
            current_time = time.time()
            fps_calc = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
            self.prev_time = current_time
            
            # YOLO detection + tracking dengan parameter yang dioptimalkan
            results = self.model.track(
                frame, 
                persist=True,
                classes=list(self.vehicle_classes.keys()),
                conf=0.856,  # Confidence threshold - cukup tinggi untuk filter false positives
                iou=0.01,   # IoU threshold untuk NMS - lebih tinggi untuk kurangi duplikat
                device=self.device,
                verbose=False
            )
            
            # Process detections
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                classes = results[0].boxes.cls.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Check line crossing
                    if self.check_line_crossing(box, track_id, height, cls):
                        self.count += 1
                        self.class_counts[cls] += 1
                        cls_name = self.vehicle_classes[cls]
                        print(f"🚗 [{frame_count:5d}] {cls_name.upper():10s} #{track_id:3d} | TOTAL: {self.count}")
                    
                    # Draw bounding box dengan warna berbeda kalau udah ke-count
                    if track_id in self.counted_ids:
                        color = (0, 255, 255)  # Yellow untuk yang udah dicount
                    else:
                        color = (0, 255, 0)  # Green untuk yang belum
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw center point
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    # Draw label dengan status counted
                    status = "✓" if track_id in self.counted_ids else ""
                    label = f'{self.vehicle_classes[cls]} #{track_id} {conf:.2f} {status}'
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0] + 10, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw info overlay
            frame = self.draw_info(frame, fps_calc)
            
            # Progress indicator
            progress = (frame_count / total_frames) * 100
            print(f"\rProgress: {progress:5.1f}% | Frame: {frame_count:5d}/{total_frames} | Count: {self.count:3d} | Active: {len(self.tracked_objects):3d}", end='')
            
            # Save frame
            if writer:
                writer.write(frame)
            
            # Show video
            if show_video:
                cv2.imshow('Real-time Vehicle Counting', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n\nStopped by user")
                    break
                elif key == ord('s'):
                    screenshot_path = f'screenshot_{frame_count}.jpg'
                    cv2.imwrite(screenshot_path, frame)
                    print(f"\n📸 Screenshot saved: {screenshot_path}")
                elif key == ord('r'):
                    self.count = 0
                    self.counted_ids.clear()
                    self.class_counts.clear()
                    print("\n🔄 Counter reset!")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n\n" + "="*70)
        print(" "*25 + "SUMMARY RESULTS")
        print("="*70)
        print(f"Total vehicles counted    : {self.count}")
        print(f"Total unique IDs tracked  : {len(self.counted_ids)}")
        print(f"Active tracked objects    : {len(self.tracked_objects)}")
        print(f"Frames processed          : {frame_count}/{total_frames}")
        print(f"\nBreakdown by class:")
        print("-"*70)
        for cls_id, cls_name in self.vehicle_classes.items():
            count = self.class_counts[cls_id]
            percentage = (count / self.count * 100) if self.count > 0 else 0
            if count > 0:
                bar = "█" * int(percentage / 2)
                print(f"  {cls_name.capitalize():<12}: {count:>4} ({percentage:5.1f}%) {bar}")
        print("="*70)


def main():
    """Main function"""
    
    # Configuration
    VIDEO_PATH = 'Screen Recording 2025-11-13 223843.mp4'
    OUTPUT_PATH = 'output_video.mp4'  # Set None kalau ga mau save
    MODEL_PATH = 'yolov8m.pt'  # yolov8n.pt (fastest) atau yolov8m.pt (balanced)
    LINE_POSITION = 0.7  # 0.3 (atas), 0.5 (tengah), 0.7 (bawah)
    DEVICE = 'cuda'  # 'cuda' atau 'cpu'
    
    print("="*70)
    print(" "*20 + "🚗 VEHICLE COUNTER - ULTRA FIXED")
    print("="*70)
    print(f"Model             : {MODEL_PATH}")
    print(f"Line Position     : {LINE_POSITION * 100:.0f}% dari atas")
    print(f"Device            : {DEVICE}")
    print(f"Tracker           : BoT-SORT (more stable)")
    print(f"Sensitivity       : HIGH (margin 3%)")
    print("="*70)
    
    # Initialize counter
    counter = VehicleCounter(
        model_path=MODEL_PATH,
        line_position=LINE_POSITION,
        device=DEVICE
    )
    
    # Process video
    counter.process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        show_video=True,
        resize_to=(960, 700)  # Set None kalau ga mau resize
    )


if __name__ == '__main__':
    main()