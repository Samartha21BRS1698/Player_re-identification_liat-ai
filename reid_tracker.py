# code to Test YOLOv11 Detection on One Frame

import os
import cv2
import numpy as np
from ultralytics import YOLO

# === PlayerTracker Class ===
class PlayerTracker:
    def __init__(self, iou_threshold=0.55, max_missing=60, dist_threshold=40, max_ids=100):
        self.players = {}  # player_id: {'bbox': tuple, 'missed': int}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.dist_threshold = dist_threshold
        self.max_ids = max_ids

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

    def center_distance(self, boxA, boxB):
        ax = (boxA[0] + boxA[2]) / 2
        ay = (boxA[1] + boxA[3]) / 2
        bx = (boxB[0] + boxB[2]) / 2
        by = (boxB[1] + boxB[3]) / 2
        return np.linalg.norm([ax - bx, ay - by])

    def update(self, detections):
        assigned_ids = [-1] * len(detections)
        matched_indices = set()

        # Match current detections with tracked players
        for pid, data in self.players.items():
            if data['missed'] >= self.max_missing:
                continue

            best_match_idx = -1
            best_score = 0

            for i, det in enumerate(detections):
                if i in matched_indices:
                    continue

                iou = self.compute_iou(data['bbox'], det)
                dist = self.center_distance(data['bbox'], det)

                if iou > self.iou_threshold or dist < self.dist_threshold:
                    if iou > best_score:
                        best_score = iou
                        best_match_idx = i

            if best_match_idx != -1:
                self.players[pid]['bbox'] = detections[best_match_idx]
                self.players[pid]['missed'] = 0
                assigned_ids[best_match_idx] = pid
                matched_indices.add(best_match_idx)

        # Assign unmatched detections
        for i, det in enumerate(detections):
            if assigned_ids[i] == -1:
                # Try to reuse recently lost IDs
                reused = False
                for pid, data in self.players.items():
                    if data['missed'] > 0 and data['missed'] < self.max_missing:
                        dist = self.center_distance(data['bbox'], det)
                        if dist < self.dist_threshold:
                            self.players[pid]['bbox'] = det
                            self.players[pid]['missed'] = 0
                            assigned_ids[i] = pid
                            reused = True
                            break

                if not reused and self.next_id < self.max_ids:
                    self.players[self.next_id] = {'bbox': det, 'missed': 0}
                    assigned_ids[i] = self.next_id
                    self.next_id += 1

        # Mark missing players
        active_ids = set(assigned_ids)
        for pid in list(self.players.keys()):
            if pid not in active_ids:
                self.players[pid]['missed'] += 1
                if self.players[pid]['missed'] > self.max_missing:
                    del self.players[pid]

        return assigned_ids

# === Main Code ===
def main():
    model = YOLO("weights/best.pt")
    cap = cv2.VideoCapture("input/15sec_input_720p.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs("output", exist_ok=True)
    out = cv2.VideoWriter("output/tracked_video.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (width, height))

    tracker = PlayerTracker()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(source=frame, conf=0.3, verbose=False)[0]
        bboxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        player_bboxes = [np.round(bbox, 1) for bbox, cls in zip(bboxes, classes) if cls == 2]

        # Update tracker
        id_assignments = tracker.update(player_bboxes)

        # Draw results
        for bbox, pid in zip(player_bboxes, id_assignments):
            if pid == -1:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {pid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Final output saved to: output/tracked_video.mp4")


# === Run the script ===
if __name__ == "__main__":
    main()
