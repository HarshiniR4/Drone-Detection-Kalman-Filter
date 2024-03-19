import cv2
from ultralytics import YOLO
import time
import os


def get_bb_video(video, model, names, object_class):
    cap = cv2.VideoCapture(video)
    centers = []
    no_frames = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        no_frames += 1
        preds = model(frame)
        person_detected = False

        for pred in preds:
            detections = []
            print("Frame number: ", no_frames)

            for r in pred.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)

                class_name = names.get(class_id)

                if class_name == object_class and score > 0.5:
                    if not person_detected:
                        person_detected = True
                        detections.append([x1, y1, x2, y2, round(score, 2), class_name])
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        centers.append((center_x, center_y))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            if not detections:
                centers.append("Not detected")

    cap.release()
    return centers


def get_bb_frame(frame, model, names, object_class):
    centers = []
    preds = model(frame)
    person_detected = False

    for pred in preds:
        detections = []

        for r in pred.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            class_name = names.get(class_id)

            if class_name == object_class and score > 0.5:
                if not person_detected:
                    person_detected = True
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    centers.append((center_x, center_y))

    if not person_detected:
        centers.append("Not detected")

    return centers






if __name__ == '__main__':
    # Get the current directory
    current_directory = os.getcwd()
    print(current_directory)

    # Go back to the parent directory
    parent_directory = r"Assignment 3 pt 2"
    print(parent_directory)

    # Set input and output directory
    video = r"Drone Videos\Drone Tracking 2.mp4"
    print(video)

    # Instantiate model
    weights_path = r"kaggle/working/drone_detection/train5/weights/best.pt"
    model = YOLO(weights_path)
    names = model.names
    print(names)

    # Example usage:
    bounding_box_centers = get_bb_video(video, model, names, object_class='Drone')
