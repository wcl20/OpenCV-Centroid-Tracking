import cv2
import imutils
import numpy as np
import time
from core.tracker import CentroidTracker
from imutils.video import VideoStream

def main():

    # Load face detection model
    prototxt = "deploy.prototxt"
    model = "res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # Setup tracker
    tracker = CentroidTracker()

    # Setup video stream. Allow 2 seconds for warm up.
    stream = VideoStream(src=0).start()
    time.sleep(2)

    while True:

        frame = stream.read()

        # Resize with aspect ratio
        frame = imutils.resize(frame, width=600)
        height, width = frame.shape[:2]

        # Detect faces
        blob = cv2.dnn.blobFromImage(frame, 1.0, (width, height), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()[0, 0]

        bboxs = []
        for detection in detections:
            # Remove detections with low confidence
            confidence = detection[2]
            if confidence > 0.9:
                bbox = detection[3:7] * np.array([width, height, width, height])
                bbox = bbox.astype("int")
                # Save bounding box for tracking
                bboxs.append(bbox)
                # Draw bounding box
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{confidence * 100: .2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        objects = tracker.update(bboxs)
        for id, centroid in objects.items():
            cx, cy = centroid
            cv2.putText(frame, f"ID {id}", (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # Display
        cv2.imshow("Centroid Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    stream.stop()










if __name__ == '__main__':
    main()
