import numpy as np
from collections import OrderedDict
from scipy.spatial import distance

class CentroidTracker:

    def __init__(self):
        # Tracking ID
        self.next_id = 0
        # Key: Object ID. Value: Object centroid
        self.objects = OrderedDict()
        # Maximum frames to deregister disappeared objects
        self.max_disappear_frames = 50
        # Key: Object ID. Value: Consecutive frame the object disappeared
        self.disappeared = OrderedDict()

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, id):
        del self.objects[id]
        del self.disappeared[id]

    def update(self, bboxs):
        if len(bboxs) == 0:
            # Increment disappear frames for each object
            for id in list(self.disappeared.keys()):
                self.disappeared[id] += 1
                # Deregister obeject if disappear for long time
                if self.disappeared[id] > self.max_disappear_frames:
                    self.deregister(id)
            return self.objects
        # Compute centroids for each bounding box
        input_centroids = np.zeros((len(bboxs), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(bboxs):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids[i] = (cx, cy)
        # If there are no tracking objects ...
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        # Otherwise, map objects to closest centroids ...
        else:
            tracking_ids = list(self.objects.keys())
            tracking_centroids = list(self.objects.values())

            # Calculate distance matrix between tracking centroids and input centroids
            # Shape of D: len(tracking_centroids) x len(input_centroids)
            D = distance.cdist(np.array(tracking_centroids), input_centroids)
            # Find closest input centroid for each tracking centroid
            # Argsort returns index of each tracking centroid
            rows = D.min(axis=1).argsort()
            # Get index of input centroid
            cols = D.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                self.objects[tracking_ids[row]] = input_centroids[col]
                self.disappeared[tracking_ids[row]] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            # If there are more tracking objects than input centroids ...
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    self.disappeared[tracking_ids[row]] += 1
                    if self.disappeared[tracking_ids[row]] > self.max_disappear_frames:
                        self.deregister(tracking_ids[row])
            # There are more input centroids than tracking objects ...
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])
        return self.objects
