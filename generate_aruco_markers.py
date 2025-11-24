import cv2 as cv
import numpy as np
import os

# Load the predefined dictionary
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

# Generate the marker
markerIDs = [25, 33, 30, 23, 50, 60, 70, 80]

for id in markerIDs:
    filename = "marker%d.png" %id
    if not os.path.isfile(filename):
        markerImage = np.zeros((200, 200), dtype=np.uint8)
        markerImage = cv.aruco.drawMarker(dictionary, id, 200, markerImage, 1);
        cv.imwrite("marker%d.png" %id, markerImage);
        print("Created marker%d" %id)
    else:
        print("marker with id %d already exists" %id)

