import cv2
import os
import time
from flask import Flask, render_template, Response
import pathlib
import numpy as np
import argparse
import tensorflow as tf
import dlib
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from trackable_object import TrackableObject
from centroidtracker import CentroidTracker
from object_detection.utils import label_map_util
from tensorflow_cumulative_object_counting import load_model, run_inference_for_single_image

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

LINK_CAMARA = os.environ["LINK_CAMARA"]

PORT = int(os.environ.get("PORT", 5000))

SAVED_MODEL = 'saved_model'
LABEL_MAP = 'labelmap.pbtxt'
print('Iniciando carga del modelo de red neuronal ...')
detection_model = load_model(SAVED_MODEL)
print('Finalizó carga del modelo de red neuronal.')

app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen():
    """Video streaming generator function."""
    print('Inicio de contador de personas.')
    print('Iniciando captura de video de la cámara ...')
    #cap = cv2.VideoCapture(LINK_CAMARA, cv2.CAP_FFMPEG)
    #cap = cv2.VideoCapture(LINK_CAMARA)
    cap = cv2.VideoCapture('768x576.avi')
    
    category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP, use_display_name=True)

    print('Iniciando detección de personas ...')
    model = detection_model
    labels = LABEL_MAP
    roi_position=0.6
    threshold=0.1 # Valor original 0.5
    x_axis=True
    skip_frames=20
    show=True
    
    counter = [0, 0, 0, 0]  # left, right, up, down
    total_frames = 0

    #ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    #ct = CentroidTracker(maxDisappeared=20, maxDistance=20)
    ct = CentroidTracker(maxDisappeared=5, maxDistance=10)
    trackers = []
    trackableObjects = {}

    while cap.isOpened():
        ret, image_np = cap.read()
        if not ret:
            print('Error. No se pudo recuperar captura de video de la cámara.')
            break

        height, width, _ = image_np.shape
        rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        status = "Waiting"
        rects = []

        if total_frames % skip_frames == 0:
            status = "Detecting"
            trackers = []

            # Actual detection.
            output_dict = run_inference_for_single_image(model, image_np)

            for i, (y_min, x_min, y_max, x_max) in enumerate(output_dict['detection_boxes']):
                if output_dict['detection_scores'][i] > threshold and (labels == None or category_index[output_dict['detection_classes'][i]]['name'] in labels):
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height))
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)
        else:
            status = "Tracking"
            for tracker in trackers:
                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

				# unpack the position object
                x_min, y_min, x_max, y_max = int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())
                
                # add the bounding box coordinates to the rectangles list
                rects.append((x_min, y_min, x_max, y_max))

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                if x_axis and not to.counted:
                    x = [c[0] for c in to.centroids]
                    direction = centroid[0] - np.mean(x)
            
                    if centroid[0] > roi_position*width and direction > 0:
                        counter[1] += 1
                        to.counted = True
                    elif centroid[0] < roi_position*width and direction < 0:
                        counter[0] += 1
                        to.counted = True
                    
                elif not x_axis and not to.counted:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)

                    if centroid[1] > roi_position*height and direction > 0:
                        counter[3] += 1
                        to.counted = True
                    elif centroid[1] < roi_position*height and direction < 0:
                        counter[2] += 1
                        to.counted = True
                
                to.centroids.append(centroid)

            trackableObjects[objectID] = to

            text = "ID {}".format(objectID)
            cv2.putText(image_np, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(image_np, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        # Original
        # Draw ROI line
        if x_axis:
            cv2.line(image_np, (int(roi_position*width), 0), (int(roi_position*width), height), (0xFF, 0, 0), 5)
        else:
            cv2.line(image_np, (0, int(roi_position*height)), (width, int(roi_position*height)), (0xFF, 0, 0), 5)
        
        # display count and status
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Original
        if x_axis:
            cv2.putText(image_np, f'Left: {counter[0]}; Right: {counter[1]}', (10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
        else:
            cv2.putText(image_np, f'Up: {counter[2]}; Down: {counter[3]}', (10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
        
        cv2.putText(image_np, 'Status: ' + status, (10, 70), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)

        if show:
            img = cv2.resize(image_np, (0,0), fx=0.5, fy=0.5) 
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)

        total_frames += 1


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)