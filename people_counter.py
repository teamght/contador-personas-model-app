# USAGE
# To read and write back out to video:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 --output output/output_01.avi
#
# To read from webcam and write back out to disk:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#	--output output/webcam_output.avi
#
# Note: If you want to test with your wireless camera or smartphone as wireless camera then you can directly put link given by application
# 	and in --input videos/ also, you have to put /video at the end of link if you are using smartphone as wireless device.

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import os
import imutils
import time
import dlib
import cv2

LINK_CAMARA = os.environ["LINK_CAMARA"]

args = {}
args['prototxt'] = 'mobilenet_ssd/MobileNetSSD_deploy.prototxt'
args['model'] = 'mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
#args['input'] = None #'videos/example_01.mp4'
args['input'] = LINK_CAMARA
args['output'] = 'output/webcam_output.avi'
args['confidence'] = 0.4
args['skip_frames'] = 30

cantidad_personas_afuera = 0
cantidad_personas_dentro = 0

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

def capturar_video():
	# if a video path was not supplied, grab a reference to the webcam
	if args['input'] is None:
		print("[INFO] starting video stream...")
		vs = cv2.VideoCapture(0)
		time.sleep(2.0)

	# otherwise, grab a reference to the video file
	else:
		print("[INFO] opening video file...")
		vs = cv2.VideoCapture(args["input"], cv2.CAP_FFMPEG)

	# initialize the video writer (we'll instantiate later if need be)
	writer = None

	# initialize the frame dimensions (we'll set them as soon as we read
	# the first frame from the video)
	W = None
	H = None

	# instantiate our centroid tracker, then initialize a list to store
	# each of our dlib correlation trackers, followed by a dictionary to
	# map each unique object ID to a TrackableObject
	ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackers = []
	trackableObjects = {}

	# initialize the total number of frames processed thus far, along
	# with the total number of objects that have moved either up or down
	totalFrames = 0
	global cantidad_personas_afuera
	global cantidad_personas_dentro

	# start the frames per second throughput estimator
	#fps = FPS().start()

	# Validar si se está obteniendo valores de la cámara web
	print('La cámara está transmiendo: {}'.format(vs.isOpened()))
	#if not vs.isOpened():
	#	yield (b'--camara_data_html\r\n'b'Content-Type: image/jpeg\r\n\r\n' +  + b'\r\n')

	# loop over frames from the video stream
	while vs.isOpened():
		# grab the next frame and handle if we are reading from either
		# VideoCapture or VideoStream
		frame = vs.read()
		#frame = frame[1] if not 'input' in args else frame
		frame = frame[1]
		
		if not frame is None:
			# if we are viewing a video and we did not grab a frame then we
			# have reached the end of the video
			#if args["input"] is not None and frame is None:
			#	break

			# resize the frame to have a maximum width of 500 pixels (the
			# less data we have, the faster we can process it), then convert
			# the frame from BGR to RGB for dlib
			frame = imutils.resize(frame, width=500)
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			# if the frame dimensions are empty, set them
			if W is None or H is None:
				(H, W) = frame.shape[:2]

			# if we are supposed to be writing a video to disk, initialize
			# the writer
			if args["output"] is not None and writer is None:
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				writer = cv2.VideoWriter(args["output"], fourcc, 30,
					(W, H), True)

			# initialize the current status along with our list of bounding
			# box rectangles returned by either (1) our object detector or
			# (2) the correlation trackers
			status = "Waiting"
			rects = []

			# check to see if we should run a more computationally expensive
			# object detection method to aid our tracker
			if totalFrames % args["skip_frames"] == 0:
				# set the status and initialize our new set of object trackers
				status = "Detecting"
				trackers = []

				# convert the frame to a blob and pass the blob through the
				# network and obtain the detections
				blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
				net.setInput(blob)
				detections = net.forward()

				# loop over the detections
				for i in np.arange(0, detections.shape[2]):
					# extract the confidence (i.e., probability) associated
					# with the prediction
					confidence = detections[0, 0, i, 2]

					# filter out weak detections by requiring a minimum
					# confidence
					if confidence > args["confidence"]:
						# extract the index of the class label from the
						# detections list
						idx = int(detections[0, 0, i, 1])

						# if the class label is not a person, ignore it
						if CLASSES[idx] != "person":
							continue

						# compute the (x, y)-coordinates of the bounding box
						# for the object
						box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
						(startX, startY, endX, endY) = box.astype("int")

						# construct a dlib rectangle object from the bounding
						# box coordinates and then start the dlib correlation
						# tracker
						tracker = dlib.correlation_tracker()
						rect = dlib.rectangle(startX, startY, endX, endY)
						tracker.start_track(rgb, rect)

						# add the tracker to our list of trackers so we can
						# utilize it during skip frames
						trackers.append(tracker)

			# otherwise, we should utilize our object *trackers* rather than
			# object *detectors* to obtain a higher frame processing throughput
			else:
				# loop over the trackers
				for tracker in trackers:
					# set the status of our system to be 'tracking' rather
					# than 'waiting' or 'detecting'
					status = "Tracking"

					# update the tracker and grab the updated position
					tracker.update(rgb)
					pos = tracker.get_position()

					# unpack the position object
					startX = int(pos.left())
					startY = int(pos.top())
					endX = int(pos.right())
					endY = int(pos.bottom())

					# add the bounding box coordinates to the rectangles list
					rects.append((startX, startY, endX, endY))

			# draw a horizontal line in the center of the frame -- once an
			# object crosses this line we will determine whether they were
			# moving 'up' or 'down'
			cv2.line(frame, ((W // 2) - 30, 0), ((W // 2) + 30, H), (50, 205, 50), 2)

			# use the centroid tracker to associate the (1) old object
			# centroids with (2) the newly computed object centroids
			objects = ct.update(rects)

			# loop over the tracked objects
			for (objectID, centroid) in objects.items():
				# check to see if a trackable object exists for the current
				# object ID
				to = trackableObjects.get(objectID, None)

				# if there is no existing trackable object, create one
				if to is None:
					to = TrackableObject(objectID, centroid)

				# otherwise, there is a trackable object so we can utilize it
				# to determine direction
				else:
					# the difference between the y-coordinate of the *current*
					# centroid and the mean of *previous* centroids will tell
					# us in which direction the object is moving (negative for
					# 'up' and positive for 'down')
					y = [c[0] for c in to.centroids]
					direction = centroid[0] - np.mean(y)
					to.centroids.append(centroid)

					# check to see if the object has been counted or not
					if not to.counted:
						# if the direction is negative (indicating the object
						# is moving up) AND the centroid is above the center
						# line, count the object
						if direction < 0 and centroid[0] < W // 2:
							cantidad_personas_dentro += 1
							to.counted = True

						# if the direction is positive (indicating the object
						# is moving down) AND the centroid is below the
						# center line, count the object
						elif direction > 0 and centroid[0] > W // 2:
							cantidad_personas_afuera += 1
							to.counted = True

				# store the trackable object in our dictionary
				trackableObjects[objectID] = to

				# draw both the ID of the object and the centroid of the
				# object on the output frame
				text = "ID {}".format(objectID)
				cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 191, 255), 2)
				cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 191, 255), -1)
				
			# construct a tuple of information we will be displaying on the
			# frame
			store = cantidad_personas_dentro - cantidad_personas_afuera
			store = store if store > 0 else 0
			info = [
				("Afuera", cantidad_personas_afuera),
				("Adentro", cantidad_personas_dentro),
				("Status", status),
				("Dentro del Edificio", store)
			]
			
			if store > 2:
					cv2.putText(frame, 'Warning',
								(50,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
				
			# loop over the info tuples and draw them on our frame
			for (i, (k, v)) in enumerate(info):
				text = "{}: {}".format(k, v)
				cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

			# check to see if we should write the frame to disk
			if writer is not None:
				writer.write(frame)

			# show the output frame
			#cv2.imshow("Frame", frame)
			#key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			#if key == ord("q"):
			#	break
			frame_html = cv2.imencode('.jpg', frame)[1].tobytes()
			yield (b'--frame_html\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_html + b'\r\n')
			#time.sleep(0.1)

			# increment the total number of frames processed thus far and
			# then update the FPS counter
			totalFrames += 1
			#fps.update()
		else:
			print('La cámara está transmiendo: {}'.format(vs.isOpened()))
			print('No hay contenido de video o de la cámara web.')
			break

	# stop the timer and display FPS information
	#fps.stop()
	#print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# check to see if we need to release the video writer pointer
	if writer is not None:
		writer.release()

	# if we are not using a video file, stop the camera video stream
	if not args.get("input", False):
		vs.stop()

	# otherwise, release the video file pointer
	else:
		vs.release()

	# close any open windows
	#cv2.destroyAllWindows()

def reset_counter_people():
	global cantidad_personas_afuera
	global cantidad_personas_dentro
	
	cantidad_personas_afuera = 0
	cantidad_personas_dentro = 0

import os
from flask import Flask, render_template, request, Response

PORT = int(os.environ.get("PORT", 5000))
app = Flask(__name__)

@app.route('/')
@app.route('/index', methods =["GET", "POST"])
def index():
	"""Video streaming home page."""
	if request.method == 'POST':
		data = request.json
		if data['reset']:
			reset_counter_people()
		return '', 200
	else:
		return render_template('index.html')

@app.route('/video_feed')
def video_feed():
	return Response(capturar_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame_html')

@app.route('/video_data', methods =["GET", "POST"])
def video_data():
	global cantidad_personas_afuera
	global cantidad_personas_dentro
	
	print('Cantidad personas afuera: {}'.format(cantidad_personas_afuera))
	print('Cantidad personas dentro: {}'.format(cantidad_personas_dentro))
	return {'cantidad_personas_afuera': cantidad_personas_afuera, 'cantidad_personas_dentro': cantidad_personas_dentro}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)