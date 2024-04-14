from YOLODetector import *

# YOLOv8
modelURL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"

classFile = "coco.yaml" #variable for class names path
threshold = 0.2 #detection threshold reference
videoPath = 1 #0 for webcam "" for video file 2 for obs virtual cam
#L_imagePath = "../renderout_L_empty.png"
#R_imagePath = "../renderout_R_empty.png"

detector = Detector() #calling the class and creating a definition
detector.downloadModel(modelURL) #passing modelURL variable into download function activating it
detector.loadModel() #activating loadModel function
detector.readClasses(classFile) #passing classFile variable into read function and activating it

detector.predictVideo(threshold)
#detector.predictImage(L_imagePath, R_imagePath, threshold)
