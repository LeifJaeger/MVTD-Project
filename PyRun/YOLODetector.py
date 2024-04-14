import cv2
import time
import os
import numpy as np
import glob
from pythonosc import udp_client
from ultralytics import YOLO
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # make sure detections are sent to GPU for better performance

client_out = udp_client.SimpleUDPClient("IP ADDRESS", 7777) # settting up variable sending over network

images_centre = [cv2.imread('../chess_imgs_centre/out.1.png'),
               cv2.imread('../chess_imgs_centre/out.2.png'),
               cv2.imread('../chess_imgs_centre/out.3.png'),
               cv2.imread('../chess_imgs_centre/out.4.png'),
               cv2.imread('../chess_imgs_centre/out.5.png'),
               cv2.imread('../chess_imgs_centre/out.6.png'),
               cv2.imread('../chess_imgs_centre/out.7.png'),
               cv2.imread('../chess_imgs_centre/out.8.png'),
               cv2.imread('../chess_imgs_centre/out.9.png'),
               cv2.imread('../chess_imgs_centre/out.10.png'),
               cv2.imread('../chess_imgs_centre/out.11.png'),
               cv2.imread('../chess_imgs_centre/out.12.png'),
               cv2.imread('../chess_imgs_centre/out.13.png'),
               cv2.imread('../chess_imgs_centre/out.14.png'),
               cv2.imread('../chess_imgs_centre/out.15.png'),
               cv2.imread('../chess_imgs_centre/out.16.png'),
               cv2.imread('../chess_imgs_centre/out.17.png'),
               cv2.imread('../chess_imgs_centre/out.18.png'),
               cv2.imread('../chess_imgs_centre/out.19.png'),
               cv2.imread('../chess_imgs_centre/out.20.png'),
               cv2.imread('../chess_imgs_centre/out.21.png'),
               cv2.imread('../chess_imgs_centre/out.22.png'),
               cv2.imread('../chess_imgs_centre/out.23.png'),
               cv2.imread('../chess_imgs_centre/out.24.png'),
               cv2.imread('../chess_imgs_centre/out.25.png'),
               cv2.imread('../chess_imgs_centre/out.26.png'),
               cv2.imread('../chess_imgs_centre/out.27.png'),
               cv2.imread('../chess_imgs_centre/out.28.png'),
               cv2.imread('../chess_imgs_centre/out.29.png'),
               cv2.imread('../chess_imgs_centre/out.30.png')] # images of the chessboard for calibration

imgsL = glob.glob("../chess_imgs_left/*.png") # images of the chessboard for left and right cameras for stereo matching
imgR = glob.glob("../chess_imgs_right/*.png")
flatL = glob.glob("../flat_left/*.png")
flatR = glob.glob("../flat_right/*.png")

square_size = 15 # real world size of chessboard squares (cm)

class Detector:
    def __init__(self):
        pass

    def readClasses(self, classesFilePath): #read class names file into an array and split by newline and print number of lines in file
        with open(classesFilePath, 'r') as f:
            self.classesList = []
            self.classesList = f.read().splitlines()
        #self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

        print(len(self.classesList))#, len(self.colorList))

    def downloadModel(self, modelURL): #download detection model from link variable, put contents into a new folder
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]

    def loadModel(self): #extract contents of new model folder and print when done
        print("Loading Model " + self.modelName)

        model = YOLO("yolov8s.pt")
        self.model = model.load("yolov8s.pt")
        self.model = model.to(device)

        print("Model " + self.modelName + " loaded successfully...")

    def createBBox(self, image, threshold): #creating bounding boxes around detections
        centX = 0
        centY = 0
        inputImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert colour of input image
        detections = self.model.predict(source=inputImg, conf=threshold) # use image to detect objects
        coord = [] # empty arrays for storing stuff later
        classScoresPerc = []
        classIndex = []
        classColor = (0, 255, 0) #GREEN
        for det in detections:
            boxes = det.boxes.cpu().numpy() # copy detection data off GPU onto CPU for augmentation
            data = boxes.data
            for xyxy in data:
                xmin, ymin, xmax, ymax = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]) # yolo layout for bounding box output
                coord.append([xmin, ymin, xmax, ymax]) # append to the array

                classConf = int(100 * xyxy[4]) # convert scores to percentage and append to array
                classScoresPerc.append(classConf)

                className = int(xyxy[5]) # read class id output and compare to names file
                classIndex.append(className)
                classLabelText = self.classesList[className]

                if classLabelText == "person": # only want if a person is detected
                    centX = int((xmin + xmax) / 2) # calculating centre of bounding box
                    centY = int((ymin + ymax) / 2)
                    lineWidth = min(int((xmax - xmin) * 0.2), int((ymax - ymin) * 0.2))
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1) # drawing lines around the detection to create the bounding box
                    cv2.line(image, (xmin, ymin), (xmin + lineWidth, ymin), classColor, thickness=5)
                    cv2.line(image, (xmin, ymin), (xmin, ymin + lineWidth), classColor, thickness=5)
                    cv2.line(image, (xmax, ymin), (xmax - lineWidth, ymin), classColor, thickness=5)
                    cv2.line(image, (xmax, ymin), (xmax, ymin + lineWidth), classColor, thickness=5)
                    cv2.line(image, (xmin, ymax), (xmin + lineWidth, ymax), classColor, thickness=5)
                    cv2.line(image, (xmin, ymax), (xmin, ymax - lineWidth), classColor, thickness=5)
                    cv2.line(image, (xmax, ymax), (xmax - lineWidth, ymax), classColor, thickness=5)
                    cv2.line(image, (xmax, ymax), (xmax, ymax - lineWidth), classColor, thickness=5)
                    cv2.circle(image, (centX, centY), radius=1, color=classColor, thickness=1)
                    displayText1 = '[ {} ][ {}% ]'.format(classLabelText, classConf)
                    displayText2 = '[ {} ][ {} ]'.format(centX, centY)
                    cv2.putText(image, displayText1, (xmin, ymin - 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, classColor, 2) # put identifying text above bounding box
                    cv2.putText(image, displayText2, (xmin, ymin - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, classColor, 2)

        return image, centX, centY

    def cam_cal(self, imgs_left, imgs_right, square_size): # calibrating cameras function
        global greyl # global define of some variables
        global greyr

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # some settings for the calibration functions

        objp = np.zeros((9*5,3), np.float32) # zero array the same size as the chessboard
        objp[:,:2] = np.mgrid[0:9,0:5].T.reshape(-1,2) * square_size
        objpoints = []

        imgpointsL = [] # empty array to be filled by detected points in the image
        imgpointsR = []

        for img_l, img_r in zip(imgs_left, imgs_right): # looks through left and right image folders
            imgL = cv2.imread(img_l)
            greyl = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            retL, cornersL = cv2.findChessboardCorners(greyl, (9,5), None) # read left image and find chessboard points in it

            imgR = cv2.imread(img_r)
            greyr = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            retR, cornersR = cv2.findChessboardCorners(greyr, (9,5), None) # read right image nad find chessboard points in it

            if retL == True:
                objpoints.append(objp)

                corners2l = cv2.cornerSubPix(greyl, cornersL, (11,11), (-1,-1), criteria) # if some points were found look in subpixels for more accuracy
                imgpointsL.append(corners2l)
                cv2.drawChessboardCorners(imgL, (9,5), corners2l, retL) # draw and show detected points
                cv2.imshow('img', imgL)
                cv2.waitKey(50)

                corners2r = cv2.cornerSubPix(greyr, cornersR, (11,11), (-1,-1), criteria)
                imgpointsR.append(corners2r)

        retvalL, mtxL, distcofL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, greyl.shape[::-1], None, None) # calibrate each camera using detected chessboard points
        retvalR, mtxR, distcofR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, greyr.shape[::-1], None, None)

        # print('\n camera calibrated: \n', retvalL)
        # print('\n camera matrix: \n', mtxL)
        # print('\n distortion coeff: \n', distcofL)
        # print('\n rotation vec: \n', rvecsL)
        # print('\n translation vec: \n', tvecsL)

        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC

        #retStereo, _, _, _, _, rota, transf, essenMatx, fundeMatx = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, mtxL, distcofL, mtxR, distcofR, greyl.shape[::-1], criteria=criteria, flags=flags)

        #rectL, rectR, projMatL, projMatR, Q, _, _ = cv2.stereoRectify(mtxL, distcofL, mtxR, distcofR, greyl.shape[::-1], rota, transf)

        return mtxL, distcofL, rvecsL, tvecsL, mtxR, distcofR, rvecsR, tvecsR

    def drawaxes(self, img, corners, imgpoints): # draws world axes on some chessboard images as a test
        corner = tuple(corners[0].ravel().astype(int))
        imgptX = tuple(imgpoints[0].ravel().astype(int))
        imgptY = tuple(imgpoints[1].ravel().astype(int))
        imgptZ = tuple(imgpoints[2].ravel().astype(int))

        img = cv2.line(img, corner, imgptX, (0, 0, 255), 7)  # x
        img = cv2.line(img, corner, imgptY, (0, 255, 0), 7)  # y
        img = cv2.line(img, corner, imgptZ, (255, 0, 0), 7)  # z
        return img

    def get_3d_coords(self, point_left, point_right, imageL, imageR, baseline, f, fov, mtxL, mtxR): # function to do all the world coordinate calculations

        heightL, widthL = imageL.shape[0], imageL.shape[1] # calculate width and height of image
        f_pixel = (widthL * 0.5) / np.tan(fov * 0.5 * np.pi/180) # calculate focal length from pixels
        disparity = point_left[0] - point_right[0] # calculate disparity between left and right centre points
        Zdist = (baseline * f_pixel) / disparity # calculate distance from cameras

        xpL = point_left[0] # x and y centre points in right and left images
        ypL = point_left[1]
        xpR = point_right[0]
        ypR = point_right[1]

        shleftmatximp = np.matrix([[mtxL[0][0], 0., mtxL[0][2]], # left intrinsic matrix with origin shift
                                   [0., mtxL[1][1], mtxL[1][2]],
                                   [0., 0., 1.]])

        leftmatximp = np.matrix([[mtxL[0][0], 0., 0.], # left intrinsice matrix without shift
                                 [0., mtxL[1][1], 0.],
                                 [0., 0., 1.]])

        shrightmatx = np.matrix([[mtxR[0][0], 0., mtxR[0][2]], # right matrix with shift
                                [0., mtxR[1][1], mtxR[1][2]],
                                [0., 0., 1.]])

        rightmatx = np.matrix([[mtxR[0][0], 0., 0.], # right matrix without shift
                              [0., mtxR[1][1], 0.],
                              [0., 0., 1.]])

        leftcamor = np.matrix([[594.], [210.], [500.]]) # origin of the cameras in world coordinates
        rightcamor = np.matrix([[606.], [210.], [500.]])

        rotatmtx = np.matrix([[1., 0., 0.], # pre calculated camera rotation matrix
                             [0., -0.4226182617, -0.906307787],
                             [0., 0.906307787, -0.4226182617]])

        pxL = np.matrix([[xpL], [ypL], [1.]]) # bounding box centre points converted to a matrix
        pxR = np.matrix([[xpR], [ypR], [1.]])

        left3d = leftcamor + Zdist*(np.dot((np.linalg.inv(np.dot(shleftmatximp, rotatmtx))), pxL)) # calculations to get world coordinate points
        left3d2 = leftcamor + Zdist*(np.dot((np.linalg.inv(np.dot(leftmatximp, rotatmtx))), pxL))
        right3d = rightcamor + Zdist * (np.dot((np.linalg.inv(np.dot(shrightmatx, rotatmtx))), pxR))
        right3d2 = rightcamor + Zdist * (np.dot((np.linalg.inv(np.dot(rightmatx, rotatmtx))), pxR))

        world_X = (float(left3d[0])+float(right3d[0]))/2
        world_Y = (float(left3d2[1])+float(right3d2[1]))/2
        world_Z = (float(((500-(left3d[2]))+left3d2[2])/2)+float(((500 - (right3d[2])) + right3d2[2]) / 2))/2

        world_coords = np.array([[world_X],
                                 [world_Y],
                                 [world_Z]])

        #print("\n world coords: \n", world_coords)

        client_out.send_message("/X", world_X) # sending calculated point values over network to TouchDesigner
        client_out.send_message("/Y", world_Y)
        client_out.send_message("/Z", world_Z)

        return

    def predictImage(self, L_imagePath, R_imagePath, threshold):

        mtxL, distcofL, rvecsL, tvecsL, mtxR, distcofR, rvecsR, tvecsR = self.cam_cal(imgsL, imgR, square_size)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((9*5,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:5].T.reshape(-1,2)

        axis = np.float32([[5,0,0], [0,-5,0], [0,0,-5]]).reshape(-1,3)

        for img in (flatL):
            img = cv2.imread(img)
            greyl = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            retL, cornersL = cv2.findChessboardCorners(greyl, (9,5), None)

            if retL == True:
                corners2l = cv2.cornerSubPix(greyl, cornersL, (11,11), (-1,-1), criteria)
                ret, rvecs, tvecs = cv2.solvePnP(objp, corners2l, mtxL, distcofL)
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtxL, distcofL)
                axesimg = self.drawaxes(img, corners2l, imgpts)
                cv2.imshow('aaaaaaaaa', axesimg)
                cv2.waitKey(1000)

        imageL = glob.glob("../test_left/*.png")
        imageR = glob.glob("../test_right/*.png")

        #self.get_3d_coords(point_left, point_right, imageL, imageR, baseline, f, alpha, mtxL, mtxR)

        key = cv2.waitKey(0) & 0xFF  # stops program and closes windows when key is pressed
        if key == ord("q"):
            cv2.destroyAllWindows()

    def predictVideo(self, threshold):

        mtxL, distcofL, rvecsL, tvecsL, mtxR, distcofR, rvecsR, tvecsR = self.cam_cal(imgsL, imgR, square_size)

        baseline = 12  # distance between cameras in cm
        f = 18 # camera focal distance
        fov = 110 # cameras field of view

        capL = cv2.VideoCapture(0) # open camera inputs
        capR = cv2.VideoCapture(1)

        if capL.isOpened() == False:
            print("ERROR finding video")
            return

        if capR.isOpened() == False:
            print("ERROR finding video")
            return

        (successL, imageL) = capL.read() # read camera inputs
        (successR, imageR) = capR.read()

        startTime = 0

        while successL:
            currentTime = time.time()

            fps = 1/(currentTime - startTime) # calculate FPS to guess performance
            startTime = currentTime

            bboxImageL, centXVL, centYVL = self.createBBox(imageL, threshold) # send video frame through detection function
            L_frame = cv2.resize(bboxImageL, (950, 534))
            cv2.putText(L_frame, "FPS: " + str(int(fps)), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2) #fps counter for video
            cv2.imshow("result left", L_frame) # show output image on screen
            cv2.imwrite("../left_out.png", L_frame) # save annotated output image

            bboxImageR, centXVR, centYVR = self.createBBox(imageR, threshold)
            R_frame = cv2.resize(bboxImageR, (950, 534))
            cv2.putText(R_frame, "FPS: " + str(int(fps)), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2) #fps counter for video
            cv2.imshow("result right", R_frame)
            cv2.imwrite("../right_out.png", R_frame)

            centrepoint_L = [centXVL, centYVL] # get centre point coordinates
            centrepoint_R = [centXVR, centYVR]

            self.get_3d_coords(centrepoint_L, centrepoint_R, imageL, imageR, baseline, f, fov, mtxL, mtxR) # use variables in world coordinate calculating function

            key = cv2.waitKey(1) & 0xFF #stops program and closes windows when key is pressed
            if key == ord("q"):
                break

            (successL, imageL) = capL.read()
            (successR, imageR) = capR.read()

        cv2.destroyAllWindows()
        capL.release()
        capR.release()

