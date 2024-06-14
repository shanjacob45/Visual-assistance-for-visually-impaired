# import modules
import cv2
import numpy as np
import threading
import pyttsx3
from datetime import datetime
import matplotlib.pyplot as plt
import queue

# initialize speech module
speech_engine = pyttsx3.init()
object_detected = queue.Queue()

cfg_name = 'data/configs.cfg'
model_name = 'data/model.weights'
class_file = 'data/classes.names'

# load model
net = cv2.dnn.readNetFromDarknet(cfg_name,model_name)

# open classes file and get classes
classes = []
with open(class_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# thread for speech processing
def speech_processing():
    last_object = ""
    last_object_detected_time = datetime.now()
    while True:
        object_name = object_detected.get()
        # if new object is detected or object i detected time is grater thane 10 sec
        if object_name != last_object or (datetime.now() - last_object_detected_time).seconds > 10:
            if(object_name == "quit"):
                return

            # speak object name
            speech_engine.say(object_name)
            speech_engine.runAndWait()
            last_object_detected_time = datetime.now()
            last_object = object_name
        
# start speech processing thread
thread = threading.Thread(target=speech_processing)
thread.start()

# start video capture from web cam
cap = cv2.VideoCapture(1)
while True:
    boxes =[]
    confidences = []
    class_ids = []

    # read frame and resize
    _, img = cap.read()
    hight,width,_ = img.shape
    img = cv2.resize(img,(640,480))
    blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)
    net.setInput(blob)
    output_layers_name = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_name)

    
    # find bounding boxes to draw rectangles around all detected object based on confidence
    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3]* hight)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    # draw recangle and label around objects
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.8,.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size =(len(boxes),3))
    if  len(indexes)>0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.putText(img,label + " " + confidence, (x,y),font,1,color,2)
            object_detected.put(label)
    
    # show output on application winow
    cv2.imshow('Realtime video',img)
    if cv2.waitKey(1) == ord('q'):
        object_detected.put("quit")
        break

# close all windows 
cap.release()
cv2.destroyAllWindows()
exit(0)
