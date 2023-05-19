import cv2
import pyttsx3
import numpy as np
import pytesseract
thres = 0.6 # Threshold to detect object
# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set properties for the voice
engine.setProperty('rate', 150)  # Speed of the voice
engine.setProperty('volume', 1)  # Volume of the voice

# Function to run speech output in a separate thread
def speak(text):
    engine.setProperty("voice", engine.getProperty('voices')[0].id)
    engine.say(text)
    engine.runAndWait()
def speakf(text):
    voices=engine.getProperty('voices')
    engine.setProperty("voice",voices[1].id)
    engine.say(text)
    engine.runAndWait()
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(240,240)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
prev_class_ids=[]
frame_count=0
while(True):
        frame_count=frame_count+1
        print(frame_count)
        success,img= cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hImg,wImg,_=gray.shape
        boxes=pytesseract.image_to_data(gray)
        if len(boxes.splitlines()) <= 1:
             break
        for x,b in enumerate(boxes.splitlines()):
             if x!=0:
                  b=b.split()
                  if len(b)==12:
                       x,y,w,h=int(b[6]),int(b[7]),int(b[8]),int(b[9])
                       cv2.rectangle(gray,(x,y),(w+x,h+y),(0,0,255),3)
                       cv2.putText(gray,b[11],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),2)
                       text=b[11]
                       speakf(text)
        classIds, confs, bbox = net.detect(img,confThreshold=thres)
       # print(classIds,bbox)
        new_class_ids = np.setdiff1d(classIds, prev_class_ids)
        disappeared_class_ids=np.setdiff1d(prev_class_ids, classIds)
        if len(classIds) != 0:
            for classId, confidence in zip(disappeared_class_ids.flatten(), confs.flatten()):
                  text = classNames[classId-1]+' disappeared'
                  speak(text)
           # for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            for classId, confidence, box in zip(new_class_ids.flatten(), confs.flatten(), bbox):
           # do something with the new object
                cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                text = classNames[classId-1]+' detected'
                speak(text)
                prev_class_ids =  classIds
        cv2.imshow("Output",img)
        cv2.imshow('Text detection',gray)
        cv2.waitKey(1)