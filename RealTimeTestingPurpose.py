import cv2
import numpy as np
import face_recognition
import os
import pickle
from datetime import datetime, time


# below method is used for creating attendance of all such persons which were captured during live sessions
def  attendance(name):
    with open ('attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for Line in myDataList:
            entry = Line.split(',')
            nameList.append(entry[0])
            # print(entry[0])

        if name not in nameList:
            timeNow = datetime.now()
            timeStr = timeNow.strftime('%H/%M/%S')
            dateStr = timeNow.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{timeStr},{dateStr}')

with open("Trained_Image.txt", "rb") as fp:  # Pickling
    images = pickle.load(fp)
fp.close()

with open("Trained_Encodings.txt", "rb") as fp:  # Pickling-> just take the list as it is from file
    encodeListKnown = pickle.load(fp)

fp.close() # close the file
    
with open("Trained_ImageName.txt", "rb") as fp:  # Pickling
    imageName = pickle.load(fp)
fp.close()
print(len(encodeListKnown))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# below code first take the image as input and recognise it using all the encodings which were present in the dataSet folder
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 1, 1)  # image,pixelSize,,Scales
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrFrame = face_recognition.face_locations(imgS)
    encodingsCurrFrame = face_recognition.face_encodings(imgS, facesCurrFrame)

    for encodingFace, faceLoc in zip(encodingsCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodingFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodingFace)

        matchIndex = np.argmin(faceDis)
        if(matches[matchIndex]):
            name = imageName[matchIndex].upper()
            cv2.rectangle(img, (faceLoc[3], faceLoc[0]),(faceLoc[1], faceLoc[2]), (0, 255, 0), 4)
            cv2.rectangle(img,(faceLoc[3],faceLoc[2]-35),(faceLoc[1],faceLoc[2]),(0,255,0),cv2.FILLED)
            cv2.putText(img, f'{name}', (faceLoc[3]+6, faceLoc[2]-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            attendance(name)
    cv2.imshow('capture', img)
    if cv2.waitKey(10) == 13 :
        break
cap.release()
cv2.destroyAllWindows()
