import cv2
import numpy as np 
import face_recognition

# this is step1
imgElon = face_recognition.load_image_file('ImageBasic/Elon Musk.jpeg') # take image in bgr form
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB);
imgTest = face_recognition.load_image_file('ImageBasic/Elon Test.jpg') # take image in bgr form
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB);
img2Test = face_recognition.load_image_file('ImageBasic/unknown.jpg') # take image in bgr form
img2Test = cv2.cvtColor(img2Test,cv2.COLOR_BGR2RGB);

# Now move to step2
faceLoc = face_recognition.face_locations(imgElon)[0] # this tells us the face locations
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2) # image,location,color,thickness

faceLocTest = face_recognition.face_locations(imgTest)[0] # this tells us the face locations
encodeElonTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2) # image,location,color,thickness

faceLocTest2 = face_recognition.face_locations(img2Test)[0] # this tells us the face locations
encodeTest2 = face_recognition.face_encodings(img2Test)[0]
cv2.rectangle(img2Test,(faceLocTest2[3],faceLocTest2[0]),(faceLocTest2[1],faceLocTest2[2]),(255,0,255),2) # image,location,color,thickness

# now we are comparing these encodings

results = face_recognition.compare_faces([encodeElon],encodeElonTest) # first parameter is the list of all faces which we have to check and second one is the testing image 
results2 = face_recognition.compare_faces([encodeElon],encodeTest2) # first parameter is the list of all faces which we have to check and second one is the testing image 
faceDis = face_recognition.face_distance([encodeElon],encodeElonTest) # first parameter is the list of all faces which we have to check and second one is the testing image 
faceDis2 = face_recognition.face_distance([encodeElon],encodeTest2) # first parameter is the list of all faces which we have to check and second one is the testing image 
# print(results,faceDis)

cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2) # image,string,origin,font,scale,color,thickness
cv2.putText(img2Test,f'{results2} {round(faceDis2[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2) # image,string,origin,font,scale,color,thickness

cv2.imshow('Elon Musk',imgElon) # this will show the image of the elon musk
cv2.imshow('Elon Test',imgTest) # this will show the image of the elon test
cv2.imshow('unknown',img2Test) # this will show the image of the elon test
cv2.waitKey(0)

