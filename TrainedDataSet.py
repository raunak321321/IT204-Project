import tkinter as tk
import cv2
import numpy as np
import face_recognition
import os
from PIL import Image, ImageTk
import pickle
from datetime import datetime

window = tk.Tk()
window.title("Face Recognition- GRP(30)")
window.geometry('900x450')
window.configure()


def training():
    path = 'dataset'
    images = []  # store all the images present inside the path
    imageName = []  # store the names of all those images

    myList = os.listdir(path)
    print(myList)
    for im in myList:
        curImg = cv2.imread(f'{path}/{im}')
        images.append(curImg)
        imageName.append(os.path.splitext(im)[0])
    with open("Trained_Image.txt", "wb") as fp:  # Pickling
        pickle.dump(images, fp)
    fp.close()
    with open("Trained_ImageName.txt", "wb") as fp:  # Pickling
        pickle.dump(imageName, fp)
    fp.close()

    def findEncodings(imagess):
        encodeList = []
        # count = 0
        for img in imagess:
            currImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # print(count)
            encode = face_recognition.face_encodings(currImg)[0]
            encodeList.append(encode)
            # count+=1
        return encodeList
    encodeListKnown = findEncodings(images)

    with open("Trained_Encodings.txt", "wb") as fp:  # Pickling
        pickle.dump(encodeListKnown, fp)
    print(len(encodeListKnown))
# File_object.write(str(images))
# File_object.close()

# with open("Trained_Image.txt", "rb") as fp:  # Pickling
#     images = pickle.load(fp)
# # print(images)


def take_img():
    cam = cv2.VideoCapture(0)
    # detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # ID = txt.get()
    Name = txt2.get()
    sampleNum = 0
    while (True):
        ret, img = cam.read()
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # faces = detector.detectMultiScale(gray, 1.3, 5)

        # for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # incrementing sample number
        # saving the captured face in the dataset folder
        cv2.imwrite("dataset/ " + Name + ".jpg", img)
        # cv2.imshow('Frame', img)
        # wait for 100 miliseconds
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break
        # break if the sample number is morethan 100
        if sampleNum > 0:
            break
        sampleNum = sampleNum + 1
    cam.release()
    cv2.destroyAllWindows()

    res = "Images Saved  : " + " Name : " + Name
    Notification.configure(text=res, bg="SpringGreen3",
                           width=50, font=('times', 18, 'bold'))
    Notification.place(x=250, y=400)


window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


def on_closing():
    from tkinter import messagebox
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()


window.protocol("WM_DELETE_WINDOW", on_closing)

Notification = tk.Label(window, text="All things good",
                        bg="Green", fg="white", width=15, height=3)

# lbl = tk.Label(window, text="Enter id", width=20, height=2,
#                fg="black", font=('times', 20, 'italic bold '))
# lbl.place(x=200, y=200)


def testVal(inStr, acttyp):
    if acttyp == '1':  # insert
        if not inStr.isdigit():
            return False
    return True


message = tk.Label(window, text="IT-204 Project Face-Recognition", bg="purple", fg="black", width=50,
                   height=2, font=('times', 20, 'italic bold '))


message.place(x=50, y=20)

# txt = tk.Entry(window, validate="key", width=20,  fg="red")
# txt['validatecommand'] = (txt.register(testVal), '%P', '%d')
# txt.place(x=550, y=210)

lbl2 = tk.Label(window, text="Name", width=20, fg="black",
                height=2, font=('times', 20, 'italic bold '))
lbl2.place(x=150, y=200)

txt2 = tk.Entry(window, width=30, fg="red")
txt2.place(x=390, y=225)

takeImg = tk.Button(window, text="Take and Save Image", command=take_img, fg="white", bg="blue",
                    width=20, height=2, activebackground="Red", font=('times', 10, 'italic bold '))
takeImg.place(x=250, y=300)

trainImg = tk.Button(window, text="Encode and Save Images", fg="white", command=training, bg="blue",
                     width=20, height=2, activebackground="Red", font=('times', 10, 'italic bold '))
trainImg.place(x=500, y=300)

window.mainloop()

# print(imageName)


# File_object.write(str(images))
# File_object.close()

# with open("Trained_ImageName.txt", "rb") as fp:  # Pickling
#     encodeListKnown = pickle.load(fp)
