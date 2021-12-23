import tkinter as tk
import cv2
import numpy as np
import face_recognition
import os
from PIL import Image, ImageTk
import pickle
from datetime import datetime

# take help of library to make program interactive
window = tk.Tk()
window.title("Face Recognition- GRP(30)")
window.geometry('900x450')
window.configure()


def training(): # trained all the images present in the dataSet
    path = 'dataset'
    images = []  # store all the images present inside the path
    imageName = []  # store the names of all those images

    myList = os.listdir(path)
    print(myList)
    for im in myList:
        curImg = cv2.imread(f'{path}/{im}')
        images.append(curImg)
        imageName.append(os.path.splitext(im)[0])
    with open("Trained_Image.txt", "wb") as fp:  # Pickling --> just dump list into the file
        pickle.dump(images, fp)
    fp.close()
    with open("Trained_ImageName.txt", "wb") as fp:  # Pickling
        pickle.dump(imageName, fp)
    fp.close()

    def findEncodings(imagess): # fetch some information from image
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
    res = "Encodings done!!"
    Notification.configure(text=res, bg="SpringGreen3",
                           width=30, font=('times', 13, 'bold'))
    Notification.place(x=300, y=130)
    on_closing()

def take_img(): # take image from camera
    cam = cv2.VideoCapture(0)
    Name = txt2.get()
    sampleNum = 0
    Id = 1
    count = 0
    while (count!=1):
        with open('IDs.txt','r') as f:
            lines = str(f.readlines())
            lines = lines[2:-2]
            lines = int(lines)
        f.close()
        count+=1
        id = lines
        Id = id
        file = open("IDs.txt", "w")
        id+=1 
        file.write(str(id))
        file.close()
        ret, img = cam.read()
        cv2.imwrite("dataset/ " + Name + "-" + str(Id) + ".jpg", img)
    cam.release()
    cv2.destroyAllWindows()

    res = "Image Saved!!" + " Your id is: " + str(Id)
    Notification.configure(text=res, bg="SpringGreen3",
                           width=30, font=('times', 13, 'bold'))
    Notification.place(x=300, y=130)


window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


def on_closing():
    from tkinter import messagebox
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()


window.protocol("WM_DELETE_WINDOW", on_closing)

Notification = tk.Label(window, text="All things good",
                        bg="Green", fg="white", width=15, height=3)


def testVal(inStr, acttyp):
    if acttyp == '1':  # insert
        if not inStr.isdigit():
            return False
    return True

# below are some gui implementations
message = tk.Label(window, text="IT-204 Project Face-Recognition", bg="purple", fg="black", width=50,
                   height=2, font=('times', 20, 'italic bold '))


message.place(x=50, y=20)

lbl2 = tk.Label(window, text="Name", width=20, fg="black",
                height=2, font=('times', 20, 'italic bold '))
lbl2.place(x=150, y=200)

txt2 = tk.Entry(window, width=30, fg="red")
txt2.place(x=390, y=225)

takeImg = tk.Button(window, text="Take and Save Image", command=take_img, fg="white", bg="blue",
                    width=20, height=2, activebackground="Red", font=('times', 10, 'italic bold '),background='green')
takeImg.place(x=250, y=300)

trainImg = tk.Button(window, text="Encode and Save Images", fg="white", command=training, bg="blue",
                     width=20, height=2, activebackground="Red", font=('times', 10, 'italic bold '),background='green')
trainImg.place(x=500, y=300)

window.mainloop()