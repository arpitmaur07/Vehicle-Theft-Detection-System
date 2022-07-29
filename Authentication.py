import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import face_recognition
import os
from playsound import playsound
from threading import Thread

path='C:/Users/Arpit Maurya/Desktop/Programs/car/1st/images'
images=[]
presonName=[]
myList=os.listdir(path)
for cu_Img in myList:
    currentImg=cv2.imread(f'{path}/{cu_Img}')
    images.append(currentImg)
    presonName.append(os.path.splitext(cu_Img)[0])
alarmOn=False

def soundAlerm(soundFile):
    while alarmOn==True:
        playsound(soundFile)
        
def registraction():
    
    root= tk.Tk()
    root.geometry("500x250")
    root.title('Registration')
    root.configure(bg="#116562")

    global entry
    def openingCamera():
        name= entry.get()
        if(len(name)==0):
            messagebox.showwarning("Invalid Name Warning", "Please Enter a Valid Name!... ")
        else:             
            entry.pack_forget()
            label.pack_forget()
            B.pack_forget()
            root.title('Register Your Face')
            root.minsize(646,530)
            root.maxsize(646,530)
            root.configure(bg='#58F')
            face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            cap= cv2.VideoCapture(0)
            if (cap.isOpened() == False):
                print("Unable to read camera feed")
                
            def captureImage():
                image=Image.fromarray(img1)  
                personName=str(name)+'.jpg'
                image.save(f'C:/Users/Arpit Maurya/Desktop/Programs/car/1st/images/{personName}')
                cap.release()
                cv2.destroyAllWindows()
                root.destroy()
                root.quit()  
                messagebox.showinfo('Registasion SuccessFul', 'Face Registasion Completed SuccessFully!!') 
            
            def exitWindow():
                cap.release()
                cv2.destroyAllWindows()
                root.destroy()
                root.quit()  

            f1=tk.LabelFrame(root,bg='red')
            f1.pack()
            l1=tk.Label(f1,bg='blue')
            l1.pack()

            b1=tk.Button(root,bg='green',fg='white',activebackground='white',activeforeground='green',text='REGISTER FACE 📷',relief=tk.RIDGE,height=2,width=30,command=captureImage)
            b1.pack(side=tk.LEFT,padx=60,pady=5)
            b2=tk.Button(root,fg='white',bg='red',activebackground='white',activeforeground='red',text='LATER ❌ ',relief=tk.RIDGE,height=2,width=20,command=exitWindow)
            b2.pack(side=tk.LEFT,padx=40,pady=5) 
              
            while True:
                img=cap.read()[1]
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces=face_cascade.detectMultiScale(gray,1.1,4,minSize=(60,60))
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
                img=cv2.flip(img,1)
                img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img=ImageTk.PhotoImage(Image.fromarray(img1))  
                l1['image']=img
                root.update()
            cap.release()
            
    label=tk.Label(root, text=" Enter your Name", font=("Courier 20 bold"),fg="white",bg='#116562',height=2)
    label.pack(pady=10,fill='x')
    entry= tk.Entry(root, width= 40)
    entry.focus_set()
    entry.pack(pady=10)
    B=tk.Button(root, text= "Next",width= 22,bg='green',fg='white',height=2,activebackground='white',activeforeground='red',font=("none 9 bold"), command= openingCamera)
    B.pack(padx=32,pady=10)
    root.mainloop()

def process():
    def faceEncodings(images):
        encodeList=[]
        for img in images:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            encode=face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList    
    encodeListKnown=(faceEncodings(images))
    print('All encoding complete!!! ')
    cap=cv2.VideoCapture(0)
    while True:
        ret, frame= cap.read()
        faces=cv2.resize(frame,(0,0), None,0.25,0.25)
        faces=cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)
        facesCurrentframe=face_recognition.face_locations(faces)
        encodeCurrentFrame=face_recognition.face_encodings(faces,facesCurrentframe)
        for encodeFace, faceLoc in (zip(encodeCurrentFrame,facesCurrentframe)):
            matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)            
            matchIndex=np.argmin(faceDis)
            if matches[matchIndex]:
                pass

            else:
                global alarmOn
                if not alarmOn:
                    alarmOn=True  
                    t=Thread(target=soundAlerm, args=("C:/Users/Arpit Maurya/Desktop/Programs/car/1st/alarm.wav",))
                    t.daemon=True
                    t.start()
        alarmOn=False    
        cv2.imshow('RECOGNIZING....',frame) 
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()        
    cv2.destroyAllWindows()
    
    
    
# ----------------------------------------Functaion Call------------------------        
  

if len(images)<=5:  
    registraction()
if(len(images)>0):
    process()
    
    