import cv2
from keras.models import load_model
import numpy as np


face_detect = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')  

def face_detection(img,size=0.5):
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    face_roi = face_detect.detectMultiScale(img_gray, 1.3,1)   
    
    class_labels = ['Fear','Angry','Neutral','Happy']
                                                              
     
    if face_roi is ():                                           
        return img
    
    
    for(x,y,w,h) in face_roi:                                     
        x = x - 5
        w = w + 10
        y = y + 7
        h = h + 2
        
        
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255), 2)  
        
        
        img_gray_crop = img_gray[y:y+h,x:x+w]  
        
        img_color_crop = img[y:y+h,x:x+w]                        
        
       
        model=load_model(r'model_2bestweights.h5')
        
        final_image = cv2.resize(img_color_crop, (48,48),interpolation=cv2.INTER_AREA)  
       
        final_image = np.expand_dims(final_image, axis = 0)    
       
        final_image = final_image/255.0    
        
        
        prediction = model.predict(final_image)  
       
        label=class_labels[prediction.argmax()]                    
        cv2.putText(frame,label, (30,80), cv2.FONT_HERSHEY_SIMPLEX,2, (18,10,200),1)  
                                                                 

     
    img_color_crop = cv2.flip(img_color_crop, 1)                 
    return img


cap = cv2.VideoCapture(0)                                         

while True:
    ret, frame = cap.read()
   
    cv2.imshow('LIVE', face_detection(frame))                     
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
