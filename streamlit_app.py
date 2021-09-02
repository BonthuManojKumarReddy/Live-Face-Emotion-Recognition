# -*- coding: utf-8 -*-
"""
@author: manoj kumar
"""
import streamlit as st
st.title('Face emotion recognition')
import av
import cv2
import numpy as np
import streamlit as st
from aiortc.contrib.media import MediaPlayer
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import load_model
from tensorflow.keras import layers


my_model=load_model('model_2bestweights.h5')


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
        
        class_labels = ['Fear','Angry','Neutral','Happy']


        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_roi = face_detect.detectMultiScale(img_gray, 1.3,1)
        if face_roi is ():
            return img

        for(x,y,w,h) in face_roi:
            x = x - 5
            w = w + 10
            y = y + 7
            h = h + 2
            
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255), 2)
            img_color_crop = img[y:y+h,x:x+w]
            img_color_crop = img[y:y+h,x:x+w]                        
            final_image = cv2.resize(img_color_crop, (48,48))
            final_image = np.expand_dims(final_image, axis = 0)
            final_image = final_image/255.0
            prediction = my_model.predict(final_image)
            label=class_labels[prediction.argmax()]
            cv2.putText(img,label, (30,80), cv2.FONT_HERSHEY_SIMPLEX,2, (18,10,200),1)    
        return img
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)



