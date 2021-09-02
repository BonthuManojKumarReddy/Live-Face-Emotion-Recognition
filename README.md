# Live-Face-Emotion-Recognition
Facial emotion recognition is the process of detecting human emotions from facial expressions. The human brain recognizes emotions automatically, and software has now been developed that can recognize emotions as well.This technology is becoming more accurate all the time, and will eventually be able to read emotions as well as our brains do.

# Introduction
Audiovisual emotion recognition is not a new problem. There has been a lot of work in visual pattern recognition for facial emotional expression recognition, as well as in signal processing for audio-based detection of emotions, and many multimodal approaches combining these cues. However, improvements in hardware, availability of datasets and wide-scale annotation infrastructure made it possible to create real affective systems a reality, and we now see applications across many domains, including robotics, HCI, healthcare, and multimedia.

# Problem Statement
The Indian education landscape has been undergoing rapid changes for the past 10 years owing to the advancement of web-based learning services, specifically, eLearning platforms. Global E-learning is estimated to witness an 8X over the next 5 years to reach USD 2B in 2021. India is expected to grow with a CAGR of 44% crossing the 10M users mark in 2021. Although the market is growing on a rapid scale, there are major challenges associated with digital learning when compared with brick and mortar classrooms. One of many challenges is how to ensure quality learning for students. Digital platforms might overpower physical classrooms in terms of content quality but when it comes to understanding whether students are able to grasp the content in a live class scenario is yet an open-end challenge. In a physical classroom during a lecturing teacher can see the faces and assess the emotion of the class and tune their lecture accordingly, whether he is going fast or slow. He can identify students who need special attention.

Digital classrooms are conducted via video telephony software program (ex-Zoom) where it’s not possible for medium scale class (25-50) to see all students and access the mood. Because of this drawback, students are not focusing on content due to lack of surveillance. While digital platforms have limitations in terms of physical surveillance but it comes with the power of data and machines which can work for you. It provides data in the form of video, audio, and texts which can be analyzed using deep learning algorithms. Deep learning backed system not only solves the surveillance issue, but it also removes the human bias from the system, and all information is no longer in the teacher’s brain rather translated in numbers that can be analyzed and tracked.

we are going to slove this with help of sequential CNN model by producing a solution to facial emotion recognition.

## Dataset Information
 Deep learning model which detects the real time emotions of students through a webcam so that teachers can understand if students are able to grasp the topic according to students' expressions or emotions and then deploy the model. The model is trained on the Face expression recognition dataset dataset .
   This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.
Here is the dataset link:-  https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset

## Dependencies

1)	Python 3
2)	Tensorflow 2.0
3)	Streamlit
4)	Streamlit-Webrtc
5)	OpenCV

## Working Sample Video
 
 ![Demo.gif](https://github.com/BonthuManojKumarReddy/Live-Face-Emotion-Recognition/blob/main/Working%20video%20in%20local%20machine%20as%20gif.gif)
 

 
 ## Model Creation
# 1. Using CNN with the help of Keras
   Deep learning is a very significant subset of machine learning because of its high performance across various domains. Convolutional Neural Network (CNN), is a powerful image processing deep learning type often using in computer vision that comprises an image and video recognition along with a recommender system and natural language processing ( NLP).
CNN uses a multilayer system consists of the input layer, output layer, and a hidden layer that comprises multiple convolutional layers, pooling layers, fully connected layers. We will discuss all layers in the next section of the article while explaining the building of CNN.

![Optional Text](https://github.com/BonthuManojKumarReddy/Live-Face-Emotion-Recognition/blob/main/Media%20files/CNN_model%20image.jpg)

Kaggle Notebook link :https://github.com/BonthuManojKumarReddy/Live-Face-Emotion-Recognition/blob/main/Kaggle%20Notebooks/emotion-detection-using-cnn%20(1).ipynb

• Initially built model has got training accuracy as 98% and validation accuracy as 75%. This means our model is overfitted.

• Second model was built by doing Data Augmentaion on the first model and done some hyperparameter tuning. This model has given training accuracy as 90% and validation accuracy as 74.2% 

 
  ## Model accuracy and loss plot
  ![](https://github.com/BonthuManojKumarReddy/Live-Face-Emotion-Recognition/blob/main/Media%20files/Loss%20and%20accuracy%20plot%20(1).jpg)
  
   ## Deployment of models
 # 1.Deployment code for OpenCV using local machine.
    using Jypyter Notebook with model saved by cnn with best weights.
  https://github.com/BonthuManojKumarReddy/Live-Face-Emotion-Recognition/tree/main/Kaggle%20Notebooks/OpenCv%20code
    
  # 2.Deployment in Streamlit webapp
    Note : please try to start by selecting device, ignore the error code shown press start()
  https://share.streamlit.io/bonthumanojkumarreddy/live-face-emotion-recognition/main
  
  
  ## Concluding Summary
   So Here, Finally I have build a Web App by Using CNN model, which as training accuracy of 74% and validation accuracy of 68%

