import cv2
import numpy as np
import pandas 
import torch
import math
from PIL import Image, ImageDraw
#import streamlit as st

import streamlit as st 
#st.markdown(""" **Sign Language Detection** """)

#File upload widget
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.markdown(""" **Abilities Infinite** """)

    
    
    from roboflow import Roboflow
    rf = Roboflow(model_format="yolov5", notebook="roboflow-yolov5")
    rf = Roboflow(api_key="X9dGHYzCX3ODnOoJl0Cn")
    project = rf.workspace("tejashri-dendi-wubeu").project("american-sign-language-letters-mjfzz")
    
    model = project.version(2).model
    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    # Predictions of model:[(x,y),w,h, class , confidence] 
    predictionX = model.predict(opencv_image).json()['predictions'][0]['x']
    predictionY = model.predict(opencv_image).json()['predictions'][0]['y']
    predictionWidth = model.predict(opencv_image).json()['predictions'][0]['width']
    predictionHeight = model.predict(opencv_image).json()['predictions'][0]['height']
    predictionclass = model.predict(opencv_image).json()['predictions'][0]['class']
    predictionConfidence = model.predict(opencv_image).json()['predictions'][0]['confidence']
    
    predictionConfidence=round(predictionConfidence, 2)
    predictionConfidence=predictionConfidence*100
    
    #Draw Bounding Box
    x0 = int(predictionX - predictionWidth / 2)
    x1 = int(predictionX + predictionWidth / 2)
    y0 = int(predictionY - predictionHeight / 2)
    y1 = int(predictionY + predictionHeight / 2)
    Outputmg=cv2.rectangle(opencv_image, (x0, y0), (x1, y1),(0, 255, 0), 2)
    print(model.predict(opencv_image).json())
    Outputmg=cv2.putText(opencv_image,str(predictionclass),(x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 0, 240), 2)
    
    Outputmg=cv2.putText(opencv_image,str(predictionConfidence),(x0+30, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 0, 240), 2)
    #Outputmg=cv2.putText(opencv_image,str(predictionclass),(x1+10, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 0, 240), 2)
    
    #Outputmg=cv2.putText(opencv_image,str(predictionConfidence),(x1+10, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 0, 240), 2)
    # show the output image
    st.image(Outputmg, channels="BGR")
