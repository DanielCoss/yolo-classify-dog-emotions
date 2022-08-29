import cv2 as cv
import numpy as np
import tensorflow as tf
from io import BytesIO
import requests
from PIL import Image


class Prediction:
    def __init__(self, image_width = 224, image_height = 224, batch_size = 32, image_channels = 3 ,model = "models\efficientnet"):
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.image_channels = image_channels
        self.image_size=(image_width, image_height)
        self.model = tf.keras.models.load_model(model)
        
        
    def __process(self,img):
        img = np.array(img).astype(float)/255

        img = cv.resize(img, self.image_size)
        prediction = self.model.predict(img.reshape(-1,self.image_width,self.image_height,self.image_channels))
        if (np.argmax(prediction[0], axis =-1) == 0):
            result = "angry"
        else:
            if (np.argmax(prediction[0], axis =-1) == 1):
                result = "happy"
            else: 
                result = "sad"
        return result
                
        
    def predictURL(self, url):
        respond = requests.get(url)
        img = Image.open(BytesIO(respond.content))
        return self.__process(img)
        
    def predictImageDirection(self, imgSource):
        img = Image.open(imgSource)
        return self.__process(img)
    
    def predictImage(self, img):
        reply = self.__process(img)
        return reply
    
    # def predictJSON(self, json):

# prediccion = Prediction()

# url = "https://i.pinimg.com/564x/9c/3d/25/9c3d252aa0bcebf1bf88d91ca043612a.jpg"
# source = "images/dog 4.jpeg"

# print (prediccion.predictImageDirection(source))