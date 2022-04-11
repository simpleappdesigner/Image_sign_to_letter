
import cv2
import numpy as np
import tensorflow.lite as tflite
from PIL import Image
import tensorflow as tf
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

#Load label and model
#build label dictionary
with open("converted_keras/labels.txt") as fl:
    label_lines=fl.read().splitlines()
labels={} # "0" --> A, "1"-->"B"
for each_l in label_lines:
    labels[each_l.split()[0]]=each_l.split()[1]

#load the model
model=tf.keras.models.load_model('converted_keras/keras_model.h5')

def predict(file_name):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(file_name).convert('RGB')#
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    #print(prediction)
    print("Max Predicted label : ",labels[str(np.argmax(prediction[0]))])
    max_3=(-prediction[0]).argsort()[:2]
    for e_label in max_3:
        print("Top", labels[str(e_label)])
    return labels[str(np.argmax(prediction[0]))]

cap = cv2.VideoCapture(0)
res, score = '', 0.0
i = 0
mem = ''
consecutive = 0
sequence = ''
c=0
while True:
    ret, img = cap.read()
    #img = cv2.flip(img, 1)    
    if ret:
        #x1, y1, x2, y2 = 200, 100, 424, 324
        #img_cropped = img[y1:y2, x1:x2]
        a = cv2.waitKey(1) # waits to see if `esc` is pressed
        #cv2.waitKey()
        img = cv2.flip(img, 1)
        cv2.imwrite('test.png', img)
        predict("test.png")
        cv2.imshow("img1", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     
        if a == 27: # when `esc` is pressed
            break
        cv2.waitKey(0)

cv2.destroyAllWindows() 
cv2.VideoCapture(0).release()