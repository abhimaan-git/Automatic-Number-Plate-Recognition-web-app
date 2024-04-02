import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import pytesseract as pt
import plotly.express as px
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xet

from glob import glob
from skimage import io
from shutil import copy
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#parsing data from XML 

path = glob('images/*xml')
labels_dict = dict(filepath=[],xmin=[],xmax=[],ymin=[],ymax=[])
for filename in path:
    info = xet.parse(filename)
    root = info.getroot()
    member_object = root.find('object')
    labels_info = member_object.find('bndbox')
    xmin = int(labels_info.find('xmin').text)
    xmax = int(labels_info.find('xmax').text)
    ymin = int(labels_info.find('ymin').text)
    ymax = int(labels_info.find('ymax').text)

    labels_dict['filepath'].append(filename)
    labels_dict['xmin'].append(xmin)
    labels_dict['xmax'].append(xmax)
    labels_dict['ymin'].append(ymin)
    labels_dict['ymax'].append(ymax)

#converting data into pandas dataframe and save in csv format

df = pd.DataFrame(labels_dict)
df.to_csv('labels.csv',index=False)
df.head()

filename = df['filepath'][0]
def getFilename(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join('images',filename_image)
    return filepath_image
#print(getFilename(filename))

image_path = list(df['filepath'].apply(getFilename))
#print(image_path[:10])


#verifying the data

# file_path = image_path[87]
# img = cv2.imread(file_path)
# img = io.imread(file_path)
# fig = px.imshow(img)
# fig.update_layout(width=600,height=500,margin=dict(l=10,r=10,b=10,t=10),xaxis_title='Figure 8 - N2.jpeg with bounding box')
# fig.add_shape(type='rect', x0=1804, x1=2493, y0=1734, y1=1882, xref='x', yref='y', line_color='cyan')

#DATA PREPROCESSING - converting image to array and resize

labels = df.iloc[:,1:].values #taking all columns 
data = []
output = []
for ind in range(len(image_path)):
    image = image_path[ind]
    img_arr = cv2.imread(image) 
    h,w,d = img_arr.shape
    #preprocessing - converting all images to 224 X 224 size
    load_image = load_img(image,target_size=(224,224))
    load_image_arr = img_to_array(load_image)
    norm_load_image_arr = load_image_arr/255.0 #Normalization
    
    #Normalization of labels
    xmin,xmax,ymin,ymax = labels[ind]
    nxmin,nxmax,nymin,nymax = xmin/w,xmax/w,ymin/h,ymax/h
    label_norm = (nxmin,nxmax,nymin,nymax) #Normalized Output
    
    #Append
    data.append(norm_load_image_arr)
    output.append(label_norm)


#SPLIT TRAIN AND TEST

#converting data to array

X=np.array(data,dtype = np.float32)
y=np.array(output,dtype = np.float32)

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

#Inception-ResNet-V2 Model Building

# inception_resnet = InceptionResNetV2(weights = "imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))

# headmodel = inception_resnet.output
# headmodel = Flatten()(headmodel)
# headmodel = Dense(500,activation='relu')(headmodel)
# headmodel = Dense(250,activation='relu')(headmodel)
# headmodel = Dense(4,activation='sigmoid')(headmodel)

# model = Model(inputs=inception_resnet.input,outputs=headmodel)

# model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4))
# model.summary()


# # #training and saving

# tfb = TensorBoard('object_detection')
# history = model.fit(x=x_train,y=y_train,batch_size=10,epochs=18,validation_data=(x_test,y_test),callbacks=[tfb])
# model.save('./object_detection.h5')

#load model
model = tf.keras.models.load_model('./object_detection.h5')
print('Model loaded Sucessfully')

#Making Predictions

path = 'images/N1.jpeg'
image = load_img(path) #python imaging library(PIL) object
image = np.array(image, dtype=np.uint8)
image1 = load_img(path,target_size=(224,224)) 
image_arr_224 = img_to_array(image1)/255.0 #convert into array and get the normalized output

#size of original image
h,w,d = image.shape
# print('Height of the image =', h)
# print('width of the image =', w)

# cv2.imshow('test',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
fig = px.imshow(image)
fig.update_layout(width=700, height=500, margin=dict(l=10, r=10, b=10, t=10), xaxis_title='TEST Image')
# print(image_arr_224.shape)

test_arr = image_arr_224.reshape(1,224,224,3)
#print(test_arr.shape)

#Making predictions

coords = model.predict(test_arr)
#print(coords)

#Denormalize the values

denorm = np.array([w,w,h,h])
coords = coords*denorm
#print(coords)

#Bounding Box

coords = coords.astype(np.int32)
coords

xmin,xmax,ymin,ymax = coords[0]
pt1 = (xmin,ymin)
pt2 = (xmax,ymax)
print(pt1,pt2)

cv2.rectangle(image,pt1,pt2,(0,255,0),3)
cv2.imshow('result',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Pipeline

path = 'images/N1.jpeg'
def object_detection(path):
    
    # Read image
    image = load_img(path) # PIL object
    image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    image1 = load_img(path,target_size=(224,224))
    
    # Data preprocessing
    image_arr_224 = img_to_array(image1)/255.0 # Convert to array & normalized
    h,w,d = image.shape
    test_arr = image_arr_224.reshape(1,224,224,3)
    
    # Make predictions
    coords = model.predict(test_arr)
    
    # Denormalize the values
    denorm = np.array([w,w,h,h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    
    # Draw bounding on top the image
    xmin, xmax,ymin,ymax = coords[0]
    pt1 =(xmin,ymin)
    pt2 =(xmax,ymax)
    print(pt1, pt2)
    cv2.rectangle(image,pt1,pt2,(0,255,0),3)
    return image, coords

image, cods = object_detection(path)

fig = px.imshow(image)
fig.update_layout(width=700, height=500, margin=dict(l=10, r=10, b=10, t=10),xaxis_title='Figure 14')

# Optical Character Recognition

# img = np.array(load_img(path))
# xmin,xmax,ymin,ymax = cods[0]
# roi = img[ymin:ymax,xmin:xmax]
# fig = px.imshow(roi)
# fig.update_layout(width=350, height=250, margin=dict(l=10, r=10, b=10, t=10),xaxis_title='Figure 15 Cropped image')
# pt.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\ tesseract.exe'
# text = pt.image_to_string(roi)
# print(text)


