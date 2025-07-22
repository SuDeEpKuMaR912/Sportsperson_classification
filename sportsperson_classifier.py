import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import cv2
import os
import shutil
import pywt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from joblib import load, dump
import json

#READING IMAGE USING OPENCV
img= cv2.imread('./test_images/sharapova1.jpg')
'''print(img.shape)                             #(555, 700, 3)
plt.imshow(img)'''

gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
'''print(gray.shape)                            #(555, 700)
plt.figure()
plt.imshow(gray, cmap='gray')
plt.show()'''

#IMPORTING HAAR CASACDE FOR FACE AND EYE DETECTION
face_cascade= cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eye_cascade= cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')

faces= face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
#print(faces)                            #[[352  38 233 233]]

(x,y,w,h)= faces[0]

#RECTANGULAR BOX ON DETECTED FACE
face_img= cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
'''plt.imshow(face_img)      
plt.show()'''

#FOR EYES
for (x,y,w,h) in faces:
    face_img= cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    roi_gray= gray[y:y+h, x:x+w]
    roi_color= face_img[y:y+h, x:x+w]
    eyes= eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

'''plt.figure()
#plt.imshow(face_img, cmap='gray')                        #THIS SHOWS COMPLETE IMAGE WITH RECTANGLE AROUND FACE AND EYE
plt.imshow(roi_color, cmap='gray')                        #THIS SHOWS CROPPED FACE
plt.show()'''

#FUNCTION FOR DOING THIS FOR OTHER IMAGES FOR TRAINING DATA
def get_cropped_image_if_2_eyes(image_path):
    img= cv2.imread(image_path)
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray= gray[y: y+h, x:x+w]
        roi_color= img[y:y+h, x:x+w]
        eyes= eye_cascade.detectMultiScale(roi_gray)
        if len(eyes)>=2:
            return roi_color
        
#TESTING THIS FUNCTION WITH SOME IMAGES
cropped_image= get_cropped_image_if_2_eyes('./test_images/sharapova1.jpg')
'''plt.imshow(cropped_image)
plt.show()'''

cropped_image2= get_cropped_image_if_2_eyes('./test_images/sharapova2.jpg')
#print(cropped_image2)

#SETTING UP PATH VARIABLE
path_to_data= "./dataset/"
path_to_cr_data= "./dataset/cropped/"

#ADD ALL DIRECTORIES INSIDE DATASETIMG FOLDER IN A LIST
img_dirs= []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)


#CREATING CROPPED FOLDER
# Skip image cropping if cropped folder already exists
if not os.path.exists(path_to_cr_data):
    os.mkdir(path_to_cr_data)
    run_crop_process = True
else:
    print("Cropped image folder already exists. Skipping cropping step.")
    run_crop_process = False


#ITERATING THROUGH IMAGE DIRECTORIES AND CROPPING IMAGES INTO CROPPED FOLDER USING FUNCTION WE MADE
cropped_image_dirs= []
celebrity_file_name_dict= {}

for img_dir in img_dirs:
    if run_crop_process:
     for img_dir in img_dirs:
        count = 1
        celebrity_name = img_dir.split('/')[-1]

        for entry in os.scandir(img_dir):
            roi_color = get_cropped_image_if_2_eyes(entry.path)
            if roi_color is not None:
                cropped_folder = os.path.join(path_to_cr_data, celebrity_name)

                if not os.path.exists(cropped_folder):
                    os.makedirs(cropped_folder)
                    cropped_image_dirs.append(cropped_folder)
                    print("Generating cropped in folder:", cropped_folder)

                cropped_file_name = celebrity_name + str(count) + ".png"
                cropped_file_path = os.path.join(cropped_folder, cropped_file_name)

                cv2.imwrite(cropped_file_path, roi_color)
                celebrity_file_name_dict.setdefault(celebrity_name, []).append(cropped_file_path)
                count += 1
    else:
    # If cropping skipped, rebuild celebrity_file_name_dict from existing files
     for entry in os.scandir(path_to_cr_data):
         if entry.is_dir():
            celebrity_name = entry.name
            file_paths = [os.path.join(entry.path, f) for f in os.listdir(entry.path) if f.endswith(".png")]
            celebrity_file_name_dict[celebrity_name] = file_paths


#WAVELET TRANSFORM FOR FEATURE GENERATION
#WAVELET TRANSFORM USES CONCEPT OF SIGNAL PROCESSING IN TERMS OF IMAGES AND FOURIER TRANSFORM
def w2d(img, mode='haar', level=1):
    imArray=img
    imArray= cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    imArray=np.float32(imArray)
    imArray/=255
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    coeffs_H= list(coeffs)
    coeffs_H[0]*=0

    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H*=255
    imArray_H=np.uint8(imArray_H)

    return imArray_H

#USING DEMO IMAGE FROM BEFORE IN THIS FUNCTION
im_har= w2d(cropped_image, 'db1', 5)
'''plt.imshow(im_har, cmap='gray')
plt.show()'''                              #WORKS!!!

#MARKING CELEBRITY NAME WITH A NUMBER SINCE MODELS TAKE NUMBERS
class_dict={}
count=0
for celebrity_name in celebrity_file_name_dict.keys():
    class_dict[celebrity_name]=count
    count=count+1

#print(class_dict)


#NOW TO IMPLEMENT THIS ON ALL OUR CROPPED IMAGES AND SEPARTING INTO FEATURES AND LABELS
x=[]
y=[]

for celebrity_name, training_files in celebrity_file_name_dict.items():
    for training_image in training_files:
        img= cv2.imread(training_image)
        if img is None:
            continue
        scaled_raw_img= cv2.resize(img, (32,32))
        img_har=w2d(img, 'db1', 5)
        scaled_img_har= cv2.resize(img_har, (32,32))
        combined_img=np.vstack((scaled_raw_img.reshape(32*32*3,1), scaled_img_har.reshape(32*32,1)))
        x.append(combined_img)
        y.append(class_dict[celebrity_name])

'''print(len(x))              #164
print(len(x[0]))           #4096
print(x)'''

x= np.array(x).reshape(len(x), 4096).astype(float)


#MODEL TRAINING
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=0)

#first just checking with SVC
pipe= Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', C=10))])
pipe.fit(x_train, y_train)

#print(pipe.score(x_test, y_test))    #0.9090909090909091
#print(len(x_test))
#print(classification_report(y_test, pipe.predict(x_test)))

#NOW HYPERTUNING USING GRIDSEARCHCV AND DIFFERENT MODELS TO FIND BEST ONE
model_params= {
    'svm': {
        'model': SVC(gamma='auto', probability=True),
        'params': {
            'svc__C': [1,10,100,1000],
            'svc__kernel': ['rbf', 'linear']
        }
    },
     'random_forest': {
          'model': RandomForestClassifier(),
          'params': {
               'randomforestclassifier__n_estimators': [1,5,10]
          }
     },
     'logistic_regression': {
          'model': LogisticRegression(),
          'params': {
               'logisticregression__C': [1,5,10]
          }
     }
}

scores=[]
best_estimators={}

for algo, mp in model_params.items():
    pipe= make_pipeline(StandardScaler(), mp['model'])
    clf= GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(x_train, y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo]= clf.best_estimator_

df=pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
#print(df)       
#print(best_estimators)

#print(best_estimators['logistic_regression'].score(x_test, y_test))    #0.9393939393939394
#print(best_estimators['random_forest'].score(x_test, y_test))           #0.8787878787878788
#print(best_estimators['svm'].score(x_test, y_test))                     #0.9696969696969697

#SO WE ARE GETTING DIFFERENT RESULTS FOR k fold cross: LogisticRegression AND Test data gives: SVM
#SO WE CAN CHOOSE ANY OF THOSE

#IM USING SVM
best_clf= best_estimators['svm']

#plotting its confusion matrix
cm= confusion_matrix(y_test, best_clf.predict(x_test))
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
#plt.show()

#SAVING OUR MODEL
dump(best_clf, 'saved_model.pkl')

#SAVE CLASS DICTIONARY
with open('class_dictionary.json', 'w') as f:
    f.write(json.dumps(class_dict))
