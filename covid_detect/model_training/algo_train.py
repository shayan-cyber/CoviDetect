#Data From Kaggle 
#Url : https://www.kaggle.com/tawsifurrahman/covid19-radiography-database




import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
img = cv2.imread('./dataset/COVID/COVID-1.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray)
# plt.show()





#wavelet transform
import pywt

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)
    imArray /= 255;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H

path_to_data = './dataset/'
folder_name_dict ={}
X =[]
y =[]
import os
count =0
for img_folder in os.scandir(path_to_data):
    path_ = img_folder.path
    folder_name_dict[path_.split('/')[-1]] = count

    count = count + 1
    for train_img in os.scandir(path_):
        img = cv2.imread(train_img.path)
        # print(img)
        scalled_raw_img = cv2.resize(img,(32,32))
        img_har = w2d(img,'db1',5)
        scalled_har_img = cv2.resize(img_har,(32,32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_har_img.reshape(32*32*1,1)))
        X.append(combined_img)
        y.append(folder_name_dict[path_.split('/')[-1]])


print(folder_name_dict)
print(len(X))
print(len(y))

#training model

X= np.array(X).reshape(len(X),4096).astype(float)

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.2)


model_params ={
    'logistic_regression':{
        'model':LogisticRegression(solver='liblinear',multi_class='auto',max_iter=20000),
        'params':{
            'logisticregression__C':[1,5,10]


        }
    },
    'svm':{
        'model':SVC(gamma='auto',probability=True),
        'params':{
            'svc__C':[1,5,10],
            'svc__kernel':['linear','rbf']
        }
    },
    'random_forest':{
        'model':RandomForestClassifier(),
        'params':{
            'randomforestclassifier__n_estimators':[1,5,10]
        }
    },
    'knn':{
        'model':KNeighborsClassifier(),
        'params':{
            'kneighborsclassifier__n_neighbors':[5,10,20],
            'kneighborsclassifier__algorithm':['auto', 'ball_tree','kd_tree','brute']
        }
    }
}
scores =[]
best_estimators ={}

for algo_names, algo in model_params.items():
    pipe = make_pipeline(StandardScaler(), algo['model'])
    clf = GridSearchCV(pipe,algo['params'],cv=5,return_train_score=False)
    clf.fit(X_train,y_train)
    scores.append(
        {
            'model':algo_names,

            'best_score':clf.best_score_,
            'best_params': clf.best_params_
        }
    )
    best_estimators[algo_names] = clf.best_estimator_#returns model with best params

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
df = pd.DataFrame(scores,columns=['model','best_score','best_params' ])
print(df)

print(best_estimators['svm'].score(X_test,y_test))
print(best_estimators['logistic_regression'].score(X_test,y_test))
best_clf_svm = best_estimators['svm']
best_clf_lr = best_estimators['logistic_regression']
import joblib
joblib.dump(best_clf_svm,"model_svm_without_viral.pkl")
# joblib.dump(best_clf_lr,"model_lr.pkl")

import json
with open('class_dict.json','w') as  f:
    f.write(json.dumps(folder_name_dict))