from django.http import request
from django.shortcuts import render,HttpResponse
import cv2
import numpy as np
import base64
import joblib
from numpy.core.defchararray import join
import sklearn 

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



def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    return img
def clf(img_base64):
    result =[]
    img = get_cv2_image_from_base64_string(img_base64)
    scalled_raw_img = cv2.resize(img, (32,32))
    img_har = w2d(img,'db1',5)
    scalled_har_img = cv2.resize(img_har,(32,32))
    combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_har_img.reshape(32*32*1,1)))
    len_img_array = 32*32*3 + 32*32
    final = combined_img.reshape(1,len_img_array).astype(float)
    with open('static/model_svm.pkl','rb') as f:
        model_ = joblib.load(f)
    result.append({
        'prediction':model_.predict(final)[0],
        'probability': np.round(model_.predict_proba(final),2).tolist()[0]
    })
    return result





# Create your views here.
def home(request):
    return render(request,'home.html')

def classify_xray(request):
    if request.method =="POST":
        print('post')
        img_string = request.POST.get('img_string','')
        pred = clf(img_string)
        keys_ = ['COVID-19', 'Other Lung Infection', 'Normal', 'Viral Pneumonia']
        print(pred)
        pred_no = pred[0]['prediction']
        prediction = keys_[pred_no]
        probability = int(float(pred[0]['probability'][pred_no])*100)
        context ={
            'prediction':prediction,
            'probability':probability
        }
    return render(request, 'classify.html',context)

# def test_(request):
#     return render(request,'test.html')

