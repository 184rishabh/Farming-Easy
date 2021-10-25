#IMPORTNG LIBRARIES
from flask import Flask,render_template,request
import keras
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import pickle
import cv2
import io
from PIL import Image
from numpy.lib.shape_base import tile
from tensorflow.python.keras.preprocessing.image import img_to_array

app = Flask(__name__)

#ROUTES 

#HOME PAGE
@app.route("/")
def home():
    title="HOME"
    return render_template('index.html',title=title)

#CROP RECOMMENDATION PAGE
@app.route("/crop")
def crop():
    title="CROP-RECOMMENDATION"
    return render_template('crop.html',title=title)

#CROP RECOMMENDATION RESULTS
@app.route('/predict',methods=['POST'])   
def predict():
    if request.method=='POST' :
       v1=request.form['nitrogen']
       v2=request.form['phosphorus']
       v3=request.form['potassium']
       v4=request.form['temp']
       v5=request.form['humidity']
       v6=request.form['ph']
       v7=request.form['rain']
       v1=int(v1)
       v2=int(v2)
       v3=int(v3)
       v4=int(v4)
       v5=int(v5)
       v6=int(v6)
       v7=int(v7)
       model =  pickle.load(open('crop_recommendation','rb'))
       y = int(model.predict([[v1,v2,v3,v4,v5,v6,v7]]))
       dict={
            1:'apple',
            2:'banana',
            3:'blackgram',
            4:'chickpea',
            5:'coconut',
            6:'coffee',
            7:'cotton',
            8:'grapes',
            9:'jute',
            10:'kidneybeans',
            11:'lentil',
            12:'maize',
            13:'mango',
            14:'mothbeans',
            15:'mungbean',
            16:'muskmelon',
            17:'orange',
            18:'papaya',
            19:'pigeonpeas',
            20:'pomegranate',
            21:'rice',
            22:'watermelon'
       }
       print(dict[y+1])
       title="CROP-RESULT"
    return  render_template('result.html',data=[dict[y+1]],title=title)

#FERTILIZER RECOMMENDATION PAGE
@app.route("/fertilizer")
def fertilizer():
    title="FERTILIZER-RECOMMENDATION"
    return render_template('fertilizer.html',title=title)

#FERTILIZER RESULTS
@app.route('/predict1',methods=['POST'])   
def predict1():
    if request.method=='POST' :
       #dict2={'Barley':0, 'Cotton':1, 'Ground Nuts':2, 'Maize':3, 'Millets':4, 'Oil seeds':5, 'Paddy':6, 'Pulses':7, 'Sugarcane':8, 'Tobacco':9, 'Wheat':10} 
       v1=request.form['moisture']
       v2=request.form['croptype'] 
       v3=request.form['nitrogen']
       v4=request.form['phosphorus']
       v5=request.form['potassium']
       
       v1=int(v1)
       v2=int(v2)
       v3=int(v3)
       v4=int(v4)
       v5=int(v5)   
       model =  pickle.load(open('fertilizer','rb'))
       y = int(model.predict([[v1,v2,v3,v4,v5]])) 
       dict1={1:'10-26-26', 2:'14-35-14', 3:'17-17-17', 4:'20-20', 5:'28-28', 6:'DAP', 7:'Urea'}
       print(v2)
       print(dict1[y+1])
       title="FERTILIZER-RESULT"
    return render_template('fresult.html',data=[dict1[y+1]],title=title)

#DISEASE PREDICTION
@app.route("/disease-predicton")
def diseasepred():
    title="DISEASE-PREDICTION"
    return render_template('disease.html',title=title)

#DISEASE RESULT 
@app.route('/disease',methods=['POST'])
def predict2():
    if request.method=='POST':
        type=request.form["crop"]
        type=int(type)
        print(type)
        if type==1:
            if request.files.get('file'):
                img_requested = request.files['file'].read()
                img = Image.open(io.BytesIO(img_requested))
                model=load_model('disease.h5')
                img = img.resize((224, 224))
                img=image.img_to_array(img)
                img=img/255
                img=np.expand_dims(img,axis=0)
                result=model.predict(img)
                print(result)
                i=result.argmax()
                disease_classes = ['Tomato___Bacterial_spot',
                                    'Tomato___Early_blight',
                                    'Tomato___Late_blight',
                                    'Tomato___Leaf_Mold',
                                    'Tomato___Septoria_leaf_spot',
                                    'Tomato___Spider_mites Two-spotted_spider_mite',
                                    'Tomato___Target_Spot',
                                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                                    'Tomato___Tomato_mosaic_virus',
                                    'Tomato___healthy'
                ]
                title="DISEASE-RESULT"
                return render_template('disease_result.html',data=[disease_classes[i]],title=title)
        elif type==2:
            if request.files.get('file'):
                
                img_requested = request.files['file'].read()
                img = Image.open(io.BytesIO(img_requested))
                model=load_model('potatod.h5')
                img = img.resize((224, 224))
                img=image.img_to_array(img)
                img=img/255
                img=np.expand_dims(img,axis=0)
                result=model.predict(img)
                print(result)
                i=result.argmax()
                print("potato")
                print(i)
                disease_classes = ['Potato___Early_blight',
                                    'Potato___Late_blight',
                                    'Potato___healthy',
                ]
                title="DISEASE-RESULT"
                return render_template('disease_result.html',data=[disease_classes[i]],title=title)
                
if __name__ == '__main__':
    app.run(host="localhost", port=3001)