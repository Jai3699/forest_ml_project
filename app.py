from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

## import ridge regressor model and standard scaler pickle
ridge=pickle.load(open('models/ridge.pkl','rb'))
scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form['Temperature'])
        RH=float(request.form['RH'])
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        scaled_data=scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge.predict(scaled_data)
        return render_template('home.html',result=result[0])

  

if __name__=='__main__':
    app.run(host="0.0.0.0")
