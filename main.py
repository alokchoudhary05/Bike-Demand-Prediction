from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.bike.pipelines.prediction_pipeline import PredictPipeline, CustomData


application = Flask(__name__)

app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html') 

## Route for prediction page
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Date=request.form['Date'],
            Hour=int(request.form['Hour']),
            Temperature_C=float(request.form['Temperature_C']),
            Humidity=float(request.form['Humidity']),
            Windspeed=float(request.form['Windspeed']),
            Dew_point_temperature=float(request.form['Dew_point_temperature']),
            Solar_Radiation=float(request.form['Solar_Radiation']),
            Rainfall=float(request.form['Rainfall']),
            Snowfall=float(request.form['Snowfall']),
            Visibility=float(request.form['Visibility']),
            Seasons=request.form['Seasons'],
            Holiday=request.form['Holiday'],
            Functioning_Day=request.form['Functioning_Day']
        )

        pred_df = data.get_data_as_data_frame()
        print("Predicting with data frame:", pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Predict Pipeline created")
        print("Mid Prediction")

        results=predict_pipeline.predict(pred_df)
        print("Prediction results:", results)
        print("after Prediction")

        return render_template('home.html',results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)    


# http://localhost:5000/predictdata