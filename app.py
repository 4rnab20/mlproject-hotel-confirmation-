from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app = application
## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData (
            no_of_adults = int(request.form.get('no_of_adults')),
            no_of_weekend_nights = int(request.form.get('no_of_weekend_nights')),
            no_of_week_nights = int(request.form.get('no_of_week_nights')),
            required_car_parking_space = int(request.form.get('required_car_parking_space')),
            lead_time = float(request.form.get('lead_time')),
            no_of_special_requests = int(request.form.get('no_of_special_requests')),
            avg_price_per_room = float(request.form.get('avg_price_per_room')),
            arrival_month = int(request.form.get('arrival_month')),
            market_segment_type = int(request.form.get('market_segment_type')),
            repeated_guest = int(request.form.get('repeated_guest'))
        )

        pred_df = data.get_data_as_data_frame()

        print(pred_df)
        print("Before Prediction")
        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results = results[0] )
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)