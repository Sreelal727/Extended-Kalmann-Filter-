import os
import urllib.request
import http
import pandas as pd
from time import sleep
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the ANN regression model from the .h5 file
model_filename = 'duty_cycle_predictor_model.h5'
loaded_model = load_model(model_filename)

base = "http://192.168.137.16/"

def transfer(my_url):   #use to send and receive data
    try:
        n = urllib.request.urlopen(base + my_url).read()
        n = n.decode("utf-8")
        return n
    except http.client.HTTPException as e:
        return e
# Create an empty list to store data
data_list = []
ct=0.0
while True:
    try:
        # Request data from the server
        res = transfer(str(ct))
        print("Received data:", res)
        response = str(res)
        
        # Split the received data
        values = response.split('-')
        if len(values) == 2:
            v, i = values
            print(v)
            vu=float(v)-float(i)
            # Convert the features to a NumPy array
            features = [[float(vu)]]
            
            # Standardize the features
            
            # Make predictions
            predicted_duty_cycle = loaded_model.predict(features)[0][0]
            print("Predicted Duty Cycle:", predicted_duty_cycle)
            if predicted_duty_cycle>1:
                predicted_duty_cycle=predicted_duty_cycle/100
            else:
                predicted_duty_cycle=predicted_duty_cycle/10
            # Send the predicted duty cycle back to the server
            ct=float(v)-predicted_duty_cycle
            
        sleep(1)
    
    except Exception as e:
        print("Error:", e)
        sleep(5)  # Add a delay in case of errors to avoid continuous requests
