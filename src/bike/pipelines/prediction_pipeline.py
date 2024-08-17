import sys
import os
import pandas as pd
from src.bike.exception import CustomException
from src.bike.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e, sys)

Date = 'Date'
Hour = 'Hour'
Temperature_C = 'Temperature(°C)'
Humidity = 'Humidity(%)'
Windspeed = 'Wind speed (m/s)'
Dew_point_temperature = 'Dew point temperature(°C)'
Solar_Radiation = 'Solar Radiation (MJ/m2)'
Rainfall = 'Rainfall(mm)'
Snowfall = 'Snowfall (cm)'
Visibility = 'Visibility (10m)'
Seasons = 'Seasons'
Holiday = 'Holiday'
Functioning_Day = 'Functioning Day'

class CustomData:
    def __init__(self,
                 Date: str,
                 Hour: int,
                 Temperature_C: float,
                 Humidity: float,
                 Windspeed: float,
                 Dew_point_temperature: float,
                 Solar_Radiation: float,
                 Rainfall: float,
                 Snowfall: float,
                 Visibility: float,
                 Seasons: str,
                 Holiday: str,
                 Functioning_Day: str):
        
        self.Date = Date
        self.Hour = Hour
        self.Temperature_C = Temperature_C
        self.Humidity = Humidity
        self.Windspeed = Windspeed
        self.Dew_point_temperature = Dew_point_temperature
        self.Solar_Radiation = Solar_Radiation
        self.Rainfall = Rainfall
        self.Snowfall = Snowfall
        self.Visibility = Visibility
        self.Seasons = Seasons
        self.Holiday = Holiday
        self.Functioning_Day = Functioning_Day

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Date": [self.Date],
                "Hour": [self.Hour],
                "Temperature(°C)": [self.Temperature_C],
                "Humidity(%)": [self.Humidity],
                "Wind speed (m/s)": [self.Windspeed],
                "Dew point temperature(°C)": [self.Dew_point_temperature],
                "Solar Radiation (MJ/m2)": [self.Solar_Radiation],
                "Rainfall(mm)": [self.Rainfall],
                "Snowfall (cm)": [self.Snowfall],
                "Visibility (10m)": [self.Visibility],
                "Seasons": [self.Seasons],
                "Holiday": [self.Holiday],
                "Functioning Day": [self.Functioning_Day]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
