import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb


app = FastAPI()

loaded_xgb_model = xgb.Booster()
loaded_xgb_model.load_model("xg_reg.model")

# with open("/Users/samiatbola-matanmi/Downloads/xgb_model.pkl", "rb") as f:
#     loaded_xgb_model = pickle.load(f)

class request_body(BaseModel):
    T1 : float
    RH_1: float
    T2 : float
    RH_2 : float
    T3 : float
    RH_3 : float
    T4: float
    RH_4: float
    T5 : float
    RH_5 : float
    T6 : float
    RH_6 : float
    T7 : float
    RH_7 : float
    T8 : float
    RH_8 : float
    T9 : float
    RH_9 : float
    T_out : float
    Press_mm_hg : float
    RH_out : float
    Windspeed : float
    Visibility : float


@app.get('/')   
def main():
    return {'message': 'Welcome to the appliance prediction API'}

@app.post("/predict")
def predict(data: request_body):
    test_data = [[
        data.T1,
        data.RH_1,
        data.T2,
        data.RH_2,
        data.T3, 
        data.RH_3,
        data.T4,
        data.RH_4,
        data.T5,
        data.RH_5,
        data.T6,
        data.RH_6,
        data.T7,
        data.RH_7,
        data.T8,
        data.RH_8,
        data.T9,
        data.RH_9,
        data.T_out,
        data.Press_mm_hg,
        data.RH_out,
        data.Windspeed,
        data.Visibility
    ]]

    data_Dmatrix = xgb.DMatrix(test_data)

    predicted = loaded_xgb_model.predict(data_Dmatrix)
    return {"predicted value", predicted.tolist()}
