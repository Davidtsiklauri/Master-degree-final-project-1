import numpy as np
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel
from sklearn.tree import DecisionTreeClassifier


# Define object we classify
class HouseInformation(BaseModel):
    lat: int
    long: int
    balcony: int
    loggia: int 
    veranda: int 
    numberOfFloors: int 
    floor: int
    numberOfRooms: int 
    numberOfBedrooms: int 
    LotArea: int 
    ceilingHeight: int
    numberOfBathrooms: int


class HousePricePredictor:
    def __init__(self):
        self.bayesian: DecisionTreeClassifier = load("./model_weights/bayesian.bin")
        self.en: DecisionTreeClassifier = load("./model_weights/en.bin")
        self.forest: DecisionTreeClassifier = load("./model_weights/forest.bin")
        self.ols: DecisionTreeClassifier = load("./model_weights/ols.bin")
        self.ridge: DecisionTreeClassifier = load("./model_weights/ridge.bin")

    def predict(self, item: HouseInformation):
        # make sure that here the order is the same as in the model training
        print("item:", item)
        x = np.array([1 if item.sex == "female" else 0, item.pclass])
        x = x.reshape(1, -1)
        # be careful to only transform and not fit
        print("numeric representation:", x)
        y = self.clf.predict_proba(x)
        print("survival probability:", y)
        # y looks now like: [[0.78 0.21]] so the second number is probability of survived
        return y[0][1]


app = FastAPI()
predictor = HousePricePredictor()



@app.get("/")
def root():
    return {"GoTo": "/docs"}


@app.post("/will_survive")
def get_house_price(request: HouseInformation):
    try:
        return {"response": predictor.predict(request)}
    except:
        raise HTTPException(status_code=418, detail="Exceptions can't be handheld by a teapot")