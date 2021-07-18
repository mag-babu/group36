import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ml_utils import load_model, predict, retrain, predictlr
from typing import List


# defining the main app
app = FastAPI(title="German Credit Score Predictor", docs_url="/")

# calling the load_model during startup.
# this will train the model and keep it loaded for prediction.
app.add_event_handler("startup", load_model)

# class which is expected in the payload
class QueryIn(BaseModel):
    durarion_in_month: float
    age_in_years: float
    credit_amount: float



# class which is returned in the response
class QueryOut(BaseModel):
    credit_risk: str

# class which is expected in the payload while re-training
class FeedbackIn(BaseModel):
    durarion_in_month: float
    age_in_years: float
    credit_amount: float
  

# Route definitions
@app.get("/ping")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"ping": "pong"}


@app.post("/credit_score", response_model=QueryOut, status_code=200)
# Route to do the prediction using the ML model defined.
# Payload: QueryIn containing the parameters
# Response: QueryOut containing the flower_class predicted (200)
def credit_score(query_data: QueryIn):
    output = {"credit_class": predict(query_data)}
    return output

#@app.post("/credit_loop", status_code=200)
# Route to further train the model based on user input in form of feedback loop
# Payload: FeedbackIn containing the parameters and correct flower class
# Response: Dict with detail confirming success (200)
#def credit_loop(data: List[FeedbackIn]):
#    retrain(data)
#    return {"detail": "credit loop successful"}


# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=8886, reload=True)
