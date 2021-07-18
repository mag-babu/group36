from fastapi.testclient import TestClient
from main import app

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong"}


# test to check if Iris Virginica is classified correctly
def test_credit_score():
    # defining a sample payload for the testcase
    payload = {
         "durarion_in_month": 6.0,
        "age_in_years": 35,
        "credit_amount": 4000,
    }
    with TestClient(app) as client:
        response = client.post("/credit_score", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"credit_class": "Good Risk"}

