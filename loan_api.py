from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np

# Define FastAPI app
app = FastAPI()

# Define the model architecture
class ModelMaker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(13, 32)
        self.hidden = torch.nn.Linear(32, 64)
        self.output = torch.nn.Linear(64, 1)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.nn.functional.relu(self.input(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.hidden(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

# Load the trained model
model_path = 'model.pth'  # Ensure the correct path
model = ModelMaker()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Given mean and standard deviation from training data (converted to float32)
mean_data = np.array([
    4.950000e-01, 5.289600e+01, 9.962139e+03, 5.000000e-01,
    6.170000e-01, 5.714460e+02, 9.840000e-01, 5.010000e-01,
    4.992500e+01, 1.862700e+01, 4.375700e+01, 2.960000e+00,
    9.740000e-01
], dtype=np.float32)

std_data = np.array([
    4.99974999e-01, 2.32612378e+01, 2.89429375e+03, 5.00000000e-01,
    4.86118298e-01, 1.57534958e+02, 8.29303322e-01, 4.99999000e-01,
    2.87958569e+01, 1.05433330e+01, 1.55637383e+01, 1.99659711e+00,
    8.18122240e-01
], dtype=np.float32)

# Mean and standard deviation of the label (interest rate)
mean_label = 17.23632
std_label = 3.0081990721360183

# Define input schema
class LoanInput(BaseModel):
    features: list[float]  # Accepts a list of 13 feature values

@app.post("/predict/")
async def predict_interest_rate(input_data: LoanInput):
    try:
        # Convert input to tensor and ensure float32
        t_input = torch.tensor(input_data.features, dtype=torch.float32)

        # Ensure correct input length
        if len(input_data.features) != 13:
            return {"error": "Input must contain exactly 13 features."}

        # Normalize the input using the given mean and std values
        t_input_normalized = (t_input - torch.tensor(mean_data, dtype=torch.float32)) / torch.tensor(std_data, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            yHat = model(t_input_normalized)

        # Denormalize the output to get actual interest rate
        yHat_denormalized = yHat.item() * std_label + mean_label

        return {
            "input": input_data.features,
            "predicted_interest_rate": round(yHat_denormalized, 2)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def home():
    return {"message": "Loan Interest Prediction API is running!"}
