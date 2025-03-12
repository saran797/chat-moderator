from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import torch
from transformers import BertTokenizer

app = FastAPI()

# Load your trained model (joblib) and other necessary files
model = joblib.load('rf_model.pkl')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class Message(BaseModel):
    text: str

def preprocess_text(text: str):
    # Preprocess the text as done during training
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    return encoded_input['input_ids']

@app.post("/predict/")
async def predict(message: Message):
    # Preprocess input message text
    input_ids = preprocess_text(message.text)

    # Extract features using BERT + BiLSTM (same method as during training)
    with torch.no_grad():
        input_ids_tensor = input_ids.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # Pass through BiLSTM and extract features (assuming you've written this part before)
        features = bilstm_model(input_ids_tensor).cpu().numpy()

    # Make a prediction using RandomForest
    prediction = model.predict(features)
    label = prediction[0]
    return {"label": label}
