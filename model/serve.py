
from flask import Flask, request
import torch

app = Flask(__name__)
model = None

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if not model:
        model = torch.load('quantized_model.pth')
    
    data = request.data
    tensor_data = torch.from_numpy(data)
    prediction = model(tensor_data)
    
    return str(torch.argmax(prediction, dim=1).item())

if __name__ == "__main__":
    app.run(port=5000)
