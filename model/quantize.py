
import torch
import torch.quantization
import mlflow

def quantize_model():
    model = torch.load('model.pth')
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(quantized_model.state_dict(), 'quantized_model.pth')

    with mlflow.start_run():
        mlflow.log_artifact('quantized_model.pth')

if __name__ == "__main__":
    quantize_model()
