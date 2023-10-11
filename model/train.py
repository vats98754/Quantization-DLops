
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    # Data loading and preprocessing
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
    
    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Start an MLflow run
    with mlflow.start_run():
        # Training loop
        for epoch in range(5):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # Log metrics, parameters, and model to MLflow
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("epochs", 5)
        mlflow.log_metric("final_loss", loss.item())
        mlflow.pytorch.log_model(model, "models")

        torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    train()
