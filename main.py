import torch
from model import PredictiveCodingNet
from data import get_dataloaders
from train import train
from test import test
from utils import predict_single_sample, plot_training

# Hyperparameters
num_epochs = 5
batch_size = 64
lr = 1e-3
inference_steps = 30
gamma = 0.1
alpha = 0.05
weight_decay = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    train_loader, test_loader, test_data = get_dataloaders(batch_size)
    model = PredictiveCodingNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        print(f"\n📦 Epoch [{epoch + 1}/{num_epochs}]")
        loss, acc = train(model, train_loader, optimizer, device, gamma, alpha)
        train_losses.append(loss)
        train_accuracies.append(acc)

    plot_training(train_losses, train_accuracies)
    test(model, test_loader, device)
    predict_single_sample(model, test_data, device)

if __name__ == "__main__":
    main()
