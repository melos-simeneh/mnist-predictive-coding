import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Hyperparameters
epochs = 5
batch_size = 64
learning_rate = 1e-3
inference_steps = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredictiveCodingNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple 3-layer MLP for MNIST (28x28=784 input)
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        # Initialize states (latent predictions) for each layer
        # These will hold internal representations during inference
        self.state1 = None
        self.state2 = None
        self.state3 = None

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        # Simple feedforward pass for evaluation/testing
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def infer(self, x, steps=10, gamma=0.1):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # Initialize states as zeros or feedforward activations
        self.state1 = torch.zeros(batch_size, 256, device=x.device, requires_grad=True)
        self.state2 = torch.zeros(batch_size, 128, device=x.device, requires_grad=True)
        self.state3 = torch.zeros(batch_size, 10, device=x.device, requires_grad=True)

        # Fixed input layer state (sensory input)
        input_state = x

        for _ in range(steps):
            # Predictions from higher layers
            pred1 = F.relu(self.fc1(input_state))
            pred2 = F.relu(self.fc2(self.state1))
            pred3 = self.fc3(self.state2)

            # Compute prediction errors
            error1 = self.state1 - pred1
            error2 = self.state2 - pred2
            error3 = self.state3 - pred3

            # Update states to reduce prediction error (gradient descent step)
            self.state1 = self.state1 - gamma * error1
            self.state2 = self.state2 - gamma * error2
            self.state3 = self.state3 - gamma * error3

        # After inference, state3 is output logits
        return self.state3

    def compute_loss(self, output, target):
        # Cross entropy loss on final output
        return F.cross_entropy(output, target)


def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Predictive coding inference step(s)
        output = model.infer(x, steps=inference_steps)

        # Compute loss
        loss = model.compute_loss(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy for this batch
        preds = output.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f"Train loss: {avg_loss:.4f} | Train accuracy: {accuracy * 100:.2f}%")
    return avg_loss, accuracy



def main():
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model initialization
    model = PredictiveCodingNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train(model, train_loader, optimizer)
    

if __name__ == "__main__":
    main()
