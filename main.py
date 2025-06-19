import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
num_epochs = 5
batch_size = 64
lr = 1e-3
inference_steps = 30
gamma = 0.1  # inference step size
alpha = 0.05  # lateral weight scale
weight_decay = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredictiveCodingNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Feedforward layers
        self.ff1 = nn.Linear(28 * 28, 256)
        self.ff2 = nn.Linear(256, 128)
        self.ff3 = nn.Linear(128, 10)

        # Feedback layers
        self.fb2 = nn.Linear(10, 128)
        self.fb1 = nn.Linear(128, 256)

        # Lateral connections
        self.lat1 = nn.Parameter(torch.randn(256, 256) * 0.01)
        self.lat2 = nn.Parameter(torch.randn(128, 128) * 0.01)

        # Latent states
        self.latent1 = None
        self.latent2 = None
        self.latent3 = None

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.ff1(x))
        x = torch.relu(self.ff2(x))
        x = self.ff3(x)
        return x

    def infer(self, x, steps=inference_steps):
        b = x.size(0)
        x = x.view(b, -1)

        self.latent1 = torch.zeros(b, 256, device=x.device)
        self.latent2 = torch.zeros(b, 128, device=x.device)
        self.latent3 = torch.zeros(b, 10, device=x.device)

        for _ in range(steps):
            ff_pred1 = torch.relu(self.ff1(x))
            ff_pred2 = torch.relu(self.ff2(self.latent1))
            ff_pred3 = self.ff3(self.latent2)

            fb_pred2 = torch.relu(self.fb2(self.latent3))
            fb_pred1 = torch.relu(self.fb1(self.latent2))

            err1 = self.latent1 - ff_pred1
            err2 = self.latent2 - ff_pred2
            err3 = self.latent3 - ff_pred3

            fb_err2 = self.latent2 - fb_pred2
            fb_err1 = self.latent1 - fb_pred1

            lat_eff1 = torch.matmul(self.latent1, (self.lat1 + self.lat1.T) / 2)
            lat_eff2 = torch.matmul(self.latent2, (self.lat2 + self.lat2.T) / 2)

            self.latent1 = self.latent1 - gamma * (err1 + fb_err1) + alpha * lat_eff1
            self.latent2 = self.latent2 - gamma * (err2 + fb_err2) + alpha * lat_eff2

            self.latent3 = self.latent3 - gamma * err3

            self.latent1 = torch.relu(self.latent1)
            self.latent2 = torch.relu(self.latent2)

        return self.latent3


    def compute_loss(self, output, target):
        return nn.functional.cross_entropy(output, target)

def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model.infer(x)
        loss = model.compute_loss(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f"Train loss: {avg_loss:.4f} | Train accuracy: {accuracy*100:.2f}%")
    return avg_loss, accuracy

def test(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"Test accuracy: {acc * 100:.2f}%")
    return acc

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = PredictiveCodingNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_loader, optimizer)
    
    test(model, test_loader)


if __name__ == "__main__":
    main()
