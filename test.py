import torch
from tqdm import tqdm

def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    print("\n")
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="🧪 Testing"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            
    acc = correct / total
    print(f"   📈 Test Accuracy: {acc * 100:.2f}%")
    return acc
