from tqdm import tqdm

def train(model, dataloader, optimizer, device, gamma, alpha):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x, y in tqdm(dataloader, desc="   🔧 Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model.infer(x, gamma=gamma, alpha=alpha)
        loss = model.compute_loss(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / len(dataloader)
    acc = correct / total
    print(f"   📉 Train Loss: {avg_loss:.4f}")
    print(f"   📊 Train Accuracy: {acc*100:.2f}%")
    return avg_loss, acc
