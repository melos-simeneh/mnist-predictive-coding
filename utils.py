import torch
import matplotlib.pyplot as plt
import random

def predict_single_sample(model, dataset, device):
    model.eval()
    idx = random.randint(0, len(dataset) - 1)
    x, y = dataset[idx]
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(1).item()

    print("\n🧠 Sample Prediction:")
    print(f"   ✅ Actual Label   : {y}")
    print(f"   🧮 Predicted Label: {pred}")

    plt.figure(figsize=(2, 2))
    plt.imshow(x.squeeze().cpu(), cmap='gray')
    plt.title(f'Actual: {y}   Predicted: {pred}')
    plt.axis('off')
    plt.show()


def plot_training(train_losses, train_accuracies):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Loss', color='red', marker='o')
    plt.title('Training Loss')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Accuracy', color='blue', marker='o')
    plt.title('Training Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_plot.png')
    print("\n📊 Training plot saved as 'training_plot.png'")
