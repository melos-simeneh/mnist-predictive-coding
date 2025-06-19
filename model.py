import torch
import torch.nn as nn
import torch.nn.functional as F

class PCLayer(nn.Module):
    def __init__(self, size_in, size_out, lateral_size=None, fb_size=None):
        super().__init__()
        self.ff = nn.Linear(size_in, size_out)
        if fb_size is not None:
            self.fb = nn.Linear(fb_size, size_out)
        else:
            self.fb = None

        if lateral_size:
            self.lat = nn.Parameter(torch.randn(size_out, size_out) * 0.01)
        else:
            self.lat = None

    def forward_ff(self, x):
        return torch.relu(self.ff(x))

    def forward_fb(self, x):
        if self.fb:
            return torch.relu(self.fb(x))
        return x

    def compute_lateral(self, latent):
        if self.lat is None:
            return 0
        lat_w = (self.lat + self.lat.T) / 2
        return torch.matmul(latent, lat_w)

class PredictiveCodingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = PCLayer(28*28, 256, lateral_size=256, fb_size=128)
        self.layer2 = PCLayer(256, 128, lateral_size=128, fb_size=10) 
        self.layer3 = PCLayer(128, 10)  # no lateral connections for output

        self.latent1 = None
        self.latent2 = None
        self.latent3 = None

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1.forward_ff(x)
        x = self.layer2.forward_ff(x)
        x = self.layer3.ff(x)
        return x

    def infer(self, x, steps=30, gamma=0.1, alpha=0.05):
        b = x.size(0)
        x = x.view(b, -1)

        self.latent1 = torch.zeros(b, 256, device=x.device)
        self.latent2 = torch.zeros(b, 128, device=x.device)
        self.latent3 = torch.zeros(b, 10, device=x.device)

        for _ in range(steps):
            ff_pred1 = self.layer1.forward_ff(x)
            ff_pred2 = self.layer2.forward_ff(self.latent1)
            ff_pred3 = self.layer3.ff(self.latent2)

            fb_pred2 = self.layer2.forward_fb(self.latent3)
            fb_pred1 = self.layer1.forward_fb(self.latent2)

            err1 = self.latent1 - ff_pred1
            err2 = self.latent2 - ff_pred2
            err3 = self.latent3 - ff_pred3

            fb_err2 = self.latent2 - fb_pred2
            fb_err1 = self.latent1 - fb_pred1

            lat_eff1 = self.layer1.compute_lateral(self.latent1)
            lat_eff2 = self.layer2.compute_lateral(self.latent2)

            self.latent1 = self.latent1 - gamma * (err1 + fb_err1) - alpha * lat_eff1
            self.latent2 = self.latent2 - gamma * (err2 + fb_err2) - alpha * lat_eff2
            self.latent3 = self.latent3 - gamma * err3

            self.latent1 = torch.relu(self.latent1)
            self.latent2 = torch.relu(self.latent2)

        return self.latent3

    def compute_loss(self, output, target):
        return F.cross_entropy(output, target)
