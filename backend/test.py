import shap, torch, numpy as np

class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 2)
    def forward(self, x):
        return torch.softmax(self.fc(x.mean(dim=1)), dim=1)

m = Dummy()
bg = torch.zeros(1, 5, 16)
inp = torch.randn(1, 5, 16)
ex = shap.GradientExplainer(m, bg)
v = ex.shap_values(inp)
print("type:", type(v))
print("shape:", np.array(v).shape)