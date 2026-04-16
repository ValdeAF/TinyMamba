import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================================
# 1. Define the parameters mapping precisely to the C code
# MAMBA_D = 16, MAMBA_N = 16
# ==========================================================
D = 16
N = 16

class SmallMambaBlock(nn.Module):
    def __init__(self, d_model=16, d_state=16):
        super().__init__()
        self.d = d_model
        self.n = d_state
        
        # MambaS6Params (the continuous transitions & skip)
        # Initialize A to be negative
        self.A = nn.Parameter(torch.arange(1, self.n + 1, dtype=torch.float32).repeat(self.d, 1) * -1.0)
        self.D_skip = nn.Parameter(torch.ones(self.d))
        
        # MambaSelectWeights (the input-dependent projections)
        self.W_delta = nn.Linear(self.d, self.d, bias=True)
        self.W_B = nn.Linear(self.d, self.n, bias=False)
        self.W_C = nn.Linear(self.d, self.n, bias=False)
        
    def forward(self, x):
        """
        A highly simplified forward pass to simulate the data flow 
        and allow PyTorch to track gradients. 
        x shape: [batch, sequence_length, D]
        """
        # x: [batch, seq_len, D]
        # In reality, this loop would contain the complex ZOH discretization
        # We simplify here just to show the gradient flow for testing the export
        delta = torch.nn.functional.softplus(self.W_delta(x)) # [B, L, D]
        B = self.W_B(x) # [B, L, N]
        C = self.W_C(x) # [B, L, N]
        
        # Fake "state" accumulation just to let the optimizer do something
        # y = C * (B * x * delta) + x * D_skip
        y = C.sum(dim=-1, keepdim=True) * B.sum(dim=-1, keepdim=True) * delta * x + x * self.D_skip
        return y


# ==========================================================
# 2. Generate Synthetic Data & Train
# ==========================================================
print("Setting up synthetic data and model...")
model = SmallMambaBlock(D, N)
optimizer = optim.AdamW(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Generate 500 fake sequences of length 10
# (Like 10-step gait strides!)
batch_size = 500
seq_len = 10
x_train = torch.rand((batch_size, seq_len, D)) # Fake normalized sensor data
y_target = torch.rand((batch_size, seq_len, D)) # Fake targets to predict

print("Starting mock training loop...")
model.train()
for epoch in range(15):
    optimizer.zero_grad()
    predictions = model(x_train)
    loss = criterion(predictions, y_target)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/15 - Fake Loss: {loss.item():.4f}")

print("\nTraining complete! Exporting weights to C...")

# ==========================================================
# 3. Export Weights to C format
# ==========================================================
def tensor_to_c_array(tensor, name):
    flat = tensor.detach().flatten().tolist()
    if len(tensor.shape) == 1:
        s = f"const float {name}[{tensor.shape[0]}] = {{\n    "
        s += ", ".join([f"{v:.6f}f" for v in flat])
        s += "\n};\n"
        return s
    elif len(tensor.shape) == 2:
        s = f"const float {name}[{tensor.shape[0]}][{tensor.shape[1]}] = {{\n"
        for i in range(tensor.shape[0]):
            row = tensor[i].detach().tolist()
            s += "    { " + ", ".join([f"{v:.6f}f" for v in row]) + " },\n"
        s += "};\n"
        return s
    return ""

with open("mamba_weights_mock.c", "w") as f:
    f.write('/* Auto-generated mock weights from PyTorch */\n')
    f.write('#include "mamba_s6.h"\n\n')
    
    # Export MambaS6Params
    f.write(tensor_to_c_array(model.A, "MAMBA_A"))
    f.write(tensor_to_c_array(model.D_skip, "MAMBA_D_SKIP"))
    
    # Export MambaSelectWeights
    f.write(tensor_to_c_array(model.W_delta.weight, "MAMBA_W_DELTA"))
    f.write(tensor_to_c_array(model.W_delta.bias, "MAMBA_B_DELTA"))
    f.write(tensor_to_c_array(model.W_B.weight, "MAMBA_W_B"))
    f.write(tensor_to_c_array(model.W_C.weight, "MAMBA_W_C"))

print("Exported to mamba_weights_mock.c! You can copy-paste these arrays into your mamba_weights.c file.")
