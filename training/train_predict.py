import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import numpy as np

# Constants
D = 6
N = 64  # Must match MAMBA_STATE_DIM in mamba_s6.h

# ==========================================================
# 1. Model Architecture (Mirror to mamba_s6.c)
# ==========================================================
class TrueMambaS6Block(nn.Module):
    def __init__(self, d_model=6, d_state=16):
        super().__init__()
        self.d = d_model
        self.n = d_state
        
        # MambaS6Params (Log-parameterized A for stability)
        # Optimized HiPPO-like diagonal spectrum initialization
        # Creates a range of timescales from short-term to long-term memory
        A_init = torch.log(torch.arange(1, self.n + 1, dtype=torch.float32).repeat(self.d, 1) * 0.2)
        self.A_log = nn.Parameter(A_init)
        self.D_skip = nn.Parameter(torch.ones(self.d) * 0.1)  # Restore skip for high-freq impacts
        self.D_skip.requires_grad = True                      # Allow learning of the bypass
        
        self.W_out = nn.Linear(self.d, self.d, bias=True)
        nn.init.eye_(self.W_out.weight)
        nn.init.constant_(self.W_out.bias, 0.0)
        
        # MambaSelectWeights (Single layer to match C framework)
        self.W_delta = nn.Linear(self.d, self.d, bias=True)
        self.W_B = nn.Linear(self.d, self.n, bias=False)
        self.W_C = nn.Linear(self.d, self.n, bias=False)
        
        # Initialization focused on signal preservation
        nn.init.constant_(self.W_delta.bias, -0.5) # Shorter memory for better responsiveness
        nn.init.xavier_uniform_(self.W_B.weight, gain=0.1)
        nn.init.xavier_uniform_(self.W_C.weight, gain=0.1)

    def forward(self, x, teach_forcing_ratio=1.0):
        """
        x shape: [batch, sequence_length, D]
        teach_forcing_ratio: Probability of using true input vs model's own previous prediction.
        """
        batch, seq_len, d = x.shape
        
        # h state shape: [batch, d_model, d_state]
        h = torch.zeros(batch, self.d, self.n, device=x.device)
        ys = []
        
        y_prev = None
        for t in range(seq_len):
            x_t_true = x[:, t, :]
            if self.training and y_prev is not None and torch.rand(1).item() > teach_forcing_ratio:
                # Simulate DPS Trigger: In production, the node only stays in open-loop 
                # if the prediction error is within the budget. If it diverges, it resets.
                err = torch.abs(y_prev.detach() - x_t_true)
                # Bound the feedback to prevent D_skip exponentially exploding the forward pass
                mask = (err < 1.0).float() 
                x_t = mask * y_prev.detach() + (1.0 - mask) * x_t_true
            else:
                x_t = x_t_true
            
            # --- 1. Selective Projections ---
            delta_t = torch.nn.functional.softplus(self.W_delta(x_t))
            B_t = self.W_B(x_t)
            C_t = self.W_C(x_t)
            
            # --- 2. ZOH Discretization ---
            delta_t_b = delta_t.unsqueeze(-1) # [batch, D, 1]
            # Enforce A < 0 using the log-parameterized A_log
            A_real = -torch.exp(self.A_log).unsqueeze(0) # [1, D, N]
            A_bar = torch.exp(delta_t_b * A_real) # [batch, D, N]
            
            B_t_b = B_t.unsqueeze(1) # [batch, 1, N]
            B_bar = (A_bar - 1.0) / A_real * B_t_b
            
            # --- 3. Hidden-state update ---
            x_t_b = x_t.unsqueeze(-1)
            h = A_bar * h + B_bar * x_t_b # [batch, D, N]
            
            # --- 4. Output projection ---
            C_t_b = C_t.unsqueeze(1) 
            y_d = self.D_skip.unsqueeze(0) * x_t + torch.sum(C_t_b * h, dim=-1) # [batch, D]
            y_final = self.W_out(y_d)
            
            ys.append(y_final.unsqueeze(1))
            y_prev = y_final
            
        y = torch.cat(ys, dim=1) # [batch, seq_len, D]
        return y


class GRUModel(nn.Module):
    def __init__(self, d_model=6, d_hidden=48):
        super().__init__()
        self.gru = nn.GRU(d_model, d_hidden, batch_first=True)
        self.fc = nn.Linear(d_hidden, d_model)
        
        # Fair Initialization
        for name, param in self.gru.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        
    def forward(self, x):
        out, _ = self.gru(x)
        y = self.fc(out)
        return y

class LSTMModel(nn.Module):
    def __init__(self, d_model=6, d_hidden=48):
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_hidden, batch_first=True)
        self.fc = nn.Linear(d_hidden, d_model)
        
        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name: nn.init.orthogonal_(param)
            elif 'weight_ih' in name: nn.init.xavier_uniform_(param)
            elif 'bias' in name: nn.init.constant_(param, 0.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

class SimpleRNNModel(nn.Module):
    def __init__(self, d_model=6, d_hidden=48):
        super().__init__()
        self.rnn = nn.RNN(d_model, d_hidden, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(d_hidden, d_model)
        
        for name, param in self.rnn.named_parameters():
            if 'weight_hh' in name: nn.init.orthogonal_(param)
            elif 'weight_ih' in name: nn.init.xavier_uniform_(param)
            elif 'bias' in name: nn.init.constant_(param, 0.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)


# 2. Data Loading & Preparation (HuGaDB)
# ==========================================================
def load_and_prepare_data(seq_len=64, data_dir="archiveIMU", max_samples=20000):
    """
    Loads raw IMU data from HuGaDB CSV files.
    All HuGaDB files share the same 40-column structure, ensuring data consistency
    across different subjects and trials.
    """
    # Try different data directory locations to be robust across Docker/Local
    if not os.path.exists(data_dir):
        alt_dirs = ["../archiveIMU", "archiveIMU", "/app/archiveIMU"]
        for adir in alt_dirs:
            if os.path.exists(adir):
                data_dir = adir
                break

    print(f"Loading HuGaDB IMU data from {data_dir} (Target: {max_samples} samples)...")

    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    # Normalize each file independently before concatenating.
    # Gyro resting bias varies between subjects — global normalization maps each
    # subject's bias to a different DC offset within [0,1], causing the constant
    # prediction offset seen on low-variance gyro channels.
    # Z-Score Normalization per file to handle heterogeneous sensor offsets
    normalized_chunks = []
    for file_name in files[:50]:
        file_path = os.path.join(data_dir, file_name)
        file_rows = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            # Skip header
            next(reader, None)
            for row in reader:
                if not row or len(row) < 40: continue
                try:
                    # Select only RIGHT FOOT: index 1-6 (Acc XYZ, Gyro XYZ)
                    features = [float(row[i]) for i in range(1, 7)]
                    file_rows.append(features)
                except ValueError:
                    continue
        if len(file_rows) < seq_len + 2:
            continue
        chunk = torch.tensor(file_rows, dtype=torch.float32)
        # Z-Score: (x - mean) / std
        f_mean = chunk.mean(dim=0)
        f_std = chunk.std(dim=0)
        f_std[f_std < 1e-4] = 1.0  # Avoid div zero
        chunk = (chunk - f_mean) / f_std
        normalized_chunks.append(chunk)

    data = torch.cat(normalized_chunks, dim=0)
    train_size = int(0.8 * len(data))
    
    # Calculate global scaling constants for the export
    data_mean = data[:train_size].mean(dim=0)
    data_std = data[:train_size].std(dim=0)
    print(f"Dataset prepared: {len(data)} samples ({len(normalized_chunks)} files, per-file normalized).")

    n_samples = min(max_samples, len(data) - seq_len - 1)
    x_seqs = torch.zeros((n_samples, seq_len, D), dtype=torch.float32)
    y_target = torch.zeros((n_samples, seq_len, D), dtype=torch.float32)

    for i in range(n_samples):
        x_seqs[i] = data[i : i + seq_len, :]
        y_target[i] = data[i + 1 : i + seq_len + 1, :]

    return x_seqs, y_target, data, data_mean, data_std, train_size


# ==========================================================
# 3. Export to C Logic
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

def export_to_c(model, x_verify, y_verify, data_min, data_max):
    print(f"Exporting weights and test data to C files...")
    # Weights & Scaling
    weight_file = "../framework/mamba_weights.c"
    with open(weight_file, "w") as f:
        f.write('/* Auto-generated true S6 weights and scaling from PyTorch */\n')
        # ------------------------------------------------------------------ #
        # Safety guard: catch macro/dimension mismatches at compile time.     #
        # The Python constants D and N are baked in here so that if the C     #
        # build defines different values the compiler emits a hard error.     #
        # ------------------------------------------------------------------ #
        f.write(f'#if MAMBA_D != {D} || MAMBA_N != {N}\n')
        f.write(f'#error "Dimension mismatch: this file was generated for MAMBA_D={D}, MAMBA_N={N}. "\\')
        f.write(f'\n       "Rebuild mamba_weights.c or update the macros in mamba_s6.h."\n')
        f.write('#endif\n\n')
        f.write('#include "mamba_s6.h"\n\n')
        # model.A_log stores log(-A); export the real A = -exp(A_log)
        import torch
        A_real = -torch.exp(model.A_log)
        f.write(tensor_to_c_array(A_real, "MAMBA_A"))
        f.write(tensor_to_c_array(model.D_skip, "MAMBA_D_SKIP"))
        f.write(tensor_to_c_array(model.W_out.weight, "MAMBA_W_OUT"))
        f.write(tensor_to_c_array(model.W_out.bias, "MAMBA_BIAS_OUT"))
        f.write(tensor_to_c_array(model.W_delta.weight, "MAMBA_W_DELTA"))
        f.write(tensor_to_c_array(model.W_delta.bias, "MAMBA_B_DELTA"))
        f.write(tensor_to_c_array(model.W_B.weight, "MAMBA_W_B"))
        f.write(tensor_to_c_array(model.W_C.weight, "MAMBA_W_C"))
        f.write("\n/* Z-Score Normalization Constants (HuGaDB Right Foot Only) */\n")
        f.write(tensor_to_c_array(data_min, "MAMBA_X_MEAN"))
        f.write(tensor_to_c_array(data_max, "MAMBA_X_STD"))

    # Test Data
    data_file = "mamba_test_data.h"
    seq_len = x_verify.shape[0]
    with open(data_file, "w") as f:
        f.write('/* Auto-generated test sequences */\n')
        f.write(f'#define TEST_SEQ_LEN {seq_len}\n\n')
        f.write(f"const float mamba_test_inputs[{seq_len}][6] = {{\n")
        for i in range(seq_len):
            row = x_verify[i].detach().tolist()
            f.write("    { " + ", ".join([f"{v:.6f}f" for v in row]) + " },\n")
        f.write("};\n\n")
        f.write(f"const float mamba_test_outputs_expected[{seq_len}][6] = {{\n")
        for i in range(seq_len):
            row = y_verify[i].detach().tolist()
            f.write("    { " + ", ".join([f"{v:.6f}f" for v in row]) + " },\n")
        f.write("};\n")


# ==========================================================
# 4. Main Tasks
# ==========================================================

def exponential_smooth(tensor, alpha=0.85):
    """
    Causal exponential moving average (EMA) applied along the sequence axis.

        s_t = alpha * x_t  +  (1 - alpha) * s_{t-1},   s_0 = x_0

    Why EMA instead of a box/MA filter for DPS:
      - MA averages the last `window` samples equally, which drags peak
        amplitude down to the window mean — destroying DPS trigger accuracy.
      - EMA keeps (alpha * 100)% of every sample's original value, so strong
        impact peaks (the DPS trigger signal) survive with minimal attenuation.
      - With alpha=0.85 only high-frequency single-sample noise spikes are
        attenuated; the underlying gait waveform is almost unchanged.

    Shape: [batch, seq_len, D]  →  [batch, seq_len, D]
    """
    out = torch.empty_like(tensor)
    out[:, 0, :] = tensor[:, 0, :]                          # initialise with first sample
    for t in range(1, tensor.shape[1]):
        out[:, t, :] = alpha * tensor[:, t, :] + (1.0 - alpha) * out[:, t - 1, :]
    return out


def train_model(model, x_train, y_train, x_val, y_val, epochs=15, batch_size=32, model_path="mamba_gait_model.pt"):
    print(f"Starting Training ({epochs} epochs, Batch Size: {batch_size})...")
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    
    # Stabilized OneCycleLR
    steps_per_epoch = (x_train.shape[0] + batch_size - 1) // batch_size
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, 
                                             epochs=epochs, steps_per_epoch=steps_per_epoch)
    
    # ------------------------------------------------------------------
    # Exponential smoothing (EMA) on training targets, alpha=0.85.
    # EMA retains 85% of each sample's amplitude so peaks/troughs used
    # by DPS triggering are preserved.  Only single-sample noise spikes
    # (which the model should NOT chase) are gently attenuated.
    # We smooth y but NOT x so the raw input signal is preserved.
    # ------------------------------------------------------------------
    EMA_ALPHA = 0.85
    y_train_smooth = exponential_smooth(y_train, alpha=EMA_ALPHA)
    y_val_smooth   = exponential_smooth(y_val,   alpha=EMA_ALPHA)
    print(f"Target smoothing: EMA alpha={EMA_ALPHA} applied to y_train and y_val.")

    # Predictive Fidelity Loss: DPS Epsilon-MSE + Correlation + Amplitude + Kinematic + FFT
    def dps_phase_aware_loss(yp, yt, epsilon=0.01):
        # 1. DPS epsilon-insensitive MSE (Tighter epsilon to push MSE lower)
        error = torch.abs(yp - yt)
        active_error = torch.clamp(error - epsilon, min=0.0)
        dps_mse = torch.mean(active_error ** 2)

        # 2. Timing (Zero Lag) — sequence-wise correlation
        yp_norm = (yp - yp.mean(dim=1, keepdim=True)) / (yp.std(dim=1, keepdim=True) + 1e-6)
        yt_norm = (yt - yt.mean(dim=1, keepdim=True)) / (yt.std(dim=1, keepdim=True) + 1e-6)
        corr_loss = 1.0 - (yp_norm * yt_norm).mean()

        # 3. Energy — penalise amplitude damping per sequence
        amp_loss = torch.abs(yp.std(dim=1) - yt.std(dim=1)).mean()

        # 4. Kinematic smoothness — penalise Δ^2y (Acceleration) mismatch
        delta_yp = yp[:, 1:, :] - yp[:, :-1, :]
        delta_yt = yt[:, 1:, :] - yt[:, :-1, :]
        delta2_yp = delta_yp[:, 1:, :] - delta_yp[:, :-1, :]
        delta2_yt = delta_yt[:, 1:, :] - delta_yt[:, :-1, :]
        kinematic_loss = nn.functional.mse_loss(delta2_yp, delta2_yt)
        
        # 5. Frequency Space Supervision (FFT)
        fft_p = torch.fft.rfft(yp, dim=1)
        fft_t = torch.fft.rfft(yt, dim=1)
        # Bypassing torch.abs() to avoid division-by-zero gradient explosions on sqrt(0)
        spectral_loss = nn.functional.mse_loss(torch.view_as_real(fft_p), torch.view_as_real(fft_t))

        # Re-weighted: Kinematic penalty reduced to prevent flattening of high-G accelerometer impacts!
        return 4.0 * dps_mse + 1.0 * corr_loss + 0.5 * amp_loss + 0.1 * kinematic_loss + 0.2 * spectral_loss

    criterion = dps_phase_aware_loss
    
    best_val_loss = float('inf')
    patience_counter = 0
    n_samples = x_train.shape[0]
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Teacher forcing ratio decays from 1.0 down to 0.85 to introduce gentle scheduled sampling
        tf_ratio = 1.0 - (0.15 * (epoch / max(1, epochs - 1)))
        
        # Simple manual progress bar
        for i in range(0, n_samples, batch_size):
            x_batch        = x_train[i : i + batch_size]
            y_batch_smooth = y_train_smooth[i : i + batch_size]  # smoothed target
            
            optimizer.zero_grad()
            
            # Add small Gaussian noise to input for robustness
            noise = torch.randn_like(x_batch) * 0.001
            if isinstance(model, TrueMambaS6Block):
                outputs = model(x_batch + noise, teach_forcing_ratio=tf_ratio)
            else:
                outputs = model(x_batch + noise)
            
            loss = criterion(outputs, y_batch_smooth)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * x_batch.size(0)
            
            # Progress update
            progress = min(100, (i + batch_size) * 100 // n_samples)
            sys.stdout.write(f"\rEpoch {epoch+1:02d}/{epochs} | [{'=' * (progress // 5)}{' ' * (20 - progress // 5)}] {progress}% - Loss: {loss.item():.5f}  ")
            sys.stdout.flush()
        
        avg_train_loss = epoch_loss / n_samples
        
        # Validation against smoothed targets (consistent with training signal)
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val), y_val_smooth).item()
        
        print(f"\nSummary - Train MSE: {avg_train_loss:.5f} - Val MSE: {val_loss:.5f}")

        # Early Stopping Logic (Stabilized)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"Checkpoint! Best Val MSE: {best_val_loss:.5f}")
        else:
            patience_counter += 1
            if patience_counter >= 6:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

def run_prediction(model, raw_data, train_size):
    print("Running prediction on validation sequence...")
    model.eval()
    with torch.no_grad():
        test_seq_len = 150
        warmup_len = 256  # Long warmup so hidden state is fully settled before scoring
        val_start_idx = train_size + 500  # Skip past the first part of val set (often calm)
        
        # Ensure we have enough data for warmup
        if val_start_idx < warmup_len:
            val_start_idx = warmup_len
            
        if val_start_idx + test_seq_len > len(raw_data):
            val_start_idx = len(raw_data) - test_seq_len - 1
            
        # Extract sequence including warmup
        input_start = val_start_idx - warmup_len
        input_end = val_start_idx + test_seq_len
        
        x_full = raw_data[input_start : input_end, :].clone().unsqueeze(0)  # [1, warmup+test, D]
        y_full_target = raw_data[input_start+1 : input_end+1, :].clone().unsqueeze(0)
        
        # Run prediction on full sequence
        y_full_pred = model(x_full)
        
        # Slice out the warmup to get the steady-state performance
        y_verify = y_full_pred[:, warmup_len:, :]
        y_verify_target = y_full_target[:, warmup_len:, :]
        x_verify = x_full[:, warmup_len:, :]
        
    mse = torch.mean((y_verify - y_verify_target)**2).item()
    print(f"Prediction Complete. Verification MSE: {mse:.6f}")
    return x_verify[0], y_verify[0], y_verify_target[0]

def benchmark_model(model, input_dim=6, seq_len=1, iterations=100):
    import time
    model.eval()
    x = torch.randn(1, seq_len, input_dim)
    
    # Warmup
    for _ in range(10):
        _ = model(x)
        
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations * 1000 # convert to ms
    return avg_time

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_all_kb = (param_size + buffer_size) / 1024
    return size_all_kb

def calculate_advanced_metrics(y_pred, y_target):
    """
    Calculates Pearson Correlation, Peak Fidelity, and a simple DTW distance.
    """
    y_p = y_pred.detach().cpu().numpy().flatten()
    y_t = y_target.detach().cpu().numpy().flatten()
    
    # 1. Pearson Correlation (Sync)
    corr = np.corrcoef(y_p, y_t)[0, 1]
    
    # 2. Peak Fidelity (How well we hit the extremes)
    peak_p = np.max(y_p) - np.min(y_p)
    peak_t = np.max(y_t) - np.min(y_t)
    peak_fid = (1.0 - abs(peak_p - peak_t) / peak_t) * 100
    
    # 3. Simple DTW (Shape Similarity)
    # Using a windowed O(N^2) approach for the 150-sample sequence
    # Flattening to 1D for simplicity in multi-channel comparison
    n, m = len(y_p), len(y_t)
    dtw_matrix = np.full((n + 1, m + 1), float('inf'))
    dtw_matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(max(1, i - 10), min(m + 1, i + 10)): # 10-sample window for speed
            cost = (y_p[i-1] - y_t[j-1])**2
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
            
    dtw_dist = dtw_matrix[n, m] / n
    
    return corr, peak_fid, dtw_dist

def generate_plot(y_pred, y_target, filename="validation_plot.png"):
    try:
        import matplotlib.pyplot as plt
        
        labels = ["RF_acc_x", "RF_acc_y", "RF_acc_z", "RF_gyro_x", "RF_gyro_y", "RF_gyro_z"]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for i in range(6):
            actual = y_target[:, i].detach().numpy()
            pred = y_pred[:, i].detach().numpy()
            
            ax = axes[i]
            ax.plot(actual, label="Actual", linestyle='-', alpha=0.8)
            ax.plot(pred, label="Pred", linestyle='--', alpha=0.8)
            ax.set_title(labels[i], fontsize=10)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)
        
        plt.suptitle(f"{filename.split('_')[0].capitalize()} Comparison: Multi-Channel Gait Prediction Dashboard", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename)
        print(f"Comprehensive dashboard saved to {filename}")
    except ImportError:
        print("matplotlib not found, skipping plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mamba/GRU Gait Training & Prediction")
    parser.add_argument('--model', type=str, default='mamba', choices=['mamba', 'gru', 'lstm', 'rnn'], 
                        help="Model architecture to use")
    parser.add_argument('--train', action='store_true', help="Train the model")
    parser.add_argument('--predict', action='store_true', help="Run prediction/inference")
    parser.add_argument('--plot', action='store_true', help="Generate visualization plot")
    parser.add_argument('--export', action='store_true', help="Export weights to C")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--max_samples', type=int, default=20000, help="Max number of sequences to train on")
    parser.add_argument('--model_path', type=str, default="mamba_gait_model.pt")
    args = parser.parse_args()

    # Set dynamic default path if not specified
    if args.model_path == "mamba_gait_model.pt" and args.model == "gru":
        args.model_path = "gru_gait_model.pt"

    # If no flags provided, show help
    if not any([args.train, args.predict, args.plot, args.export]):
        parser.print_help()
        sys.exit(0)

    if args.model == 'mamba':
        print("Using Mamba S6 model architecture.")
        model = TrueMambaS6Block(D, N)
    elif args.model == 'gru':
        print("Using GRU model architecture.")
        model = GRUModel(D)
    elif args.model == 'lstm':
        print("Using LSTM model architecture.")
        model = LSTMModel(D)
    elif args.model == 'rnn':
        print("Using Simple RNN model architecture.")
        model = SimpleRNNModel(D)
    
    # Set dynamic default path if not specified
    if args.model_path == "mamba_gait_model.pt" and args.model != "mamba":
        args.model_path = f"{args.model}_gait_model.pt"
    
    # Load data with user-specified max_samples
    x_seqs, y_target, raw_data, d_min, d_max, raw_train_size = load_and_prepare_data(max_samples=args.max_samples)
    train_size = int(0.8 * x_seqs.shape[0])  # sequence-level split for training
    
    if args.train:
        x_train, y_train = x_seqs[:train_size], y_target[:train_size]
        x_val, y_val = x_seqs[train_size:], y_target[train_size:]
        train_model(model, x_train, y_train, x_val, y_val, epochs=args.epochs, model_path=args.model_path)
    
    # Load model if we are doing anything other than training (or in addition to training)
    if os.path.exists(args.model_path):
        print(f"Loading weights from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
    elif not args.train:
        print(f"Error: Model file {args.model_path} not found. Train it first with --train.")
        sys.exit(1)

    x_v, y_v, y_t = None, None, None
    if args.predict or args.plot or args.export:
        x_v, y_v, y_t = run_prediction(model, raw_data, train_size)
        
        # Report Metrics
        size_kb = get_model_size(model)
        latency_ms = benchmark_model(model)
        corr, peak_fid, dtw_dist = calculate_advanced_metrics(y_v, y_t)

        print(f"\n--- {args.model.upper()} Metrics ---")
        print(f"Model Weight Size: {size_kb:.2f} KB")
        print(f"Inference Latency (per step): {latency_ms:.4f} ms")
        print(f"Correlation (Sync): {corr:.4f}")
        print(f"Peak Fidelity: {peak_fid:.2f}%")
        print(f"DTW Distance (Shape): {dtw_dist:.6f}")
        
        if args.model == 'mamba':
            print(f"Hidden State RAM (D*N): {D*N*4/1024:.2f} KB")
        elif args.model == 'lstm':
            # LSTM has h and c states
            print(f"Hidden State RAM (h+c): {48*2*4/1024:.2f} KB")
        else:
            # GRU/RNN only have h state
            print(f"Hidden State RAM (h): {48*4/1024:.2f} KB")
        print("------------------------\n")

    if args.plot and y_v is not None:
        plot_name = f"{args.model}_validation_plot.png"
        generate_plot(y_v, y_t, filename=plot_name)
        
    if args.export and y_v is not None:
        if args.model == 'mamba':
            export_to_c(model, x_v, y_v, d_min, d_max)
        else:
            print("Warning: C export is only supported for Mamba models. Skipping export.")
