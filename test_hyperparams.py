import torch
import torch.nn as nn
from model import LyricsLSTM
from data import preprocess_data, SPECIAL_TOKENS
import time
import math
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Open results file
results_file = open("hyperparameter_results.txt", "w")

# Write header
results_file.write(f"HYPERPARAMETER SENSITIVITY TEST\n")
results_file.write("=" * 80 + "\n\n")

print("Loading data...")
results_file.write("Loading data...")
train_loader, val_loader, test_loader, vocab = preprocess_data()
vocab_size = len(vocab)
pad_idx = SPECIAL_TOKENS["<pad>"]

# Create smaller subsets for faster testing
def create_small_loaders(train_loader, val_loader, sample_size=20):
    """Create smaller loaders with only sample_size batches"""
    train_data = list(train_loader)[:sample_size]
    val_data = list(val_loader)[:sample_size]
    return train_data, val_data

print("\nCreating smaller dataset subset for quick testing...")
results_file.write(" (Done)\n")
results_file.write("Creating smaller dataset subset for quick testing...\n\n")

train_data_small, val_data_small = create_small_loaders(train_loader, val_loader, sample_size=20)

# Test parameters - only most impactful ones
learning_rates = [1e-4, 3e-4]
hidden_dims = [128, 256]

results = []

table_header = f"{'LR':<12} {'Hidden Dim':<12} {'Train Loss':<14} {'Train PPL':<12} {'Val Loss':<14} {'Val PPL':<12} {'Time (s)':<10}"
print("\n" + table_header)
print("=" * 100)
results_file.write(table_header + "\n")
results_file.write("=" * 100 + "\n")

for lr in learning_rates:
    for hidden_dim in hidden_dims:
        print(f"{lr:<12} {hidden_dim:<12} ", end="", flush=True)
        results_file.write(f"{lr:<12} {hidden_dim:<12} ")
        results_file.flush()
        
        # Create model
        model = LyricsLSTM(
            vocab_size=vocab_size,
            embed_dim=128,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=0.3,
            pad_idx=pad_idx
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        
        # Train for 3 epochs on small subset
        start = time.time()
        train_loss = 0
        val_loss = 0
        
        for epoch in range(3):
            model.train()
            epoch_train_loss = 0
            for x, y in train_data_small:
                x = x.to(device)
                y = y.to(device)
                
                optimizer.zero_grad()
                logits, _ = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            
            train_loss = epoch_train_loss / len(train_data_small)
            
            # Evaluate
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for x, y in val_data_small:
                    x = x.to(device)
                    y = y.to(device)
                    logits, _ = model(x)
                    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                    epoch_val_loss += loss.item()
            
            val_loss = epoch_val_loss / len(val_data_small)
        
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - start
        
        print(f"{train_loss:<14.4f} {train_ppl:<12.2f} {val_loss:<14.4f} {val_ppl:<12.2f} {elapsed:<10.2f}")
        results_file.write(f"{train_loss:<14.4f} {train_ppl:<12.2f} {val_loss:<14.4f} {val_ppl:<12.2f} {elapsed:<10.2f}\n")
        results_file.flush()
        
        results.append({
            "lr": lr,
            "hidden_dim": hidden_dim,
            "train_loss": train_loss,
            "train_ppl": train_ppl,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "time": elapsed
        })

# Find best configuration
best = min(results, key=lambda x: x["val_loss"])
worst = max(results, key=lambda x: x["val_loss"])

print("\n" + "=" * 70)
results_file.write("\n" + "=" * 70 + "\n\n")

summary = f"""SUMMARY:
--------
Best Configuration (Lowest Validation Loss):
  Learning Rate: {best['lr']}
  Hidden Dimension: {best['hidden_dim']}
  Training Loss: {best['train_loss']:.4f} (Perplexity: {best['train_ppl']:.2f})
  Validation Loss: {best['val_loss']:.4f} (Perplexity: {best['val_ppl']:.2f})
  Training Time: {best['time']:.2f}s

Worst Configuration (Highest Validation Loss):
  Learning Rate: {worst['lr']}
  Hidden Dimension: {worst['hidden_dim']}
  Training Loss: {worst['train_loss']:.4f} (Perplexity: {worst['train_ppl']:.2f})
  Validation Loss: {worst['val_loss']:.4f} (Perplexity: {worst['val_ppl']:.2f})
  Training Time: {worst['time']:.2f}s

Performance Difference:
  Validation Loss Difference: {worst['val_loss'] - best['val_loss']:.4f}
  Perplexity Difference: {worst['val_ppl'] - best['val_ppl']:.2f}
  Improvement: {((worst['val_loss'] - best['val_loss']) / worst['val_loss'] * 100):.2f}%


"""

print(summary)
results_file.write(summary)

# Analysis
analysis = f"""ANALYSIS:
---------
Test Configuration:
  - Only 20 batches per epoch (small subset for quick testing)
  - 3 epochs of training per configuration
  - 2 learning rates tested
  - 2 hidden dimensions tested

Metrics Explanation:
  - Loss: Cross-entropy loss (lower is better, directly measurable)
  - Perplexity: e^(loss) - intuitive metric for language models
    * PPL = 1 means perfect predictions
    * PPL = vocab_size means random guessing
    * Lower PPL = better predictions

Key Findings:
  - Best configuration uses: LR={best['lr']}, Hidden={best['hidden_dim']}
  - Validation Perplexity: {best['val_ppl']:.2f} (model is {best['val_ppl']:.0f}x confused vs perfect)

"""

print(analysis)
results_file.write(analysis)

results_file.close()
print("\nResults saved to: hyperparameter_results.txt")

