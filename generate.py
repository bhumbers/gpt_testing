import torch 

from v2 import *
from v2 import BigramLanguageModel

from typing import cast

from datetime import datetime

filename = "model_004800.pt"
device="cuda"
print(f"Loading model '{filename}...")
m = cast(BigramLanguageModel, torch.load("model_004800.pt"))
print("Model loaded")

m.to(device)

# generate from the model
n_tokens = 10000
torch.manual_seed(datetime.now().timestamp())
print(f"Generating up to {n_tokens} tokens...")
print("---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=n_tokens)[0].tolist()))
