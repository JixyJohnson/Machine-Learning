import torch
import torch.nn as nn

# Input: 3 words, each represented by 4 numbers
x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0]
])

# Create Query, Key and Value
query = nn.Linear(4, 4)
key = nn.Linear(4, 4)
value = nn.Linear(4, 4)

Q = query(x)
K = key(x)
V = value(x)

# Calculate attention scores
scores = torch.matmul(Q, K.T)

# Scale scores
scores = scores / (4 ** 0.5)

# Convert scores into probabilities
attention = torch.softmax(scores, dim=-1)

# Calculate final attention output
output = torch.matmul(attention, V)

print("Input:")
print(x)

print("\nAttention Weights:")
print(attention)

print("\nTransformer Output:")
print(output)