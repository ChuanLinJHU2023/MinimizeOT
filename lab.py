import torch

# Create tensors with requires_grad=True to track computation
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Define a simple operation
z = x * y

# First backward pass
z.backward()
print(f"Gradient of x after first backward: {x.grad}")
# Reset gradients manually for demonstration
x.grad.zero_()

# Second backward pass without retain_graph
try:
    z.backward()
except RuntimeError as e:
    print(f"Error during second backward: {e}")