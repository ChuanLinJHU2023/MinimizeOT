from visualizations import *
from problems import *
import numpy as np
from models import *
from problems import *
from distances import *
from scipy.spatial import distance
from visualizations import *
import ot

# Step 1: Get data
X_source, y_source, X_target, y_target = create_domain_adaptation_problem(n_samples=100,
                                                                noise_level=0.1, source_ratio=0.7, target_ratio=0.3)

def ot_domain_adaptation(X_s, y_s, X_t, y_t):
    n_source = X_s.shape[0]
    n_target = X_t.shape[0]
    M = ot.dist(X_s, X_t, metric='sqeuclidean')
    a = np.ones((n_source,)) / n_source
    b = np.ones((n_target,)) / n_target
    transport_matrix = ot.emd(a, b, M)
    row_sums = transport_matrix.sum(axis=1, keepdims=True)
    # Tip: transport_matrix -> n_source x n_target
    # Tip: row_sums -> n_source x 1
    X_s_adapted = (transport_matrix / row_sums) @ X_t
    assert X_s_adapted.shape[0] == X_s.shape[0]
    return X_s_adapted, transport_matrix

X_source_adapted = ot_domain_adaptation(X_source, y_source, X_target, y_target)[0]
learning_rate = 0.001
num_epochs = 30000
num_epochs_per_print = 1000
list_of_num_hidden_units = [16]
model = SimpleClassifier(list_of_num_hidden_units)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# X_tensor.shape = (500, 2), y_tensor.shape = (100, 1)
X_tensor = torch.tensor(X_source_adapted, dtype=torch.float32)
y_tensor = torch.tensor(y_source.reshape(-1, 1), dtype=torch.float32)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    # print(outputs.shape, y_tensor.shape)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % num_epochs_per_print == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

# To predict for a new datapoint x_new:
# x_new = torch.tensor([[value1, value2]], dtype=torch.float32)
# pred = model(x_new)
# predicted_label = 1 if pred.item() > 0.5 else 0

visualize_domains([X_source, X_source_adapted, X_target], [y_source, y_source, y_target],\
                  ['Source Domain', 'Source Domain Adapted', "Target Domain"], \
                  x_limit=(-3, 3), y_limit=(-3, 3), with_model=model)