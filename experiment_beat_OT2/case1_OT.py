import numpy as np
from models import *
from problems import *
from distances import *
from scipy.spatial import distance
from visualizations import *
from evaluate import *

# Step 1: Get data
n_samples=100
X_source, y_source, X_target, y_target = create_domain_adaptation_problem(n_samples=n_samples,  theta=0, noise_level=0.1,
                                                                          horizontally_stretch=1.5,
                                                                          pos_x_interval_source=(0.5, 2.5),
                                                                          pos_y_interval_source=(-np.inf, np.inf),
                                                                          neg_x_interval_source=(-np.inf, np.inf),
                                                                          neg_y_interval_source=(-np.inf, np.inf),
                                                                          pos_x_interval_target=(-np.inf, np.inf),
                                                                          pos_y_interval_target=(-np.inf, np.inf),
                                                                          neg_x_interval_target=(-1, 1),
                                                                          neg_y_interval_target=(-np.inf, np.inf),
                                                                          )


# Step 2: Set Hyper Parameter
learning_rate = 0.001
num_epochs = 30000
num_epochs_per_print = num_epochs/10
list_of_num_hidden_units = [16]
model = SimpleClassifier(list_of_num_hidden_units)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Step 3: Train Classifier
def ot_domain_adaptation(X_s, X_t):
    n_source = X_s.shape[0]
    n_target = X_t.shape[0]
    M = ot.dist(X_s, X_t, metric='sqeuclidean')
    a = np.ones((n_source,)) / n_source
    b = np.ones((n_target,)) / n_target
    transport_matrix = ot.emd(a, b, M)
    row_sums = transport_matrix.sum(axis=1, keepdims=True)
    X_s_adapted = (transport_matrix / row_sums) @ X_t
    assert X_s_adapted.shape[0] == X_s.shape[0]
    return X_s_adapted, transport_matrix
X_source_adapted = ot_domain_adaptation(X_source, X_target)[0]
X_tensor = torch.tensor(X_source_adapted, dtype=torch.float32)
y_tensor = torch.tensor(y_source.reshape(-1, 1), dtype=torch.float32)


for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % num_epochs_per_print == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')



# Step 4: Visualize
print_hyper_parameters(n_samples, None, None, learning_rate, num_epochs, list_of_num_hidden_units)
evaluate_and_print_for_binary_classification(X_target, y_target, model)
visualize_domains([X_source, X_source_adapted, X_target], [y_source, y_source, y_target],
                  ['Source Domain', 'Source Domain Adapted', "Target Domain"],
                  x_limit=(-2.5, 3.5), y_limit=(-3, 3), with_model=model)




