import numpy as np

from models import *
from problems import *
from distances import *
from scipy.spatial import distance

# Step 1: Get data
X_source, y_source, X_target, y_target = create_domain_adaptation_problem(n_samples=100, noise_level=0.1)


# Step 2: Set hyper-parameter
hyper_parameter_n_classes = 2
hyper_parameter_p = 2
hyper_parameter_c = 2
learning_rate = 0.1
num_epochs = 5000
num_epochs_per_print = 1000
speed_up_options = {"msg":False}
list_of_num_hidden_units = [16]
model = SimpleClassifier(list_of_num_hidden_units)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Step 3: Train Model
X_source_tensor = torch.tensor(X_source, dtype=torch.float32)
y_source_tensor = torch.tensor(y_source.reshape(-1, 1), dtype=torch.float32)
X_target_tensor = torch.tensor(X_target, dtype=torch.float32)
y_target_tensor = torch.tensor(y_target.reshape(-1, 1), dtype=torch.float32)
costs_X_tensor = torch.tensor(
    distance.cdist(X_source, X_target, metric='minkowski', p=hyper_parameter_p) ** hyper_parameter_p)


pred_y_target = None
causal_distance = None
transport_plan = None
transport_plan_tensor = None
approx_costs_Y_tensor = None
approx_costs_tensor = None
approx_causal_distance = None

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    new_pred_y_target_tensor = model(X_target_tensor)
    new_pred_y_target = new_pred_y_target_tensor.detach().numpy().reshape(-1)
    if pred_y_target is None or not np.array_equal(new_pred_y_target, pred_y_target):
        pred_y_target = new_pred_y_target
        causal_distance, transport_plan = calculate_causal_distance_between_datasets(
            X_source, y_source, X_target, (pred_y_target>=0.5).astype(np.int64), hyper_parameter_n_classes,
            order_parameter_p=hyper_parameter_p, scaling_parameter_c=hyper_parameter_c, options=speed_up_options
        )
        transport_plan_tensor = torch.tensor(transport_plan)
        approx_costs_Y_tensor = \
            torch.abs(y_source_tensor.reshape(-1,1) - new_pred_y_target_tensor.reshape(1,-1)) * hyper_parameter_c ** hyper_parameter_p
        approx_costs_tensor = costs_X_tensor + approx_costs_Y_tensor
        approx_causal_distance = torch.sum(approx_costs_tensor * transport_plan_tensor)
    approx_causal_distance.backward()
    optimizer.step()


