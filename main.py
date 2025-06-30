import numpy as np
from ot.backend import torch
from scipy.stats import hypsecant_gen

from models import *
from problems import *
from distances import *
from scipy.spatial import distance

from trash.trash import true_causal_distance

# Step 1: Get data
X_source, y_source, X_target, y_target = create_domain_adaptation_problem(n_samples=100, noise_level=0.1)


# Step 2: Set hyper-parameter
class_number_n = 2
hyper_parameter_p = 2
hyper_parameter_c = 2
learning_rate = 0.1
num_epochs = 5000
num_epochs_per_print = 1000
speed_up_options = None
list_of_num_hidden_units = [16]
model = SimpleClassifier(list_of_num_hidden_units)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Step 3: Train Model
X_source_tensor = torch.tensor(X_source, dtype=torch.float32)
y_source_tensor = torch.tensor(y_source.reshape(-1, 1), dtype=torch.float32)
X_target_tensor = torch.tensor(X_target, dtype=torch.float32)
y_target_tensor = torch.tensor(y_target.reshape(-1, 1), dtype=torch.float32)



y_target_prediction = None
transport_plan = None
true_causal_distance = None

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    new_y_target_prediction_tensor = model(X_target_tensor)
    new_y_target_prediction = new_y_target_prediction_tensor.detach().numpy()
    if y_target_prediction is None or not np.array_equal(new_y_target_prediction, y_target_prediction):
        true_causal_distance, transport_plan = calculate_causal_distance_between_datasets(
            X_source, y_source, X_target, new_y_target_prediction, class_number_n,
            order_parameter_p=hyper_parameter_p, scaling_parameter_c=hyper_parameter_c, options=speed_up_options
        )
        y_target_prediction = new_y_target_prediction
    transport_plan_tensor = torch.tensor(transport_plan)
    costs_X_tensor = torch.tensor(distance.cdist(X_source, X_target, metric='minkowski', p=hyper_parameter_p) ** hyper_parameter_p)
    costs_Y_approximate_tensor = torch.abs(y_source_tensor.reshape(-1,1) - new_y_target_prediction_tensor.reshape(1,-1)) * hyper_parameter_c ** hyper_parameter_p
    costs_approximate_tensor = costs_X_tensor + costs_Y_approximate_tensor
    approximate_causal_distance = torch.sum(costs_approximate_tensor * transport_plan_tensor)
    approximate_causal_distance.backward()
    optimizer.step()


