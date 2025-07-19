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
                                                                          pos_x_interval_target=(-0.2, 1.7),
                                                                          pos_y_interval_target=(-np.inf, np.inf),
                                                                          neg_x_interval_target=(-0.2, 1.7),
                                                                          neg_y_interval_target=(-np.inf, np.inf),
                                                                          pos_x_interval_source=(1.3, np.inf),
                                                                          pos_y_interval_source=(-np.inf, np.inf),
                                                                          neg_x_interval_source=(-np.inf, 0.2),
                                                                          neg_y_interval_source=(-np.inf, np.inf),
                                                                          )





# Step 2: Set hyper-parameter
hyper_parameter_n_classes = 2
hyper_parameter_p = 2
hyper_parameter_c = 0.2
learning_rate = 1e-6
num_epochs = 600000
num_epochs_per_print = num_epochs/10
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
ideal_causal_distance, _ = calculate_causal_distance_between_datasets(
            X_source, y_source, X_target, y_target, hyper_parameter_n_classes,
            order_parameter_p=hyper_parameter_p, scaling_parameter_c=hyper_parameter_c, options=speed_up_options
        )


prediction_change_happens_in_n_epochs = 0
pred_y_target = None
pred_y_target_tensor = None
binary_pred_y_target = None
# Tip: pred_y_target = [0.6, 0.3, 0.8] ---> binary_pred_y_target = [1,0,1]
causal_distance = None
transport_plan = None
transport_plan_tensor = None
approx_costs_Y_tensor = None
approx_costs_tensor = None
approx_causal_distance = None
# Tip: We calculate causal_distance first with true costs and then calculate approx_causal_distance with approximate costs
# causal_distance is our real target to minimize while approximate_causal_distance can give us gradient
# the difference of causal_distance and approximate causal_distance lies only in costs.
# Their plans are the same, which is based on true costs
for epoch in range(1, num_epochs+1):
    model.train()
    new_pred_y_target_tensor = model(X_target_tensor)
    new_pred_y_target = new_pred_y_target_tensor.detach().numpy().reshape(-1)
    new_binary_pred_y_target = (new_pred_y_target>=0.5).astype(np.int64)
    if binary_pred_y_target is None or not np.array_equal(binary_pred_y_target, new_binary_pred_y_target):
        prediction_change_happens_in_n_epochs += 1
        pred_y_target = new_pred_y_target
        pred_y_target_tensor = new_pred_y_target_tensor
        binary_pred_y_target = new_binary_pred_y_target
        binary_pred_y_target_tensor = torch.tensor(binary_pred_y_target, dtype=torch.float32)
        causal_distance, transport_plan = calculate_causal_distance_between_datasets(
            X_source, y_source, X_target, binary_pred_y_target, hyper_parameter_n_classes,
            order_parameter_p=hyper_parameter_p, scaling_parameter_c=hyper_parameter_c, options=speed_up_options
        )
        transport_plan_tensor = torch.tensor(transport_plan)
        approx_costs_Y_tensor = \
            torch.abs(y_source_tensor.reshape(-1,1) - pred_y_target_tensor.reshape(1,-1)) * hyper_parameter_c ** hyper_parameter_p
        costs_Y_tensor = \
            torch.abs(y_source_tensor.reshape(-1,1) - binary_pred_y_target_tensor.reshape(1,-1)) * hyper_parameter_c ** hyper_parameter_p
        approx_costs_tensor = costs_X_tensor + approx_costs_Y_tensor
        approx_causal_distance = torch.sum(approx_costs_tensor * transport_plan_tensor)
        optimizer.zero_grad()
        approx_causal_distance.backward()
    optimizer.step()
    if epoch % num_epochs_per_print == 0:
        X_part = torch.sum(costs_X_tensor * transport_plan_tensor).item()
        Y_part = torch.sum(costs_Y_tensor * transport_plan_tensor).item()
        print(
            f'Epoch [{epoch}/{num_epochs}]({prediction_change_happens_in_n_epochs}), '
            f'True Loss: {causal_distance:.4f} (X: {X_part:.4f}, Y:{Y_part:.4f}), '
            f'Approximate Loss: {approx_causal_distance.item():.4f}, '
            f'Ideal Loss: {ideal_causal_distance:.4f}'
        )



# Step 4: Visualize the Classifier Result
print_hyper_parameters(n_samples, hyper_parameter_p, hyper_parameter_c, learning_rate, num_epochs, list_of_num_hidden_units)
evaluate_and_print_for_binary_classification(X_target, y_target, model)
visualize_domains([X_source, X_target], [y_source, y_target],
                  [f'Source Domain c={hyper_parameter_c} overflow={ideal_causal_distance/causal_distance*100 - 100:.2f}%',
                   f"Target Domain c={hyper_parameter_c} overflow={ideal_causal_distance/causal_distance*100 - 100:.2f}%"],
                  x_limit=(-2.5, 3.5), y_limit=(-3, 3), with_model=model)




