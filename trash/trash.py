for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    y_target_prediction_tensor = model(X_target_tensor)
    y_target_prediction = y_target_prediction_tensor.numpy()
    true_causal_distance, redundant_transport_plan = calculate_causal_distance_between_datasets(X_source, y_source,
                                                                                                X_target,
                                                                                                y_target_prediction,
                                                                                                class_number_n,
                                                                                                order_parameter_p=hyper_parameter_p,
                                                                                                scaling_parameter_c=hyper_parameter_c,
                                                                                                options=speed_up_options)
    transport_plan_tensor = torch.tensor(reduce_redundant_transport_matrix(redundant_transport_plan, y_source, y_target))
    costs_X_tensor = torch.tensor(distance.cdist(X_source, X_target, metric='minkowski', p=hyper_parameter_p) ** hyper_parameter_p)
    costs_Y_approximate_tensor = torch.abs(y_source_tensor.reshape(-1,1) - y_target_prediction_tensor.reshape(1,-1))*hyper_parameter_c**hyper_parameter_p
    costs_approximate_tensor = costs_X_tensor + costs_Y_approximate_tensor
    approximate_causal_distance = torch.sum(costs_approximate_tensor * transport_plan_tensor)
    approximate_causal_distance.backward()
    optimizer.step()

