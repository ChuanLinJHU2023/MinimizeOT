import pulp

# Define the problem
prob = pulp.LpProblem("Simple LP Problem", pulp.LpMinimize)

# Decision variables
x = pulp.LpVariable('x', lowBound=0)
y = pulp.LpVariable('y', lowBound=0)

# Objective function
prob += 2 * x + 3 * y, "Total Cost"

# Constraints
prob += x + y >= 4, "Constraint 1"
prob += x - y <= 1, "Constraint 2"

# Solve using GUROBI
# prob.solve(pulp.GUROBI(msg=True, gapRel=0.3))
prob.solve(pulp.GUROBI(msg=True))

# Output the results
print(f"Status: {pulp.LpStatus[prob.status]}")
print(f"x = {pulp.value(x)}")
print(f"y = {pulp.value(y)}")