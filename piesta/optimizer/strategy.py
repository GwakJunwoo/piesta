import numpy as np

class Strategy:
    def __init__(self, name, expected_returns, covariances, macro_data=None):
        self.name = name
        self.expected_returns = expected_returns
        self.covariances = covariances
        self.macro_data = macro_data

    def get_optimal_weights(self, constraints=None):
        n = self.expected_returns.shape[0]
        x0 = np.full(n, 1.0 / n)
        bounds = Bounds(0.0, 1.0)

        # Define the objective function
        def objective(x):
            return -np.dot(x, self.expected_returns)

        # Define the inequality constraints
        constraints_list = []
        if constraints is not None:
            for c in constraints:
                if c['type'] == 'le':
                    constraints_list.append(
                        {'type': 'ineq', 'fun': lambda x, c=c: c['rhs'] - np.dot(x, c['coefficients'])})
                elif c['type'] == 'ge':
                    constraints_list.append(
                        {'type': 'ineq', 'fun': lambda x, c=c: np.dot(x, c['coefficients']) - c['rhs']})
                else:
                    constraints_list.append(
                        {'type': 'eq', 'fun': lambda x, c=c: np.dot(x, c['coefficients']) - c['rhs']})

        # Define the bounds and constraints
        bounds_and_constraints = constraints_list.copy()
        bounds_and_constraints.append(bounds)

        # Optimize the weights using the SLSQP algorithm
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)

        return result.x
    
class Custom_Strategy:
    def __init__(self, name, returns, covariances, factors, macro_data):
        self.name = name
        self.returns = returns
        self.covariances = covariances
        self.factors = factors
        self.macro_data = macro_data

    def get_optimal_weights(self, start_date, end_date, universe):
        # Calculate expected returns and covariance matrix
        expected_returns = self.returns.loc[start_date:end_date, universe].mean()
        covariances = self.covariances.loc[start_date:end_date, universe, universe]

        # Calculate the constraints
        constraints = constraints_manager(universe)

        # Run the optimizer
        weights = run_optimization(expected_returns, covariances, constraints)

        return weights

