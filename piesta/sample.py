import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import pandas as pd

class Asset:
    def __init__(self, name, price, country=None):
        self.name = name
        self.price = price
        self.country = country

class AssetClass:
    def __init__(self, name, assets, parent=None):
        self.name = name
        self.assets = assets
        self.parent = parent
        
        # create a dictionary to map asset names to their metadata
        self.asset_metadata = {}
        for asset in assets:
            self.asset_metadata[asset.name] = {'price': asset.price, 'country': asset.country}
        
        # if parent exists, add the current asset class to its children
        if parent:
            parent.add_child(self)
        
        self.children = []
        
    def add_child(self, child):
        self.children.append(child)
        
    def get_asset_metadata(self, asset_name):
        return self.asset_metadata[asset_name]
        
    def __repr__(self):
        return f'{self.name}'
        

# create sample assets
stock1 = Asset('Stock1', pd.Series([10, 12, 15, 13]), 'USA')
stock2 = Asset('Stock2', pd.Series([20, 22, 25, 23]), 'Korea')
bond1 = Asset('Bond1', pd.Series([100, 102, 105, 103]), 'USA')
bond2 = Asset('Bond2', pd.Series([90, 92, 95, 93]), 'Korea')
commodity1 = Asset('Commodity1', pd.Series([50, 52, 55, 53]))
commodity2 = Asset('Commodity2', pd.Series([80, 82, 85, 83]))
alt1 = Asset('Alt1', pd.Series([30, 32, 35, 33]), 'USA')
alt2 = Asset('Alt2', pd.Series([70, 72, 75, 73]), 'Korea')

# create asset classes
L1 = AssetClass('L1', [stock1, stock2, bond1, bond2, commodity1, commodity2, alt1, alt2])
L2_stock = AssetClass('L2_stock', [stock1, stock2], parent=L1)
L2_bond = AssetClass('L2_bond', [bond1, bond2], parent=L1)
L2_commodity = AssetClass('L2_commodity', [commodity1, commodity2], parent=L1)
L2_alt = AssetClass('L2_alt', [alt1, alt2], parent=L1)
L3_stock_USA = AssetClass('L3_stock_USA', [stock1, bond1, alt1], parent=L2_stock)
L3_stock_Korea = AssetClass('L3_stock_Korea', [stock2, bond2, alt2], parent=L2_stock)
L3_bond_USA = AssetClass('L3_bond_USA', [bond1, alt1], parent=L2_bond)
L3_bond_Korea = AssetClass('L3_bond_Korea', [bond2, alt2], parent=L2_bond)


class Strategy:
    def __init__(self, name, expected_returns, covariance_matrix):
        """
        Initialize the Strategy object.

        Args:
        - name (str): name of the strategy.
        - expected_returns (np.ndarray): a 1D array containing the expected returns of each asset.
        - covariance_matrix (np.ndarray): a 2D array containing the covariance matrix of the assets.
        """
        self.name = name
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.weights = None

    def optimize(self, constraints=None):
        """
        Optimize the portfolio using the expected returns and covariance matrix.

        Args:
        - constraints (list): a list of Constraint objects to apply to the optimization.

        Returns:
        - weights (np.ndarray): a 1D array containing the optimal weights of each asset.
        """
        # define the objective function
        def objective_function(weights):
            return -np.dot(weights, self.expected_returns)

        # define the constraint functions
        constraint_functions = []
        if constraints is not None:
            for constraint in constraints:
                constraint_functions.append(constraint.apply_constraint)

        # set up the optimization problem
        num_assets = len(self.expected_returns)
        initial_weights = np.ones(num_assets) / num_assets
        bounds = [(0, 1) for i in range(num_assets)]
        constraints = [{'type': 'eq', 'fun': constraint} for constraint in constraint_functions]

        # solve the optimization problem
        result = minimize(objective_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        self.weights = result.x

        return self.weights


    def apply_constraints(self, portfolio):
        # Apply constraints to portfolio
        if self.max_volatility:
            portfolio = MaxVolatilityConstraint(self.max_volatility).apply(portfolio)
        if self.min_volatility:
            portfolio = MinVolatilityConstraint(self.min_volatility).apply(portfolio)
        if self.target_volatility:
            portfolio = TargetVolatilityConstraint(self.target_volatility).apply(portfolio)
        
        # Additional constraints could be added here
        
        return portfolio


class AssetAllocationPipeline:
    def __init__(self, strategies=None):
        self.strategies = strategies if strategies else []

    def add_strategy(self, strategy):
        self.strategies.append(strategy)

    def remove_strategy(self, strategy):
        self.strategies.remove(strategy)

    def optimize(self, constraints=None):
        portfolio = None

        for idx, strategy in enumerate(self.strategies):
            if portfolio is None:
                portfolio = strategy.optimize()
            else:
                # apply constraints from the previous strategy
                prev_constraints = constraints[idx-1] if constraints else None
                prev_allocations = portfolio.allocations
                strategy_constraints = []
                for asset in prev_allocations:
                    if asset.level == strategy.asset_level:
                        strategy_constraints.append(asset)

                strategy.set_constraints(strategy_constraints, prev_allocations)

                # optimize the strategy
                strategy_result = strategy.optimize()

                # apply constraints for the next strategy
                next_constraints = []
                for asset in prev_allocations:
                    if asset.level > strategy.asset_level:
                        next_constraints.append(asset)

                strategy_result.apply_constraints(next_constraints)

                # merge the results with the previous portfolio
                portfolio.merge(strategy_result)

        return portfolio


class ConstraintsManager:
    def __init__(self, constraints=None):
        self.constraints = constraints if constraints else []

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def remove_constraint(self, constraint):
        self.constraints.remove(constraint)

    def apply(self, expected_returns, covariances):
        for constraint in self.constraints:
            constraint.apply(expected_returns, covariances)

class Constraint:
    def __init__(self, name):
        self.name = name
    
    def apply(self, asset_allocation, previous_asset_allocation=None):
        """
        Applies the constraint to the given asset allocation.
        
        Parameters
        ----------
        asset_allocation : dict
            A dictionary of asset allocation where keys are asset names and values are allocation percentages.
        
        previous_asset_allocation : dict, optional
            A dictionary of asset allocation from the previous strategy object. Used as a constraint condition
            for the current strategy object. Default is None.
            
        Returns
        -------
        dict
            The updated asset allocation dictionary after applying the constraint.
        """
        pass

def calculate_portfolio_volatility(weights, covariance):
    """
    Calculates the volatility of a portfolio given the weight vector and covariance matrix.
    
    Parameters:
    weights (ndarray): a 1D numpy array of asset weights
    covariance (ndarray): a 2D numpy array of asset covariances
    
    Returns:
    float: the volatility of the portfolio
    """
    return np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))

def calculate_weights_with_max_volatility(expected_returns, covariance, max_volatility):
    """
    Calculates the weights of a portfolio that maximizes expected returns subject to a maximum volatility constraint.
    
    Parameters:
    expected_returns (ndarray): a 1D numpy array of expected returns for each asset
    covariance (ndarray): a 2D numpy array of asset covariances
    max_volatility (float): the maximum volatility allowed
    
    Returns:
    ndarray: a 1D numpy array of asset weights
    """
    num_assets = len(expected_returns)
    
    # set up optimization problem
    from scipy.optimize import minimize
    
    def objective(weights):
        return -np.dot(expected_returns, weights)
    
    def constraint(weights):
        return max_volatility - calculate_portfolio_volatility(weights, covariance)
    
    cons = [{'type': 'ineq', 'fun': constraint}]
    bounds = tuple((0, 1) for i in range(num_assets))
    initial_guess = np.ones(num_assets) / num_assets
    
    # solve optimization problem
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=cons)
    
    return result.x

def calculate_weights_with_min_volatility(expected_returns, covariances, min_volatility):
    n = len(expected_returns)
    x0 = np.repeat(1/n, n)
    bounds = ((0, 1),)*n
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: calculate_portfolio_volatility(x, covariances) - min_volatility}]
    results = minimize(lambda x: -np.dot(x, expected_returns), x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return results.x

class MaxVolatilityConstraint(Constraint):
    def __init__(self, max_volatility):
        self.max_volatility = max_volatility

    def apply(self, expected_returns, covariances, weights):
        # apply max volatility constraint
        portfolio_volatility = calculate_portfolio_volatility(covariances, weights)
        if portfolio_volatility > self.max_volatility:
            # adjust weights to meet max volatility constraint
            weights = calculate_weights_with_max_volatility(covariances, self.max_volatility)
        return weights

class MinVolatilityConstraint(Constraint):
    def __init__(self, min_volatility):
        self.min_volatility = min_volatility

    def apply(self, weights, cov_matrix):
        portfolio_volatility = calculate_portfolio_volatility(weights, cov_matrix)

        # Check if portfolio volatility meets the minimum requirement
        if portfolio_volatility < self.min_volatility:
            # Adjust weights to meet minimum volatility requirement
            weights = calculate_weights_with_min_volatility(weights, cov_matrix, self.min_volatility)

        return weights

class TargetVolatilityConstraint(Constraint):
    def __init__(self, target_volatility):
        self.target_volatility = target_volatility

    def apply(self, expected_returns, covariances):
        # apply target volatility constraint
        pass

from graphviz import Digraph

class PipelineVisualizer:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def visualize(self):
        dot = Digraph()

        # add strategies to graph
        for strategy in self.pipeline.strategies:
            dot.node(str(id(strategy)), strategy.__class__.__name__)

        # add constraints to graph
        for constraint in self.pipeline.constraints:
            dot.node(str(id(constraint)), constraint.__class__.__name__)

        # add edges from constraints to strategies
        for i, constraint in enumerate(self.pipeline.constraints):
            for strategy in self.pipeline.strategies[i:]:
                dot.edge(str(id(constraint)), str(id(strategy)))

        # render graph
        dot.render('pipeline.gv', view=True)

class Portfolio:
    def __init__(self, cash, positions=None):
        self.cash = cash
        self.positions = positions if positions is not None else {}

    def __repr__(self):
        return f"Portfolio(cash={self.cash}, positions={self.positions})"

    def buy(self, asset, quantity, price):
        cost = quantity * price
        if self.cash < cost:
            raise ValueError("Insufficient cash")
        if asset in self.positions:
            self.positions[asset] += quantity
        else:
            self.positions[asset] = quantity
        self.cash -= cost

    def sell(self, asset, quantity, price):
        if asset not in self.positions:
            raise ValueError("Asset not in portfolio")
        if self.positions[asset] < quantity:
            raise ValueError("Insufficient quantity")
        revenue = quantity * price
        self.positions[asset] -= quantity
        self.cash += revenue

    def value(self, prices):
        value = self.cash
        for asset, quantity in self.positions.items():
            if asset in prices:
                value += quantity * prices[asset]
        return value

    def liquidate(self, prices):
        for asset, quantity in self.positions.items():
            if asset in prices:
                self.cash += quantity * prices[asset]
        self.positions = {}



class Backtest:
    def __init__(self, pipeline, start_date, end_date, initial_capital, transaction_cost=0, slippage=0):
        self.pipeline = pipeline
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.strategy = None
        self.weights = None
        self.portfolio = None
        self.performance = None

    def run(self):
        # Run the pipeline to get the strategy and weights
        self.strategy, self.weights = self.pipeline.run(self.start_date, self.end_date)

        # Create an empty portfolio with the initial capital
        self.portfolio = Portfolio(self.initial_capital)

        # Create a list to store the portfolio values over time
        portfolio_values = []

        # Loop over the dates and execute the strategy
        for date in self.strategy.index:
            # Get the target weights for the current date
            target_weights = self.weights.loc[date]

            # Calculate the order book based on the target weights and current portfolio
            order_book = self.portfolio.get_order_book(target_weights)

            # Execute the orders with transaction costs and slippage
            executed_orders = self.portfolio.execute_orders(order_book, self.transaction_cost, self.slippage)

            # Update the portfolio with the executed orders
            self.portfolio.update_portfolio(executed_orders)

            # Append the current portfolio value to the list of portfolio values
            portfolio_values.append(self.portfolio.get_portfolio_value())

        # Convert the portfolio values to a Pandas Series and set the date index
        self.performance = pd.Series(portfolio_values, index=self.strategy.index)

    def plot_performance(self):
        # Plot the portfolio performance
        plt.plot(self.performance)
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.show()


from typing import List
from datetime import datetime
import pandas as pd
import numpy as np

class PipelineEngine:
    def __init__(self, strategies: List[Strategy], constraints: List[Constraint], start_date: datetime, end_date: datetime):
        self.strategies = strategies
        self.constraints = constraints
        self.start_date = start_date
        self.end_date = end_date
        
    def run_pipeline(self):
        # Run the pipeline for each strategy
        for strategy in self.strategies:
            strategy.run_pipeline(self.start_date, self.end_date)

        # Apply constraints to the results of each strategy
        for constraint in self.constraints:
            constraint.apply_constraint(self.strategies)

    def backtest(self, initial_capital: float, rebalance_freq: str, transaction_cost: float, slippage: float):
        # Combine the pipeline results from all the strategies
        pipeline_df = pd.concat([strategy.pipeline_df for strategy in self.strategies], axis=1)

        # Perform initial setup for the backtest
        rebalance_dates = pd.date_range(start=self.start_date, end=self.end_date, freq=rebalance_freq)
        current_date = self.start_date
        current_weights = None
        current_prices = None
        transaction_history = pd.DataFrame(columns=['Date', 'Ticker', 'Order Type', 'Shares', 'Price'])
        cash_history = pd.DataFrame(columns=['Date', 'Cash'])
        position_history = pd.DataFrame(columns=['Date', 'Ticker', 'Shares', 'Price'])
        portfolio_history = pd.DataFrame(columns=['Date', 'Portfolio Value', 'Cash'])
        previous_portfolio_value = initial_capital

        # Perform the backtest loop
        for rebalance_date in rebalance_dates:
            # Extract the relevant slice of the pipeline DataFrame
            pipeline_slice = pipeline_df.loc[current_date:rebalance_date]

            # If this is the first iteration, calculate initial weights based on the first slice of the pipeline
            if current_weights is None:
                current_weights = self._calculate_weights(pipeline_slice)
                current_prices = self._get_current_prices(current_date, current_weights)

            # Calculate the current portfolio value
            current_portfolio_value = self._calculate_portfolio_value(current_prices, current_weights)

            # Calculate transaction costs and update the current portfolio value
            if previous_portfolio_value is not None:
                transaction_costs = self._calculate_transaction_costs(current_weights, current_prices, previous_portfolio_value, transaction_cost, slippage)
                current_portfolio_value -= transaction_costs
                transaction_history = transaction_history.append(transaction_costs, ignore_index=True)

            # Update the cash and position history
            cash_history = cash_history.append({'Date': rebalance_date, 'Cash': current_portfolio_value}, ignore_index=True)
            position_history = position_history.append(self._get_position_history(current_date, current_prices, current_weights), ignore_index=True)

            # Calculate the new weights and prices for the next iteration
            new_weights = self._calculate_weights(pipeline_slice)
            new_prices = self._get_current_prices(rebalance_date, new_weights)

            # Update the portfolio value with the new prices
            new_portfolio_value = self._calculate_portfolio_value(new_prices, new_weights)

            # Calculate transaction costs and update the new portfolio value
            transaction_costs = self._calculate_transaction_costs(new_weights, new_prices, current_portfolio_value, transaction_cost, slippage)
            new_portfolio_value -= transaction_costs
            transaction_history = transaction_history.append(transaction_costs, ignore_index=True)

            # Update the portfolio and cash history with the new portfolio value
            portfolio_history = portfolio_history.append({'Date': rebalance
