import numpy as np
import pandas as pd
from typing import List, Callable, Dict

from typing import List, Dict, Optional, Callable
import pandas as pd
from data.asset import Universe

class AssetAllocationPipeline:
    def __init__(self, universe: Dict, strategies: List[Callable]):
        self.universe = universe
        self.strategies = strategies

    def run_pipeline(self, data: pd.DataFrame) -> Dict:
        allocation_results = {}
        constraints = {}
        universe_by_layer = self.universe.get_universe_by_layer()

        for i, (layer, assets) in enumerate(universe_by_layer.items()):
            if i < len(self.strategies):
                strategy = self.strategies[i]
                layer_data = data[assets]  # Filter data for assets in the current layer
                strategy_result = strategy(self.universe, layer_data, constraints)
                allocation_results[layer] = strategy_result

                if i < len(self.strategies) - 1:
                    constraints[layer] = strategy_result

        return allocation_results

class PipelineBuilder:
    def __init__(self, universe: Dict):
        self.universe = universe
        self.strategies = []

    def add_strategy(self, strategy: Callable):
        self.strategies.append(strategy)

    def build_pipeline(self):
        return AssetAllocationPipeline(self.universe, self.strategies)

# Sample strategy function
def mean_variance_strategy(universe: Universe, data: pd.DataFrame, constraints: Optional[Dict] = None) -> pd.Series:
    optimizer = MeanVarianceOptimizer()
    if constraints is not None:
        # Enforce the constraints obtained from the previous allocation pipeline run
        prev_weights = constraints
        lb = {asset: max(0, prev_weights[asset]-0.05) for asset in prev_weights}
        ub = {asset: min(1, prev_weights[asset]+0.05) for asset in prev_weights}
        weights = optimizer.optimize(data, lower_bounds=lb, upper_bounds=ub)
    else:
        weights = optimizer.optimize(data)

    return weights

import numpy as np
import pandas as pd
from typing import Optional, Dict


from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint

class Optimizer:
    def optimize(self, data: pd.DataFrame, constraints: Optional[Dict] = None) -> pd.Series:
        pass

class MeanVarianceOptimizer(Optimizer):
    def __init__(self, return_estimate: str = 'mean', risk_estimate: str = 'sample_cov'):
        self.return_estimate = return_estimate
        self.risk_estimate = risk_estimate

    def optimize(self, data: pd.DataFrame, constraints: Optional[Dict] = None) -> pd.Series:
        if constraints is not None:
            lb = [constraints[asset]['lb'] for asset in data.columns]
            ub = [constraints[asset]['ub'] for asset in data.columns]
            bounds = tuple(zip(lb, ub))
        else:
            bounds = tuple((0, 1) for _ in range(data.shape[1]))

        returns = data.mean()
        cov_matrix = data.cov()
        
        def objective(weights):
            return np.dot(weights.T, returns) / np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        initial_guess = np.array([1 / data.shape[1] for _ in range(data.shape[1])])
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        return pd.Series(result.x, index=data.columns)

class NaiveOptimizer(Optimizer):
    def optimize(self, data, constraints=None):
        weights = {}
        for asset_class in self.universe.get_universe_by_layer().values():
            num_assets = len(asset_class)
            weight = 1 / num_assets
            for asset in asset_class:
                weights[asset] = weight
        return weights



# Sample usage
universe = Universe()  # Assuming you're using the Universe class provided earlier
strategies = [mean_variance_strategy]  # List of strategy functions

pipeline = AssetAllocationPipeline(universe, strategies)
data = pd.DataFrame()  # Load your data here
allocation_results = pipeline.run_pipeline(data)

print(allocation_results)

import numpy as np
import pandas as pd
from typing import List, Callable, Dict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from typing import List, Dict, Callable
from datetime import datetime, timedelta

class Backtest:
    def __init__(self, pipeline: AssetAllocationPipeline, data: pd.DataFrame, start_date: datetime, end_date: datetime, rebalancing_frequency: timedelta, transaction_cost: float = 0.0, slippage: float = 0.0, benchmark: pd.DataFrame = None):
        self.pipeline = pipeline
        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        self.rebalancing_frequency = rebalancing_frequency
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.benchmark = benchmark

    def run_backtest(self) -> Dict:
        backtest_results = {}
        current_date = self.start_date
        weights = None
        assets = self.pipeline.universe.get_last_layer()
        rebalancing_dates = pd.date_range(self.start_date, self.end_date, freq=self.rebalancing_frequency)

        while current_date <= self.end_date:
            date_data = self.data.loc[self.data.index == current_date]
            if not date_data.empty:
                # Check if it's time to rebalance
                if current_date in rebalancing_dates or weights is None:
                    allocation_results = self.pipeline.run_pipeline(date_data)
                    new_weights = allocation_results['L{}'.format(len(allocation_results))]

                    # Apply transaction costs and slippage if there are previous weights
                    if weights is not None:
                        turnover = np.abs(weights - new_weights).sum() / 2
                        cost = turnover * (self.transaction_cost + self.slippage)
                    else:
                        cost = 0

                    weights = new_weights

                asset_returns = date_data[assets].pct_change()
                weighted_returns = asset_returns.mul(weights).sum() - cost
                backtest_results[current_date] = weighted_returns.values[0]

            current_date += timedelta(days=1)

        # ...
        # Rest of the method remains the same

        backtest_df = pd.Series(backtest_results)
        if self.benchmark is not None:
            benchmark_returns = self.benchmark.loc[self.start_date:self.end_date].pct_change()
            backtest_df = pd.concat([backtest_df, benchmark_returns], axis=1, join='inner')
            backtest_df.columns = ['Strategy', 'Benchmark']
            backtest_df['Excess Returns'] = backtest_df['Strategy'] - backtest_df['Benchmark']
            backtest_df = backtest_df.dropna()

        results = {}
        results['Returns'] = backtest_df['Strategy'].sum()
        results['Volatility'] = backtest_df['Strategy'].std() * np.sqrt(252)
        results['Sharpe Ratio'] = results['Returns'] / results['Volatility']
        results['Max Drawdown'] = (backtest_df['Strategy'].cummax() - backtest_df['Strategy']).max()
        results['Turnover'] = (np.abs(backtest_df['Strategy'].diff()) / 2).mean() / backtest_df['Strategy'].mean()

        if self.benchmark is not None:
            results['Benchmark Returns'] = backtest_df['Benchmark'].sum()
            results['Excess Returns'] = backtest_df['Excess Returns'].sum()
            results['Information Ratio'] = results['Excess Returns'] / backtest_df['Excess Returns'].std()
            results['Tracking Error'] = backtest_df['Excess Returns'].std()

        return results
