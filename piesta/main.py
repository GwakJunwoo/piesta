import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime
from Strategy import CustomStrategy
from Pipeline import AssetAllocationPipeline
from PipelineEngine import PipelineEngine
from Constraints import ConstraintsManager
from Backtesting import BacktestEngine


class AutoAssetAllocationService:
    def __init__(self, start_date: str, end_date: str,
                 universe: List[str], macro_data: pd.DataFrame):
        self.start_date = start_date
        self.end_date = end_date
        self.universe = universe
        self.macro_data = macro_data
        self.constraints_manager = ConstraintsManager()
        self.strategies: Dict[str, CustomStrategy] = {}
        self.pipeline = AssetAllocationPipeline(self.universe, self.macro_data)
        self.pipeline_engine = PipelineEngine(self.pipeline)

    def add_strategy(self, name: str, strategy: CustomStrategy):
        self.strategies[name] = strategy

    def remove_strategy(self, name: str):
        del self.strategies[name]

    def update_strategy(self, name: str, strategy: CustomStrategy):
        if name not in self.strategies:
            raise ValueError(f"Strategy {name} not found.")
        self.strategies[name] = strategy

    def run(self) -> pd.DataFrame:
        pipeline_results = self.pipeline_engine.run_pipeline(self.start_date, self.end_date)
        pipeline_output = pipeline_results['pipeline_output']
        constraints = pipeline_results['constraints']

        for name, strategy in self.strategies.items():
            expected_returns = pipeline_output[strategy.returns].to_numpy()
            covariance_matrix = pipeline_output[strategy.covariance].to_numpy()
            constraints = self.constraints_manager.apply_constraints(constraints, strategy.constraints)
            optimal_weights = strategy.get_optimal_weights(expected_returns, covariance_matrix, constraints)
            pipeline_output[name] = optimal_weights

        return pipeline_output


def main():
    # Sample usage
    start_date = '2019-01-01'
    end_date = '2022-01-01'
    universe = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    macro_data = pd.DataFrame(np.random.randn(1096, 3), index=pd.date_range(start_date, end_date, freq='D'),
                              columns=['gdp_growth', 'inflation', 'unemployment'])

    # Initialize the AutoAssetAllocationService
    auto_asset_allocation_service = AutoAssetAllocationService(start_date, end_date, universe, macro_data)

    # Define the strategies to be used
    strategy_1 = CustomStrategy('strategy_1', ['AAPL', 'MSFT'], ['gdp_growth'], ['inflation'], [0.5, 0.5],
                                {'gdp_growth': (0.0, None)})
    strategy_2 = CustomStrategy('strategy_2', ['GOOG', 'AMZN'], ['inflation'], ['unemployment'], [0.7, 0.3],
                                {'unemployment': (None, 0.03)})
    auto_asset_allocation_service.add_strategy('strategy_1', strategy_1)
    auto_asset_allocation_service.add_strategy('strategy_2', strategy_2)

    # Run the AutoAssetAllocationService
    allocation_results = auto_asset_allocation_service.run()
    print(allocation_results)


if __name__ == '__main__':
    main()
