import pandas as pd
import numpy as np
from data.asset import Universe
from optimize import MeanVarianceOptimizer, GoalBasedOptimizer, BlackLittermanOptimizer, NaiveOptimizer, RiskParityOptimizer, CustomOptimizer

def mean_variance_strategy(universe: Universe, data: pd.DataFrame) -> pd.Series:
    assets = universe.get_last_layer()
    optimizer = MeanVarianceOptimizer()
    return optimizer.optimize(data[assets])