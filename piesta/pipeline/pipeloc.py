from scipy.optimize import minimize, LinearConstraint
import numpy as np
import pandas as pd
from typing import List, Callable, Dict, Optional
from abc import ABC, abstractmethod
from data.asset import Universe
from graphviz import Digraph

class AssetAllocationPipeline:
    def __init__(self, universe: Universe, strategies: List[Callable]):
        self.universe = universe
        self.strategies = strategies

    def run_pipeline(self, data: pd.DataFrame) -> Dict:
        allocation_results = {}
        for i, strategy in enumerate(self.strategies):
            layer = 'L{}'.format(i + 1)
            allocation_results[layer] = strategy(self.universe, data)
        return allocation_results

    def diagram(self, filename=None, save_png=False):
        g = Digraph('G', filename=filename or 'pipeline_diagram', format='png')
        g.attr(rankdir='LR', size='8,5')

        for i, strategy in enumerate(self.strategies):
            strategy_name = strategy.__name__
            g.node(f'S{i}', label=strategy_name, shape='box')

            if i > 0:
                g.edge(f'S{i - 1}', f'S{i}')

        if save_png:
            g.render(filename=filename or 'pipeline_diagram', view=False)
        else:
            g.view()
