from collections import defaultdict
from typing import Dict, List, Optional
import pandas as pd
from itertools import islice 
from graphviz import Diagraph

class Pipeline:
    def __init__(self, columns: Optional[Dict[str, Term]] = None,
                 screen: Optional[Filter] = None):
        self.columns = columns or {}
        self.screen = screen

    def add(self, name: str, term: Term):
        self.columns[name] = term

    def remove(self, name: str):
        del self.columns[name]

    def compile(self):
        execution_plan = defaultdict(set)

        def add_edge(input_term, output_term):
            execution_plan[input_term].add(output_term)

        def add_term(term):
            for input_term in term.inputs:
                add_edge(input_term, term)
            for output_term in term.outputs:
                add_edge(term, output_term)

        # Add graph edges for each column.
        for term in self.columns.values():
            term.traverse_postorder(add_term)

        return execution_plan

    def visualize(self):
        from graphviz import Digraph

        execution_plan = self.compile()

        dot = Digraph()

        # Add nodes for each term.
        for term in execution_plan:
            dot.node(term.name, label=term.name)

        # Add edges for each dependency.
        for term, dependents in execution_plan.items():
            for dependent in dependents:
                dot.edge(term.name, dependent.name)

        return dot

class Portfolio_Pipeline(Pipeline):
    '''
    pf_pipeline = Portfolio_Pipeline(
        [('SAA', 'GBI', {rf_rate:'UST10YR',}),
         ('TAA', 'BL', {rf_rate:'UST2YR',}),
         ('AP', 'MVO', {rf_rate:'UST3MO',})])
    '''
    def __init__(self, steps):
        self.steps = steps

    def _iter(self,):
        for idx, name, cvx_opt, params in enumerate(islice(self.steps, 0, len(self.steps))):
            yield idx, name, cvx_opt, params
    
    def plot_pipeline(self, display='diagram', ):
        graph = Digraph()
        for _, (name, _, _) in self._iter():
            graph.node(f'{name}:{cvx_opt}')

        for idx in range(len(self.steps)-1):
            graph.edge(
    

    def set_universe(self, screen_rule):
        self.universe = Universe(screen_rule)

    def run(self, start_date, end_date, rebalancing_rule = 'MS'):
        self.backtest = Backtest(self.steps, self.universe)
        self.backtest.run(start_date, end_date, rebalancing_rule)
        return self.backtest.result

    def plot_backtest(self,):
        return self.backtest.plot()

    def get_backtest_stats(self,):
        return self.batcktest.stats


class Backtest:
    def __init__(self, steps, universe):
        self.universe = universe
        self.steps = steps
        self.weights = {}
        self.strategy = {}
        for name, cvx_opt, params in self._iter():
            self.strategy.append(Strategy(name, cvx_opt, params))

    def _iter(self,):
        for name, cvx_opt, params in islice(self.steps, 0, len(self.steps)):
            yield name, cvx_opt, params
        
    def run(self, start_date, end_date, rebalancing_rule):
        for rebalancing_date in pd.date_range(start_date, end_date).resample(rebalancing_rule).first():
            for strategy in self.strategy:
                weights = strategy.get_optimal_weights(constraint=weights if strategy.name in ['TAA', 'AP'] else None)
            self.weights[rebalancing_date] = weights
