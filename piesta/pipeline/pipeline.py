from collections import defaultdict
from typing import Dict, List, Optional, Callable
import pandas as pd
from itertools import islice 
from graphviz import Digraph


class Portfolio_Pipeline:
    '''
    Examples
    ------------------------------
    pf_pipeline = Portfolio_Pipeline(
        [('SAA', GBI(), {"rf_rate":'UST10YR',}),
         ('TAA', Black_Litterman(), {"rf_rate":'UST2YR',"prior":prior}),
         ('AP', MVO(), {"rf_rate":'UST3MO',})])
    pf_pipeine.set_universe(Universe)
    pf_pipeline.Backtest()
    pf_pipeline.plot_backtest()
    '''

    def __init__(self, steps:List[Callable]):
        self.steps = steps

    def _iter(self,):
        # Generate (idx, (step, model_name, params) tuples from self.steps
        for idx, (step, model, params) in enumerate(self.steps):
            yield idx, step, model, params

    def _log_message(self, step_idx):
        name,_ = self.steps[step_idx]
        return f'(step {step_idx+1} of {len(self.steps)} Processing {name}'
    
    def _get_constraints(self, model, params):
        model.optimize()

    def set_universe(self, screen_rule):
        self.universe = Universe(screen_rule)

    def run(self, start_date, end_date, rebalancing_rule = 'MS'):
        for step_idx, step, model, params in self._iter():
            model = Backtest(self.universe.step, model)
            model.run(start_date, end_date, rebalancing_rule)
            model.weights

    def plot_pipeline(self,):
        pipeline = Digraph(name='pipeline', filename='pipeline', format='png', directory='./')
        pipeline.attr('node', {'shape':'box', "color":'black', "penwidth":'1'}, label='Portfolio Pipeline')
        for i in range(0,2):
            pipeline.edges(f'[{self.steps[i][0]}]{self.steps[i][1]}', f'[{self.steps[i+1][0]}]{self.steps[i+1][1]}', label = "Constraints")
        pipeline.view()

    def plot_backtest(self,):
        return

    def get_backtest_stats(self,):
        return self.batcktest.stats


class Backtest:
    def __init__(self, universe):
        self.universe = universe
        self.weights = {}
    
    @property
    def result(self,):
        return self._result
    
    def _load_data(self,):
        pass

    def _iter(self,):
        for name, cvx_opt, params in islice(self.steps, 0, len(self.steps)):
            yield name, cvx_opt, params
        
    def run(self, start_date, end_date, rebalancing_rule):
        for rebalancing_date in pd.date_range(start_date, end_date, freq=rebalancing_rule):
            for strategy in self.pipeline:
                weights = strategy.get_optimal_weights(params, constraint=weights if strategy.name in ['TAA', 'AP'] else None)
            self.weights[rebalancing_date] = weights

    def plot(self,):
        pass