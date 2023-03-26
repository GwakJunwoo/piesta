from collections import defaultdict
from typing import Dict, List, Optional
import pandas as pd
from itertools import islice 
from graphviz import Diagraph

class Portfolio_Pipeline(Pipeline):
    '''
    pf_pipeline = Portfolio_Pipeline(
        [('SAA', 'GBI', {"rf_rate":'UST10YR',}),
         ('TAA', 'BL', {"rf_rate":'UST2YR',}),
         ('AP', 'MVO', {"rf_rate":'UST3MO',})])
    '''

    def __init__(self, steps:list):
        self.steps = steps

    def plot_pipeline(self,):
        pipeline = Diagraph(name='pipeline', filename='pipeline', format='png', directory='./')
        pipeline.attr('node', {'shape':'box', "color":'black', "penwidth":'1'}, label='Portfolio Pipeline')
        for i in range(0,2):
            pipeline.edges(f'[{self.steps[i][0]}]{self.steps[i][1]}', f'[{self.steps[i+1][0]}]{self.steps[i+1][1]}', label = "Constraints")
        pipeline.view()

    def set_universe(self, screen_rule):
        self.universe = Universe(screen_rule)

    def run_backtest(self, start_date, end_date, rebalancing_rule = 'MS'):
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
        self.pipeline = []
        for name, cvx_opt, params in self._iter():
            name = Strategy(cvx_opt, params); self.pipeline.append(name)
    
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