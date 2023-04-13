from typing import Dict, List, Optional, Callable
import pandas as pd
from itertools import islice 
from graphviz import Digraph
from piesta.data.asset import Universe, Loader
from IPython.display import diplay_png, Image
import os
import matplotlib.pyplot as plt


# Portfolio_Pipeline
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

    def __len__(self,):
        return len(self.steps)
    
    def _iter(self,):
        # Generate (idx, (step, model_name, params) tuples from self.steps
        for idx, (step, model, params) in enumerate(self.steps):
            yield idx, step, model, params

    def _sorted_univ(self, step_idx):
        _sorted_univ = [self.universe.get_universe_by_layer()['L0'], self.universe.get_universe_by_layer()['L1'],
                        self.universe.get_last_layer()]
        return _sorted_univ[step_idx]
            
    def _run(self, start_date, end_date, rebalancing_rule):
        constraint = None
        self._bt_engine = []
        for step_idx, step, model, params in self._iter():
            bt_engine = Backtest(self._sorted_univ[step_idx], model, params)
            bt_engine.set_constraints(constraint)
            bt_engine.run(start_date, end_date, rebalancing_rule)
            self._bt_engine[step_idx] = bt_engine
            constraint = bt_engine._historical_weights

    def set_universe(self,univ:Universe):
        self.universe = univ

    def run(self, start_date, end_date, rebalancing_rule = 'MS'):
        self._run(start_date, end_date, rebalancing_rule,)
        return self
    
    def plot_pipeline(self, save=False, inline=True):
        pipeline = Digraph(name='pipeline', filename='pipeline', format='png', directory='./')
        pipeline.attr('node', {'shape':'box', "color":'black', "penwidth":'1'}, label='Portfolio Pipeline')
        for i in range(0,2):
            pipeline.edge(f'[{self.steps[i][0]}]{self.steps[i][1]}', f'[{self.steps[i+1][0]}]{self.steps[i+1][1]}', label='constraints')
        
        pipeline.render(filename='pipeline', view=False)
        diplay_png(Image(pipeline.filename))
        if not save:
            os.remove(pipeline.filename)

    def plot_backtest_reuslt(self,):
        fig, ax = plt.subplots(figsize=(15,4))
        ax.plot(self._bt_engine[-1].results)

    def get_backtest_stats(self,):
        return self._bt_engine[-1].bt_statistics


# Backtest Module
class Backtest:
    def __init__(self, model, univ:Universe, params):
        self.model = model
        self.universe = univ
        self.constraints = None
        self.params = params
    
    def _load_data(self,):
        Loader(self.universe)
    
    def _run(self, start_date, end_date, reblancing_rule="MS"):
        self._historical_weights = {}
        for rebalancing_date in pd.date_range(start_date, end_date, freq=reblancing_rule):
            weights = self.model.optimize(self.constraints[rebalancing_date])
            self._historical_weights[rebalancing_date] = weights

    @property
    def historical_weights(self,):
        return self._hisotrical_weights
    
    @property
    def bt_statistics(self,):
        return self._bt_statistics
    
    @property
    def results(self,):
        return self._results
    
    def set_constraints(self, constraint):
        self.constraints = constraint
  

    def run(self, start_date, end_date, rebalancing_rule):
        self._run(start_date, end_date, rebalancing_rule)
        return self