import numpy as np
import pandas as pd
from functools import reduce
import operator
from copy import deepcopy
from typing import List, Optional, Dict, DefaultDict

class Universe:
    def __init__(self, universe : Optional[Dict] = None):
        self._update(universe)

    def _update(self, universe : Optional[Dict] = None):
        self._universe = universe or self.sample_universe()
        self._depth = self.dict_depth(self._universe)
        self._hierarchy_list = self.dict_keys_by_layer(self._universe)
        self._last_assets = self.lowest_values_hierarchy(self._universe)
        self._hierarchy = DefaultDict(Optional[Dict or List])

        cnt = 0
        for level in range(self._depth-1):
            self._hierarchy[f'L{cnt}'] = self._hierarchy_list[level]
            cnt += 1
        self._hierarchy[f'L{cnt+1}'] = self._last_assets

    def delete_assets_step(self, name, d):
        if isinstance(d, dict):
            for key, value in list(d.items()):
                if key == name:
                    del d[key]
                else:
                    self.delete_assets(name, value)
        elif isinstance(d, list):
            for item in d:
                self.delete_assets(name, item)

        elif isinstance(d, str):
            del d

    def delete_assets(self, name, d):
        self.delete_assets_step(name, d)
        self._update(self._universe)

    def dict_depth(self, d : Dict) -> int:
        if isinstance(d, dict):
            return 1 + (max(map(self.dict_depth, d.values())) if d else 0)
        return 1
    
    def dict_keys_by_layer(self, d):
        result = [[]]
        for key, value in d.items():
            if isinstance(value, dict):
                sub_result = self.dict_keys_by_layer(value)
                for i in range(len(sub_result)):
                    if i >= len(result) - 1:
                        result.append([])
                    result[i+1].extend(sub_result[i])
            result[0].append(key)
        return result

    def lowest_values(self, d):
        if isinstance(d, dict):
            return [value for val in d.values() for value in self.lowest_values(val)]
        else:
            return [d]

    def lowest_values_hierarchy(self, d):
        return list(reduce(operator.add, self.lowest_values(d) if d else []))

    # Generate sample universe dict -> Dict[Dict[List]]
    def sample_universe(self) -> Dict:
        dict_universe = dict()

        # Sample L1, L2 Hierarchy
        level_1 = ['Stock', 'Bond', 'Alternative', 'Commodity', 'Currency']
        level_2 = [['Korea', 'US', 'Europe', 'Japan', 'China'], 
                    ['Developed', 'Emerging'], 
                    ['Real estate', 'Hedge fund'],
                    ['Metal', 'Grains', 'Energy'],
                    ['USDKRW', 'USDJPY', 'USDEUR']]
        sample_size = len(level_1) * len(level_2) * 2
        sample_num = 0

        # formatting to Dict[Dict[List]]
        for i, l1 in enumerate(level_1):
            tmp_dict = dict()
            for j, l2 in enumerate(level_2[i]):
                tmp_dict[l2] = [f'Ticker_{sample_num}', f'Ticker_{sample_num + 1}']
                sample_num += 2
            dict_universe[l1] = tmp_dict

        return dict_universe

test = Universe()
print(test._hierarchy)
print(test._hierarchy_list)
print(test._universe)

print("=======================")
test.delete_assets('Korea', test._universe)
print(test._hierarchy)
print(test._hierarchy_list)
print(test._universe)