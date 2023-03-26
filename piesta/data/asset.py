from functools import reduce
import operator
from graphviz import Digraph
from typing import List, Optional, Dict, DefaultDict

class Universe:
    """
    The Universe class represents a collection of assets grouped by type. It has the following methods: 

    __init__(self, universe: Optional[Dict] = None): Initializes a new instance of the Universe class with the given universe dict. If no universe is provided, a sample universe is generated.
    remove(self, name): Deletes all assets with the given name in the universe dict and updates the internal state of the Universe object.
    diagram(self, filename=None, save_png=False): Creates a diagram of the Universe object using the Digraph class from the graphviz package. The diagram shows the hierarchy of the assets by type. 
    The diagram can be saved to a file and/or displayed using the view() method.
    
    _update(self, universe: Optional[Dict] = None): Updates the internal state of the Universe object with the given universe dict.
    _remove(self, name, d): Recursively deletes all assets with the given name in the given dictionary.
    _recur_remove_func(self, name, d): Recursively deletes all assets with the given name in the given dictionary.
    _get_depth(self, d: Dict) -> int: Returns the depth of the given dictionary.
    _get_keys_by_layer(self, d): Returns a list of keys at each layer of the given dictionary hierarchy.
    _get_bottom_values(self, d): Returns the lowest values of the given dictionary hierarchy.
    _get_bottom_values_by_layer(self, d): Returns a flattened list of the lowest values in the given dictionary hierarchy.
    _generate_sample(self) -> Dict: Generates a sample universe dictionary.
    """

    def __init__(self, universe : Optional[Dict] = None):
        self._update(universe)

    def remove(self, name):
        self._remove(name, self._universe)

    def diagram(self, filename = None, save_png = False):
        g = Digraph('G', filename= filename or'universe_diagram', format='png')
        for asset_type in self._universe:
            with g.subgraph(name=f'cluster_{asset_type}') as cluster:
                cluster.attr(label=asset_type)
                for asset_region in self._universe[asset_type]:
                    for ticker in self._universe[asset_type][asset_region]:
                        cluster.node(ticker, label=f"{ticker}\n({asset_region})")
        if save_png:
            g.render(filename= filename or 'universe_diagram', view=False)
        else:
            g.view()

    def _remove(self, name, d):
        if self._universe is not None:
            self._recur_remove_func(name, d)
            self._update(self._universe)

    
    def _update(self, universe : Optional[Dict] = None):
        self._universe = universe or self._generate_sample()
        self._depth = self._get_depth(self._universe)
        self._hierarchy_list = self._get_keys_by_layer(self._universe)
        self._last_assets = self._get_bottom_values_by_layer(self._universe)
        self._hierarchy = DefaultDict(Optional[Dict or List])

        cnt = 0
        for level in range(self._depth-1):
            self._hierarchy[f'L{cnt}'] = self._hierarchy_list[level]
            cnt += 1
        self._hierarchy[f'L{cnt+1}'] = self._last_assets


    def _recur_remove_func(self, name, d):
        if isinstance(d, dict):
            for key, value in list(d.items()):
                if key == name:
                    del d[key]
                else:
                    self._remove(name, value)
        elif isinstance(d, list):
            for item in d:
                self._remove(name, item)

        elif isinstance(d, str):
            del d


    def _get_depth(self, d : Dict) -> int:
        if isinstance(d, dict):
            return 1 + (max(map(self._get_depth, d.values())) if d else 0)
        return 1
    

    def _get_keys_by_layer(self, d):
        result = [[]]
        for key, value in d.items():
            if isinstance(value, dict):
                sub_result = self._get_keys_by_layer(value)
                for i in range(len(sub_result)):
                    if i >= len(result) - 1:
                        result.append([])
                    result[i+1].extend(sub_result[i])
            result[0].append(key)
        return result


    def _get_bottom_values(self, d):
        if isinstance(d, dict):
            return [value for val in d.values() for value in self._get_bottom_values(val)]
        else:
            return [d]


    def _get_bottom_values_by_layer(self, d):
        return list(reduce(operator.add, self._get_bottom_values(d) if d else []))

    def _generate_sample(self) -> Dict:
        _universe = {
            'Stock': {
                'Korea': ['Ticker_0', 'Ticker_1'],
                'US': ['Ticker_2', 'Ticker_3'],
                'Europe': ['Ticker_4', 'Ticker_5'],
                'Japan': ['Ticker_6', 'Ticker_7'],
                'China': ['Ticker_8', 'Ticker_9']
            },
            'Bond': {
                'Developed': ['Ticker_10', 'Ticker_11'],
                'Emerging': ['Ticker_12', 'Ticker_13']
            },
            'Alternative': {
                'Real estate': ['Ticker_14', 'Ticker_15'],
                'Hedge fund': ['Ticker_16', 'Ticker_17']
            },
            'Commodity': {
                'Metal': ['Ticker_18', 'Ticker_19'],
                'Grains': ['Ticker_20', 'Ticker_21'],
                'Energy': ['Ticker_22', 'Ticker_23']
            },
            'Currency': {
                'USDKRW': ['Ticker_24', 'Ticker_25'],
                'USDJPY': ['Ticker_26', 'Ticker_27'],
                'USDEUR': ['Ticker_28', 'Ticker_29']
            }
        }
        return _universe



"""test = Universe()
print(test._hierarchy)
print(test._hierarchy_list)
print(test._universe)

print("=======================")
test.remove('Korea')
print(test._hierarchy)
print(test._hierarchy_list)
print(test._universe)

test.diagram()
"""