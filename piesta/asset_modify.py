import operator
from functools import reduce
import operator
from typing import Dict, List, Optional
from collections import defaultdict


class Universe:
    def __init__(self, universe: Optional[Dict] = None):
        """
        Initializes a new instance of the Universe class with the given universe dict.
        """
        self._update(universe)

    def _update(self, universe: Optional[Dict] = None):
        """
        Updates the internal state of the Universe object with the given universe dict.
        """
        self._universe = universe or self.sample_universe()
        self._depth = self.dict_depth(self._universe)
        self._hierarchy_list = self.dict_keys_by_layer(self._universe)
        self._last_assets = self.lowest_values_hierarchy(self._universe)
        self._hierarchy = defaultdict(Optional[Dict or List])
        cnt = 0
        for level in range(self._depth - 1):
            self._hierarchy[f"L{cnt}"] = self._hierarchy_list[level]
            cnt += 1
        self._hierarchy[f"L{cnt + 1}"] = self._last_assets

    def delete_assets_step(self, name: str, d: Dict):
        """
        Recursively deletes all assets with the given name in the given dictionary.
        """
        if isinstance(d, dict):
            for key, value in list(d.items()):
                if key == name:
                    del d[key]
                else:
                    self.delete_assets_step(name, value)
        elif isinstance(d, list):
            for item in d:
                self.delete_assets_step(name, item)

    def delete_assets(self, name: str):
        """
        Deletes all assets with the given name in the universe dict and updates the internal state of the Universe object.
        """
        self.delete_assets_step(name, self._universe)
        self._update()

    def dict_depth(self, d: Dict) -> int:
        """
        Returns the depth of the given dictionary.
        """
        if isinstance(d, dict):
            return 1 + (max(map(self.dict_depth, d.values())) if d else 0)
        return 1

    def dict_keys_by_layer(self, d: Dict) -> List[List[str]]:
        """
        Returns a list of keys at each layer of the given dictionary hierarchy.
        """
        result = [[] for _ in range(self.dict_depth(d))]
        stack = [(d, 0)]
        while stack:
            node, depth = stack.pop()
            result[depth].extend(node.keys())
            for child in node.values():
                if isinstance(child, dict):
                    stack.append((child, depth + 1))
        return result

    def lowest_values(self, d: Dict) -> List:
        """
        Returns the lowest values of the given dictionary hierarchy.
        """
        if isinstance(d, dict):
            return [value for val in d.values() for value in self.lowest_values(val)]
        else:
            return [d]

    def lowest_values_hierarchy(self, d: Dict) -> List:
        """
        Returns a flattened list of the lowest values in the given dictionary hierarchy.
        """
        return list(reduce(operator.add, self.lowest_values(d) if d else []))

    def sample_universe(self) -> Dict:
        dict_universe = {}

        # Sample L1, L2 Hierarchy
        level_1 = ['Stock', 'Bond', 'Alternative', 'Commodity', 'Currency']
        level_2 = {'Stock': ['Korea', 'US', 'Europe', 'Japan', 'China'],
                'Bond': ['Developed', 'Emerging'],
                'Alternative': ['Real estate', 'Hedge fund'],
                'Commodity': ['Metal', 'Grains', 'Energy'],
                'Currency': ['USDKRW', 'USDJPY', 'USDEUR']}
        sample_size = len(level_1) * len(level_2) * 2
        sample_num = 0

        # formatting to Dict[Dict[List]]
        for i, l1 in enumerate(level_1):
            tmp_dict = {}
            for j, l2 in enumerate(level_2[l1]):
                tmp_dict[l2] = [f'Ticker_{sample_num}', f'Ticker_{sample_num + 1}']
                sample_num += 2
            dict_universe[l1] = tmp_dict

        return dict_universe


test = Universe()
print(test._hierarchy)
print(test._hierarchy_list)
print(test._universe)

print("=======================")
test.delete_assets('Korea')
print(test._hierarchy)
print(test._hierarchy_list)
print(test._universe)