from collections import defaultdict
from typing import Dict, List, Optional


class Term:
    def __init__(self, name: str):
        self.name = name
        self.inputs = set()
        self.outputs = set()

    def _compute(self, inputs):
        raise NotImplementedError

    def compute(self, inputs):
        computed_inputs = [i.compute(inputs) for i in self.inputs]
        return self._compute(computed_inputs)

    def _traverse(self, visited, func):
        visited.add(self)
        for input_term in self.inputs:
            if input_term not in visited:
                input_term._traverse(visited, func)
        func(self)

    def traverse_postorder(self, func):
        visited = set()
        self._traverse(visited, func)

    def __repr__(self):
        return f"{type(self).__name__}({self.name})"


class Filter(Term):
    def __init__(self, name: str, predicate):
        super().__init__(name)
        self.predicate = predicate

    def _compute(self, inputs):
        return inputs[0] & self.predicate


class Factor(Term):
    def __init__(self, name: str, compute_fn):
        super().__init__(name)
        self.compute_fn = compute_fn

    def _compute(self, inputs):
        return self.compute_fn(*inputs)


class Pipeline:
    def __init__(self, columns: Optional[Dict[str, Term]] = None,
                 screen: Optional[Filter] = None):
        self.columns = columns or {}
        self.screen = screen

    def add(self, name: str, term: Term):
        self.columns[name] = term

    def remove(self, name: str):
        del self.columns[name]

    def attach_strategy(self, strategy):
        for name, term in strategy.items():
            self.add(name, term)

    def detach_strategy(self, strategy):
        for name in strategy:
            self.remove(name)

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
        """
        Visualize the pipeline as a graph using Graphviz.
        """
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

import networkx as nx
import matplotlib.pyplot as plt


def visualize_pipeline(strategy_objects):
    """
    Create a graph visualization of the pipeline for the given list of strategy objects.

    Parameters
    ----------
    strategy_objects: list[Custom_Strategy]
        A list of Custom_Strategy objects representing the pipeline.

    Returns
    -------
    None
    """
    # Create a directed graph to represent the pipeline
    graph = nx.DiGraph()

    # Add nodes for each strategy object and its columns
    for i, strategy in enumerate(strategy_objects):
        # Add a node for the strategy object
        graph.add_node(f"Strategy {i}")

        # Add nodes for each column of the strategy object
        for col in strategy.columns:
            graph.add_node(col)
            graph.add_edge(f"Strategy {i}", col)

    # Add edges for constraints between strategy objects
    for i in range(len(strategy_objects) - 1):
        # Get the constraints from the current strategy object
        constraints = strategy_objects[i].constraints

        # Add edges from columns in the current strategy object to columns in the next strategy object
        for col1 in constraints:
            for col2 in strategy_objects[i + 1].columns:
                if constraints[col1].dependencies.intersection(strategy_objects[i + 1].columns[col2].dependencies):
                    graph.add_edge(col1, col2)

    # Draw the graph
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_color="lightblue")
    nx.draw_networkx_labels(graph, pos)
    nx.draw_networkx_edges(graph, pos, edge_color="gray", arrowsize=10, arrowstyle="->")
    plt.show()

class AssetAllocationPipeline:
    def __init__(self, universe, start_date, end_date, frequency='daily'):
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.pipeline = None
        self.columns = {}
        self.screen = None
        self.constraints = []
        
    def add_column(self, name, term):
        self.columns[name] = term
        
    def set_screen(self, screen):
        self.screen = screen
        
    def constraints_manager(self, constraints):
        self.constraints = constraints

    def run_pipeline(self):
        data = {}
        dates = self.universe[self.universe['date'].between(self.start_date, self.end_date)]['date'].unique()
        
        for date in dates:
            # Get data for the current date
            date_mask = (self.universe['date'] == date)
            current_data = self.universe[date_mask]
            
            # Filter by the screen
            if self.screen is not None:
                current_data = current_data[self.screen(current_data)]
            
            # Compute pipeline columns
            if self.pipeline is not None:
                output = engine.run_pipeline(self.pipeline, start_date=date, end_date=date)
                current_data = current_data.merge(output, on='asset')
                
            data[date] = current_data
                
        return data
