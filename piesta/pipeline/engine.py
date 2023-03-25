
class Engine:
    def __init__(self):
        self.asset_data = {}
        
    def add_asset(self, asset_name, asset_data):
        self.asset_data[asset_name] = asset_data
        
    def run(self, pipeline_class, start_date, end_date, constraints_manager=None):
        pipeline = pipeline_class()
        for name, term in pipeline.columns.items():
            asset_data = self.asset_data[term.asset_name]
            data = asset_data[term.column]
            term.set_data(data)
            pipeline.add_column(name, term)
            
        if constraints_manager is not None:
            composite_constraint = constraints_manager.get_composite_constraint()
            pipeline.set_screen(composite_constraint)
            
        return pipeline.run_pipeline(start_date, end_date)

class PipelineEngine:
    """
    Engine for computing pipeline results.
    """
    def __init__(self):
        self._cache = {}
        
    def run_pipeline(self, pipeline, start_date, end_date):
        """
        Compute pipeline results over a range of dates.
        """
        dates = pipeline._compute_dates(start_date, end_date)
        results = {}
        
        for date in dates:
            if date in self._cache:
                # Use cached results if available.
                results[date] = self._cache[date]
            else:
                # Compute pipeline results for the date.
                columns = pipeline.columns
                graph = pipeline.graph
                dependencies = pipeline.dependencies
                outputs = {}
                inputs = {}

                # Initialize input values.
                for node in graph:
                    inputs[node] = {}

                # Populate input values.
                for node in graph:
                    for dep in dependencies[node]:
                        if dep in columns:
                            inputs[node][dep] = results[date - 1][dep]
                        else:
                            inputs[node][dep] = pipeline.inputs[dep][date]

                # Compute output values.
                for node in graph:
                    outputs[node] = graph[node](*[inputs[node][dep] for dep in dependencies[node]])

                # Filter results by the pipeline screen.
                screen = pipeline.screen
                mask = screen.mask(*[outputs[col] for col in screen.inputs])
                masked_outputs = {col: outputs[col][mask] for col in outputs}

                # Save results to the cache.
                results[date] = masked_outputs
                self._cache[date] = masked_outputs
        
        return results
