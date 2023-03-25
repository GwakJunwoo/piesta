class ConstraintsManager:
    def __init__(self):
        self.constraints = []
    
    def add_constraint(self, constraint):
        self.constraints.append(constraint)
    
    def get_constraints(self, dt, assets):
        constraints = []
        for constraint in self.constraints:
            if constraint.active(dt):
                constraints.extend(constraint.get_assets_constraints(dt, assets))
        return constraints
