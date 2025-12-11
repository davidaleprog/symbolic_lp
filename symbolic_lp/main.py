from .lib import (
    Variable,
    Parameter,
    LinearExpression,
    LinearConstraint,
    ParameterExpression,
)
import numpy as np
import torch
import concurrent.futures
import copy


class Model:
    def __init__(self):
        self.variables = []
        self.parameters = []
        self.constraints = []

    def add_var(self, name, size=1):
        v = Variable(name, size)
        if isinstance(size, tuple):
            for idx in np.ndindex(size):
                v._array[idx].index = len(self.variables)
                self.variables.append(v._array[idx])
        elif isinstance(size, int) and size > 1:
            for i in range(size):
                v._array[i].index = len(self.variables)
                self.variables.append(v._array[i])
        else:
            v.index = len(self.variables)
            self.variables.append(v)
        return v

    def add_param(self, name, size=1):
        if "[" in name or "]" in name:
            raise ValueError("Parameter name cannot contain '[]' characters.")
        p = Parameter(name, size)
        if isinstance(size, tuple):
            for idx in np.ndindex(size):
                p._array[idx].index = len(self.parameters)
                self.parameters.append(p._array[idx])
        elif isinstance(size, int) and size > 1:
            for i in range(size):
                p._array[i].index = len(self.parameters)
                self.parameters.append(p._array[i])
        else:
            p.index = len(self.parameters)
            self.parameters.append(p)
        return p

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def get_vector_x(self):
        return np.array([v.name for v in self.variables])

    def build_matrices(self, param_values=None):
        n = len(self.variables)
        self._set_param_values(param_values)
        self._check_param_values()
        A = []
        b = []
        for c in self.constraints:
            # Initialize a row of A
            row = np.zeros(n)
            expr = self._move_to_lhs(c)  # get linear expression as : lhs - rhs <= 0
            self._fill_row_from_expr(row, expr)
            rhs_val = -self._eval_to_float(expr.constant)  # move constant to rhs
            if c.sense == "<=":
                A.append(row)
                b.append(rhs_val)
            elif c.sense == ">=":
                A.append(-row)
                b.append(-rhs_val)
            elif c.sense == "=":
                A.append(row)
                b.append(rhs_val)
                A.append(-row)
                b.append(-rhs_val)
        return np.array(A), np.array(b)

    def _set_param_values(self, param_values: dict):
        """Set parameter values from the provided dictionary.

        Supports param_values like {"p1": [1,2,3], "p2": [4,5]} for parameters named p1[0], p1[1], ...
        """
        if param_values is None:
            return
        # Build a mapping from base name to list of parameters
        name_map = {}
        for p in self.parameters:
            # Split name like "p1[0]" into base "p1" and index 0
            if "[" in p.name and p.name.endswith("]"):
                base, idx = p.name.split("[", 1)
                idx = int(idx[:-1])
                name_map.setdefault(base, []).append((idx, p))
            else:
                name_map.setdefault(p.name, []).append((None, p))
        # Set values for each base name
        for key, val in param_values.items():
            if key not in name_map:
                raise ValueError(f"Parameter '{key}' not found in model.")
            plist = name_map[key]
            if all(idx is not None for idx, _ in plist):
                # Sort by index to match order
                plist = sorted(plist, key=lambda x: x[0])
                if len(val) != len(plist):
                    raise ValueError(
                        f"Length mismatch for parameter '{key}': expected {len(plist)}, got {len(val)}"
                    )
                for (idx, p), v in zip(plist, val):
                    p.set_value(v)
            else:
                # Scalar parameter
                plist[0][1].set_value(val)

    def _check_param_values(self):
        """Returns an error if any parameter has no value set."""
        for p in self.parameters:
            if p.value is None:
                raise ValueError(
                    f"Parameter {p.name} has no value. Please provide values for all parameters."
                )

    def _eval_to_float(self, val):
        """Convert any value type to float or array of floats."""
        if isinstance(val, Parameter):
            v = val._get_value()
            if isinstance(v, (np.ndarray, torch.Tensor)):
                return np.asarray(v, dtype=float)
            return float(v)
        elif isinstance(val, ParameterExpression):
            result = val.evaluate()
            if isinstance(result, (np.ndarray, torch.Tensor)):
                return np.asarray(result, dtype=float)
            return float(result)
        elif isinstance(val, (np.ndarray, torch.Tensor)):
            return np.asarray(val, dtype=float)
        elif isinstance(val, (int, float)):
            return float(val)
        else:
            return float(val)

    def _move_to_lhs(self, constraint):
        """Move constraint to LHS form."""
        return constraint.lhs - constraint.rhs

    def _fill_row_from_expr(self, row, expr):
        """
        Fill the row array with coefficients from the expression.
        If expr is a numpy array of LinearExpression, sum them to a single LinearExpression.
        """
        if isinstance(expr, np.ndarray):
            # Sum all LinearExpressions in the array
            expr_sum = None
            for e in expr.flat:
                if expr_sum is None:
                    expr_sum = e
                else:
                    expr_sum = expr_sum + e
            expr = expr_sum
        for v, coeff in expr.coeffs.items():
            if not isinstance(v, Variable):
                raise ValueError(f"Non-variable {v} found in expression coefficients")
            row[v.index] = self._eval_to_float(coeff)

    def __repr__(self):
        return f"Model(Variables: {self.variables}, Parameters: {self.parameters}, Constraints: {self.constraints})"

    def __repr__(self):
        return f"Model(Variables: {self.variables}, Parameters: {self.parameters}, Constraints: {self.constraints})"

    def __call__(self, param_values=None):
        first_param_val = (
            param_values[next(iter(param_values))] if param_values else None
        )
        if (
            isinstance(first_param_val, (torch.Tensor, np.ndarray))
            and first_param_val.ndim > 1
        ):
            batch_size = first_param_val.shape[0]
            param_batches = [
                {k: v[i] for k, v in param_values.items()} for i in range(batch_size)
            ]

            def build_for_batch(param_batch):
                model_copy = copy.deepcopy(self) # avoid mutation of model state across threads
                return model_copy.build_matrices(param_batch)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(build_for_batch, param_batches))
            A_list, b_list = zip(*results)
            return np.array(A_list), np.array(b_list)
        else:
            return self.build_matrices(param_values)
