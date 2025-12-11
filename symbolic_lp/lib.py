import numpy as np


class Parameter:
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.index = None
        if isinstance(size, tuple):
            # Batched ND array of scalar Parameters
            self._array = np.empty(size, dtype=object)
            for idx in np.ndindex(size):
                pname = f"{name}{list(idx)}"
                self._array[idx] = Parameter(pname, 1)
        elif isinstance(size, int) and size > 1:
            self._array = np.array([Parameter(f"{name}[{i}]", 1) for i in range(size)])
        else:
            self._array = None
        self.value = None  # Only used for scalar

    def __pow__(self, power):
        if self._array is not None:
            return ParameterExpression(
                list(self._array.flat), lambda vals: np.array(vals) ** power
            )
        return ParameterExpression([self], lambda vals: vals[0] ** power)

    def __truediv__(self, other):
        if self._array is not None:
            return ParameterExpression(
                list(self._array.flat), lambda vals: np.array(vals) / other
            )
        if isinstance(other, (int, float, np.ndarray, Parameter, ParameterExpression)):
            return self * (other**-1)
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        if self._array is not None:
            return ParameterExpression(
                list(self._array.flat), lambda vals: other / np.array(vals)
            )
        if isinstance(other, (int, float, np.ndarray, Parameter, ParameterExpression)):
            return other * (self**-1)
        else:
            return NotImplemented

    def _get_value(self):
        if self._array is not None:
            # Return numpy array of all values
            return np.array([p._get_value() for p in self._array.flat]).reshape(
                self.size
            )
        if self.value is None:
            raise ValueError(f"Parameter {self.name} has no value set")
        return self.value

    def set_value(self, value):
        if self._array is not None:
            arr = np.array(value)
            # Accept shape (n,) for size n or (n,) for size (n,)
            expected_shape = self.size if isinstance(self.size, tuple) else (self.size,)
            if arr.shape != expected_shape:
                # Accept shape (n,) for size n
                if isinstance(self.size, int) and arr.shape == (self.size,):
                    pass
                else:
                    raise ValueError(
                        f"Value shape {arr.shape} does not match parameter size {self.size}"
                    )
            for idx in np.ndindex(arr.shape):
                self._array[idx].set_value(arr[idx])
            self.value = arr
        else:
            self.value = value

    def __matmul__(self, other):
        # Optional: implement matrix multiplication if needed
        if self._vector is not None:
            # Example: dot product with another vector
            if isinstance(other, (list, np.ndarray)) and len(other) == self.size:
                return sum(self._vector[i] * other[i] for i in range(self.size))
            elif (
                isinstance(other, Parameter)
                and other._vector is not None
                and other.size == self.size
            ):
                return sum(self._vector[i] * other._vector[i] for i in range(self.size))
            else:
                raise ValueError("Incompatible shapes for @ operator")
        else:
            # Scalar case: treat as multiplication
            return self * other

    def __mul__(self, other):
        if self._array is not None:
            arr_flat = list(self._array.flat)
            if (
                isinstance(other, (list, np.ndarray))
                and np.array(other).shape == self.size
            ):
                return ParameterExpression(
                    arr_flat,
                    lambda vals: np.array(vals)
                    * np.array(other).reshape(self.size).flat,
                )
            elif (
                isinstance(other, Parameter)
                and other._array is not None
                and other.size == self.size
            ):
                return ParameterExpression(
                    arr_flat + list(other._array.flat),
                    lambda vals: np.array(vals[: len(arr_flat)])
                    * np.array(vals[len(arr_flat) :]),
                )
            else:
                return ParameterExpression(
                    arr_flat, lambda vals: np.array(vals) * other
                )
        elif isinstance(other, (int, float, np.ndarray)):
            return ParameterExpression([self], lambda vals: vals[0] * other)
        elif isinstance(other, LinearExpression):
            return other * self
        elif isinstance(other, Parameter):
            return ParameterExpression([self, other], lambda vals: vals[0] * vals[1])
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if self._array is not None:
            arr_flat = list(self._array.flat)
            shape = self.size if isinstance(self.size, tuple) else (self.size,)
            if isinstance(other, (list, np.ndarray)) and np.array(other).shape == shape:
                # Elementwise add with array
                return ParameterExpression(
                    arr_flat,
                    lambda vals: np.array(vals).reshape(shape) + np.array(other),
                )
            elif (
                isinstance(other, Parameter)
                and other._array is not None
                and other.size == self.size
            ):
                other_flat = list(other._array.flat)
                return ParameterExpression(
                    arr_flat + other_flat,
                    lambda vals: np.array(vals[: len(arr_flat)]).reshape(shape)
                    + np.array(vals[len(arr_flat) :]).reshape(shape),
                )
            else:
                # Add scalar or Parameter to all elements
                return ParameterExpression(
                    arr_flat, lambda vals: np.array(vals).reshape(shape) + other
                )
        elif isinstance(other, (int, float, np.ndarray)):
            return ParameterExpression([self], lambda vals: vals[0] + other)
        elif isinstance(other, LinearExpression):
            return other + self
        elif isinstance(other, Parameter):
            return ParameterExpression([self, other], lambda vals: vals[0] + vals[1])
        else:
            return NotImplemented

    def __radd__(self, other):
        # Support int + ParameterExpression and similar
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return (-1 * self) + other

    def __neg__(self):
        return -1 * self

    def __le__(self, rhs):
        return LinearExpression({}, self) <= rhs

    def __ge__(self, rhs):
        return LinearExpression({}, self) >= rhs

    def __eq__(self, other):
        return LinearExpression({}, self).__eq__(other)

    def __getitem__(self, idx):
        if self._array is not None:
            return self._array[idx]
        if idx == 0:
            return self
        raise IndexError(f"Scalar Parameter only supports index 0, got {idx}")

    def __repr__(self):
        if self._array is not None:
            return f"Parameter({self.name}, size={self.size})"
        return self.name


class ParameterExpression:
    """Represents an expression involving only parameters (evaluates to a scalar)."""

    def __init__(self, params, eval_fn):
        self.params = params  # list of Parameter objects
        self.eval_fn = eval_fn  # function to evaluate given parameter values

    def evaluate(self):
        """Evaluate to a numeric value or array, preserving shape if possible."""
        values = [p._get_value() for p in self.params]
        result = self.eval_fn(values)
        # If all params are Parameter arrays with the same shape, try to reshape result
        if len(self.params) > 0 and all(
            hasattr(p, "size") and p._array is not None for p in self.params
        ):
            # Use the shape of the first param
            shape = (
                self.params[0].size
                if isinstance(self.params[0].size, tuple)
                else (self.params[0].size,)
            )
            try:
                return np.array(result).reshape(shape)
            except Exception:
                return result
        return result

    def __mul__(self, other):
        if isinstance(other, Variable):
            return LinearExpression({other: self}, 0)
        elif isinstance(other, LinearExpression):
            return other * self
        elif isinstance(
            other, (int, float, np.ndarray, Parameter, ParameterExpression)
        ):
            return ParameterExpression(
                self.params + ([other] if isinstance(other, Parameter) else []),
                lambda vals: self.eval_fn(vals[: len(self.params)])
                * (
                    vals[-1]
                    if isinstance(other, Parameter)
                    else (
                        other.evaluate()
                        if isinstance(other, ParameterExpression)
                        else other
                    )
                ),
            )
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, Variable):
            return LinearExpression({other: 1}, self)
        elif isinstance(other, LinearExpression):
            return other + self
        elif isinstance(other, (int, float, np.ndarray, Parameter)):
            return ParameterExpression(
                self.params + ([other] if isinstance(other, Parameter) else []),
                lambda vals: self.eval_fn(vals[: len(self.params)])
                + (vals[-1] if isinstance(other, Parameter) else other),
            )
        elif isinstance(other, ParameterExpression):
            # Compose two ParameterExpressions
            return ParameterExpression(
                self.params + other.params,
                lambda vals: self.eval_fn(vals[: len(self.params)])
                + other.eval_fn(vals[len(self.params) :]),
            )
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, Variable):
            return LinearExpression({other: self}, 0)
        elif isinstance(other, LinearExpression):
            return other * self
        elif isinstance(other, (int, float, np.ndarray, Parameter)):
            return ParameterExpression(
                self.params + ([other] if isinstance(other, Parameter) else []),
                lambda vals: self.eval_fn(vals[: len(self.params)])
                * (vals[-1] if isinstance(other, Parameter) else other),
            )
        elif isinstance(other, ParameterExpression):
            # Compose two ParameterExpressions
            return ParameterExpression(
                self.params + other.params,
                lambda vals: self.eval_fn(vals[: len(self.params)])
                * other.eval_fn(vals[len(self.params) :]),
            )
        return NotImplemented

    def __eq__(self, rhs):
        # Support ParameterExpression == rhs
        return LinearConstraint(LinearExpression({}, self), rhs, "=")


class Variable:
    def __truediv__(self, other):
        # Variable / scalar or Variable / Parameter
        if isinstance(other, (int, float, np.ndarray)):
            return LinearExpression({self: 1 / other}, 0)
        elif isinstance(other, Parameter):
            return LinearExpression({self: 1 / other}, 0)
        else:
            return NotImplemented

    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.index = None
        if isinstance(size, tuple):
            self._array = np.empty(size, dtype=object)
            for idx in np.ndindex(size):
                vname = f"{name}{list(idx)}"
                self._array[idx] = Variable(vname, 1)
        elif isinstance(size, int) and size > 1:
            self._array = np.array([Variable(f"{name}[{i}]", 1) for i in range(size)])
        else:
            self._array = None

    def __hash__(self):
        return hash((self.name, self.index))

    def __mul__(self, other):
        if self._array is not None:
            arr_flat = list(self._array.flat)
            # Elementwise Variable * other
            if isinstance(other, np.ndarray) and other.shape == self.size:
                return sum([arr_flat[i] * other.flat[i] for i in range(len(arr_flat))])
            elif isinstance(other, (list, tuple)) and len(other) == len(arr_flat):
                return sum([arr_flat[i] * other[i] for i in range(len(arr_flat))])
            else:
                return sum([v * other for v in arr_flat])
        if isinstance(other, Variable):
            return LinearExpression({self: 1, other: 1}, 0)
        elif isinstance(other, Parameter):
            return LinearExpression({self: other}, 0)
        elif isinstance(other, ParameterExpression):
            return LinearExpression({self: other}, 0)
        elif isinstance(other, LinearExpression):
            return NotImplemented
        elif isinstance(other, (int, float, np.ndarray)):
            return LinearExpression({self: other}, 0)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if self._array is not None:
            # For batched Variable, treat the whole Variable as a single symbolic object for addition with constants
            if isinstance(other, (int, float, np.ndarray)):
                return LinearExpression({self: 1}, other)
            elif isinstance(other, Variable):
                if other.name == self.name and other.size == self.size:
                    # Same batched Variable
                    return LinearExpression({self: 2}, 0)
                return LinearExpression({self: 1, other: 1}, 0)
            elif isinstance(other, Parameter):
                return LinearExpression({self: 1}, other)
            elif isinstance(other, ParameterExpression):
                return LinearExpression({self: 1}, other)
            elif isinstance(other, LinearExpression):
                return other + self
            else:
                # Fallback: elementwise addition (legacy, not used in most tests)
                arr_flat = list(self._array.flat)
                return sum([v + other for v in arr_flat])
        if isinstance(other, Variable):
            return LinearExpression({self: 1, other: 1}, 0)
        elif isinstance(other, Parameter):
            return LinearExpression({self: 1}, other)
        elif isinstance(other, ParameterExpression):
            return LinearExpression({self: 1}, other)
        elif isinstance(other, LinearExpression):
            return other + self
        elif isinstance(other, (int, float, np.ndarray)):
            return LinearExpression({self: 1}, other)
        else:
            return NotImplemented

    def __radd__(self, other):
        # Ensure commutativity for batched Variable + constant
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return (-1 * self) + other

    def _to_expr(self):
        return LinearExpression({self: 1}, 0)

    def __le__(self, rhs):
        return self._to_expr() <= rhs

    def __ge__(self, rhs):
        return self._to_expr() >= rhs

    def __eq__(self, rhs):
        return self._to_expr().__eq__(rhs)

    def __getitem__(self, idx):
        if self._array is not None:
            return self._array[idx]
        if idx == 0:
            return self
        raise IndexError(f"Scalar Variable only supports index 0, got {idx}")

    def __repr__(self):
        if self._array is not None:
            return f"Variable({self.name}, size={self.size})"
        return self.name


class LinearExpression:
    def __init__(self, coeffs=None, constant=0):
        # coeffs: dict {Variable: coefficient (int/float/Parameter/ParameterExpression)}
        # constant: int/float/Parameter/ParameterExpression
        self.coeffs = coeffs or {}
        self.constant = constant

    def __add__(self, other):
        if isinstance(other, LinearExpression):
            new_coeffs = self.coeffs.copy()
            for v, c in other.coeffs.items():
                new_coeffs[v] = new_coeffs.get(v, 0) + c
            return LinearExpression(new_coeffs, self.constant + other.constant)
        elif isinstance(other, Variable):
            return self + other._to_expr()
        elif isinstance(other, ParameterExpression):
            # Add ParameterExpression as symbolic constant
            return LinearExpression(self.coeffs, self.constant + other)
        elif isinstance(other, (Parameter, int, float, np.ndarray)):
            return LinearExpression(self.coeffs, self.constant + other)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return (-1 * self) + other

    def __mul__(self, scalar):
        if isinstance(scalar, Variable):
            return NotImplemented
        # scalar can be Parameter, ParameterExpression, or numeric
        new_coeffs = {}
        for v, c in self.coeffs.items():
            new_coeffs[v] = c * scalar
        return LinearExpression(new_coeffs, self.constant * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, other):
        # Support division by Parameter, ParameterExpression, or numeric
        if isinstance(other, (Parameter, ParameterExpression, int, float, np.ndarray)):
            new_coeffs = {v: c / other for v, c in self.coeffs.items()}
            return LinearExpression(new_coeffs, self.constant / other)
        return NotImplemented

    def __rtruediv__(self, other):
        # Support right division: numeric / LinearExpression (not generally meaningful, but for completeness)
        return NotImplemented

    def __le__(self, rhs):
        return LinearConstraint(self, rhs, "<=")

    def __ge__(self, rhs):
        return LinearConstraint(self, rhs, ">=")

    def __eq__(self, rhs):
        return LinearConstraint(self, rhs, "=")

    def __repr__(self):
        return f"Expr({self.coeffs} + {self.constant})"


class LinearConstraint:
    def __init__(self, lhs, rhs, sense):
        if not isinstance(lhs, LinearExpression):
            if isinstance(lhs, Variable):
                lhs = LinearExpression({lhs: 1}, 0)
            else:
                lhs = LinearExpression({}, lhs)
        if not isinstance(rhs, LinearExpression):
            if isinstance(rhs, Variable):
                rhs = LinearExpression({rhs: 1}, 0)
            else:
                rhs = LinearExpression({}, rhs)
        self.lhs = lhs
        self.rhs = rhs
        self.sense = sense

    def __repr__(self):
        return f"{self.lhs} {self.sense} {self.rhs}"
