import pytest
from symbolic_lp import (
    Parameter,
    Variable,
    LinearExpression,
    LinearConstraint,
    ParameterExpression,
)
import numpy as np


@pytest.fixture
def param1():
    return Parameter("p1", 1)


@pytest.fixture
def param2():
    return Parameter("p2", 1)


@pytest.fixture
def param_array1():
    return Parameter("p_array1", 3)


@pytest.fixture
def param_array2():
    return Parameter("p_array2", 3)


@pytest.fixture
def batched_param1():
    return Parameter("batched_p1", (2, 3))


@pytest.fixture
def batched_param2():
    return Parameter("batched_p2", (2, 3))


class Testparameters:
    def test_parameter_creation(self, param1, param_array1, batched_param1):
        assert param1.name == "p1"
        assert param1.size == 1

        assert param_array1.name == "p_array1"
        assert param_array1.size == 3
        assert isinstance(param_array1[0], Parameter)
        assert param_array1[0].name == "p_array1[0]"

        assert batched_param1.name == "batched_p1"
        assert batched_param1.size == (2, 3)

    def test_parameter_set_value_scalar(self, param1, param_array1, batched_param1):
        param1.set_value(5)
        assert param1.value == 5

        param_array1.set_value(np.array([1, 2, 3]))
        assert np.array_equal(param_array1.value, np.array([1, 2, 3]))

        batched_param1.set_value(np.ones((2, 3)))
        assert np.array_equal(batched_param1.value, np.ones((2, 3)))

    def test_parameter_addition(self, param1, param2):
        expr = param1 + param2
        assert isinstance(expr, ParameterExpression)
        assert expr.params == [param1, param2]
        param1.set_value(3)
        param2.set_value(4)
        value = expr.evaluate()
        assert value == 7

    def test_parameter_array_addition(self, param_array1, param_array2):

        expr_array = param_array1 + param_array2
        assert isinstance(expr_array, ParameterExpression)
        param_array1.set_value(np.array([1, 2, 3]))
        param_array2.set_value(np.array([4, 5, 6]))
        value_array = expr_array.evaluate()
        assert np.array_equal(value_array, np.array([5, 7, 9]))

    def test_batched_parameter_array_addition(self, batched_param1, batched_param2):

        expr_array = batched_param1 + batched_param2
        assert isinstance(
            expr_array, ParameterExpression
        ), f"type(expr_array)={type(expr_array)}"
        batched_param1.set_value(np.array([[1, 2, 3], [4, 5, 6]]))
        batched_param2.set_value(np.array([[10, 20, 30], [40, 50, 60]]))
        value_array = expr_array.evaluate()
        assert np.array_equal(
            value_array, np.array([[11, 22, 33], [44, 55, 66]])
        ), f"value_array={value_array}"

    def test_parameter_multiplication(self, param1, param_array1):
        expr = param1 * 3
        assert isinstance(expr, ParameterExpression)
        assert expr.params == [param1]
        param1.set_value(4)
        value = expr.evaluate()
        assert value == 12

        expr_array = param_array1 * 2
        assert isinstance(expr_array, ParameterExpression)
        param_array1.set_value(np.array([1, 2, 3]))
        value_array = expr_array.evaluate()
        assert np.array_equal(value_array, np.array([2, 4, 6]))

    def test_parameter_rmultiplication(self, param1):
        expr = 4 * param1
        assert isinstance(expr, ParameterExpression)
        assert expr.params == [param1]
        param1.set_value(5)
        value = expr.evaluate()
        assert value == 20

    def test_parameter_division_by_parameter(self, param1, param2):
        expr = param1 / param2
        assert isinstance(expr, ParameterExpression)
        assert expr.params == [param1, param2]
        param1.set_value(10)
        param2.set_value(2)
        value = expr.evaluate()
        assert value == 5

    def test_parameter_division_by_scalar(self, param1):
        expr = param1 / 4
        assert isinstance(expr, ParameterExpression)
        assert expr.params == [param1]
        param1.set_value(8)
        value = expr.evaluate()
        assert value == 2

    # ParameterVector is deprecated; use Parameter(size>1) for vector parameters


class TestParameterExpression:
    def test_parameter_expression_creation(self, param1, param2):
        expr = ParameterExpression([param1, param2], lambda vals: vals[0] + 2 * vals[1])
        assert expr.params == [param1, param2]

    def test_parameter_expression_evaluation(self, param1, param2):
        expr = ParameterExpression([param1, param2], lambda vals: vals[0] + 2 * vals[1])
        param1.set_value(3)
        param2.set_value(4)
        value = expr.evaluate()
        assert value == 3 + 2 * 4


@pytest.fixture
def variable():
    return Variable("x", 1)


@pytest.fixture
def variable2():
    return Variable("y", 1)


@pytest.fixture
def variable_array():
    return Variable("z", 3)


@pytest.fixture
def variable_batched():
    return Variable("w1", (2, 3))


@pytest.fixture
def variable_batched2():
    return Variable("w2", (2, 3))


class TestVariable:
    def test_variable_creation(self):
        v = Variable("x", (2, 3))
        assert v.name == "x"
        assert v.size == (2, 3)

    def test_variable_multiplication(self, variable):
        v = variable
        expr = v * 3
        assert isinstance(expr, LinearExpression)
        assert expr.coeffs[v] == 3
        assert expr.constant == 0

    def test_parameter_variable_multiplication(self, param1, variable):
        expr = param1 * variable
        assert isinstance(expr, LinearExpression)
        assert expr.coeffs[variable] == param1
        assert expr.constant == 0

    def test_variable_param_expression_multiplication(self, variable, param1):
        v = variable
        p = param1
        expr = (p + 2) * v
        assert isinstance(expr, LinearExpression)
        assert isinstance(expr.coeffs[v], ParameterExpression)
        assert expr.constant == 0
        p.set_value(3)
        value = expr.coeffs[v].evaluate()
        assert value == 5  # 3 + 2

    def test_variable_rmultiplication(self, variable):
        v = variable
        expr = 3 * v
        assert isinstance(expr, LinearExpression)
        assert expr.coeffs[v] == 3
        assert expr.constant == 0

    def test_division_by_scalar(self, variable):
        v = variable
        expr = v / 4
        assert isinstance(expr, LinearExpression)
        assert expr.coeffs[v] == 0.25
        assert expr.constant == 0

    def test_variable_division_by_parameter(self, variable, param1):
        expr = variable / param1
        assert isinstance(expr, LinearExpression)
        assert isinstance(expr.coeffs[variable], ParameterExpression)
        assert expr.constant == 0
        param1.set_value(4)
        value = expr.coeffs[variable].evaluate()
        assert value == 0.25  # 1/4

    def test_variable_addition(self, variable, variable2):
        expr = variable + variable2
        assert isinstance(expr, LinearExpression)
        assert expr.coeffs[variable] == 1
        assert expr.coeffs[variable2] == 1
        assert expr.constant == 0

    def test_variable_addition_with_parameter(self, variable, param1):
        p = param1
        expr = variable + p
        assert expr.coeffs[variable] == 1

        expr2 = p + variable
        assert expr2.coeffs[variable] == 1

    def test_variable_addition_with_constant(self, variable):
        expr = 5 + variable
        assert expr.coeffs[variable] == 1
        assert expr.constant == 5

        expr2 = variable + 5
        assert expr2.coeffs[variable] == 1
        assert expr2.constant == 5

    def test_batched_variable_addition_with_constant(self, variable_batched):
        expr = 3 + variable_batched
        assert expr.coeffs[variable_batched] == 1
        assert expr.constant == 3

        expr2 = variable_batched + 7
        assert expr2.coeffs[variable_batched] == 1
        assert expr2.constant == 7

    def test_batched_variable_addition(self, variable_batched, variable_batched2):
        v = variable_batched
        expr = v + v
        assert isinstance(expr, LinearExpression)
        assert expr.coeffs[v] == 2
        assert expr.constant == 0

        expr2 = variable_batched + variable_batched2
        assert isinstance(expr2, LinearExpression)
        assert expr2.coeffs[variable_batched] == 1
        assert expr2.coeffs[variable_batched2] == 1
        assert expr2.constant == 0

    def test_variable_subtraction(self, variable, variable2):
        expr = variable - variable2
        assert expr.coeffs[variable] == 1
        assert expr.coeffs[variable2] == -1
        assert expr.constant == 0

    def test_variable_rsubtraction(self, variable):
        expr = 10 - variable
        assert expr.coeffs[variable] == -1
        assert expr.constant == 10

    def test_variable_less_than_or_equal(self, variable):
        constraint = variable <= 5
        assert isinstance(constraint, LinearConstraint)
        assert constraint.sense == "<="
        assert constraint.lhs.coeffs[variable] == 1
        assert constraint.rhs.constant == 5

    def test_variable_greater_than_or_equal(self, variable):
        constraint = variable >= 5
        assert isinstance(constraint, LinearConstraint)
        assert constraint.sense == ">="
        assert constraint.lhs.coeffs[variable] == 1
        assert constraint.rhs.constant == 5

    def test_variable_equality(self, variable):
        constraint = variable == 5
        assert isinstance(constraint, LinearConstraint)
        assert constraint.sense == "="


class TestLinearExpression:
    def test_empty_expression(self):
        expr = LinearExpression()
        assert expr.coeffs == {}
        assert expr.constant == 0

    def test_expression_with_coeffs(self, variable):
        expr = LinearExpression({variable: 3}, 5)
        assert expr.coeffs[variable] == 3
        assert expr.constant == 5

    def test_expression_addition(self):
        v1 = Variable("x", 1)
        v2 = Variable("y", 1)
        expr1 = LinearExpression({v1: 2}, 3)
        expr2 = LinearExpression({v2: 4}, 5)
        result = expr1 + expr2
        assert result.coeffs[v1] == 2
        assert result.coeffs[v2] == 4
        assert result.constant == 8

    def test_expression_addition_same_variable(self, variable):
        expr1 = LinearExpression({variable: 2}, 0)
        expr2 = LinearExpression({variable: 3}, 0)
        result = expr1 + expr2
        assert result.coeffs[variable] == 5

    def test_expression_add_variable(self):
        v1 = Variable("x", 1)
        v2 = Variable("y", 1)
        expr = LinearExpression({v1: 2}, 0)
        result = expr + v2
        assert result.coeffs[v1] == 2
        assert result.coeffs[v2] == 1

    def test_expression_add_constant(self, variable):
        expr = LinearExpression({variable: 2}, 3)
        result = expr + 7
        assert result.coeffs[variable] == 2
        assert result.constant == 10

    def test_expression_subtraction(self, variable, variable2):
        expr1 = LinearExpression({variable: 5}, 10)
        expr2 = LinearExpression({variable2: 3}, 2)
        result = expr1 - expr2
        assert result.coeffs[variable] == 5
        assert result.coeffs[variable2] == -3
        assert result.constant == 8

    def test_expression_multiplication(self, variable):
        expr = LinearExpression({variable: 2}, 3)
        result = expr * 4
        assert result.coeffs[variable] == 8
        assert result.constant == 12

        result = 4 * expr
        assert result.coeffs[variable] == 8
        assert result.constant == 12

    def test_expression_less_than_or_equal(self, variable):
        expr = LinearExpression({variable: 2}, 3)
        constraint = expr <= 10
        assert isinstance(constraint, LinearConstraint)
        assert constraint.sense == "<="
        assert constraint.lhs.coeffs[variable] == 2
        assert constraint.rhs.constant == 10

        constraint2 = expr == 7
        assert isinstance(constraint2, LinearConstraint)
        assert constraint2.sense == "="
        assert constraint2.lhs.coeffs[variable] == 2
        assert constraint2.rhs.constant == 7

        constraint3 = expr >= 5
        assert isinstance(constraint3, LinearConstraint)
        assert constraint3.sense == ">="
        assert constraint3.lhs.coeffs[variable] == 2
        assert constraint3.lhs.constant == 3
        assert constraint3.rhs.constant == 5

    def test_variable_array_parameter_linear_expression(
        self, variable_array, param_array1
    ):
        expr = variable_array[0] * param_array1[0] + variable_array[1] * param_array1[1]
        assert isinstance(expr, LinearExpression)
        assert expr.coeffs[variable_array[0]] == param_array1[0]
        assert expr.coeffs[variable_array[1]] == param_array1[1]
        assert expr.constant == 0


class TestLinearConstraint:
    def test_constraint_creation(self, variable):
        expr = LinearExpression({variable: 2}, 3)
        constraint = LinearConstraint(expr, 10, "<=")
        assert constraint.lhs.coeffs[variable] == 2
        assert constraint.lhs.constant == 3
        assert constraint.rhs.constant == 10
        assert constraint.sense == "<="

    def test_symbolic_constraint_creation(self, variable, param1):
        constraint = 2 * variable + param1 <= 15
        assert isinstance(constraint, LinearConstraint)
        assert constraint.sense == "<="

        # Check the structure before evaluation
        assert variable in constraint.lhs.coeffs
        assert constraint.lhs.coeffs[variable] == 2
        assert isinstance(constraint.lhs.constant, ParameterExpression)
        assert constraint.lhs.constant.params == [param1]

    def test_batched_variable_parameter_constraint(
        self, variable_batched, variable_batched2, batched_param1, batched_param2
    ):
        constraint = (
            variable_batched[:, 1] @ batched_param1[:, 1] + variable_batched2
            <= batched_param2
        )
        assert isinstance(constraint, LinearConstraint)
        assert constraint.sense == "<="

        assert variable_batched[0, 1] in constraint.lhs.coeffs
        assert constraint.lhs.coeffs[variable_batched[0, 1]] == batched_param1[0, 1]
        assert constraint.lhs.coeffs[variable_batched[1, 1]] == batched_param1[1, 1]
        assert constraint.lhs.coeffs[variable_batched2] == 1
        assert constraint.rhs.constant == batched_param2

        constraint_i = (
            variable_batched[0, 1] * batched_param1[0, 1] + variable_batched2[0, 1]
            <= batched_param2[0, 1] + 2
        )
        assert isinstance(constraint_i, LinearConstraint)
        assert isinstance(constraint_i.rhs.constant, ParameterExpression)
        assert constraint_i.sense == "<="
        batched_param1.set_value(np.array([[1, 2, 3], [4, 5, 6]]))
        batched_param2.set_value(np.array([[10, 20, 30], [40, 50, 60]]))
        assert constraint_i.lhs.coeffs[variable_batched[0, 1]] == 2
        assert constraint_i.rhs.constant.evaluate() == 22


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=symbolic_lp.lib"])
