from symbolic_lp import Model
import pytest
import numpy as np
import torch
class TestModel:

    @pytest.fixture(autouse=True)
    def model(self):
        return Model()
        
    def test_model_creation(self, model):
        assert model.variables == []
        assert model.constraints == []

    def test_add_variables(self, model):
        x = model.add_var("x", 1)
        y = model.add_var("y", 1)
        z = model.add_var("z", 1)
        assert x.index == 0
        assert y.index == 1        
        assert z.index == 2
        assert len(model.variables) == 3

    def test_add_constraint(self, model):
        x = model.add_var("x", 1)
        constraint = x <= 5
        model.add_constraint(constraint)
        assert len(model.constraints) == 1

    def test_move_to_lhs(self, model):
        x = model.add_var("x", 2)
        p = model.add_param("p", 2)
        constraint = x[0] + p[0] <= x[1] + p[1] + 3
        lhs = model._move_to_lhs(constraint)
        expected_lhs = x[0] - x[1] + p[0] - p[1] - 3
        assert lhs.coeffs == expected_lhs.coeffs
        assert lhs.constant == expected_lhs.constant
        
    def test_eval_to_float(self, model):
        p = model.add_param("p", 1)
        p.value = 7
        expr = 3 * p + 5
        val = model._eval_to_float(expr)
        assert val == 26  # 3*7 + 5 = 26

    def test_get_vector_x(self, model):
        x = model.add_var("x", 1)
        y = model.add_var("y", 1)
        z = model.add_var("z", 1)
        vec = model.get_vector_x()
        assert list(vec) == ["x", "y", "z"]

    def test_build_matrices_simple_le_constraint(self, model):
        """Test: x <= 5 becomes row [1], b [5]"""

        x = model.add_var("x", 1)
        model.add_constraint(x <= 5)
        A, b = model.get_matrices()
        assert A.shape == (1, 1)
        assert np.allclose(A, [[1]])
        assert np.allclose(b, [5])

    def test_build_matrices_simple_ge_constraint(self, model):
        """Test: x >= 3 becomes -x <= -3, so row [-1], b [-3]"""

        x = model.add_var("x", 1)
        model.add_constraint(x >= 3)
        A, b = model.get_matrices()
        assert A.shape == (1, 1)
        assert np.allclose(A, [[-1]])
        assert np.allclose(b, [-3])

    def test_build_matrices_equality_constraint(self, model):
        """Test: x = 5 becomes two constraints: x <= 5 and -x <= -5"""

        x = model.add_var("x", 1)
        model.add_constraint(x == 5)
        A, b = model.get_matrices()
        assert A.shape == (2, 1)
        assert np.allclose(A, [[1], [-1]])
        assert np.allclose(b, [5, -5])

    def test_build_matrices_multiple_variables(self, model):
        """Test: 2x + 3y <= 10"""

        x = model.add_var("x")
        y = model.add_var("y")
        model.add_constraint(2*x + 3*y <= 10)
        A, b = model.get_matrices()
        assert A.shape == (1, 2)
        assert np.allclose(A, [[2, 3]])
        assert np.allclose(b, [10])

    def test_build_matrices_with_constant_in_lhs(self, model):
        """Test: 2x + 5 <= 10 becomes 2x <= 5"""

        x = model.add_var("x")
        model.add_constraint(2*x + 5 <= 10)
        A, b = model.get_matrices()
        assert A.shape == (1, 1)
        assert np.allclose(A, [[2]])
        assert np.allclose(b, [5])

    def test_build_matrices_complex_expression(self, model):
        """Test: 3x - 2y + 7 <= 15 becomes 3x - 2y <= 8"""

        x = model.add_var("x")
        y = model.add_var("y")
        model.add_constraint(3*x - 2*y + 7 <= 15)
        A, b = model.get_matrices()
        assert A.shape == (1, 2)
        assert np.allclose(A, [[3, -2]])
        assert np.allclose(b, [8])

    def test_build_matrices_multiple_constraints(self, model):
        """Test multiple constraints"""

        x = model.add_var("x")
        y = model.add_var("y")
        model.add_constraint(x + y <= 10)
        model.add_constraint(x >= 0)
        model.add_constraint(y >= 0)
        A, b = model.get_matrices()
        assert A.shape == (3, 2)
        # x + y <= 10: [1, 1]
        # x >= 0 → -x <= 0: [-1, 0]
        # y >= 0 → -y <= 0: [0, -1]
        expected_A = [[1, 1], [-1, 0], [0, -1]]
        expected_b = [10, 0, 0]
        assert np.allclose(A, expected_A)
        assert np.allclose(b, expected_b)

    def test_build_matrices_rhs_with_expression(self, model):
        """Test: x <= 2y + 5 becomes x - 2y <= 5"""

        x = model.add_var("x")
        y = model.add_var("y")
        model.add_constraint(x <= 2*y + 5)
        A, b = model.get_matrices()
        assert A.shape == (1, 2)
        assert np.allclose(A, [[1, -2]])
        assert np.allclose(b, [5])

    def test_build_matrices_expression_both_sides(self, model):
        """Test: 2x + 3 <= 4y - 7 becomes 2x - 4y <= -10"""

        x = model.add_var("x")
        y = model.add_var("y")
        model.add_constraint(2*x + 3 <= 4*y - 7)
        A, b = model.get_matrices()
        assert A.shape == (1, 2)
        assert np.allclose(A, [[2, -4]])
        assert np.allclose(b, [-10])
    
    def test_build_matrices_with_parametrized_constraint(self, model):
        """
        Build model with parameterized constraint, then build matrices with specific parameter values."""
        x = model.add_var("x")
        p = model.add_param("p", 1)
        model.add_constraint(x + p <= 10)
        A, b = model.get_matrices({"p": 4})
        assert A.shape == (1, 1)
        assert np.allclose(A, [[1]])
        assert np.allclose(b, [10 - p.value])
    
    def test_build_matrices_from_complex_parameterized_constraint(self, model):
        """
        Test building matrices from a constraint involving parameters on both sides.
        e.g., x + p1 <= p2 + 5
        """
        x1 = model.add_var("x1")
        x2 = model.add_var("x2")
        p1 = model.add_param("p1")
        p2 = model.add_param("p2", 1)
        p3 = model.add_param("p3")
        model.add_constraint(x1*9 + p1*x2 + p2 - p3<=  + 5)
        A, b = model.get_matrices({"p1": 3, "p2": 8, "p3": 2})
        assert A.shape == (1, 2)
        assert np.allclose(A, [[9, p1.value]])
        assert np.allclose(b, [5 + p3.value - p2.value])

    
    def test_set_param_values(self, model):
        p1 = model.add_param("p1", 3)
        p2 = model.add_param("p2", 2)
        model._set_param_values({"p1": [1, 2, 3], "p2": [4, 5]})
        print(model)
        assert p1 == [1, 2, 3], p1
        assert model.parameters[0].value == 1
        assert p1[1].value == 2
        assert p1[2].value == 3
        assert p2[0].value == 4
        assert p2[1].value == 5

    def test_check_param_values(self, model):
        """ Test that ValueError is raised if any parameter has no value set."""
        p1 = model.add_param("p1", 3)
        p2 = model.add_param("p2", 2)
        try:
            model._check_param_values()
            pytest.fail("Expected ValueError not raised")
        except ValueError:
            pass
        model._set_param_values({"p1": [1, 2, 3], "p2": [4, 5]})
        print(model)
        try :
            model._check_param_values()  # Should not raise
        except ValueError:
            pytest.fail("Unexpected ValueError raised")

    def test_test_fill_row_from_expr(self, model):
        x = model.add_var("x", 2)
        y = model.add_var("y", 1)
        expr = 3*x[0] - 2*x[1] + 4*y + 5
        row = np.zeros(len(model.variables))
        model._fill_row_from_expr(row, expr)
        assert np.allclose(row, [3, -2, 4])

    def test_integration_lp_arrays(self, model):
        x = model.add_var("x", 4)
        p = model.add_param("p", 4)
        model.add_constraint(x[0] + 2*x[1] + p[0] <= 10)
        model.add_constraint(3*x[2] - p[2]*x[3] >= p[1] + 5)
        A, b = model.get_matrices({"p": [1, 4, 9, 0]})
        assert p[2].value == 9
        # assert A.shape == (2, 4), f"Expected shape (2,4), got {A.shape}"
        assert np.allclose(A, [[1, 2, 0, 0], [0, 0, -3, 9]]), f"Unexpected A: {A}"
        assert np.allclose(b, [10 - 1, -5 - 4]), f"Unexpected b: {b}"

    
    def test_batched_lp_simple(self, model):
        x = model.add_var("x", 2)
        p = model.add_param("p", 2)
        model.add_constraint(p[0]*x[0] + 2*x[1] <= 8)
        model.add_constraint(3*x[0] - p[1]*x[1] >= 4)
        p_value = torch.tensor([[2, 3], [1, 2], [4, 5]])
        A, b = model.get_matrices({"p": p_value})
        assert p_value.shape == (3, 2)
        assert A.shape == (3, 2, 2)
        assert np.allclose(A[0], [[2, 2], [-3, 3]])
        assert np.allclose(A[1], [[1, 2], [-3, 2]])
        assert np.allclose(A[2], [[4, 2], [-3, 5]])
        assert np.allclose(b, [[8, -4], [8, -4], [8, -4]])
    
    def test_battery_storage_like_problem(self, model):
        """Test a problem similar to battery storage constraints"""
        u_c = model.add_var("u_c", 10)  # charge
        u_d = model.add_var("u_d", 10)  # discharge
        s = model.add_var("s", 11)      # state
        g = model.add_var("g", 10)      # grid

        netload = model.add_param("netload", 10)
        
        model.add_constraint(s[0] == 0.5)  # Initial state

        # demand balance
        for i in range(10):
            model.add_constraint(g[i] + u_d[i] - u_c[i] == netload[i])
            model.add_constraint(g[i] >= 0)
            model.add_constraint(u_c[i] >= 0)
            model.add_constraint(u_d[i] >= 0)
            model.add_constraint(u_c[i] <= 10)
            model.add_constraint(u_d[i] <= 10)
            model.add_constraint(s[i] >= 0)
            model.add_constraint(s[i] <= 1)
            model.add_constraint(s[i+1] == s[i] + (0.9*u_c[i] - 1.0*u_d[i])*0.1)
        
        A, b = model.get_matrices({"netload": [2.0]*10})
        nb_constraints = 7*10 + 2*2*10 + 2  # 7 <= per time step, 2 == per timstep, 2 for initial and final state
        assert A.shape == (nb_constraints, len(model.variables))
        assert b.shape == (nb_constraints,)
        print("A : ", A)



    def test_microgrid_example(self, model):
        horizon_size = 5
        timestep_in_hour = 0.25
        rho_c = 0.9
        rho_d = 0.9
        S_bar = 1.0
        G_bar = 5.0
        U_bar = 2.0
        U_underbar = 2.0


        # Define the variables
        u_c = model.add_var("u_c", horizon_size)
        u_d = model.add_var("u_d", horizon_size)
        s = model.add_var("s", horizon_size + 1)
        g_b = model.add_var("g_b", horizon_size)
        g_s = model.add_var("g_s", horizon_size)
        g_slack = model.add_var("g_slack", horizon_size)

        # Define the parameters
        w = model.add_param("w", horizon_size)
        soc0 = model.add_param("soc0")

        # Define the constraints
        model.add_constraint(s[0] == soc0)
        for i in range(1, horizon_size + 1):
            model.add_constraint(s[i] == s[i-1] + (timestep_in_hour * (rho_c * u_c[i-1] - (1 / rho_d) * u_d[i-1])) / S_bar)
            model.add_constraint(s[i] <= 1)
            model.add_constraint(s[i] >= 0)

        for i in range(horizon_size):
            model.add_constraint(g_b[i] - g_s[i] == w[i] + u_c[i] - u_d[i])
            model.add_constraint(g_b[i] - g_slack[i] <= G_bar)
            model.add_constraint(u_c[i] <= U_bar)
            model.add_constraint(u_d[i] <= U_underbar)
            model.add_constraint(g_b[i] >= 0)
            model.add_constraint(g_s[i] >= 0)
            model.add_constraint(g_slack[i] >= 0)
            model.add_constraint(u_c[i] >= 0)
            model.add_constraint(u_d[i] >= 0)

        A, b = model.get_matrices({"w": [0.5]*horizon_size, "soc0": 0.2})
        expected_num_constraints = (
            14*horizon_size + 
            2  # s[0] == soc0
        )
        assert A.shape == (expected_num_constraints, len(model.variables))

        
    def test_batched_microgrid(self):
        microgrid_model = Model()
        u_c = microgrid_model.add_var("u_c", 3)  # Charge power over 3 time steps
        u_d = microgrid_model.add_var("u_d", 3)  # Discharge power over 3 time steps
        s = microgrid_model.add_var("s", 4)      # State of charge over 4 time steps
        g = microgrid_model.add_var("g", 3)  # Grid
        g_DC = microgrid_model.add_var("g_DC", 3)  # Demand charge slack variable over 3 time steps

        soc0 = microgrid_model.add_param("soc0", 1)  # Initial state of charge
        rho_c = microgrid_model.add_param("rho_c", 1)  # Charge efficiency
        rho_d = microgrid_model.add_param("rho_d", 1)  # Discharge efficiency
        U_bar = microgrid_model.add_param("U_bar", 1)  # Max charge power
        U_underbar = microgrid_model.add_param("U_underbar", 1)  # Max discharge power
        S_bar = microgrid_model.add_param("S_bar", 1)  # Battery capacity
        G_bar = microgrid_model.add_param("G_bar", 1)  # Max grid import power
        M = microgrid_model.add_param("M", 1)  # Demand charge price
        w = microgrid_model.add_param("w", 3)  # Net demand over 4 time steps

        # Constraints
        microgrid_model.add_constraint(s[0] == soc0)
        for t in range(3):
            microgrid_model.add_constraint(
                s[t+1] == s[t] + (rho_c * u_c[t] - (1 / rho_d) * u_d[t]) / S_bar
            )
            microgrid_model.add_constraint(
                g[t]== w[t] + u_c[t] - u_d[t]
            )
            microgrid_model.add_constraint(
                g[t] - g_DC[t] <= G_bar
            )
            microgrid_model.add_constraint(
                u_c[t] <= U_bar
            )
            microgrid_model.add_constraint(
                u_d[t] <= U_underbar
            )
            microgrid_model.add_constraint(
                s[t+1] <= 1
            )
            microgrid_model.add_constraint(
                s[t+1] >= 0
            )

        A, b = microgrid_model.get_matrices(param_values={
            "soc0": torch.ones(10,1)*0.5,
            "rho_c": torch.ones(10,1)*0.9,
            "rho_d": torch.ones(10,1)*0.9,
            "U_bar": torch.ones(10,1)*4.0,
            "U_underbar": torch.ones(10,1)*5.0,
            "S_bar": torch.ones(10,1)*20.0,
            "G_bar": torch.ones(10,1)*10.0,
            "M": torch.ones(10,1)*2.0,
            "w": torch.randn(10,3)*3.0
        })
        

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=symbolic_lp.main"])