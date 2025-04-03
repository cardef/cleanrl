import pytest
import torch
import numpy as np

# Check if torch.func is available, otherwise skip tests
try:
    from torch.func import grad, vmap, jacrev, hessian

    TORCH_FUNC_AVAILABLE = True
except ImportError:
    try:
        from functorch import grad, vmap, jacrev, hessian

        TORCH_FUNC_AVAILABLE = True
    except ImportError:
        TORCH_FUNC_AVAILABLE = False

# Import the function to test from the main script
# Assuming the script is runnable and calculate_a_star_quad_approx is accessible
# Adjust the import path if necessary based on your project structure.
# Running pytest from the root directory should make this import work.
from cleanrl.hjb import calculate_a_star_quad_approx


# --- Configuration ---
OBS_DIM = 3
ACTION_DIM = 2
BATCH_SIZE = 5
DEVICE = torch.device("cpu")  # Tests run on CPU for simplicity
ATOL = 1e-6  # Absolute tolerance for comparisons

# --- 1. Define Analytical Dummy Functions (NumPy) ---
# Choose simple forms for easy analytical derivatives

# V(s) = 0.5 * s^T P s + k^T s + c (Quadratic Value Function)
P_np = np.diag(np.arange(1, OBS_DIM + 1)) * 0.5  # Example positive definite P
k_np = np.arange(OBS_DIM).reshape(-1, 1) * 0.1  # Example linear term k
c_np = 5.0  # Example constant term c


def V_dummy_analytical(s_np):
    """Calculates V(s) = 0.5 * s @ P @ s.T + k.T @ s.T + c for a batch."""
    s_np = s_np.reshape(s_np.shape[0], -1)  # Ensure [batch, dim]
    quad_term = 0.5 * np.diag(s_np @ P_np @ s_np.T)
    linear_term = (k_np.T @ s_np.T).squeeze()
    return quad_term + linear_term + c_np


def dVds_analytical(s_np):
    """Calculates dV/ds = s^T P + k^T for a batch."""
    s_np = s_np.reshape(s_np.shape[0], -1)  # Ensure [batch, dim]
    # Note: d/dx (0.5 x'Px) = x'P if P is symmetric. If not, 0.5 * x'(P+P')
    # P_np is diagonal, hence symmetric.
    grad_val = s_np @ P_np + k_np.T
    return grad_val  # Shape [batch, dim]


# R(s, a) = -0.5 * a^T Q a - 0.1 * s^T M s - d^T a (Quadratic Reward)
Q_np = np.diag(np.arange(1, ACTION_DIM + 1)) * 1.0  # Example positive definite Q
M_np = np.diag(np.arange(1, OBS_DIM + 1)) * 0.1  # Example state cost M
d_np = np.arange(ACTION_DIM).reshape(-1, 1) * 0.2  # Example linear action cost d


def R_dummy_analytical(s_np, a_np):
    """Calculates R(s,a) = -0.5*aQa -0.1*sMs - da for a batch."""
    s_np = s_np.reshape(s_np.shape[0], -1)
    a_np = a_np.reshape(a_np.shape[0], -1)
    quad_a_term = -0.5 * np.diag(a_np @ Q_np @ a_np.T)
    quad_s_term = -0.1 * np.diag(s_np @ M_np @ s_np.T)
    linear_a_term = (d_np.T @ a_np.T).squeeze()
    return quad_a_term + quad_s_term - linear_a_term


def dRda_analytical(s_np, a_np):
    """Calculates dR/da = -a^T Q - d^T for a batch."""
    a_np = a_np.reshape(a_np.shape[0], -1)
    # Note: d/dx (-0.5 x'Qx) = -x'Q if Q symmetric. Q_np is diagonal.
    grad_val = -a_np @ Q_np - d_np.T
    return grad_val  # Shape [batch, action_dim]


def d2Rda2_analytical(s_np, a_np):
    """Calculates d^2R/da^2 = -Q (constant Hessian) for a batch."""
    # Since Q_np is diagonal, the Hessian is constant and diagonal.
    # Return shape [batch, action_dim, action_dim]
    hessian_val = -Q_np
    return np.tile(hessian_val, (s_np.shape[0], 1, 1))


# f(s, a) = f1(s) + f2(s) @ a (Control-Affine Dynamics)
# f1(s) = A s + b
A_np = np.eye(OBS_DIM) * 0.95 + np.random.randn(OBS_DIM, OBS_DIM) * 0.01
b_np = np.arange(OBS_DIM).reshape(-1, 1) * 0.05
# f2(s) = B (constant control matrix)
B_np = np.random.randn(OBS_DIM, ACTION_DIM) * 0.5


def f1_dummy_analytical(s_np):
    """Calculates f1(s) = A @ s.T + b for a batch."""
    s_np = s_np.reshape(s_np.shape[0], -1)
    f1_val = (A_np @ s_np.T + b_np).T  # Transpose result back to [batch, obs_dim]
    return f1_val


def f2_dummy_analytical(s_np):
    """Calculates f2(s) = B (constant) for a batch."""
    # Return shape [batch, obs_dim, action_dim]
    return np.tile(B_np, (s_np.shape[0], 1, 1))


def f_dummy_analytical(s_np, a_np):
    """Calculates f(s, a) = f1(s) + f2(s) @ a for a batch."""
    s_np = s_np.reshape(s_np.shape[0], -1)
    a_np = a_np.reshape(a_np.shape[0], -1)
    f1_val = f1_dummy_analytical(s_np)
    f2_val = f2_dummy_analytical(s_np)
    # Need batch matrix multiplication: [b, o, a] @ [b, a, 1] -> [b, o, 1] -> [b, o]
    f2a_term = (f2_val @ a_np[:, :, np.newaxis]).squeeze(-1)
    return f1_val + f2a_term


# --- 2. Define PyTorch Dummy Functions ---
P_torch = torch.tensor(P_np, dtype=torch.float32, device=DEVICE)
k_torch = torch.tensor(k_np, dtype=torch.float32, device=DEVICE)
c_torch = torch.tensor(c_np, dtype=torch.float32, device=DEVICE)
Q_torch = torch.tensor(Q_np, dtype=torch.float32, device=DEVICE)
M_torch = torch.tensor(M_np, dtype=torch.float32, device=DEVICE)
d_torch = torch.tensor(d_np, dtype=torch.float32, device=DEVICE)
A_torch = torch.tensor(A_np, dtype=torch.float32, device=DEVICE)
b_torch = torch.tensor(b_np.squeeze(), dtype=torch.float32, device=DEVICE) # Squeeze to make 1D
B_torch = torch.tensor(B_np, dtype=torch.float32, device=DEVICE)


# Wrapper for V to handle single input for grad
def V_dummy_torch_scalar(s_tensor):
    """Input: 1D tensor s"""
    s_tensor = s_tensor.float() # Ensure float
    quad_term = 0.5 * s_tensor @ P_torch @ s_tensor
    linear_term = k_torch.T @ s_tensor
    return (quad_term + linear_term + c_torch).squeeze()


# Wrapper for R to handle single inputs for grad/hessian
def R_dummy_torch_scalar(s_tensor, a_tensor):
    """Input: 1D tensors s, a"""
    s_tensor = s_tensor.float()
    a_tensor = a_tensor.float()
    quad_a_term = -0.5 * a_tensor @ Q_torch @ a_tensor
    quad_s_term = -0.1 * s_tensor @ M_torch @ s_tensor
    linear_a_term = d_torch.T @ a_tensor
    return (quad_a_term + quad_s_term - linear_a_term).squeeze()


# Wrapper for f to handle single inputs for jacrev
# Note: torch.func jacrev expects func(y, x) for dy/dx
# Our ODE func is f(t, s, a). We need df/da.
# We create a wrapper f_wrapper(s, a) assuming t=0
def f_dummy_torch_wrapper(s_tensor, a_tensor):
    """Input: 1D tensors s, a. Output: 1D tensor f(s,a)"""
    s_tensor = s_tensor.float()
    a_tensor = a_tensor.float()
    # Use direct matmul: [obs,obs]@[obs] + [obs] -> [obs]
    f1_val = A_torch @ s_tensor + b_torch
    f2_val = B_torch  # Constant B [obs_dim, action_dim]
    # Use direct matmul: [obs,act]@[act] -> [obs]
    f2a_term = f2_val @ a_tensor
    return f1_val + f2a_term # [obs]


# --- 3. Test Fixtures ---
@pytest.fixture
def sample_data():
    s_np = np.random.rand(BATCH_SIZE, OBS_DIM).astype(np.float32)
    a_np = np.random.rand(BATCH_SIZE, ACTION_DIM).astype(np.float32)
    s_torch = torch.tensor(s_np, device=DEVICE)
    a_torch = torch.tensor(a_np, device=DEVICE)
    return s_np, a_np, s_torch, a_torch


# --- 4. Tests ---

# Skip all tests in this file if torch.func is not available
pytestmark = pytest.mark.skipif(
    not TORCH_FUNC_AVAILABLE, reason="torch.func or functorch not available"
)


def test_V_gradient(sample_data):
    """Compare analytical dV/ds with numerical gradient from torch.func."""
    s_np, _, s_torch, _ = sample_data

    # Numerical gradient using vmap(grad(...))
    compute_value_grad_func_dummy = grad(V_dummy_torch_scalar)
    dVds_numerical = vmap(compute_value_grad_func_dummy)(s_torch)

    # Analytical gradient
    dVds_analytical_val = dVds_analytical(s_np)

    assert np.allclose(
        dVds_numerical.cpu().numpy(), dVds_analytical_val, atol=ATOL
    ), "dV/ds mismatch"
    print("\nTest dV/ds: PASSED")


def test_R_gradient_hessian(sample_data):
    """Compare analytical dR/da and d2R/da2 with numerical results."""
    s_np, a_np, s_torch, a_torch = sample_data

    # --- Gradient dR/da ---
    compute_reward_grad_func_dummy = grad(R_dummy_torch_scalar, argnums=1)
    dRda_numerical = vmap(compute_reward_grad_func_dummy)(s_torch, a_torch)
    dRda_analytical_val = dRda_analytical(s_np, a_np)
    assert np.allclose(
        dRda_numerical.cpu().numpy(), dRda_analytical_val, atol=ATOL
    ), "dR/da mismatch"
    print("Test dR/da: PASSED")

    # --- Hessian d2R/da2 ---
    compute_reward_hessian_func_dummy = hessian(R_dummy_torch_scalar, argnums=1)
    d2Rda2_numerical = vmap(compute_reward_hessian_func_dummy)(s_torch, a_torch)
    d2Rda2_analytical_val = d2Rda2_analytical(s_np, a_np)
    assert np.allclose(
        d2Rda2_numerical.cpu().numpy(), d2Rda2_analytical_val, atol=ATOL
    ), "d2R/da2 mismatch"
    print("Test d2R/da2: PASSED")


def test_f_jacobian(sample_data):
    """Compare analytical df/da (f2) with numerical Jacobian from torch.func."""
    s_np, a_np, s_torch, a_torch = sample_data

    # Numerical Jacobian using vmap(jacrev(...))
    # jacrev(f, argnums=1) computes df/da for f(s, a)
    # vmap applies this over the batch dimension.
    dfda_numerical = vmap(jacrev(f_dummy_torch_wrapper, argnums=1))(s_torch, a_torch) # Expect [batch, obs_dim, action_dim]

    # Analytical Jacobian (f2)
    f2_analytical_val = f2_dummy_analytical(s_np) # Expect [batch, obs_dim, action_dim]

    assert np.allclose(
        dfda_numerical.cpu().numpy(), f2_analytical_val, atol=ATOL
    ), "df/da (f2) mismatch"
    print("Test df/da (f2): PASSED")


def test_a_star_calculation(sample_data):
    """Test the calculate_a_star_quad_approx function using analytical derivatives."""
    s_np, _, s_torch, _ = sample_data
    action_space_low_t = torch.tensor([-1.0] * ACTION_DIM, device=DEVICE)
    action_space_high_t = torch.tensor([1.0] * ACTION_DIM, device=DEVICE)
    hessian_reg = 1e-3

    # --- Get Analytical Derivatives for the batch ---
    dVdx_analytical_batch = torch.tensor(dVds_analytical(s_np), device=DEVICE).float()

    # Calculate c1 = -dR/da|_{a=0} and c2 = -d2R/da2|_{a=0}
    a_zeros_np = np.zeros((BATCH_SIZE, ACTION_DIM))
    c1_analytical_batch = torch.tensor(
        -dRda_analytical(s_np, a_zeros_np), device=DEVICE
    ).float()
    c2_analytical_batch = torch.tensor(
        -d2Rda2_analytical(s_np, a_zeros_np), device=DEVICE
    ).float()

    # Get f2^T
    f2_analytical_batch = torch.tensor(
        f2_dummy_analytical(s_np), device=DEVICE
    ).float()
    f2_T_analytical_batch = torch.permute(
        f2_analytical_batch, (0, 2, 1)
    )  # [batch, action_dim, obs_dim]

    # Regularize c2
    c2_reg_analytical = c2_analytical_batch + torch.eye(
        ACTION_DIM, device=DEVICE
    ) * hessian_reg

    # --- Call the function under test ---
    a_star_calculated = calculate_a_star_quad_approx(
        dVdx_analytical_batch,
        f2_T_analytical_batch,
        c1_analytical_batch,
        c2_reg_analytical,
        action_space_low_t,
        action_space_high_t,
    )

    # --- Calculate Expected a* Analytically ---
    # a* = argmin_a [ R(s,a) + <dV/ds, f(s,a)> ]
    # For our quadratic R and affine f, this is a quadratic program in 'a'.
    # R(s,a) + <dVds, f1 + f2 a>
    # = -0.5 aQa - 0.1 sMs - da + <dVds, f1> + <dVds, f2 a>
    # = -0.5 aQa + (<dVds, f2> - d)a + constant_terms(s)
    # The minimum occurs when d/da = 0:
    # -Qa + (f2^T dVds - d) = 0  => Qa = f2^T dVds - d
    # => a* = Q^{-1} (f2^T dVds - d)
    # Note: This matches the formula used in calculate_a_star_quad_approx:
    # a* = - c2_reg^{-1} * (c1 + f2^T dVdx)
    #    = - (-Q + reg*I)^{-1} * (-dR/da|_{a=0} + f2^T dVdx)
    #    =   (Q + reg*I)^{-1} * (-(-d^T) + f2^T dVdx)  (since dR/da|_{a=0} = -d^T)
    #    =   (Q + reg*I)^{-1} * (d + f2^T dVdx)
    # This seems slightly different. Let's re-derive from the HJB perspective:
    # H = R(s,a) + <V_s, f(s,a)>
    # We approximate H quadratically around a=0:
    # H(a) approx H(0) + H_a(0) a + 0.5 a^T H_aa(0) a
    # H_a = R_a + <V_s, f_a> = R_a + <V_s, f2> = R_a + V_s^T f2
    # H_aa = R_aa + <V_s, f_aa> = R_aa (since f is affine in a, f_aa=0)
    # Maximize H => set dH/da = 0 => H_a(0) + H_aa(0) a = 0
    # => a* = - H_aa(0)^{-1} H_a(0)
    # => a* = - R_aa(0)^{-1} (R_a(0) + V_s^T f2)
    # => a* = - c2^{-1} (c1 + dVdx^T f2)  <-- This matches the code's intention!
    # Where c1 = R_a(0) = -dR/da|_{a=0} from the code's perspective (neg signs differ)
    # And   c2 = R_aa(0) = -d2R/da2|_{a=0} from the code's perspective

    # Let's use the code's formula structure with our analytical parts:
    # c1_a = -dRda_analytical(s_np, a_zeros_np) # [batch, action_dim]
    # c2_a = -d2Rda2_analytical(s_np, a_zeros_np) # [batch, action_dim, action_dim]
    # f2_T_a = np.transpose(f2_dummy_analytical(s_np), (0, 2, 1)) # [b, act, obs]
    # dVdx_a = dVds_analytical(s_np) # [batch, obs_dim]

    # term1 = c1_a + (dVdx_a @ f2_T_a.transpose(0, 2, 1)) # Check dimensions carefully
    # Need batch matmul: [b, obs] @ [b, obs, act] -> [b, act]
    # f2_T_a is [b, act, obs]. dVdx_a is [b, obs].
    # We need dVdx @ f2. Let's use torch for batch matmul ease.
    # term1_torch = c1_analytical_batch + torch.bmm(dVdx_analytical_batch.unsqueeze(1), f2_analytical_batch).squeeze(1)
    term1_torch = c1_analytical_batch + torch.bmm(
        f2_T_analytical_batch, dVdx_analytical_batch.unsqueeze(-1)
    ).squeeze(-1)

    # Solve (c2_a + reg*I) a* = -term1
    c2_reg_a_torch = c2_analytical_batch + torch.eye(
        ACTION_DIM, device=DEVICE
    ) * hessian_reg
    # Use torch.linalg.solve for batch solve
    a_star_expected_unclamped = torch.linalg.solve(
        c2_reg_a_torch, -term1_torch.unsqueeze(-1)
    ).squeeze(-1)

    # Clamp expected result
    a_star_expected_clamped = torch.clamp(
        a_star_expected_unclamped, action_space_low_t, action_space_high_t
    )

    # --- Compare ---
    assert a_star_calculated is not None, "a* calculation returned None unexpectedly"
    assert torch.allclose(
        a_star_calculated, a_star_expected_clamped, atol=ATOL
    ), "a* calculation mismatch"
    print("Test a* calculation: PASSED")


def test_hjb_residual(sample_data):
    """Test the HJB residual calculation using analytical functions."""
    s_np, _, s_torch, _ = sample_data
    rho = 0.0 # Assume zero discount rate for simplicity in this analytical test

    # --- Get Analytical Derivatives/Values ---
    dVdx_a = dVds_analytical(s_np) # [batch, obs_dim]
    V_s_a = V_dummy_analytical(s_np) # [batch]

    # Calculate analytical a* (unclamped)
    # a* = Q^{-1} (f2^T dVds - d)
    # Need Q inverse
    try:
        Q_inv_np = np.linalg.inv(Q_np)
    except np.linalg.LinAlgError:
        pytest.skip("Analytical Q matrix is singular, cannot compute analytical a*.")

    f2_a = f2_dummy_analytical(s_np) # [batch, obs_dim, action_dim]
    f2_T_a = np.transpose(f2_a, (0, 2, 1)) # [batch, action_dim, obs_dim]
    d_T_a = d_np.T # [1, action_dim]

    # Batch calculation: term = (dVdx @ f2) - d^T
    # dVdx [b, obs], f2 [b, obs, act] -> bmm(dVdx.unsqueeze(1), f2) -> [b, 1, act] -> squeeze(1) -> [b, act]
    dVdx_f2 = np.einsum('bi,bio->bo', dVdx_a, f2_a) # More robust batch matmul [b, act]
    term_for_a_star = dVdx_f2 - d_T_a # [b, act]

    # Batch calculation: a* = term @ Q_inv^T (since Q is symmetric Q_inv^T = Q_inv)
    # term [b, act], Q_inv [act, act] -> term @ Q_inv -> [b, act]
    a_star_analytical_unclamped = term_for_a_star @ Q_inv_np # [b, act]

    # --- Calculate HJB components with a* ---
    R_s_astar = R_dummy_analytical(s_np, a_star_analytical_unclamped) # [batch]
    f_s_astar = f_dummy_analytical(s_np, a_star_analytical_unclamped) # [batch, obs_dim]

    # Calculate <dVds, f(s, a*)> using batch dot product (einsum)
    dVds_dot_f = np.einsum('bi,bi->b', dVdx_a, f_s_astar) # [batch]

    # --- Calculate HJB Residual ---
    H_s_astar = R_s_astar + dVds_dot_f # Hamiltonian at optimal action a* (since rho=0)

    # --- Calculate Hamiltonian at a=0 ---
    a_zeros_np = np.zeros((BATCH_SIZE, ACTION_DIM))
    R_s_zero = R_dummy_analytical(s_np, a_zeros_np) # [batch]
    f_s_zero = f_dummy_analytical(s_np, a_zeros_np) # [batch, obs_dim]
    dVds_dot_f_zero = np.einsum('bi,bi->b', dVdx_a, f_s_zero) # [batch]
    H_s_zero = R_s_zero + dVds_dot_f_zero # [batch]

    # --- Assert H(s, a*) >= H(s, 0) ---
    # a* should maximize the Hamiltonian, so H(s, a*) must be >= H(s, 0)
    assert np.all(H_s_astar >= H_s_zero - ATOL), "H(s, a*) should be >= H(s, 0)"
    print("Test HJB residual components (H(a*) >= H(0)): PASSED")


def test_a_star_none_inputs(sample_data):
    """Test calculate_a_star_quad_approx returns None if inputs are None."""
    s_np, _, s_torch, _ = sample_data
    action_space_low_t = torch.tensor([-1.0] * ACTION_DIM, device=DEVICE)
    action_space_high_t = torch.tensor([1.0] * ACTION_DIM, device=DEVICE)

    # Get valid analytical derivatives first
    dVdx_analytical_batch = torch.tensor(dVds_analytical(s_np), device=DEVICE).float()
    a_zeros_np = np.zeros((BATCH_SIZE, ACTION_DIM))
    c1_analytical_batch = torch.tensor(
        -dRda_analytical(s_np, a_zeros_np), device=DEVICE
    ).float()
    c2_analytical_batch = torch.tensor(
        -d2Rda2_analytical(s_np, a_zeros_np), device=DEVICE
    ).float()
    f2_analytical_batch = torch.tensor(
        f2_dummy_analytical(s_np), device=DEVICE
    ).float()
    f2_T_analytical_batch = torch.permute(
        f2_analytical_batch, (0, 2, 1)
    )
    c2_reg_analytical = c2_analytical_batch + torch.eye(
        ACTION_DIM, device=DEVICE
    ) * 1e-3 # Add some regularization

    # Test with None f2_transpose
    a_star_none_f2 = calculate_a_star_quad_approx(
        dVdx_analytical_batch,
        None, # f2_transpose is None
        c1_analytical_batch,
        c2_reg_analytical,
        action_space_low_t,
        action_space_high_t,
    )
    assert a_star_none_f2 is None, "Should return None when f2_transpose is None"
    print("Test a* None f2: PASSED")

    # Test with None c1
    a_star_none_c1 = calculate_a_star_quad_approx(
        dVdx_analytical_batch,
        f2_T_analytical_batch,
        None, # c1 is None
        c2_reg_analytical,
        action_space_low_t,
        action_space_high_t,
    )
    assert a_star_none_c1 is None, "Should return None when c1 is None"
    print("Test a* None c1: PASSED")

    # Test with None c2_reg
    a_star_none_c2 = calculate_a_star_quad_approx(
        dVdx_analytical_batch,
        f2_T_analytical_batch,
        c1_analytical_batch,
        None, # c2_reg is None
        action_space_low_t,
        action_space_high_t,
    )
    assert a_star_none_c2 is None, "Should return None when c2_reg is None"
    print("Test a* None c2: PASSED")


def test_a_star_singular_hessian(sample_data):
    """Test calculate_a_star uses pseudo-inverse for singular Hessian."""
    s_np, _, s_torch, _ = sample_data
    action_space_low_t = torch.tensor([-1.0] * ACTION_DIM, device=DEVICE)
    action_space_high_t = torch.tensor([1.0] * ACTION_DIM, device=DEVICE)
    hessian_reg = 0.0 # Use zero regularization to force singularity if c2 is singular

    # Get analytical derivatives
    dVdx_analytical_batch = torch.tensor(dVds_analytical(s_np), device=DEVICE).float()
    a_zeros_np = np.zeros((BATCH_SIZE, ACTION_DIM))
    c1_analytical_batch = torch.tensor(
        -dRda_analytical(s_np, a_zeros_np), device=DEVICE
    ).float()
    f2_analytical_batch = torch.tensor(
        f2_dummy_analytical(s_np), device=DEVICE
    ).float()
    f2_T_analytical_batch = torch.permute(
        f2_analytical_batch, (0, 2, 1)
    )

    # Create a singular c2 (e.g., rank deficient if ACTION_DIM > 1)
    # For ACTION_DIM=2, make one eigenvalue zero. Use -d2R/da2 = Q
    singular_Q = np.array([[1.0, 0.0], [0.0, 0.0]]) # Singular matrix
    c2_singular_np = np.tile(singular_Q, (BATCH_SIZE, 1, 1))
    c2_singular_batch = torch.tensor(c2_singular_np, device=DEVICE).float()

    # c2_reg will also be singular since hessian_reg is 0
    c2_reg_singular = c2_singular_batch + torch.eye(
        ACTION_DIM, device=DEVICE
    ) * hessian_reg

    # --- Call the function under test ---
    # Expect a warning about pinv, but should not raise an error
    print("\nTesting singular Hessian (expect LinAlgError warning -> pinv):")
    a_star_calculated = calculate_a_star_quad_approx(
        dVdx_analytical_batch,
        f2_T_analytical_batch,
        c1_analytical_batch,
        c2_reg_singular, # Pass the singular matrix
        action_space_low_t,
        action_space_high_t,
    )

    # --- Assert output is not None (pseudo-inverse should have worked) ---
    assert a_star_calculated is not None, "a* calculation returned None for singular Hessian (pinv should have run)"

    # --- Optional: Calculate Expected a* Analytically using pseudo-inverse ---
    term1_torch = c1_analytical_batch + torch.bmm(
        f2_T_analytical_batch, dVdx_analytical_batch.unsqueeze(-1)
    ).squeeze(-1)
    c2_reg_pinv_torch = torch.linalg.pinv(c2_reg_singular)
    a_star_expected_unclamped = torch.bmm(
        c2_reg_pinv_torch, -term1_torch.unsqueeze(-1)
    ).squeeze(-1)
    a_star_expected_clamped = torch.clamp(
        a_star_expected_unclamped, action_space_low_t, action_space_high_t
    )

    assert torch.allclose(
        a_star_calculated, a_star_expected_clamped, atol=ATOL
    ), "a* calculation mismatch with analytical pinv result"

    print("Test a* singular Hessian (pinv): PASSED")


def test_a_star_clamping(sample_data):
    """Test calculate_a_star clamps the result correctly."""
    s_np, _, s_torch, _ = sample_data
    # Use wider bounds temporarily to calculate unclamped a*
    wide_bounds_low = torch.tensor([-100.0] * ACTION_DIM, device=DEVICE)
    wide_bounds_high = torch.tensor([100.0] * ACTION_DIM, device=DEVICE)
    # Target bounds for clamping check
    target_bounds_low = torch.tensor([-0.5] * ACTION_DIM, device=DEVICE)
    target_bounds_high = torch.tensor([0.5] * ACTION_DIM, device=DEVICE)
    hessian_reg = 1e-3

    # Get analytical derivatives
    dVdx_analytical_batch = torch.tensor(dVds_analytical(s_np), device=DEVICE).float()
    a_zeros_np = np.zeros((BATCH_SIZE, ACTION_DIM))
    # --- Modify c1 to force a* outside target bounds ---
    # Original c1 = -dR/da|a=0 = d^T
    # a* approx = (Q+reg*I)^-1 * (d + f2^T dVdx)
    # Make c1 large positive -> term1 large positive -> a* large negative
    # Make c1 large negative -> term1 large negative -> a* large positive
    c1_force_large_neg = torch.ones_like(dVdx_analytical_batch[:, 0:ACTION_DIM]) * 1000.0
    c1_force_large_pos = torch.ones_like(dVdx_analytical_batch[:, 0:ACTION_DIM]) * -1000.0

    c2_analytical_batch = torch.tensor(
        -d2Rda2_analytical(s_np, a_zeros_np), device=DEVICE
    ).float()
    f2_analytical_batch = torch.tensor(
        f2_dummy_analytical(s_np), device=DEVICE
    ).float()
    f2_T_analytical_batch = torch.permute(
        f2_analytical_batch, (0, 2, 1)
    )
    c2_reg_analytical = c2_analytical_batch + torch.eye(
        ACTION_DIM, device=DEVICE
    ) * hessian_reg

    # --- Case 1: Force a* below lower bound ---
    a_star_neg = calculate_a_star_quad_approx(
        dVdx_analytical_batch,
        f2_T_analytical_batch,
        c1_force_large_neg, # Use modified c1
        c2_reg_analytical,
        target_bounds_low, # Target bounds
        target_bounds_high,
    )
    assert a_star_neg is not None, "a* calculation failed for negative force case"
    assert torch.all(a_star_neg >= target_bounds_low - ATOL), "a* not clamped to lower bound"
    # Check if it actually clamped (at least one element should be close to the bound)
    assert torch.any(torch.isclose(a_star_neg, target_bounds_low, atol=ATOL)), "a* seems not to have been clamped low"
    print("Test a* clamping (low): PASSED")

    # --- Case 2: Force a* above upper bound ---
    a_star_pos = calculate_a_star_quad_approx(
        dVdx_analytical_batch,
        f2_T_analytical_batch,
        c1_force_large_pos, # Use modified c1
        c2_reg_analytical,
        target_bounds_low, # Target bounds
        target_bounds_high,
    )
    assert a_star_pos is not None, "a* calculation failed for positive force case"
    assert torch.all(a_star_pos <= target_bounds_high + ATOL), "a* not clamped to upper bound"
    # Check if it actually clamped (at least one element should be close to the bound)
    assert torch.any(torch.isclose(a_star_pos, target_bounds_high, atol=ATOL)), "a* seems not to have been clamped high"
    print("Test a* clamping (high): PASSED")


def test_get_f1_direct(sample_data):
    """Test the get_f1 helper function directly."""
    s_np, _, s_torch, _ = sample_data

    # Define a dummy ODE function matching the signature expected by get_f1
    # This uses the analytical torch tensors defined earlier (A_torch, b_torch)
    def dummy_ode_func(t, s_norm_batch, a_batch):
        # f(s, a) = A s + b + B a
        # f1 is f(s, 0) = A s + b
        # Ensure s_norm_batch is [batch, obs_dim]
        s_norm_batch = s_norm_batch.reshape(BATCH_SIZE, OBS_DIM)
        # A @ s.T -> [obs, obs] @ [obs, batch] -> [obs, batch] -> .T -> [batch, obs]
        f1_val = (A_torch @ s_norm_batch.T).T + b_torch.unsqueeze(0) # Add broadcastable b
        # B @ a.T -> [obs, act] @ [act, batch] -> [obs, batch] -> .T -> [batch, obs]
        f2a_val = (B_torch @ a_batch.T).T
        return f1_val + f2a_val

    # Call the actual get_f1 function from hjb.py
    from cleanrl.hjb import get_f1
    f1_numerical = get_f1(s_torch, dummy_ode_func, ACTION_DIM)

    # Calculate analytical f1
    f1_analytical = f1_dummy_analytical(s_np)

    assert np.allclose(
        f1_numerical.cpu().numpy(), f1_analytical, atol=ATOL
    ), "get_f1 output mismatch"
    print("Test get_f1 direct: PASSED")
