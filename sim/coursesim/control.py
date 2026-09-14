"""Provided finite-horizon LQR and model-predictive controllers.

Students configure costs, bounds, and horizons through this stable interface;
the underlying solvers are provided course infrastructure.

Contents:

  riccati_lqr(A, B, Q, R, Qf, N)   finite-horizon discrete LQR recursion
  QRCost(Q, R, Qf)                 quadratic tracking cost
  MPCController(dynamics_fn, cost, u_bounds, horizon)
      .solve(x0, ref_traj) -> MPCResult(u0, predicted_traj,
                                        solve_time_s, converged)
  tracking_cost(states, ref)       the course's one tracking-error number

Like the rest of coursesim, everything here is pure numpy, deterministic,
and RNG-free: same inputs, same outputs bit-for-bit. The single exception
is `MPCResult.solve_time_s`, which is wall-clock by nature; autograders
never assert on it (course MPC interface, Determinism).

Units are SI throughout (meters, radians, seconds), matching the
simulator. Angles are treated as plain reals by the quadratic cost and the
finite-difference linearization, so reference trajectories must keep angle
components away from the +-pi wrap seam.

This module is not re-exported from `coursesim/__init__` (the F1-frozen
surface); import it explicitly, like viz:

    from coursesim.control import MPCController, QRCost, riccati_lqr
"""
import time
from dataclasses import dataclass

import numpy as np

__all__ = ["riccati_lqr", "QRCost", "MPCResult", "MPCController",
           "tracking_cost"]


def riccati_lqr(A, B, Q, R, Qf, N):
    """Finite-horizon discrete-time LQR via the backward Riccati recursion.

    Solves min sum_{k=0}^{N-1} (x_k^T Q x_k + u_k^T R u_k) + x_N^T Qf x_N
    subject to x_{k+1} = A x_k + B u_k. The optimal policy is the
    time-varying linear feedback u_k = -Ks[k] @ x_k.

    Arguments (all numpy arrays, any consistent units):
      A   (n, n) discrete-time state matrix
      B   (n, m) discrete-time input matrix
      Q   (n, n) stage state cost, symmetric PSD
      R   (m, m) stage control cost, symmetric PD
      Qf  (n, n) terminal state cost, symmetric PSD
      N   int, horizon length in steps

    Returns (Ks, Ps):
      Ks  list of N gain matrices, Ks[k] shape (m, n); u_k = -Ks[k] @ x_k
      Ps  list of N+1 cost-to-go matrices, Ps[k] shape (n, n); the optimal
          cost from state x at step k is x^T Ps[k] x, and Ps[N] == Qf.

    The recursion, executed for k = N-1 down to 0 with P = Ps[k+1]:
      Ks[k] = (R + B^T P B)^{-1} B^T P A
      Ps[k] = Q + A^T P A - A^T P B Ks[k]
    Deterministic, pure numpy; a scalar system is the (1, 1) case and
    matches the hand-executed Riccati step from lecture digit for digit.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    Qf = np.asarray(Qf, dtype=float)
    N = int(N)

    Ks = [None] * N
    Ps = [None] * (N + 1)
    P = Qf.copy()
    Ps[N] = Qf.copy()
    for k in range(N - 1, -1, -1):
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)
        P = Q + A.T @ P @ A - (BtP @ A).T @ K
        P = 0.5 * (P + P.T)  # keep symmetric against float drift
        Ks[k] = K
        Ps[k] = P
    return Ks, Ps


class QRCost:
    """Quadratic tracking cost QRCost(Q, R, Qf), per course MPC interface v0.1.

    Stage cost     0.5 * (x - x_ref)^T Q (x - x_ref) + 0.5 * u^T R u
    Terminal cost  0.5 * (x - x_ref)^T Qf (x - x_ref)

    Q, Qf are (n, n) symmetric PSD; R is (m, m) symmetric PD. All are
    coerced to float64 numpy arrays; scalars/1-D input are treated as
    (1, 1) matrices. Students may tune these matrices. The
    0.5 factor is a convention only: scaling every matrix by the same
    constant leaves the optimal controls and gains unchanged.
    """

    def __init__(self, Q, R, Qf):
        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        self.Qf = np.atleast_2d(np.asarray(Qf, dtype=float))
        for name, M in (("Q", self.Q), ("R", self.R), ("Qf", self.Qf)):
            if M.ndim != 2 or M.shape[0] != M.shape[1]:
                raise ValueError(f"{name} must be square, got {M.shape}")

    def stage(self, x, u, x_ref):
        """Stage cost at one step: state tracking plus control effort."""
        dx = x - x_ref
        return 0.5 * float(dx @ self.Q @ dx + u @ self.R @ u)

    def terminal(self, x, x_ref):
        """Terminal cost at the end of the horizon."""
        dx = x - x_ref
        return 0.5 * float(dx @ self.Qf @ dx)


@dataclass
class MPCResult:
    """One solve's output, per course MPC interface v0.1.

    u0              (m,) first control of the optimized sequence; the only
                    field a receding-horizon loop needs.
    predicted_traj  (H+1, n) the solver's predicted states,
                    predicted_traj[0] == x0. For plotting and debugging.
    solve_time_s    wall-clock seconds for this solve, measured inside the
                    controller (perf_counter around the optimization only).
                    The sole nondeterministic field; never asserted on.
    converged       False means the iteration limit hit before tolerance;
                    u0 is still the best iterate and still usable.
    """
    u0: np.ndarray
    predicted_traj: np.ndarray
    solve_time_s: float
    converged: bool


class MPCController:
    """Receding-horizon trajectory optimizer, per course MPC interface v0.1.

        ctrl = MPCController(dynamics_fn, cost, u_bounds, horizon)
        result = ctrl.solve(x0, ref_traj)   # -> MPCResult

    Arguments:
      dynamics_fn  the pure step function from coursesim.dynamics with
                   params and dt bound: dynamics_fn(state, control) ->
                   next state. The controller predicts with the same
                   function the simulator integrates; no second model.
      cost         a QRCost(Q, R, Qf).
      u_bounds     (u_min, u_max) arrays of shape (m,); elementwise clamps
                   applied to every control in the forward rollout, so the
                   returned u0 always respects them.
      horizon      int H, number of steps. The HW4 sweep variable.

    Backend: iterative LQR (iLQR). Each iteration linearizes the dynamics
    along the current trajectory (central finite differences), runs a
    Riccati-style backward pass for a feedback gain K and feedforward step
    d per stage (with Levenberg-style regularization on the control
    Hessian), then line-searches a closed-loop forward rollout
    u <- clip(u + alpha d + K (x_new - x)). Convergence means the
    predicted cost improvement fell below `tol` relative to the current
    cost. On a linear system with this quadratic cost, one iteration
    reproduces the finite-horizon LQR policy exactly.

    Deterministic: no RNG anywhere, fixed iteration and line-search
    schedules, so same construction plus same solve inputs give the same
    u0 and predicted_traj bit-for-bit. Only solve_time_s varies.

    The keyword-only knobs (max_iters, tol) are staff/debug dials with
    course defaults; the student-facing surface is the four positional
    arguments above, frozen at v0.1.
    """

    #: line-search step sizes, largest first (halving schedule)
    _ALPHAS = tuple(0.5 ** i for i in range(11))
    _REG_INIT = 1e-8   # initial Levenberg regularization on Quu
    _REG_MAX = 1e8     # give up escalating beyond this
    _FD_EPS = 1e-6     # central-difference step for the linearization

    def __init__(self, dynamics_fn, cost, u_bounds, horizon, *,
                 max_iters=50, tol=1e-6):
        self.dynamics_fn = dynamics_fn
        self.cost = cost
        u_min, u_max = u_bounds
        self.u_min = np.asarray(u_min, dtype=float).reshape(-1)
        self.u_max = np.asarray(u_max, dtype=float).reshape(-1)
        if self.u_min.shape != self.u_max.shape:
            raise ValueError("u_min and u_max must have the same shape")
        if np.any(self.u_min > self.u_max):
            raise ValueError("u_bounds must satisfy u_min <= u_max")
        self.horizon = int(horizon)
        if self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        #: (H, m) optimized control sequence from the most recent solve.
        #: Debugging aid only. Applying it open loop instead of re-solving
        #: each step is exactly the bug course MPC interface warns about.
        self.last_u_traj = None

    # ------------------------------------------------------------- solve

    def solve(self, x0, ref_traj):
        """Optimize a length-H control sequence from x0 tracking ref_traj.

        x0        (n,) current state estimate.
        ref_traj  (H+1, n) reference states for this window; the CALLER
                  pads by repeating the final state if its reference runs
                  short near the goal (course MPC interface, Semantics).

        Returns MPCResult. Deterministic except solve_time_s.
        """
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        ref = np.asarray(ref_traj, dtype=float)
        H, n, m = self.horizon, x0.size, self.u_min.size
        if ref.shape != (H + 1, n):
            raise ValueError(
                f"ref_traj must have shape ({H + 1}, {n}) = (H+1, n), got "
                f"{ref.shape}; pad a short reference by repeating its "
                "final state")

        t_start = time.perf_counter()
        u = np.clip(np.zeros((H, m)), self.u_min, self.u_max)
        x, u, J = self._rollout(x0, u, None, None, 0.0, ref)
        rho = self._REG_INIT
        converged = False

        for _ in range(self.max_iters):
            fx, fu = self._linearize(x, u)
            # Backward pass, escalating regularization until Quu is PD.
            while True:
                bp = self._backward(x, u, ref, fx, fu, rho)
                if bp is not None:
                    break
                rho *= 10.0
                if rho > self._REG_MAX:
                    break
            if bp is None:
                break  # hopelessly ill-conditioned; return best iterate
            K, d, dJ1, dJ2 = bp

            # Converged: the predicted improvement is negligible.
            if abs(dJ1) < self.tol * (1.0 + abs(J)):
                converged = True
                break

            # Line search on the closed-loop rollout.
            improved = False
            for alpha in self._ALPHAS:
                x_new, u_new, J_new = self._rollout(x0, u, x, (K, d),
                                                    alpha, ref)
                if J_new < J - 1e-12:
                    delta = J - J_new
                    x, u, J = x_new, u_new, J_new
                    improved = True
                    rho = max(rho * 0.1, self._REG_INIT)
                    break
            if improved:
                if delta < self.tol * (1.0 + abs(J)):
                    converged = True
                    break
            else:
                expected = -(dJ1 + 0.5 * dJ2)
                if abs(expected) < self.tol * (1.0 + abs(J)):
                    converged = True
                    break
                rho *= 10.0
                if rho > self._REG_MAX:
                    break  # stalled (typically hard against the clamps)

        solve_time_s = time.perf_counter() - t_start
        self.last_u_traj = u.copy()
        return MPCResult(u0=u[0].copy(), predicted_traj=x.copy(),
                         solve_time_s=solve_time_s, converged=converged)

    # ---------------------------------------------------------- internals

    def _rollout(self, x0, u_bar, x_bar, Kd, alpha, ref):
        """Forward rollout with elementwise control clamping.

        With Kd=None: open-loop rollout of u_bar. Otherwise closed-loop
        u_k = clip(u_bar[k] + alpha d[k] + K[k] (x_k - x_bar[k])).
        Returns (x (H+1, n), u (H, m), total cost J).
        """
        H, n, m = self.horizon, x0.size, self.u_min.size
        x = np.empty((H + 1, n))
        u = np.empty((H, m))
        x[0] = x0
        J = 0.0
        for k in range(H):
            if Kd is None:
                uk = u_bar[k]
            else:
                K, d = Kd
                uk = u_bar[k] + alpha * d[k] + K[k] @ (x[k] - x_bar[k])
            uk = np.clip(uk, self.u_min, self.u_max)
            u[k] = uk
            J += self.cost.stage(x[k], uk, ref[k])
            x[k + 1] = np.asarray(self.dynamics_fn(x[k], uk),
                                  dtype=float).reshape(-1)
        J += self.cost.terminal(x[H], ref[H])
        return x, u, J

    def _linearize(self, x, u):
        """Central finite-difference Jacobians of the dynamics along the
        trajectory. Returns (fx (H, n, n), fu (H, n, m))."""
        H, n, m = self.horizon, x.shape[1], u.shape[1]
        eps = self._FD_EPS
        fx = np.empty((H, n, n))
        fu = np.empty((H, n, m))
        f = self.dynamics_fn
        for k in range(H):
            xk, uk = x[k], u[k]
            for i in range(n):
                dx = np.zeros(n)
                dx[i] = eps
                fx[k, :, i] = (np.asarray(f(xk + dx, uk), dtype=float)
                               - np.asarray(f(xk - dx, uk), dtype=float)) \
                    / (2.0 * eps)
            for j in range(m):
                du = np.zeros(m)
                du[j] = eps
                fu[k, :, j] = (np.asarray(f(xk, uk + du), dtype=float)
                               - np.asarray(f(xk, uk - du), dtype=float)) \
                    / (2.0 * eps)
        return fx, fu

    def _backward(self, x, u, ref, fx, fu, rho):
        """Riccati-style backward pass along the current trajectory.

        Returns (K (H, m, n), d (H, m), dJ1, dJ2) where dJ1 = sum d^T q_u
        and dJ2 = sum d^T Quu d feed the expected-improvement estimate, or
        None if any regularized Quu fails its Cholesky (caller escalates
        rho and retries).
        """
        H, n, m = self.horizon, x.shape[1], u.shape[1]
        Q, R, Qf = self.cost.Q, self.cost.R, self.cost.Qf
        K = np.empty((H, m, n))
        d = np.empty((H, m))
        P = Qf.copy()
        p = Qf @ (x[H] - ref[H])
        dJ1 = 0.0
        dJ2 = 0.0
        for k in range(H - 1, -1, -1):
            A, B = fx[k], fu[k]
            qx = Q @ (x[k] - ref[k]) + A.T @ p
            qu = R @ u[k] + B.T @ p
            Qxx = Q + A.T @ P @ A
            Quu = R + B.T @ P @ B + rho * np.eye(m)
            Qux = B.T @ P @ A
            try:
                np.linalg.cholesky(Quu)
            except np.linalg.LinAlgError:
                return None
            Kk = -np.linalg.solve(Quu, Qux)
            dk = -np.linalg.solve(Quu, qu)
            K[k] = Kk
            d[k] = dk
            P = Qxx + Kk.T @ Quu @ Kk + Kk.T @ Qux + Qux.T @ Kk
            P = 0.5 * (P + P.T)
            p = qx + Kk.T @ Quu @ dk + Kk.T @ qu + Qux.T @ dk
            dJ1 += float(dk @ qu)
            dJ2 += float(dk @ Quu @ dk)
        return K, d, dJ1, dJ2


def tracking_cost(states, ref):
    """The course's one tracking-error number (HW4 Task 2/3, autograder).

    sum_k (x_k - xref_k)^2 + (y_k - yref_k)^2 over the EXECUTED states,
    i.e. the squared planar position error summed over the states the
    simulator actually visited, index-aligned with the reference. Provided
    so every student and the autograder compute the same number.

    states  (T, n) executed states, states[:, 0:2] = (x, y) in meters
    ref     (T_ref, n) reference states, same layout; compared over the
            first min(T, T_ref) rows.

    Returns a float (meters squared).
    """
    states = np.asarray(states, dtype=float)
    ref = np.asarray(ref, dtype=float)
    T = min(states.shape[0], ref.shape[0])
    d = states[:T, :2] - ref[:T, :2]
    return float(np.sum(d * d))
