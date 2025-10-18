
import math
import warnings
from functools import partial
from typing import Optional, Union, Tuple # Added Tuple

import numpy as np
# Remove pot import
# import ot as pot
import torch

# --- JAX Imports ---
import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import jax.dlpack
# Import other solvers if needed (unbalanced, partial, gromov...)
# from ott.solvers.linear import sinkhorn_unbalanced
# from ott.problems.linear import partial_linear_problem
# from ott.solvers.linear import partial_sinkhorn



class OTPlanSampler:
    """OTPlanSampler using JAX OTT for plan calculation."""

    def __init__(
        self,
        method: str,
        reg: float = 0.05, # Corresponds to epsilon in OTT Sinkhorn
        reg_m: float = 1.0, # For unbalanced
        normalize_cost: bool = False, # Less common/needed in OTT? Check docs.
        # num_threads removed (not applicable to JAX solvers)
        warn: bool = True,
        # --- OTT Specific additions ---
        ott_threshold: float = 1e-4, # Sinkhorn convergence threshold
        ott_lse_mode: bool = True, # Use log-sum-exp for stability
        ott_max_iterations: int = 100,
    ) -> None:
        """Initialize the OTPlanSampler class with JAX OTT."""
        self.method = method
        self.epsilon = reg # Map reg to epsilon
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost # May need custom handling if required
        self.warn = warn

        # --- Configure JAX OTT Solver ---
        # Note: Needs refinement based on exact methods needed
        if method == "sinkhorn":
            self.solver = sinkhorn.Sinkhorn(
                threshold=ott_threshold,
                lse_mode=ott_lse_mode,
                max_iterations=ott_max_iterations,
                # Specify cost_fn if not default squared Euclidean? Here we assume PointCloud handles it.
            )
        # elif method == "unbalanced":
        #     self.solver = sinkhorn_unbalanced.SinkhornUnbalanced(
        #         epsilon=self.epsilon, # Use epsilon here
        #         reg_m=self.reg_m,
        #         threshold=ott_threshold,
        #         lse_mode=ott_lse_mode,
        #         max_iterations=ott_max_iterations,
        #     )
        # elif method == "partial":
        #     # Requires defining mass parameter 'm' for partial problem
        #     # self.solver = partial_sinkhorn.PartialSinkhorn(...) TBD
        #     raise NotImplementedError("Partial OT with JAX OTT needs specific setup.")
        elif method == "exact":
            # No direct equivalent for pot.emd's exactness in standard JAX OTT Sinkhorn.
            # Option 1: Use Sinkhorn with very small epsilon (approximates EMD)
            warnings.warn(f"Method 'exact' requested. Approximating with JAX OTT Sinkhorn using small epsilon={reg}. Solution is not guaranteed to be exact EMD.")
            # Use the Sinkhorn solver configured above, maybe with adjusted params if needed
            self.solver = sinkhorn.Sinkhorn(
                threshold=ott_threshold, # Maybe lower threshold for 'exact'?
                lse_mode=ott_lse_mode,
                max_iterations=ott_max_iterations, # Maybe higher iterations?
            )
            # Option 2: Raise error or use an external JAX-compatible LP solver if available
            # raise NotImplementedError("Exact EMD solver not directly available in JAX OTT's core Sinkhorn. Consider approximation or external LP solver.")
        else:
            raise ValueError(f"Unknown method: {method}. JAX OTT setup needs specific solver.")

        # Store other config if needed
        self.ott_lse_mode = ott_lse_mode

    @staticmethod
    def _pytorch_to_jax(tensor: torch.Tensor) -> jax.Array:
        """Helper to convert PyTorch tensor to JAX array.
           Attempts zero-copy GPU transfer via DLPack if possible."""
        if tensor.device.type == 'cuda':
            try:
                # Attempt zero-copy transfer from PyTorch GPU tensor to JAX GPU array
                # Requires PyTorch and JAX versions with compatible DLPack support
                return jax.dlpack.from_dlpack(torch.to_dlpack(tensor))
            except Exception as e:
                warnings.warn(f"DLPack GPU->JAX failed (Error: {e}). Falling back to CPU transfer. "
                              "Check PyTorch/JAX/CUDA versions for DLPack compatibility.", UserWarning)
                # Fallback to CPU transfer if DLPack fails
                return jnp.asarray(tensor.detach().cpu().numpy())
        else: # If tensor is already on CPU
             return jnp.asarray(tensor.numpy()) # Directly convert from CPU tensor

    @staticmethod
    def _jax_to_numpy(arr: jax.Array) -> np.ndarray:
        """Helper to convert JAX array to NumPy array (on CPU)."""
        return np.array(arr)

    def get_map(self, x0: torch.Tensor, x1: torch.Tensor) -> jax.Array: # Returns JAX array
        """Compute the OT plan using JAX OTT.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim) - PyTorch Tensor
        x1 : Tensor, shape (bs, *dim) - PyTorch Tensor

        Returns
        -------
        P : jax.Array, shape (bs, bs) - JAX array representing the OT plan
        """
        bs0, bs1 = x0.shape[0], x1.shape[0]

        # --- Convert PyTorch Tensors to JAX Arrays ---
        jax_x0 = self._pytorch_to_jax(x0)
        jax_x1 = self._pytorch_to_jax(x1)

        # Reshape if needed (using JAX)
        if jax_x0.ndim > 2:
            jax_x0 = jax_x0.reshape(bs0, -1)
        if jax_x1.ndim > 2:
            jax_x1 = jax_x1.reshape(bs1, -1)

        # --- Define OTT Geometry (PointCloud implies Squared Euclidean cost) ---
        # Epsilon (reg) is passed here for Sinkhorn's stability/computation
        geom = pointcloud.PointCloud(jax_x0, jax_x1, epsilon=self.epsilon)

        # --- Define Marginals (as JAX arrays) ---
        a = jnp.ones(bs0) / bs0
        b = jnp.ones(bs1) / bs1

        # --- Define Linear Problem ---
        ot_prob = linear_problem.LinearProblem(geom, a=a, b=b)

        # --- Solve using the initialized solver ---
        sol = self.solver(ot_prob)

        # --- Check Convergence (Optional but recommended) ---
        if not sol.converged:
            num_iters_run = sol.n_iters if hasattr(sol, 'n_iters') else 'N/A' # Check if n_iters exists too, just in case
            warnings.warn(f"JAX OTT Solver ({self.method}) did not converge. "
                        f"Iterations Run: {num_iters_run}, Threshold: {self.solver.threshold}. "
                        f"Max Iterations: {self.solver.max_iterations if hasattr(self.solver, 'max_iterations') else 'N/A'}")



        # --- Extract Transport Plan Matrix ---
        P = sol.matrix # This is the transport plan as a JAX array

        # --- Handle Numerical Issues (Check Nans/Infs, Sum) ---
        # Note: JAX operations inside JIT might behave differently with NaNs
        # It's often better to check convergence status from the solver
        # if not jnp.all(jnp.isfinite(P)):
        #     print("ERROR: JAX OTT plan P is not finite")
        #     # Add more debug info if needed
        # if jnp.abs(P.sum() - 1.0) > 1e-6: # Check sum for balanced case
        #      if self.warn:
        #           warnings.warn("Numerical errors in OT plan sum, potentially reverting.")
        #      # Example: return jnp.ones_like(P) / P.size

        return P # Return the JAX array

    # Modified to accept JAX key and JAX plan, return JAX indices
    def sample_map(self, key: jax.random.PRNGKey, pi: jax.Array, batch_size: int, replace=True) -> Tuple[jax.Array, jax.Array]:
        r"""Draw source and target indices from pi using JAX.

        Parameters
        ----------
        key : jax.random.PRNGKey
            JAX random key.
        pi : jax.Array, shape (n, m)
            Represents the OT plan (JAX array).
        batch_size : int
            Number of samples to draw.
        replace : bool
            Sampling with or without replacement. (Note: JAX choice handles this)

        Returns
        -------
        (i_s, i_j) : tuple of jax.Arrays, shape (batch_size,)
            Represents the indices of source and target data samples from pi.
        """
        n, m = pi.shape
        p = pi.ravel() # Flatten using JAX
        # Ensure normalization (handle potential small numerical errors)
        p = jnp.maximum(p, 0.0) # Ensure non-negative
        p_sum = p.sum()
        # Use a safe division, defaulting to uniform if sum is near zero
        p = jax.lax.cond(
             p_sum > 1e-9,
             lambda x: x / x.sum(),
             lambda x: jnp.ones_like(x) / x.size,
             p
        )

        # Sample flattened indices using JAX
        choices = jax.random.choice(
            key,
            n * m, # Number of elements to choose from (0 to n*m - 1)
            shape=(batch_size,),
            p=p,
            replace=replace
        )

        # Calculate row (i) and column (j) indices using JAX
        # Equivalent to np.divmod(choices, m)
        i_s = choices // m
        i_j = choices % m
        return i_s, i_j

    # Modified to accept JAX key, convert types, return PyTorch tensors
    def sample_plan(self, key: jax.random.PRNGKey, x0: torch.Tensor, x1: torch.Tensor, replace=True, batch_size=None) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Compute OT plan with JAX OTT and sample PyTorch tensors."""
        # Get OT plan as a JAX array
        pi_jax = self.get_map(x0, x1)

        if batch_size is None:
            batch_size = x0.shape[0] # Use source batch size

        # Sample indices as JAX arrays
        i_jax, j_jax = self.sample_map(pi_jax, batch_size, replace=replace)

        # --- Convert JAX indices to NumPy/CPU for indexing PyTorch tensors ---
        # This is necessary if x0, x1 remain PyTorch tensors
        i_np = self._jax_to_numpy(i_jax)
        j_np = self._jax_to_numpy(j_jax)

        # Index original PyTorch tensors
        return x0[i_np], x1[j_np]

    # Modified similarly to sample_plan
    def sample_plan_with_labels(self, key: jax.random.PRNGKey, x0: torch.Tensor, x1: torch.Tensor, y0=None, y1=None, replace=True, batch_size=None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""Compute OT plan with JAX OTT and sample labeled PyTorch tensors."""
        pi_jax = self.get_map(x0, x1)

        if batch_size is None:
            batch_size = x0.shape[0]

        i_jax, j_jax = self.sample_map(pi_jax, batch_size, replace=replace)

        # Convert indices for PyTorch indexing
        i_np = self._jax_to_numpy(i_jax)
        j_np = self._jax_to_numpy(j_jax)

        # Index original PyTorch tensors (data and labels)
        return (
            x0[i_np],
            x1[j_np],
            y0[i_np] if y0 is not None else None,
            y1[j_np] if y1 is not None else None,
        )

    # Needs significant rework for JAX random keys and array handling
    def sample_trajectory(self, key: jax.random.PRNGKey, X: torch.Tensor) -> torch.Tensor:
        """Compute OT trajectories using JAX OTT.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Base JAX random key.
        X : Tensor, (bs, times, *dim) - PyTorch Tensor
            Populations of samples over time.

        Returns
        -------
        to_return : Tensor, (bs, times, *dim) - PyTorch Tensor
            Represents the OT sampled trajectories over time.
        """
        bs = X.shape[0]
        times = X.shape[1]
        # It might be more efficient to convert X to JAX once
        X_jax = self._pytorch_to_jax(X) # Shape (bs, times, *dim)

        # --- Compute all plans first (can be JITted if wrapped) ---
        pis_jax = []
        for t in range(times - 1):
            # Pass relevant slices of JAX tensor
            pi_t = self.get_map(X_jax[:, t], X_jax[:, t + 1]) # Pass JAX arrays directly if get_map accepts them
            pis_jax.append(pi_t)

        # --- Sample trajectories using JAX ---
        # Initial indices (JAX array)
        current_indices = jnp.arange(bs)
        all_indices = [current_indices]

        # Loop through plans, splitting key at each step
        for t in range(times - 1):
            key, subkey = jax.random.split(key)
            pi_t = pis_jax[t] # Shape (bs, bs) - Plan from X_t to X_{t+1}

            # Sample next indices based on current indices and plan pi_t
            # We need to sample j for each current i based on pi_t[i, :]
            # This requires vmapping the sampling process

            # Define sampling for a single row i
            def sample_for_row(k, row_idx):
                p_row = pi_t[row_idx] # Get probabilities for current sample i
                p_row = jnp.maximum(p_row, 0.0)
                 # Safe normalization per row
                p_row = jax.lax.cond(
                    p_row.sum() > 1e-9,
                    lambda x: x / x.sum(),
                    lambda x: jnp.ones_like(x) / x.size,
                    p_row
                )
                # Sample the next index j based on p_row
                return jax.random.choice(k, pi_t.shape[1], p=p_row) # Sample column index

            # Generate keys for each row in the vmap
            row_keys = jax.random.split(subkey, bs)
            # Use vmap to apply sampling across all current indices
            next_indices = jax.vmap(sample_for_row)(row_keys, current_indices)

            all_indices.append(next_indices)
            current_indices = next_indices # Update for next step

        # --- Gather results using JAX indices ---
        # all_indices is a list of JAX arrays, len=times, each shape=(bs,)
        # We want to gather X_jax[all_indices[t], t] for each t
        # Stack indices first: indices_stack shape (times, bs)
        indices_stack = jnp.stack(all_indices, axis=0)

        # Use vmap or manual indexing to gather
        # Need shape (bs, times, *dim)
        # Gather X_jax[:, t][indices_stack[t]] - incorrect indexing
        # Try: X_jax[indices_stack[:, i], jnp.arange(times), ...] ? - Need careful indexing
        # Easiest might be a loop or vmap:
        def gather_at_time(t):
             # Gather samples at time t using indices[t]
             # X_jax[:, t] has shape (bs, *dim)
             # indices_stack[t] has shape (bs,)
             return X_jax[:, t][indices_stack[t]]

        # Apply vmap over the time dimension
        # Need to adjust X_jax if it's not already (times, bs, *dim) for this vmap
        # Alternative: stack gathered results
        gathered_list = []
        for t in range(times):
            gathered_list.append(X_jax[:, t][indices_stack[t]]) # Gather for each time step
        
        result_jax = jnp.stack(gathered_list, axis=1) # Stack along time axis -> (bs, times, *dim)


        # --- Convert final result back to PyTorch Tensor ---
        return torch.from_numpy(self._jax_to_numpy(result_jax)).to(X.device) # Move back to original device

