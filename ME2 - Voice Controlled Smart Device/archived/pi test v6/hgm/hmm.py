"""Left-to-right 3-state HMM per phone, with Baum-Welch (forward-backward) for
transition probabilities and GMM EM for emissions.

Each phone is modeled as a 3-state L2R HMM (entry, hold, exit) with a small
amount of self-loop + rightward structure (no skipping, no backtracking) -- the
standard HTK/3-state topology. Transitions:
    state 0 -> 0 (a00), 0 -> 1 (a01)
    state 1 -> 1 (a11), 1 -> 2 (a12)
    state 2 -> 2 (a22)
Start state is 0; the HMM can terminate in any state (we add an implicit end).
"""
from __future__ import annotations
import numpy as np
from .gmm import GMM

N_STATES = 3
EPS = 1e-6


class PhoneHMM:
    def __init__(self, phone: str, dim: int, n_comp: int, seed: int = 0):
        self.phone = phone
        self.dim = dim
        self.n_comp = n_comp
        # transition matrix (3x3), rows sum to 1
        a = np.array([
            [0.80, 0.20, 0.00],
            [0.00, 0.80, 0.20],
            [0.00, 0.00, 1.00],
        ], dtype=np.float64)
        self.log_a = np.log(a + EPS)
        self.states = [GMM(n_comp, dim, seed=seed + i) for i in range(N_STATES)]
        self.seed = seed

    # ------------------------------------------------------------------ #
    def log_emit(self, X: np.ndarray) -> np.ndarray:
        """(T, 3) log emission per state."""
        return np.stack([s.log_likelihood(X) for s in self.states], axis=1)

    # ------------------------------------------------------------------ #
    def _forward(self, log_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass. log_b: (T, 3). Returns (log_alpha (T,3), log_scale (T,))."""
        T = log_b.shape[0]
        log_alpha = np.full((T, N_STATES), -np.inf)
        log_scale = np.zeros(T)
        # init
        log_alpha[0] = log_b[0]  # start-state prior folded in by caller
        log_scale[0] = _logsumexp(log_alpha[0])
        log_alpha[0] -= log_scale[0]
        for t in range(1, T):
            # alpha[t, j] = logsumexp_i (log_alpha[t-1, i] + log_a[i, j]) + log_b[t, j]
            la = log_alpha[t - 1][:, None] + self.log_a          # (3,3)
            prev = _logsumexp_rows(la)                            # (3,) over i
            log_alpha[t] = prev + log_b[t]
            log_scale[t] = _logsumexp(log_alpha[t])
            log_alpha[t] -= log_scale[t]
        return log_alpha, log_scale

    def _backward(self, log_b: np.ndarray, log_scale: np.ndarray) -> np.ndarray:
        T = log_b.shape[0]
        log_beta = np.zeros((T, N_STATES))
        for t in range(T - 2, -1, -1):
            la = self.log_a + log_b[t + 1][None, :]              # (3,3) a[i,j]+b[j]
            log_beta[t] = _logsumexp_rows(la)
        return log_beta

    # ------------------------------------------------------------------ #
    def baum_welch(self, X: np.ndarray, log_b: np.ndarray, n_iter: int = 3):
        """Update transitions (and re-run GMM EM on state segments)."""
        T = log_b.shape[0]
        if T < 3:
            return
        log_alpha, log_scale = self._forward(log_b)
        log_beta = self._backward(log_b, log_scale)
        log_post = log_alpha + log_beta - log_scale[:, None]     # (T,3) gamma
        # xi: (T-1, 3, 3)
        log_xi = np.zeros((T - 1, N_STATES, N_STATES))
        for t in range(T - 1):
            log_xi[t] = (log_alpha[t][:, None] + self.log_a +
                         log_b[t + 1][None, :] + log_beta[t + 1][None, :])
        log_xi -= _logsumexp_rows2(log_xi)[:, None, None]
        xi = np.exp(log_xi)
        gamma = np.exp(log_post - _logsumexp_rows(log_post)[:, None])
        # re-estimate transitions (rows sum to 1)
        a = np.zeros((N_STATES, N_STATES))
        for t in range(T - 1):
            a += xi[t]
        a += EPS
        a /= a.sum(axis=1, keepdims=True)
        self.log_a = np.log(a)
        # re-estimate emissions per state using gamma weights
        for j in range(N_STATES):
            w = gamma[:, j]
            wsum = w.sum()
            if wsum < 1e-6:
                continue
            self.states[j].em_update_weighted(X, w / wsum)

    # ------------------------------------------------------------------ #
    def viterbi(self, log_b: np.ndarray) -> tuple[np.ndarray, float]:
        """Return (state_path (T,), log_prob)."""
        T = log_b.shape[0]
        delta = np.full((T, N_STATES), -np.inf)
        psi = np.zeros((T, N_STATES), dtype=int)
        delta[0] = log_b[0]
        for t in range(1, T):
            cand = delta[t - 1][:, None] + self.log_a
            psi[t] = cand.argmax(axis=0)
            delta[t] = cand[psi[t], np.arange(N_STATES)] + log_b[t]
        path = np.zeros(T, dtype=int)
        path[T - 1] = delta[T - 1].argmax()
        logp = delta[T - 1].max()
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1][path[t + 1]]
        return path, float(logp)

    # ------------------------------------------------------------------ #
    def to_dict(self):
        return {"phone": self.phone, "log_a": self.log_a,
                "states": [s.to_dict() for s in self.states]}

    @classmethod
    def from_dict(cls, d):
        h = cls(d["phone"], dim=0, n_comp=0)
        h.log_a = np.asarray(d["log_a"], dtype=np.float64)
        h.states = [GMM.from_dict(s) for s in d["states"]]
        h.dim = h.states[0].dim
        h.n_comp = h.states[0].n_comp
        return h


def _logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True) + 1e-300)
    if axis is None:
        return float(out.ravel()[0])
    return out.squeeze(-1) if out.shape[-1] == 1 else out


def _logsumexp_rows(a):
    """logsumexp over last axis -> (...,)."""
    m = a.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=-1, keepdims=True) + 1e-300)).squeeze(-1)


def _logsumexp_rows2(a):
    return _logsumexp_rows(a.reshape(a.shape[0], -1)).reshape(a.shape[0])
