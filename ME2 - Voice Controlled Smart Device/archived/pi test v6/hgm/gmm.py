"""Gaussian Mixture Model with diagonal covariance (numpy, vectorized).

This is the emission model of each HMM state. A GMM is a weighted sum of K
Gaussians: p(x) = sum_k w_k N(x | mu_k, diag(var_k)). We store log-variances
for numerical stability and compute log-likelihoods with the full log-sum-exp
over components.
"""
from __future__ import annotations
import numpy as np

LOG2PI = float(np.log(2.0 * np.pi))


class GMM:
    def __init__(self, n_comp: int, dim: int, seed: int = 0):
        self.n_comp = int(n_comp)
        self.dim = int(dim)
        rng = np.random.default_rng(seed)
        # start with unit-ish variance, zero mean; EM will move them
        self.w = np.full(n_comp, 1.0 / n_comp, dtype=np.float64)
        self.mu = rng.normal(0.0, 1.0, size=(n_comp, dim)).astype(np.float64)
        self.logvar = np.zeros((n_comp, dim), dtype=np.float64)

    # ------------------------------------------------------------------ #
    def log_component(self, X: np.ndarray) -> np.ndarray:
        """(T, K) log N(x_t | mu_k, diag(var_k))."""
        # diff: (T, D) - (K, D) -> (T, K, D)
        diff = X[:, None, :] - self.mu[None, :, :]
        var = np.exp(self.logvar)                      # (K, D)
        lpdf = -0.5 * (diff ** 2) / var[None, :, :] - 0.5 * self.logvar[None, :, :] - 0.5 * LOG2PI
        return lpdf.sum(axis=2)                        # (T, K)

    def log_posterior(self, X: np.ndarray) -> np.ndarray:
        """(T, K) log responsibility (log w_k + log N)."""
        return self.log_component(X) + np.log(self.w)[None, :]

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """(T,) log p(x_t) under the mixture."""
        lp = self.log_posterior(X)                     # (T, K)
        m = lp.max(axis=1, keepdims=True)
        return (m + np.log(np.exp(lp - m).sum(axis=1, keepdims=True))).ravel()

    # ------------------------------------------------------------------ #
    def em_update(self, X: np.ndarray, eps: float = 1e-9,
                  var_floor: float = 1e-6):
        """One EM step. X: (T, D)."""
        lp = self.log_posterior(X)                     # (T, K)
        m = lp.max(axis=1, keepdims=True)
        r = np.exp(lp - m)                             # (T, K) unnormalized
        r = r / r.sum(axis=1, keepdims=True)           # responsibilities
        Nk = r.sum(axis=0) + eps                       # (K,)
        self.w = Nk / (Nk.sum() + eps * self.n_comp)
        self.mu = (r.T @ X) / Nk[:, None]
        var = (r.T @ (X ** 2)) / Nk[:, None] - self.mu ** 2
        var = np.clip(var, var_floor, None)
        self.logvar = np.log(var)

    def em_update_weighted(self, X: np.ndarray, w: np.ndarray,
                           var_floor: float = 1e-6):
        """EM step with per-frame weights w (T,) (e.g. Viterbi posteriors)."""
        r = np.exp(self.log_posterior(X))              # (T, K)
        r = r / (r.sum(axis=1, keepdims=True) + 1e-300)
        r = r * w[:, None]
        Nk = r.sum(axis=0) + 1e-9
        self.w = Nk / (Nk.sum() + 1e-9 * self.n_comp)
        self.mu = (r.T @ X) / Nk[:, None]
        var = (r.T @ (X ** 2)) / Nk[:, None] - self.mu ** 2
        self.logvar = np.log(np.clip(var, var_floor, None))

    def update_from_stats(self, Nk: np.ndarray, mu_acc: np.ndarray,
                          x2_acc: np.ndarray, eps: float = 1e-9,
                          var_floor: float = 1e-6):
        """EM update directly from accumulated sufficient statistics.

        Nk     : (K,)      effective frame count per component
        mu_acc : (K, D)    sum of weighted x_t per component
        x2_acc : (K, D)    sum of weighted x_t^2 per component
        mean = mu_acc/Nk ; var = x2_acc/Nk - mean^2 (two-moment form).

        var_floor : scalar or (D,) per-dimension lower bound on the variance.
                    A floor proportional to the feature scale (not 1e-6) is
                    what stops a component from collapsing to a point and
                    starving the other phones.
        """
        Nk = Nk + eps
        mu = mu_acc / Nk[:, None]
        var = x2_acc / Nk[:, None] - mu ** 2
        var = np.clip(var, var_floor, None)
        w = Nk / (Nk.sum() + eps * self.n_comp)
        self.w = w
        self.mu = mu
        self.logvar = np.log(var)

    def initialize_from_data(self, X: np.ndarray, seed: int = 0,
                             kmeans_iters: int = 4):
        """k-means++ init + a few k-means refinement steps.

        Far more robust than picking K random points: the means start spread
        across the data (k-means++) and then settle into the real clusters
        (k-means), so each component actually represents a distinct region of
        feature space. This is what prevents the bootstrap from collapsing.
        """
        rng = np.random.default_rng(seed)
        T, D = X.shape
        K = self.n_comp
        if T == 0:
            return
        if T <= K:
            self.mu = X.copy()
            self.w = np.full(min(K, T), 1.0 / max(T, 1))
            self.logvar = np.tile(np.log(X.var(axis=0) + 1e-3),
                                  (self.mu.shape[0], 1))
            return
        # ---- k-means++ seeding ----
        mu = np.empty((K, D), dtype=np.float64)
        mu[0] = X[rng.integers(T)]
        closest = ((X - mu[0]) ** 2).sum(axis=1)
        for k in range(1, K):
            total = closest.sum()
            if total <= 0:
                mu[k] = X[rng.integers(T)]
            else:
                probs = closest / total
                mu[k] = X[rng.choice(T, p=probs)]
            d2 = ((X - mu[k]) ** 2).sum(axis=1)
            closest = np.minimum(closest, d2)
        # ---- k-means refinement ----
        for _ in range(kmeans_iters):
            dist = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)
            lab = dist.argmin(axis=1)
            counts = np.bincount(lab, minlength=K)
            new_mu = mu.copy()
            for k in range(K):
                if counts[k] > 0:
                    new_mu[k] = X[lab == k].mean(axis=0)
                else:
                    new_mu[k] = X[rng.integers(T)]
            mu = 0.5 * mu + 0.5 * new_mu
        self.mu = mu
        # ---- per-component variance from within-cluster spread, floored at a
        # fraction of the GLOBAL per-dim variance. A 1e-4 floor is ~5 orders of
        # magnitude below the real variance of 39-d cepstral features, so the
        # Gaussian density at each mean explodes (log-likelihood -> +4000) and
        # the forward-backward posteriors become garbage. 5% of the global
        # per-dim variance is a stable, scale-aware lower bound.
        dist = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)
        lab = dist.argmin(axis=1)
        gvar = X.var(axis=0)
        floor = np.maximum(0.05 * gvar, 1e-3)
        var = np.tile(gvar, (K, 1))
        for k in range(K):
            m = lab == k
            if m.sum() > 1:
                var[k] = np.maximum(X[m].var(axis=0), floor)
        self.logvar = np.log(var)
        self.w = np.full(K, 1.0 / K)

    # ------------------------------------------------------------------ #
    def to_dict(self):
        return {"n_comp": self.n_comp, "dim": self.dim,
                "w": self.w, "mu": self.mu, "logvar": self.logvar}

    @classmethod
    def from_dict(cls, d):
        g = cls(d["n_comp"], d["dim"])
        g.w = np.asarray(d["w"], dtype=np.float64)
        g.mu = np.asarray(d["mu"], dtype=np.float64)
        g.logvar = np.asarray(d["logvar"], dtype=np.float64)
        return g
