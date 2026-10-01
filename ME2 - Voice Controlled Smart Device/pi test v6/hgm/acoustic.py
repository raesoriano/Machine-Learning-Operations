"""Acoustic model: one 3-state left-to-right HMM per phone; each HMM state is a
diagonal GMM. Trained by bootstrap EM (the classic HTK approach):

  bootstrap : proportional segmentation (split each phone's frames into 3
              equal blocks) -> GMMs initialized from data.
  round n   : SOFT forced alignment (forward-backward posteriors) over each
              utterance's known phone sequence -> accumulate sufficient
              statistics per (phone, state) -> one weighted GMM update +
              L2R-masked transition re-estimation.

Soft (posterior) alignment is what HTK uses; it is far more robust than hard
Viterbi alignment, which collapses when a phone's GMMs are still rough (a
frame gets dumped entirely into one state/phone and the others starve).

The model is *tied*: every occurrence of a phone (across all words and
utterances) shares the same HMM, which is what makes it work with limited data.
"""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from .hmm import PhoneHMM, N_STATES, EPS
from .gmm import GMM

NEG = -1e30          # finite log sentinel (same convention as decode.py)
EXIT_LOG = 0.0       # structural inter-phone exit edge (prob 1, log 1)


def _lsep2(a, b):
    """Elementwise logaddexp."""
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def _lsep_all(x):
    """logsumexp over ALL elements of x -> scalar (per-frame scaling)."""
    m = x.max()
    return float(m + np.log(np.exp(x - m).sum()))


class AcousticModel:
    def __init__(self, phones: list[str], dim: int, n_comp: int = 4, seed: int = 0):
        self.phones = list(phones)
        self.phone2idx = {p: i for i, p in enumerate(phones)}
        self.n_phones = len(phones)
        self.dim = dim
        self.n_comp = n_comp
        self.seed = seed
        self.hmms = {p: PhoneHMM(p, dim, n_comp, seed=seed + i)
                     for i, p in enumerate(phones)}

    # ------------------------------------------------------------------ #
    def _collect_frames(self, utterances):
        """utterances: list of (X (T,D), phone_seq list[str]).
        Returns dict phone -> (n,D) of all frames proportionally assigned to it
        (bootstrap initialization only)."""
        acc = {p: [] for p in self.phones}
        for X, seq in utterances:
            T = X.shape[0]
            L = len(seq)
            if T < L:
                seq = (seq * (T // L + 1))[:T]
                L = len(seq)
            bounds = np.round(np.linspace(0, T, L + 1)).astype(int)
            for i, ph in enumerate(seq):
                a, b = bounds[i], bounds[i + 1]
                if b > a and ph in acc:
                    acc[ph].append(X[a:b])
        return {p: (np.concatenate(v, axis=0) if v else None) for p, v in acc.items()}

    # ------------------------------------------------------------------ #
    # forced-alignment forward-backward over the concatenated 3L-state L2R
    # graph for ONE utterance with a known phone sequence. Unnormalized
    # (log-domain) forward/backward; gamma and xi are proper posteriors.
    # ------------------------------------------------------------------ #
    def _fb(self, log_emit: np.ndarray, la: np.ndarray):
        """SCALED forward-backward over the concatenated 3L-state L2R graph.

        log_emit: (T, L, 3). la: (L, 3, 3) per-position phone transition.
        Returns (gamma (T,L,3), xi (T-1,L,3,3), logZ) where gamma and xi are
        in PROBABILITY domain (each row normalized), as the EM update expects.

        Scaling (the standard HTK fix): alpha and beta are renormalized every
        frame, so they stay O(1) and the products never overflow/underflow --
        the unnormalized version overflows for utterances longer than a few
        hundred frames (log-emissions of +/-30 over T frames span exp(30T)).
        """
        T, L, _ = log_emit.shape
        a00 = la[:, 0, 0]; a01 = la[:, 0, 1]; a11 = la[:, 1, 1]
        a12 = la[:, 1, 2]; a22 = la[:, 2, 2]

        # ---- scaled forward ----
        alpha = np.full((T, L, 3), NEG)
        log_scale = np.zeros(T)
        alpha[0, 0, 0] = log_emit[0, 0, 0]
        log_scale[0] = _lsep_all(alpha[0])
        alpha[0] -= log_scale[0]
        for t in range(1, T):
            p0 = alpha[t - 1, :, 0] + a00
            p0[1:] = _lsep2(p0[1:], alpha[t - 1, :-1, 2] + EXIT_LOG)
            p1 = _lsep2(alpha[t - 1, :, 0] + a01, alpha[t - 1, :, 1] + a11)
            p2 = _lsep2(alpha[t - 1, :, 1] + a12, alpha[t - 1, :, 2] + a22)
            alpha[t, :, 0] = p0 + log_emit[t, :, 0]
            alpha[t, :, 1] = p1 + log_emit[t, :, 1]
            alpha[t, :, 2] = p2 + log_emit[t, :, 2]
            log_scale[t] = _lsep_all(alpha[t])
            alpha[t] -= log_scale[t]
        logZ = float(log_scale.sum())

        # ---- scaled backward (beta[T-1] = 0 = log 1) ----
        beta = np.zeros((T, L, 3))
        log_scale_b = np.zeros(T)
        for t in range(T - 2, -1, -1):
            b0 = _lsep2(la[:, 0, 0] + log_emit[t + 1, :, 0] + beta[t + 1, :, 0],
                        la[:, 0, 1] + log_emit[t + 1, :, 1] + beta[t + 1, :, 1])
            b1 = _lsep2(la[:, 1, 1] + log_emit[t + 1, :, 1] + beta[t + 1, :, 1],
                        la[:, 1, 2] + log_emit[t + 1, :, 2] + beta[t + 1, :, 2])
            b2s = la[:, 2, 2] + log_emit[t + 1, :, 2] + beta[t + 1, :, 2]
            b2n = log_emit[t + 1, 1:, 0] + beta[t + 1, 1:, 0] + EXIT_LOG  # (L-1,)
            b2 = b2s.copy()
            b2[:L - 1] = _lsep2(b2s[:L - 1], b2n)
            beta[t, :, 0] = b0
            beta[t, :, 1] = b1
            beta[t, :, 2] = b2
            log_scale_b[t] = _lsep_all(beta[t])
            beta[t] -= log_scale_b[t]

        # ---- state posteriors: log gamma_t(i) = a_t(i) + b_t(i) + log_scale[t]
        log_gamma = alpha + beta + log_scale[:, None, None]
        gm = log_gamma.max(axis=2, keepdims=True)
        gamma = np.exp(log_gamma - gm)
        gamma /= gamma.sum(axis=2, keepdims=True) + 1e-300

        # ---- transition posteriors (banded only; exit kept structural)
        # log xi_t(i,j) = a_t(i) + log a_ij + emit_{t+1}(j) + b_{t+1}(j)
        xi = np.zeros((T - 1, L, 3, 3))
        if T > 1:
            log_xi = np.full((T - 1, L, 3, 3), NEG)
            ae = alpha[:-1]; be = beta[1:]; le = log_emit[1:]
            log_xi[:, :, 0, 0] = ae[:, :, 0] + a00[None] + le[:, :, 0] + be[:, :, 0]
            log_xi[:, :, 0, 1] = ae[:, :, 0] + a01[None] + le[:, :, 1] + be[:, :, 1]
            log_xi[:, :, 1, 1] = ae[:, :, 1] + a11[None] + le[:, :, 1] + be[:, :, 1]
            log_xi[:, :, 1, 2] = ae[:, :, 1] + a12[None] + le[:, :, 2] + be[:, :, 2]
            log_xi[:, :, 2, 2] = ae[:, :, 2] + a22[None] + le[:, :, 2] + be[:, :, 2]
            xm = log_xi.max(axis=(2, 3), keepdims=True)
            xi = np.exp(log_xi - xm)
            xi /= xi.sum(axis=(2, 3), keepdims=True) + 1e-300
        return gamma, xi, logZ

    # ------------------------------------------------------------------ #
    def train(self, utterances, rounds: int = 10, verbose: bool = True):
        """Soft forced-alignment EM. utterances: list of (X (T,D), seq [str])."""
        # ---- bootstrap: initialize every GMM from proportionally split frames
        frames0 = self._collect_frames(utterances)
        self._boot_frames = frames0
        for pi, p in enumerate(self.phones):
            X = frames0[p]
            if X is None or len(X) < 2:
                continue
            for s, st in enumerate(self.hmms[p].states):
                st.initialize_from_data(X, seed=self.seed + pi * 7 + s)

        # per-dimension variance floor proportional to the feature scale.
        # A 1e-6 floor (the old value) is ~4 orders of magnitude below the
        # real variance of cepstral features, so components collapsed to
        # points and starved the other phones. 5% of the pooled per-dim
        # variance is a sane lower bound.
        pooled = np.concatenate([frames0[p] for p in self.phones
                                 if frames0[p] is not None and len(frames0[p]) > 0])
        self._var_floor = np.maximum(pooled.var(axis=0) * 0.05, 1e-3)

        K = self.n_comp
        D = self.dim
        log_a_stacked = np.stack([self.hmms[p].log_a for p in self.phones], axis=0)

        for r in range(rounds):
            Nk = np.zeros((self.n_phones, N_STATES, K))
            mu_acc = np.zeros((self.n_phones, N_STATES, K, D))
            x2_acc = np.zeros((self.n_phones, N_STATES, K, D))
            trans = np.zeros((self.n_phones, N_STATES, N_STATES))

            for X, seq in utterances:
                T = X.shape[0]
                L = len(seq)
                if T < L:
                    seq = (seq * (T // L + 1))[:T]
                    L = len(seq)
                if T < 2:
                    continue
                phone_idx = np.array([self.phone2idx[w] for w in seq], dtype=np.int64)
                logli = self.frame_logcomp(X)              # (T,P,3,K)
                LL = logsumexp(logli, axis=3)           # (T,P,3)
                log_emit = LL[:, phone_idx, :]             # (T,L,3)
                la = log_a_stacked[phone_idx]              # (L,3,3)
                gamma, xi, logZ = self._fb(log_emit, la)
                if not np.isfinite(logZ):
                    continue
                rresp = logli[:, phone_idx, :, :]          # (T,L,3,K)
                m = rresp.max(axis=3, keepdims=True)
                rresp = np.exp(rresp - m)
                rresp /= rresp.sum(axis=3, keepdims=True)  # (T,L,3,K)

                X2 = X * X
                for p in np.unique(phone_idx):
                    idx = np.where(phone_idx == p)[0]
                    w = gamma[:, idx, :, None] * rresp[:, idx, :, :]  # (T,npos,3,K)
                    c = w.sum(axis=1)                                  # (T,3,K)
                    Nk[p] += c.sum(axis=0)
                    mu_acc[p] += np.einsum("tsk,td->skd", c, X)
                    x2_acc[p] += np.einsum("tsk,td->skd", c, X2)
                    trans[p] += xi[:, idx, :, :].sum(axis=(0, 1))

            # ---- GMM update from accumulated stats (one EM step)
            for pi, p in enumerate(self.phones):
                if Nk[pi].sum() < 1.0:
                    continue  # starved this round: keep previous params
                for s in range(N_STATES):
                    self.hmms[p].states[s].update_from_stats(
                        Nk[pi, s], mu_acc[pi, s], x2_acc[pi, s],
                        var_floor=self._var_floor)
            # ---- phone-dropout re-init: a phone that kept starving is
            # re-seeded from its bootstrap frames instead of dying
            for pi, p in enumerate(self.phones):
                if Nk[pi].sum() < 5.0:
                    X = self._boot_frames.get(p)
                    if X is not None and len(X) >= 2:
                        for s, st in enumerate(self.hmms[p].states):
                            st.initialize_from_data(
                                X, seed=self.seed + pi * 7 + s)
            # ---- transition re-estimation (keep L2R structure)
            mask = np.zeros((N_STATES, N_STATES))
            mask[0, 0] = mask[0, 1] = mask[1, 1] = mask[1, 2] = mask[2, 2] = 1.0
            for pi, p in enumerate(self.phones):
                a = trans[pi]
                if a.sum() < 1:
                    continue
                a = a * mask + EPS
                a /= a.sum(axis=1, keepdims=True)
                self.hmms[p].log_a = np.log(a)

            if verbose:
                tot = float(Nk.sum())
                nph = int((Nk.sum(axis=(1, 2)) > 3.0).sum())
                print(f"  [AM] round {r}: {tot:.0f} eff phone-frames "
                      f"({nph}/{self.n_phones} phones active)", flush=True)

    # ------------------------------------------------------------------ #
    def frame_loglik(self, X: np.ndarray) -> np.ndarray:
        """(T, n_phones, 3) log-likelihood of frame t under phone p, state s."""
        return logsumexp(self.frame_logcomp(X), axis=3)

    def frame_logcomp(self, X: np.ndarray) -> np.ndarray:
        """Vectorized (T, n_phones, 3, K) log-responsibility (log w + log N).

        Stacks every (phone, state) GMM into (P, 3, K, D) arrays and computes
        all component log-likelihoods in a few numpy ops (no Python loops).
        """
        P = self.n_phones
        K = self.n_comp
        D = X.shape[1]
        T = X.shape[0]
        mu = np.zeros((P, N_STATES, K, D))
        logvar = np.zeros((P, N_STATES, K, D))
        logw = np.zeros((P, N_STATES, K))
        for pi, p in enumerate(self.phones):
            for s in range(N_STATES):
                g = self.hmms[p].states[s]
                mu[pi, s] = g.mu
                logvar[pi, s] = g.logvar
                logw[pi, s] = np.log(g.w + 1e-30)
        invvar = np.exp(-logvar)                                    # (P,3,K,D)
        # STABLE quadratic form: quad = sum_d (x_d - mu_d)^2 / var_d, computed
        # as the squared distance (never the expanded x^2 - 2xmu + mu^2 form,
        # which cancels to a small number and flips sign in float64 when the
        # cepstral features are large, e.g. c0 with std ~20).
        diff = X[:, None, None, None, :] - mu[None, :, :, :, :]     # (T,P,3,K,D)
        quad = (diff ** 2 * invvar[None, :, :, :, :]).sum(axis=4)   # (T,P,3,K)
        # log N(x|mu,diag(var)) = -0.5*(quad + sum_d logvar_d)  [global
        # -0.5*D*log2pi constant omitted: it cancels in every logsumexp/softmax]
        return (-0.5 * quad - 0.5 * logvar.sum(axis=3)) + logw[None, :, :, :]

    def frame_loglik_fast(self, X: np.ndarray) -> np.ndarray:
        """Vectorized (T, n_phones, 3) log-likelihood -- no Python loops."""
        return logsumexp(self.frame_logcomp(X), axis=3)

    # ------------------------------------------------------------------ #
    def save(self, path: str):
        data = {
            "phones": np.array(self.phones),
            "dim": np.int64(self.dim),
            "n_comp": np.int64(self.n_comp),
            "seed": np.int64(self.seed),
        }
        for p in self.phones:
            d = self.hmms[p].to_dict()
            data[f"log_a|{p}"] = d["log_a"]
            for s in range(N_STATES):
                g = d["states"][s]
                data[f"w|{p}|{s}"] = g["w"]
                data[f"mu|{p}|{s}"] = g["mu"]
                data[f"logvar|{p}|{s}"] = g["logvar"]
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: str):
        z = np.load(path, allow_pickle=False)
        phones = [str(p) for p in z["phones"]]
        am = cls(phones, dim=int(z["dim"]), n_comp=int(z["n_comp"]),
                 seed=int(z["seed"]))
        for p in phones:
            hmm = am.hmms[p]
            hmm.log_a = np.asarray(z[f"log_a|{p}"], dtype=np.float64)
            for s in range(N_STATES):
                g = hmm.states[s]
                g.w = np.asarray(z[f"w|{p}|{s}"], dtype=np.float64)
                g.mu = np.asarray(z[f"mu|{p}|{s}"], dtype=np.float64)
                g.logvar = np.asarray(z[f"logvar|{p}|{s}"], dtype=np.float64)
        return am
