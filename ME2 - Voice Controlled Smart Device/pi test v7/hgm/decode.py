"""Decoder: two-level Viterbi search over (word FSA x phone HMMs).

The search state is (fsa_node, phone_pos, hmm_state):
  * fsa_node  : node of the word-level FSA (grammar or free unigram loop).
  * phone_pos : position within the current word's phone sequence; -1 means we
                are at a word boundary (in a silence region).
  * hmm_state : state (0/1/2) of the current phone's 3-state L2R HMM. In a
                silence node it is the state of the dedicated SIL HMM.

Silence is modeled explicitly: the FSA's start and end nodes are SIL regions.
While the search sits in a SIL node it emits frames of the SIL phone (a
3-state HMM trained on real silence), which is how leading/trailing silence is
handled without padding the dictionary.

Emissions come from the acoustic model (per-frame, per-phone, per-state
log-likelihoods); word transitions come from the FSA (language model). The best
path through the grammar FSA yields the recognized command phrase; the best path
through the free unigram FSA yields "what I actually heard"; the gap between the
two drives REJECT.
"""
from __future__ import annotations
import numpy as np
from .hmm import N_STATES
from .acoustic import AcousticModel
from .dict import SIL


class FSA:
    """Weighted word FSA. Each edge emits a word (word_idx >= 0) or is a
    non-emitting final edge (word_idx == -1). Nodes in `sil_nodes` emit the SIL
    phone while the search dwells in them."""
    def __init__(self, n_nodes: int, end_node: int,
                 adj: list[list[tuple[int, int, float]]],
                 init_logw: np.ndarray, final_logw: np.ndarray,
                 sil_nodes: set[int]):
        self.n_nodes = n_nodes
        self.end_node = end_node
        self.adj = adj
        self.init_logw = init_logw
        self.final_logw = final_logw
        self.sil_nodes = sil_nodes


def grammar_fsa(lm) -> FSA:
    return FSA(lm.n_nodes, lm.end_node, lm.adj, lm.init_logw, lm.final_logw,
               lm.sil_nodes)


def free_fsa(words: list[str], log_unigram: np.ndarray) -> FSA:
    """Unigram loop with leading/trailing SIL: any word, any number of times."""
    V = len(words)
    start, end = 0, V + 1
    n_nodes = V + 2
    adj = [[] for _ in range(n_nodes)]
    init = np.full(n_nodes, -np.inf)
    fin = np.full(n_nodes, -np.inf)
    init[start] = 0.0
    fin[end] = 0.0
    for w in range(V):
        node = w + 1
        adj[start].append((node, w, float(log_unigram[w])))
        for w2 in range(V):
            adj[node].append((w2 + 1, w2, float(log_unigram[w2])))
        adj[node].append((end, -1, 0.0))
    adj[start].append((end, -1, 0.0))
    return FSA(n_nodes, end, adj, init, fin, sil_nodes={start, end})


class Decoder:
    def __init__(self, am: AcousticModel, dictionary: dict[str, list[str]],
                 words: list[str], beam: int = 4000):
        self.am = am
        self.dict = dictionary
        self.words = words
        self.w2i = {w: i for i, w in enumerate(words)}
        self.beam = beam
        self.phone2idx = am.phone2idx
        self.sil_idx = am.phone2idx[SIL]

    def _frame_ll(self, X: np.ndarray) -> np.ndarray:
        """(T, n_phones, 3)."""
        return self.am.frame_loglik(X)

    # ------------------------------------------------------------------ #
    def decode(self, fsa: FSA, node2word: dict[int, int],
               X: np.ndarray) -> tuple[list[int], float]:
        """Viterbi over (fsa_node, phone_pos, hmm_state).
        Returns (word_idx sequence, logprob)."""
        LL = self._frame_ll(X)
        T = LL.shape[0]
        NEG = -1e30
        sil_la = self.am.hmms[SIL].log_a
        # state -> (logprob, prev_state, word_emitted_into_this_state)
        active = {(0, -1, 0): (LL[0, self.sil_idx, 0], None, -1)}
        history = []
        for t in range(T):
            new: dict = {}
            for state, (lp, _prev, _w) in active.items():
                v, i, s = state
                if i == -1:
                    # at a word boundary (silence region) at node v
                    if v in fsa.sil_nodes:
                        # dwell: emit a SIL frame, move within the SIL HMM
                        for s2 in range(s, min(s + 1, N_STATES - 1) + 1):
                            wts = sil_la[s, s2]
                            if not np.isfinite(wts):
                                continue
                            key = (v, -1, s2)
                            val = lp + wts + LL[t, self.sil_idx, s2]
                            if val > new.get(key, (NEG, None, -1))[0]:
                                new[key] = (val, state, -1)
                    # take an FSA edge (start a word / exit to end)
                    for (v2, w, logw) in fsa.adj[v]:
                        if w >= 0:
                            ph = self.dict[self.words[w]][0]
                            key = (v2, 0, 0)
                            val = lp + logw + LL[t, self.phone2idx[ph], 0]
                            if val > new.get(key, (NEG, None, -1))[0]:
                                new[key] = (val, state, w)
                        else:
                            # non-emitting edge: enter a SIL node at state 0
                            s_enter = 0 if v2 in fsa.sil_nodes else -1
                            key = (v2, -1, s_enter)
                            val = lp + logw
                            if val > new.get(key, (NEG, None, -1))[0]:
                                new[key] = (val, state, -1)
                else:
                    w = node2word[v]
                    seq = self.dict[self.words[w]]
                    pidx = self.phone2idx[seq[i]]
                    la = self.am.hmms[seq[i]].log_a
                    for s2 in range(s, min(s + 1, N_STATES - 1) + 1):
                        if s2 > N_STATES - 1:
                            continue
                        wts = la[s, s2]
                        if not np.isfinite(wts):
                            continue
                        # within-phone transition to state s2
                        key = (v, i, s2)
                        val = lp + wts + LL[t, pidx, s2]
                        if val > new.get(key, (NEG, None, -1))[0]:
                            new[key] = (val, state, -1)
                        # reached exit state: advance to next phone / finish word
                        if s2 == 2:
                            if i < len(seq) - 1:
                                p2 = self.phone2idx[seq[i + 1]]
                                key = (v, i + 1, 0)
                                val = lp + wts + LL[t, p2, 0]
                                if val > new.get(key, (NEG, None, -1))[0]:
                                    new[key] = (val, state, -1)
                            else:
                                key = (v, -1, -1)
                                val = lp + wts
                                if val > new.get(key, (NEG, None, -1))[0]:
                                    new[key] = (val, state, -1)
            if len(new) > self.beam:
                items = sorted(new.items(), key=lambda kv: -kv[1][0])[:self.beam]
                new = dict(items)
            history.append(active)
            active = new
        # terminal: best state at the FSA end node (a SIL region)
        best_lp = NEG
        best_state = None
        for state, (lp, _p, _w) in active.items():
            v, i, s = state
            if v == fsa.end_node and i == -1:
                cand = lp + fsa.final_logw[v]
                if cand > best_lp:
                    best_lp = cand
                    best_state = state
        if best_state is None:
            return [], NEG
        # backtrack, collecting emitted words
        words = []
        state = best_state
        cur = active
        for t in range(T - 1, -1, -1):
            lp, prev, w = cur[state]
            if w >= 0:
                words.append(w)
            state = prev
            if state is None:
                break
            cur = history[t]
        words.reverse()
        return words, float(best_lp)

    # ------------------------------------------------------------------ #
    def decode_constrained(self, lm, X: np.ndarray) -> tuple[list[str], float, int]:
        """Returns (words, logprob, phrase_idx)."""
        fsa = grammar_fsa(lm)
        node2word = {}
        for p, ph in enumerate(lm.phrases):
            for i in range(len(ph)):
                node2word[lm.node_of[(p, i)]] = self.w2i[ph[i]]
        words_idx, lp = self.decode(fsa, node2word, X)
        words = [self.words[i] for i in words_idx]
        pidx = -1
        for p, ph in enumerate(lm.phrases):
            if ph == words:
                pidx = p
                break
        return words, lp, pidx

    def decode_free(self, lm, X: np.ndarray) -> tuple[list[str], float]:
        fsa = free_fsa(self.words, lm.log_unigram)
        node2word = {w + 1: w for w in range(len(self.words))}
        words_idx, lp = self.decode(fsa, node2word, X)
        return [self.words[i] for i in words_idx], lp
