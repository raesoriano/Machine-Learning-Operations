"""Fast vectorized two-level Viterbi decoder.

The reference decoder (decode.py) runs Viterbi over a Python dict of active
states, which is correct but slow (~5 s/clip on a Pi). This module precomputes
the full (FSA x phone-HMM) state graph once per FSA and then runs the Viterbi
forward pass as a handful of numpy gathers per frame, which is ~50-150 ms/clip.

State = (fsa_node, ppos, s):
  * fsa_node : node of the word-level FSA.
  * ppos     : phone position within the current word; -1 = word boundary.
  * s        : HMM state 0/1/2; -1 = silent (non-emitting) boundary.

Emission convention (matches the reference decoder): the frame is emitted by
the state we transition INTO.
  * phone state (v, ppos, s)      -> emits phone(v, ppos), state s.
  * SIL boundary (v, -1, s)       -> emits SIL, state s.
  * silent boundary (v, -1, -1)   -> emits nothing (zero-width).
"""
from __future__ import annotations
import numpy as np
from .hmm import N_STATES

NEG = -1e30


class ViterbiGraph:
    """Precomputed state graph + vectorized Viterbi for one FSA."""

    def __init__(self, fsa, node2word, dictionary, words, phone2idx, sil_idx,
                 phone_log_a, sil_log_a):
        self.fsa = fsa
        self.n_nodes = fsa.n_nodes
        self.end_node = fsa.end_node
        self.adj = fsa.adj
        self.sil_nodes = fsa.sil_nodes
        self.init_logw = fsa.init_logw
        self.final_logw = fsa.final_logw
        self.dictionary = dictionary
        self.words = words
        self.phone2idx = phone2idx
        self.sil_idx = sil_idx
        self.sil_log_a = sil_log_a
        self.node2word = node2word
        self.phone_log_a = phone_log_a

        # ---- word info per node ---- #
        self.node_word = {}
        self.node_seq = {}
        for v in range(self.n_nodes):
            w = node2word.get(v, -1)
            if w is not None and w >= 0:
                self.node_word[v] = w
                self.node_seq[v] = dictionary[words[w]]

        # ---- build state list ---- #
        states = []
        state_idx = {}

        def add(node, ppos, s):
            key = (node, ppos, s)
            if key not in state_idx:
                state_idx[key] = len(states)
                states.append(key)
            return state_idx[key]

        for v in range(self.n_nodes):
            if v in self.sil_nodes:
                for s in range(N_STATES):
                    add(v, -1, s)
            else:
                add(v, -1, -1)
            if v in self.node_word:
                seq = self.node_seq[v]
                for ppos in range(len(seq)):
                    for s in range(N_STATES):
                        add(v, ppos, s)

        self.states = states
        self.state_idx = state_idx
        N = len(states)
        self.N = N

        # ---- emission + word-introduction arrays ---- #
        emit_phone = np.full(N, -1, dtype=np.int32)
        emit_state = np.full(N, -1, dtype=np.int32)
        word_of = np.full(N, -1, dtype=np.int32)
        for idx, (node, ppos, s) in enumerate(states):
            if ppos == -1:
                if node in self.sil_nodes and s >= 0:
                    emit_phone[idx] = sil_idx
                    emit_state[idx] = s
            else:
                seq = self.node_seq[node]
                emit_phone[idx] = phone2idx[seq[ppos]]
                emit_state[idx] = s
                if ppos == 0:
                    word_of[idx] = self.node_word[node]
        self.emit_phone = emit_phone
        self.emit_state = emit_state
        self.word_of = word_of

        # ---- incoming edges per node ---- #
        self._incoming = [[] for _ in range(self.n_nodes)]
        for u in range(self.n_nodes):
            for (v2, w, logw) in self.adj[u]:
                self._incoming[v2].append((u, w, logw))

        # ---- predecessor lists ---- #
        def preds_of(node, ppos, s):
            out = []
            if ppos == -1:
                if node in self.sil_nodes and s >= 0:
                    # SIL self-loop + advance
                    if np.isfinite(sil_log_a[s, s]):
                        out.append((state_idx[(node, -1, s)], sil_log_a[s, s]))
                    if s > 0 and np.isfinite(sil_log_a[s - 1, s]):
                        out.append((state_idx[(node, -1, s - 1)],
                                    sil_log_a[s - 1, s]))
                    # non-emitting FSA edges into this sil node
                    for (u, w, logw) in self._incoming[node]:
                        if w < 0:
                            if u in self.sil_nodes:
                                for s2 in range(N_STATES):
                                    out.append((state_idx[(u, -1, s2)], logw))
                            else:
                                out.append((state_idx[(u, -1, -1)], logw))
                else:
                    # silent boundary: reached by word completion
                    if node in self.node_word:
                        seq = self.node_seq[node]
                        last = len(seq) - 1
                        la = phone_log_a[seq[last]]
                        if np.isfinite(la[2, 2]):
                            out.append((state_idx[(node, last, 2)], la[2, 2]))
            else:
                seq = self.node_seq[node]
                la = phone_log_a[seq[ppos]]
                # within-phone self-loop + advance
                if np.isfinite(la[s, s]):
                    out.append((state_idx[(node, ppos, s)], la[s, s]))
                if s > 0 and np.isfinite(la[s - 1, s]):
                    out.append((state_idx[(node, ppos, s - 1)], la[s - 1, s]))
                if s == 0:
                    if ppos == 0:
                        # word start: from the entering node's boundary
                        for (u, w, logw) in self._incoming[node]:
                            if w >= 0 and w == self.node_word[node]:
                                if u in self.sil_nodes:
                                    for s2 in range(N_STATES):
                                        out.append((state_idx[(u, -1, s2)], logw))
                                else:
                                    out.append((state_idx[(u, -1, -1)], logw))
                    else:
                        # phone advance from previous phone's exit state
                        la_prev = phone_log_a[seq[ppos - 1]]
                        if np.isfinite(la_prev[2, 2]):
                            out.append((state_idx[(node, ppos - 1, 2)],
                                        la_prev[2, 2]))
            return out

        pred_lists = [preds_of(node, ppos, s) for (node, ppos, s) in states]
        Kmax = max((len(pl) for pl in pred_lists), default=0)
        self.Kmax = Kmax
        pred_idx = np.full((N, max(Kmax, 1)), -1, dtype=np.int32)
        pred_logw = np.full((N, max(Kmax, 1)), NEG, dtype=np.float64)
        for idx, pl in enumerate(pred_lists):
            for k, (pi, lw) in enumerate(pl):
                pred_idx[idx, k] = pi
                pred_logw[idx, k] = lw
        self.pred_idx = pred_idx
        self.pred_logw = pred_logw

        # ---- init / final ---- #
        self.init_lp = np.full(N, NEG)
        for v in range(self.n_nodes):
            if self.init_logw[v] > NEG / 2:
                if v in self.sil_nodes:
                    self.init_lp[state_idx[(v, -1, 0)]] = self.init_logw[v]
                else:
                    self.init_lp[state_idx[(v, -1, -1)]] = self.init_logw[v]
        self.final_lp = np.full(N, NEG)
        for v in range(self.n_nodes):
            if self.final_logw[v] > NEG / 2:
                if v in self.sil_nodes:
                    for s in range(N_STATES):
                        self.final_lp[state_idx[(v, -1, s)]] = self.final_logw[v]
                else:
                    self.final_lp[state_idx[(v, -1, -1)]] = self.final_logw[v]

        # precompute emission gather indices
        m = emit_phone >= 0
        self._em_mask = m
        self._em_ph = emit_phone[m]
        self._em_st = emit_state[m]

    # ------------------------------------------------------------------ #
    def _emit(self, LL_t: np.ndarray) -> np.ndarray:
        out = np.zeros(self.N)
        out[self._em_mask] = LL_t[self._em_ph, self._em_st]
        return out

    def viterbi(self, LL: np.ndarray):
        """LL: (T, n_phones, 3). Returns (word_idx list, logprob)."""
        T = LL.shape[0]
        N = self.N
        arange = np.arange(N)
        lp = self.init_lp + self._emit(LL[0])
        backptr = np.zeros((T, N), dtype=np.int16)
        for t in range(1, T):
            emit = self._emit(LL[t])
            cand = lp[self.pred_idx] + self.pred_logw      # (N, Kmax)
            best_k = cand.argmax(axis=1)
            backptr[t] = best_k
            lp = cand[arange, best_k] + emit
        score = lp + self.final_lp
        end_idx = int(score.argmax())
        if score[end_idx] < NEG / 2:
            return [], NEG
        words = []
        state = end_idx
        for t in range(T - 1, -1, -1):
            w = self.word_of[state]
            if w >= 0:
                words.append(int(w))
            if t > 0:
                k = backptr[t][state]
                state = int(self.pred_idx[state, k])
        words.reverse()
        return words, float(score[end_idx])


class FastDecoder:
    """Drop-in fast replacement for decode.Decoder.

    Same public API:
        decode_constrained(lm, X) -> (words, logprob, phrase_idx)
        decode_free(lm, X)        -> (words, logprob)
    """

    def __init__(self, am, dictionary, words):
        from .decode import grammar_fsa, free_fsa
        self.am = am
        self.dictionary = dictionary
        self.words = words
        self.w2i = {w: i for i, w in enumerate(words)}
        self.phone2idx = am.phone2idx
        self.sil_idx = am.phone2idx["SIL"]
        # phone_log_a: (n_phones, 3, 3)
        self.phone_log_a = np.stack(
            [am.hmms[p].log_a for p in am.phones], axis=0)
        self.sil_log_a = am.hmms["SIL"].log_a

        self._grammar = None
        self._free = None

    def _build(self, lm):
        if self._grammar is None:
            fsa = grammar_fsa(lm)
            node2word = {}
            for p, ph in enumerate(lm.phrases):
                for i in range(len(ph)):
                    node2word[lm.node_of[(p, i)]] = self.w2i[ph[i]]
            self._grammar = ViterbiGraph(
                fsa, node2word, self.dictionary, self.words,
                self.phone2idx, self.sil_idx, self.phone_log_a, self.sil_log_a)
        if self._free is None:
            fsa = free_fsa(self.words, lm.log_unigram)
            node2word = {w + 1: w for w in range(len(self.words))}
            self._free = ViterbiGraph(
                fsa, node2word, self.dictionary, self.words,
                self.phone2idx, self.sil_idx, self.phone_log_a, self.sil_log_a)

    def decode_constrained(self, lm, X):
        self._build(lm)
        LL = self.am.frame_loglik_fast(X)
        words_idx, lp = self._grammar.viterbi(LL)
        words = [self.words[i] for i in words_idx]
        pidx = -1
        for p, ph in enumerate(lm.phrases):
            if ph == words:
                pidx = p
                break
        return words, lp, pidx

    def decode_free(self, lm, X):
        self._build(lm)
        LL = self.am.frame_loglik_fast(X)
        words_idx, lp = self._free.viterbi(LL)
        return [self.words[i] for i in words_idx], lp
