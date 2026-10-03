"""Language model: word bigram estimated from the command grammar, plus the
constrained word FSA (finite-state acceptor) that the decoder searches.

The grammar is a flat set of complete command phrases (103 phrases, 32 command
classes). The LM has two roles:
  * constrained decode : the FSA only allows word sequences that are valid
    grammar phrases (this is what keeps the search space tiny and the result a
    real command).
  * free decode        : a flat unigram over the word inventory, used to report
    "what I actually heard" and to drive REJECT (if free decodes far better
    than the grammar, the utterance is not one of our commands).

FSA topology (silence-aware):
    node 0        : start, a SIL region (leading silence)
    (p, i)        : inside phrase p, after emitting word i-1
    node E        : end, a SIL region (trailing silence)
    0 -> (p,0)    : emits phrase's first word, weight = log phrase prior
    (p,i)->(p,i+1): emits next word, weight = log bigram prob
    (p,n) -> E    : non-emitting (exit into trailing silence)
The decoder emits the SIL phone while it sits in node 0 or node E, which is how
leading/trailing silence is modeled without padding the dictionary.
"""
from __future__ import annotations
import math
import numpy as np

BOS = "<s>"
EOS = "</s>"


class LanguageModel:
    def __init__(self, phrases, class_of: list[str],
                 words: list[str], alpha: float = 1.0):
        # accept phrases as strings or token lists; store as token lists
        self.phrases = [p.split() if isinstance(p, str) else list(p)
                        for p in phrases]
        self.class_of = class_of
        self.words = list(words)
        self.alpha = alpha
        self.w2i = {w: i for i, w in enumerate(words)}
        V = len(self.words)

        # ---------------- bigram counts ---------------- #
        self.bg_count = np.zeros((V, V), dtype=np.float64)
        self.unigram = np.zeros(V, dtype=np.float64)
        for ph in self.phrases:
            for b in ph:
                self.unigram[self.w2i[b]] += 1
            for a, b in zip(ph[:-1], ph[1:]):
                self.bg_count[self.w2i[a], self.w2i[b]] += 1
        self.bg_count += alpha
        self.unigram += alpha
        self.bg_prob = self.bg_count / self.bg_count.sum(axis=1, keepdims=True)
        self.unigram_prob = self.unigram / self.unigram.sum()
        self.log_unigram = np.log(self.unigram_prob)

        # ---------------- phrase priors ---------------- #
        P = len(phrases)
        self.log_phrase_prior = np.full(P, -math.log(P))

        # ---------------- word FSA ---------------- #
        self.n_phrases = P
        self.start_node = 0
        self.end_node = 1
        # node id for (phrase p, position i) = 2 + p * (len(phrases[p]) + 1) + i
        self.node_of = {}
        node = 2
        for p, ph in enumerate(self.phrases):
            for i in range(len(ph) + 1):
                self.node_of[(p, i)] = node
                node += 1
        self.n_nodes = node
        self.sil_nodes = {0, 1}
        self.edges = []
        self.init_logw = np.full(self.n_nodes, -np.inf)
        self.final_logw = np.full(self.n_nodes, -np.inf)
        self.init_logw[0] = 0.0
        self.final_logw[self.end_node] = 0.0
        for p, ph in enumerate(self.phrases):
            n = len(ph)
            prev = self.start_node
            for i in range(n):
                cur = self.node_of[(p, i)]
                w = ph[i]
                if i == 0:
                    logw = self.log_phrase_prior[p]
                else:
                    logw = math.log(self.bg_prob[self.w2i[ph[i - 1]], self.w2i[w]] + 1e-30)
                self.edges.append((prev, cur, self.w2i[w], logw))
                prev = cur
            # last word complete -> exit into trailing SIL (end node)
            self.edges.append((prev, self.end_node, -1, -2.0))
        self.edges = np.array(self.edges, dtype=np.float64)
        self.adj = [[] for _ in range(self.n_nodes)]
        for e in self.edges:
            self.adj[int(e[0])].append((int(e[1]), int(e[2]), float(e[3])))
    # ------------------------------------------------------------------ #
    def free_logprob(self, word: str) -> float:
        i = self.w2i.get(word)
        if i is None:
            return -20.0
        return float(self.log_unigram[i])

    # ------------------------------------------------------------------ #
    def save(self, path: str):
        np.savez_compressed(path,
                            phrases=np.array(self.phrases, dtype=object),
                            class_of=np.array(self.class_of),
                            words=np.array(self.words),
                            log_unigram=self.log_unigram,
                            bg_prob=self.bg_prob,
                            log_phrase_prior=self.log_phrase_prior,
                            init_logw=self.init_logw,
                            final_logw=self.final_logw,
                            edges=self.edges,
                            n_nodes=np.int64(self.n_nodes),
                            end_node=np.int64(self.end_node),
                            )

    @classmethod
    def load(cls, path: str):
        z = np.load(path, allow_pickle=True)
        phrases = [list(map(str, p)) for p in z["phrases"]]
        lm = cls(phrases, [str(c) for c in z["class_of"]],
                 [str(w) for w in z["words"]])
        # restore FSA arrays (rebuild adj + node_of deterministically)
        lm.edges = np.asarray(z["edges"], dtype=np.float64)
        lm.init_logw = np.asarray(z["init_logw"])
        lm.final_logw = np.asarray(z["final_logw"])
        lm.n_nodes = int(z["n_nodes"])
        lm.end_node = int(z["end_node"])
        node = 2
        for p, ph in enumerate(phrases):
            for i in range(len(ph) + 1):
                lm.node_of[(p, i)] = node
                node += 1
        lm.adj = [[] for _ in range(lm.n_nodes)]
        for e in lm.edges:
            lm.adj[int(e[0])].append((int(e[1]), int(e[2]), float(e[3])))
        return lm
