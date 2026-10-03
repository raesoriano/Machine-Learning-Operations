"""pi test v6 -- hgm: a from-scratch Hidden-Markov-Model / Gaussian-Mixture-Model
speech recognizer, inspired by PocketSphinx but with no PocketSphinx dependency.

Three components (classic HTK/Sphinx architecture):
  1. Acoustic model  (hmm.py + gmm.py + acoustic.py): raw waveform -> per-frame
     phone-state log-likelihoods, via per-phone 3-state left-to-right HMMs whose
     states are diagonal-covariance Gaussian Mixture Models.
  2. Phonetic dictionary (dict.py): word -> phoneme (ARPAbet) sequence.
  3. Language model (lm.py): word bigram / grammar FSA that constrains the
     search space to the command set.

decode.py ties them together: a two-level Viterbi (word FSA x phone HMMs) yields
the best command phrase; a flat-unigram free decode yields "what I heard"; the
gap between the two drives REJECT.
"""
__version__ = "6.0.0"
