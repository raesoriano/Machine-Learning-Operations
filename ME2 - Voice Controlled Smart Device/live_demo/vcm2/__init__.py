"""VCM v2 - two-stage voice command model.

Stage 1 (ASR):      audio -> transcript (words actually spoken)
                    Reuses the ME2 me2_v6 transcript-trained CTC model.
Stage 2 (classify): transcript -> one of 31 commands | REJECT
                    A NEW learned text classifier (TF-IDF + LinearSVC),
                    trained on the ME2 manifest transcript->command mapping.

This replaces v1's rule-based parser/canonicalizer with a learned classifier,
and splits the unlearnable "audio -> canonical phrase" task (v1's failure)
into two learnable tasks.
"""
__version__ = "2.0"
