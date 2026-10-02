"""Deterministic command parser for the tiny VCM.

Pure code: no model, no network. Turns the VCM's constrained transcript into
a device command: {intent, slots}. The same parser is used for training
labels, benchmark scoring, and the RPi runtime service.
"""
from .parser import parse, Command  # noqa: F401
