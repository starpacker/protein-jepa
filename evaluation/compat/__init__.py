"""Vendored, frozen copies of model code versions needed to load LEGACY
checkpoints whose architecture was later rewritten in the working tree.

Nothing here is imported by the main pipeline; it exists only so the evaluator
can load old checkpoints reproducibly.  Each module records the git revision
it was copied from.
"""
