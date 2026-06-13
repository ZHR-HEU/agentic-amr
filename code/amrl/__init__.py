"""amrl 鈥?LLM acquisition-policy controller for online AMR active learning.

IQ-free RF control-plane: a (possibly frozen LLM) controller reads deterministic
RF/Candidate state cards and emits soft weights over acquisition criteria;
deterministic tools select/label/update. The controller never sees raw IQ and
never classifies.

See refine-logs/FINAL_PROPOSAL.md (method M1-M8) and EXPERIMENT_PLAN.md.
"""
__version__ = "0.0.1"
