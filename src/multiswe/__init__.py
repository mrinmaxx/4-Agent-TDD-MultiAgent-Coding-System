"""multiswe — a 4-agent Test-Driven-Development coding system built on mini-swe-agent.

Pipeline:  Planner (Oracle) -> Test Architect (Constrainer) -> [syntax gate]
           -> Implementer (Solver) -> [execution gate] <-> Reviewer (Debugger)

The correctness signal is a *deterministic runtime gate* (pytest over brute-force-verified
fuzz tests), never an LLM-invented expected value.
"""
