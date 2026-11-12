"""Small agent workflows built without a framework.

LangGraph/LangChain would add hundreds of MB of dependencies and start-up time for a few loops. The graphs
here are tiny and explicit: a writer/evaluator loop with a round limit and a stop condition, and a supervisor
that splits a team analysis into per-person workers and combines their results.
"""
