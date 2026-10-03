"""Specialist agents behind Saheli's brain (contract v1).

The brain is the only one that talks to people. It hands a task to a specialist agent (shopping,
pharmacy, rides); the agent works on a channel (the store's own connector first, a browser as fallback)
and reports a structured status. Every report passes the guard, which enforces the money and safety
rules in code whatever the agent says. Metrics and cost are recorded on every task.
"""
