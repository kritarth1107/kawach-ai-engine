"""A commit in someone else's area needs their Reviewed-by. Author-agent is always required."""

from scripts.owners_check import needed, owner_of, rules

TEXT = """
```owners
app/brain/loop.py	Brain Engineer
app/jobs/	Reliability Engineer
eval/	Saheli Quality Engineer
```
"""


def test_longest_prefix_wins():
    rows = rules(TEXT)
    assert owner_of("app/jobs/saheli_night.py", rows) == "Reliability Engineer"
    assert owner_of("tests/test_x.py", rows) is None


def test_own_area_needs_no_review():
    rows = rules(TEXT)
    message = "Move a job.\n\nAuthor-agent: Reliability Engineer\n"
    assert needed(["app/jobs/saheli_night.py"], message, rows) == []


def test_another_area_needs_the_owner():
    rows = rules(TEXT)
    message = "Touch the loop.\n\nAuthor-agent: Reliability Engineer\n"
    assert needed(["app/brain/loop.py"], message, rows) == ["missing Reviewed-by: Brain Engineer"]


def test_a_review_clears_it():
    rows = rules(TEXT)
    message = "Touch the loop.\n\nAuthor-agent: Reliability Engineer\nReviewed-by: Brain Engineer\n"
    assert needed(["app/brain/loop.py"], message, rows) == []


def test_author_is_required():
    rows = rules(TEXT)
    assert needed(["app/jobs/x.py"], "no trailer\n", rows) == [
        "missing Author-agent", "missing Reviewed-by: Reliability Engineer"]
