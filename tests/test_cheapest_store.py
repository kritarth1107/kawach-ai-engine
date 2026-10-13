"""The options update names the cheapest store plainly (founder 2026-10-13: "I can get it cheapest at XYZ"): the same
items, the quantities asked, as each store prices them; a store with fewer of the items or only a near match is not the
cheapest of the same thing."""

from app.tasks import runtime
from app.tasks.models import Task


def _store(service, price, *, exact=True, qty=2, name="Zerodol MR Tablet (strip of 10)"):
    t = Task(service=service, kind="order", status="awaiting_confirm", phase="browse", input_needed="go", goal="Zerodol MR",
             details={"items": [{"name": "Zerodol MR", "qty": qty}], "compare": "g1",
                      "chosen": [{"oid": "p1", "name": name, "price": f"₹{price}", "exact_match": exact}]},
             result={"items": [{"oid": "p1", "name": name, "price": f"₹{price}", "available": True}]})
    return t


def test_the_cheapest_store_is_named_with_its_total_for_the_quantity_asked():
    group = [_store("apollo", 61), _store("1mg", 58.5), _store("pharmeasy", 63)]
    out = runtime._compare_summary(group)
    assert "Cheapest: Tata 1mg, ₹117 for 2 x Zerodol MR Tablet (strip of 10)" in out
    assert 'cheapest at Tata 1mg' in out


def test_a_near_match_is_never_the_cheapest_of_the_same_thing():
    group = [_store("apollo", 61), _store("1mg", 20, exact=False, name="Zerodol P Tablet")]
    assert "Cheapest: Apollo 24|7 Pharmacy, ₹122" in runtime._compare_summary(group)


def test_one_store_with_prices_names_no_cheapest():
    assert "Cheapest" not in runtime._compare_summary([_store("apollo", 61)])


def test_the_cheapest_ride_fare_across_apps():
    def ride(service, fares):
        return Task(service=service, kind="ride", status="awaiting_confirm", phase="browse", input_needed="go", goal="Cab",
                    details={"pickup": "Home", "drop": "Station", "compare": "g2"},
                    result={"options": [{"type": t, "fare": f"₹{f}", "eta": "4 min"} for t, f in fares]})
    out = runtime._compare_summary([ride("uber", [("Uber Go", 236), ("Auto", 142)]), ride("ola", [("Auto", 128), ("Mini", 210)])])
    assert "Cheapest: Ola, Auto ₹128" in out
