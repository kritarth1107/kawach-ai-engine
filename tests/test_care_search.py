"""Lab tests and doctors: matching, ordering and the family's city (the live calls are checked by hand, see journal)."""

from app.tasks import care_search


def test_test_words_and_matching():
    w = care_search._wanted("CBC")
    assert care_search._matches("HEMOGRAM - 6 PART (DIFF)", w) and care_search._matches("Complete Blood Count", w)
    assert not care_search._matches("Hepatitis B Surface Antigen", care_search._wanted("HbA1c"))
    assert care_search._matches("Glycosylated Hemoglobin (HbA1c) Test", care_search._wanted("sugar ka teen mahine wala test hba1c"))


def test_city_of_the_saved_place():
    assert care_search.city_of({"full": "Sunita Park, Labhandih, Raipur, Chhattisgarh 492001"}) == "Raipur"
    assert care_search.city_of({"city": "Pune"}) == "Pune"
    assert care_search.city_of({}) == "Raipur"
