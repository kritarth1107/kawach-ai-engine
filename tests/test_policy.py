from app.brain import policy


def test_alert_reasons_and_confidence():
    assert policy.alert_decision("red_flag", 0.5).whatsapp
    assert not policy.alert_decision("safety", 0.6).whatsapp
    assert policy.alert_decision("safety", 0.8).whatsapp
    d = policy.alert_decision("missed_lunch", 1.0)
    assert not d.whatsapp and d.dashboard


def test_issue_ref_is_stable_per_day():
    assert policy.issue_ref("red_flag", "Fall in Bathroom!", "2026-10-02") == "alert:red_flag:fall-in-bathroom:2026-10-02"


def test_vital_red_flags():
    assert policy.vital_red_flag("bp", "192/118")
    assert policy.vital_red_flag("bp", "85/55")
    assert policy.vital_red_flag("bp", "140/90") is None
    assert policy.vital_red_flag("sugar", "62 mg/dL")
    assert policy.vital_red_flag("sugar", "180") is None
    assert policy.vital_red_flag("temperature", "103.5 F")
    assert policy.vital_red_flag("temperature", "39.8")
    assert policy.vital_red_flag("spo2", "89%")


def test_reply_guard():
    problems = policy.reply_problems("Aaj paneer bana lijiye, Maa!", known_text="dal chawal", avoid_words=["maa"], user_text="kya banau")
    assert any("paneer" in p for p in problems) and any("maa" in p for p in problems)
    assert not policy.reply_problems("Besan chilla bana lijiye", known_text="loves besan chilla", avoid_words=[], user_text="")
    assert policy.reply_problems("I checked the browser", known_text="", avoid_words=[], user_text="")
    assert policy.reply_problems("Total ₹450 hoga", known_text="", avoid_words=[], user_text="")
    assert not policy.reply_problems("Total ₹450 hoga", known_text='{"total": "₹450"}', avoid_words=[], user_text="")


def test_order_conflicts_use_allergen_families():
    assert policy.order_conflicts("2 strawberry milkshake", ["milk"], [])
    assert policy.order_conflicts("paneer tikka", ["milk"], [])
    assert policy.order_conflicts("kaju katli 250g", ["nuts"], [])
    assert not policy.order_conflicts("nariyal paani", ["milk"], [])
    assert policy.order_conflicts("Haldiram bhujia", [], ["bhujia"])


def test_language_mismatch():
    assert policy.reply_problems("The ride has been cancelled. Have a safe trip!", known_text="", avoid_words=[], user_text="Cab cancel kar do, beta aa raha hai")
    assert not policy.reply_problems("ठीक है, राइड कैंसल कर दी है।", known_text="", avoid_words=[], user_text="Cab cancel kar do, beta aa raha hai")
    assert not policy.reply_problems("Done, cancelled the ride.", known_text="", avoid_words=[], user_text="Please cancel the cab")
