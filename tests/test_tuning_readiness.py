from app.learn import tuning_readiness as tuning


async def test_readiness_names_what_is_missing(db):
    out = await tuning.readiness(db)
    assert out["ready"] is False and out["goodReplies"] == 0
    assert any("agreed to share" in m for m in out["missing"]) and any("more good replies" in m for m in out["missing"])
