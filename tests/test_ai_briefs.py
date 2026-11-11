"""Tests for the number checker in common/ai_briefs.py. Run with: python -m pytest tests"""
from common.ai_briefs import check, drop_ungrounded_sentences, fact_sheet, strip_citations

SUMMARY = {
    "removal_drop": {"NC": -0.875, "CA": -0.753},
    "lost_rx": 386771.0,
    "spend_musd": 335.5,
    "net_impact_usd": 21_300_000_000,
    "recovered_share": 0.568,
    "median_gap_points": -4.8,
    "label": "not a number",
}
FACTS = fact_sheet(SUMMARY)
ID = {f.key: f.fid for f in FACTS}


def statuses(text):
    return [(c.text, c.status) for c in check(text, FACTS).checks]


def test_fact_sheet_skips_text():
    assert "label" not in ID and len(FACTS) == 7


def test_cited_percent_and_sign():
    out = statuses(f"Use fell 88% [{ID['removal_drop.NC']}] and 57% recovered [{ID['recovered_share']}].")
    assert [s for _, s in out] == ["grounded", "grounded"]


def test_scaled_units():
    out = statuses(f"About 387k [{ID['lost_rx']}] lost scripts, $335.5M [{ID['spend_musd']}] spend, "
                   f"$21.3B [{ID['net_impact_usd']}] impact, also written as 386,771 [{ID['lost_rx']}].")
    assert all(s == "grounded" for _, s in out), out


def test_uncited_but_real_number():
    assert statuses("The gap was 4.8 points.") == [("4.8 points", "uncited")]


def test_invented_number_is_caught():
    out = statuses(f"Use fell 91% [{ID['removal_drop.NC']}].")
    assert out[0][1] == "ungrounded"


def test_wrong_precision_is_caught():
    # 0.568 is 56.8%, so "56.2%" must fail even though "56%" would pass
    assert statuses("Recovery was 56.2%.")[0][1] == "ungrounded"
    assert statuses("Recovery was 57%.")[0][1] == "uncited"


def test_years_and_small_counts_are_exempt():
    assert {s for _, s in statuses("In 2025, 4 states ran the test in Phase 3.")} == {"exempt"}


def test_drop_and_strip():
    text = "Real claim 57% [F5]. Made up 63% here.\n- Bullet with 99%.\n- Fine bullet."
    rep = check(text, FACTS)
    cleaned, removed = drop_ungrounded_sentences(text, rep)
    assert "63%" not in cleaned and "99%" not in cleaned and "Fine bullet" in cleaned and len(removed) == 2
    assert strip_citations("57% [F5] and [F2]") == "57% and"


def test_citation_groups_are_split_and_stripped():
    from common.ai_briefs import split_citation_groups
    t = split_citation_groups(f"Drops of 88% and 93% [{ID['removal_drop.NC']}, {ID['removal_drop.CA']}].")
    assert "[F1][F2]" in t.replace(" ", "")
    assert strip_citations("Drops of 88% [F4, F6].") == "Drops of 88%."


def test_claim_removal_keeps_structure():
    from common.ai_briefs import remove_sentences, sentences, tidy
    text = "**Headline here is fine.**\n\n**What we found**\n- First claim is right. Second claim is wrong.\n- Third claim is wrong too.\n\n**What could change this**\n- Only claim is wrong here."
    s = sentences(text)
    bad = [{"line": li, "part": pi} for li, pi, sent in s if "wrong" in sent]
    out = tidy(remove_sentences(text, bad))
    assert "First claim is right." in out and "wrong" not in out
    assert "What could change this" not in out  # heading left empty is removed
    assert "What we found" in out
