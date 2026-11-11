"""Draft plain-English briefs from each project's results, then check every number.

Each project writes its headline results to `outputs/tables/summary.json`. This
module turns that file into short briefs for two readers:

  brand_lead        a commercial decision-maker: what happened, what to do, what it is worth
  analytics_review  a sceptical analyst: what was estimated, how, and how far to trust it

An open-weights language model, served locally by Ollama, writes the drafts. A
language model can write a fluent sentence around a number it made up, so the
draft is not trusted. Every number in it is checked against the summary file:

1. The model is given a numbered fact sheet ("[F7] nc_recovered_share = 0.568")
   and told to cite the fact behind each number, like "57% [F7]".
2. The checker extracts every number from the draft. A number passes if it
   matches the cited fact after allowed conversions (0.568 -> 57%, 386771 ->
   387k, 21300000000 -> $21.3B, sign dropped for "fell 88%"). A number with no
   citation passes only if it matches some fact, and is reported as uncited.
3. If any number fails, the model gets the list back and rewrites. After the
   last attempt, sentences that still contain an unsupported number are
   removed, and the removal is recorded.

The published brief has the citations stripped. A copy with citations and a
JSON report of every check sit beside it, so a reader can audit any sentence.

Terms
-----
Grounded   the number traces to a value in summary.json.
Uncited    grounded, but the model did not say which fact it used.
Ungrounded the number matches nothing in summary.json: treated as invented.

Usage
-----
    python -m common.ai_briefs                       # all projects, Ollama at localhost:11434
    python -m common.ai_briefs 07-test-and-control-experiments --model qwen2.5:7b
    python -m common.ai_briefs --check path/to/brief.md 07-test-and-control-experiments
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# One paragraph of context per project, so the model knows what the numbers mean.
# Kept short and factual: the numbers themselves come only from summary.json.
CONTEXT = {
    "01-glp1-market-and-targeting": (
        "Medicare Part D GLP-1 market sizing and prescriber targeting, built on CMS prescriber-level "
        "Part D data. Sizes the gap between high-volume primary care prescribers who under-use GLP-1s "
        "and their peers, ranks prescribers for rep calls, predicts next-year growth with gradient-boosted "
        "trees explained by SHAP values, and fits growth curves to forecast the class."),
    "02-ira-negotiation-exposure": (
        "Exposure to Medicare drug price negotiation under the Inflation Reduction Act. Estimates the net "
        "revenue impact of negotiated prices by manufacturer, and screens for the drugs most likely to be "
        "selected next, back-tested against actual CMS selections."),
    "03-loe-erosion-forecast": (
        "Loss of exclusivity: how fast brand prescriptions erode after generic entry, using Medicaid state "
        "drug utilisation data. Builds analog erosion curves, a Bayesian hierarchical model with two erosion "
        "regimes, conformal prediction intervals, and applies them to Eliquis generic entry in 2028."),
    "04-obesity-pipeline-intelligence": (
        "Obesity drug pipeline competitive intelligence from ClinicalTrials.gov. Counts molecules by "
        "mechanism and phase, simulates the number of new US approvals by 2030 with Bayesian phase-success "
        "rates, models trial durations with survival analysis, and tracks comorbidity themes in new trials."),
    "05-omnichannel-hcp-engagement": (
        "Omnichannel engagement of prescribers (HCPs). Real data: CMS Open Payments promotional payments "
        "joined to Medicare Part D prescribing, giving reach, share of voice and a propensity-matched "
        "estimate of payment effects. Simulated data with a known truth: multi-touch attribution, uplift "
        "modelling, email cadence and a contextual bandit for next-best action, each graded against the truth."),
    "06-marketing-mix-model": (
        "Bayesian marketing mix model on a simulated brand with known true channel effects: adstock "
        "carry-over, saturation, regional baselines, calibration to lift tests, and budget optimisation. "
        "Because the truth is known, the recovered ROI and the optimiser's promises are checked against it."),
    "07-test-and-control-experiments": (
        "Test-and-control analysis on real Medicaid data: states that removed or restored coverage of "
        "anti-obesity GLP-1 medicines, measured with synthetic control and difference-in-differences, "
        "placebo tests, an event study for spillover into diabetes GLP-1 use, and power analysis for "
        "designing a state pilot. Use is measured as prescriptions per 10,000 Medicaid prescriptions of any "
        "drug. Pilot sizes are the smallest lift detectable with 80% power."),
}

AUDIENCES = {
    "brand_lead": {
        "title": "Brief for a brand or commercial lead",
        "ask": ("Write for a US pharma brand lead who has five minutes. Structure: a one-sentence headline; "
                "'What we found' with three or four bullets; 'What to do' with two or three concrete actions; "
                "'What could change this' with one or two bullets. 150 to 230 words."),
    },
    "analytics_review": {
        "title": "Brief for an analytics reviewer",
        "ask": ("Write for a sceptical senior analyst reviewing the work. Structure: 'Estimated' (what was "
                "measured, with the key numbers); 'How' (the methods, one bullet each); 'How far to trust it' "
                "(validation results, interval widths, known weaknesses). 160 to 250 words."),
    },
}

SYSTEM = """You write short, plain-English briefs from analysis results.

Rules:
- Use only numbers from the fact sheet. Every number you write must be followed by the fact it came from in square brackets, e.g. "fell 88% [F3]" or "$21.3B [F9]".
- You may convert a fact to a percentage (0.568 -> 57%), round it, drop a minus sign when the verb says the direction ("fell 88%"), or show it in thousands, millions or billions.
- Do not add, subtract, multiply or average facts. Do not write any number that is not in the fact sheet, including dates and counts, unless it appears in the project description.
- Prefer short sentences and ordinary words. No hype ("game-changing", "unlock", "leverage"). No emojis.
- Output Markdown only, starting with the headline. No preamble."""


# ---------------------------------------------------------------- fact sheet

@dataclass
class Fact:
    fid: str
    key: str
    value: float


def flatten(obj, prefix: str = "") -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += flatten(v, f"{prefix}[{i}]")
    elif isinstance(obj, bool) or obj is None:
        pass
    elif isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.append((prefix, float(obj)))
    elif isinstance(obj, str):
        try:
            out.append((prefix, float(obj.replace(",", ""))))
        except ValueError:
            pass
    return out


def fact_sheet(summary: dict) -> list[Fact]:
    return [Fact(f"F{i}", k, v) for i, (k, v) in enumerate(flatten(summary), 1)]


def render_facts(facts: list[Fact]) -> str:
    def fmt(v: float) -> str:
        return f"{v:,.0f}" if abs(v) >= 1000 and v == int(v) else f"{v:.6g}"
    return "\n".join(f"[{f.fid}] {f.key} = {fmt(f.value)}" for f in facts)


# ---------------------------------------------------------------- checking

NUM = re.compile(
    r"(?P<cur>\$)?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s?(?P<unit>%|pp\b|percentage points?|points?|bn\b|billion|B\b|million|mn\b|M\b|k\b|K\b|thousand)?"
    r"(?P<cite>(?:\s?\[F\d+\])+)?")
CITE = re.compile(r"\[F(\d+)\]")
SCALE = {"bn": 1e9, "billion": 1e9, "B": 1e9, "million": 1e6, "mn": 1e6, "M": 1e6, "k": 1e3, "K": 1e3, "thousand": 1e3}


@dataclass
class NumberCheck:
    text: str
    value: float
    cited: list[str]
    status: str            # grounded | uncited | ungrounded | exempt
    matched: str | None = None


@dataclass
class Report:
    checks: list[NumberCheck] = field(default_factory=list)

    @property
    def ungrounded(self) -> list[NumberCheck]:
        return [c for c in self.checks if c.status == "ungrounded"]

    def counts(self) -> dict:
        out = {"grounded": 0, "uncited": 0, "ungrounded": 0, "exempt": 0}
        for c in self.checks:
            out[c.status] += 1
        return out


def _decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def matches(written: str, unit: str | None, cur: bool, fact: float) -> bool:
    """Does a number as written in a brief match a fact, at the precision written?"""
    x = float(written.replace(",", ""))
    tol = 0.5 * 10 ** (-_decimals(written)) * 1.001
    bases = (fact, abs(fact))
    if unit in SCALE:
        s = SCALE[unit]
        return any(abs(b * k - x * s) <= tol * s for b in bases for k in (1, 1e3, 1e6, 1e9))
    if unit == "%" or unit is None:
        return any(abs(c - x) <= tol for b in bases for c in (b, b * 100))
    return False


def check(text: str, facts: list[Fact], context: str = "") -> Report:
    by_id = {f.fid: f for f in facts}
    ctx_numbers = {m.group("num").replace(",", "") for m in NUM.finditer(context)}
    rep = Report()
    for m in NUM.finditer(text):
        raw, unit, cur = m.group("num"), m.group("unit"), bool(m.group("cur"))
        start = m.start()
        # skip the citation markers themselves and Markdown list numbers
        if text[max(0, start - 2):start].endswith("[F") or text[max(0, start - 1):start] == "F":
            continue
        cited = [f"F{c}" for c in CITE.findall(m.group("cite") or "")]
        if not cited:  # a citation a few words later still counts, if no other number comes first
            ahead = text[m.end(): m.end() + 60]
            cm = CITE.search(ahead)
            if cm and not re.search(r"\d", re.sub(r"\[F\d+\]", "", ahead[: cm.start()])):
                cited = [f"F{cm.group(1)}"]
        value = float(raw.replace(",", ""))
        if unit and unit.startswith(("point", "percentage")):
            unit = None
        # exempt: years and quarters, and small counts written as numerals
        if (unit is None and not cur and value == int(value) and (1990 <= value <= 2040 or value <= 12)
                and not cited) or raw.replace(",", "") in ctx_numbers:
            rep.checks.append(NumberCheck(m.group(0).strip(), value, cited, "exempt"))
            continue
        status, hit = "ungrounded", None
        for fid in cited:
            if fid in by_id and matches(raw, unit, cur, by_id[fid].value):
                status, hit = "grounded", fid
                break
        if status == "ungrounded":
            for f in facts:
                if matches(raw, unit, cur, f.value):
                    status, hit = "uncited", f.fid
                    break
        rep.checks.append(NumberCheck(m.group(0).strip(), value, cited, status, hit))
    return rep


def strip_citations(text: str) -> str:
    return re.sub(r"\s?\[F\d+\]", "", text)


def drop_ungrounded_sentences(text: str, rep: Report) -> tuple[str, list[str]]:
    bad = {c.text for c in rep.ungrounded}
    removed, keep_lines = [], []
    for line in text.splitlines():
        parts = re.split(r"(?<=[.!?])\s+", line)
        kept = []
        for p in parts:
            if any(b in p for b in bad):
                removed.append(p.strip())
            else:
                kept.append(p)
        new = " ".join(kept).rstrip()
        if new.strip() in ("-", "*", ""):
            if line.strip() == "":
                keep_lines.append("")
            continue
        keep_lines.append(new)
    return "\n".join(keep_lines), removed


# ---------------------------------------------------------------- generation

def ollama_chat(messages: list[dict], model: str, host: str) -> str:
    import requests

    r = requests.post(f"{host}/api/chat", json={
        "model": model, "messages": messages, "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192, "seed": 7},
    }, timeout=1800)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def draft(project: str, audience: str, model: str, host: str, attempts: int = 3) -> dict:
    summary = json.loads((ROOT / project / "outputs" / "tables" / "summary.json").read_text())
    facts = fact_sheet(summary)
    ctx = CONTEXT[project]
    user = (f"Project description:\n{ctx}\n\nFact sheet (from summary.json):\n{render_facts(facts)}\n\n"
            f"Task: {AUDIENCES[audience]['ask']}")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    history = []
    text, rep = "", Report()
    for attempt in range(1, attempts + 1):
        text = ollama_chat(messages, model, host)
        rep = check(text, facts, ctx)
        history.append({"attempt": attempt, **rep.counts(), "ungrounded": [c.text for c in rep.ungrounded]})
        if not rep.ungrounded:
            break
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": (
                         "These numbers do not match any fact in the fact sheet: "
                         + "; ".join(c.text for c in rep.ungrounded)
                         + ". Rewrite the whole brief. Use only fact-sheet numbers, each with its [F] citation, "
                           "or remove the claim.")}]
    removed: list[str] = []
    if rep.ungrounded:
        text, removed = drop_ungrounded_sentences(text, rep)
        rep = check(text, facts, ctx)
    return {"project": project, "audience": audience, "model": model, "facts": len(facts),
            "attempts": history, "removed_sentences": removed, "final": rep.counts(),
            "checks": [c.__dict__ for c in rep.checks], "text_cited": text}


def write(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    a = result["audience"]
    c = result["final"]
    head = (f"<!-- Drafted by the open-weights model {result['model']} from outputs/tables/summary.json on "
            f"{dt.date.today().isoformat()}. Numbers checked: {c['grounded'] + c['uncited']} traced to the "
            f"summary, {c['ungrounded']} unsupported, {len(result['removed_sentences'])} sentences removed. "
            f"Audit trail: {a}.cited.md and grounding.json. -->\n\n")
    (out_dir / f"{a}.md").write_text(f"# {AUDIENCES[a]['title']}\n\n" + head + strip_citations(result["text_cited"]) + "\n")
    (out_dir / f"{a}.cited.md").write_text(result["text_cited"] + "\n")


def index() -> str:
    """A table of every brief and how its numbers fared, for the repository README."""
    rows = ["| Project | Brief | Numbers traced to results | Unsupported | Sentences removed | Drafts needed |",
            "|---|---|---|---|---|---|"]
    tot = {"ok": 0, "bad": 0, "removed": 0}
    for project in sorted(CONTEXT):
        g = ROOT / project / "outputs" / "briefs" / "grounding.json"
        if not g.exists():
            continue
        for r in json.loads(g.read_text()):
            c = r["final"]
            ok = c["grounded"] + c["uncited"]
            tot["ok"] += ok
            tot["bad"] += c["ungrounded"]
            tot["removed"] += len(r["removed_sentences"])
            first = r["attempts"][0]
            rows.append(f"| {project[:2]} | [{AUDIENCES[r['audience']]['title'].replace('Brief for ', '')}]"
                        f"({project}/outputs/briefs/{r['audience']}.md) | {ok} | {c['ungrounded']} | "
                        f"{len(r['removed_sentences'])} | {len(r['attempts'])} (first draft: {first['ungrounded']} unsupported) |")
    rows.append(f"| | **All** | **{tot['ok']}** | **{tot['bad']}** | **{tot['removed']}** | |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("projects", nargs="*", default=sorted(CONTEXT))
    ap.add_argument("--model", default=os.environ.get("BRIEF_MODEL", "qwen2.5:7b"))
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--check", metavar="BRIEF", help="only check an existing brief against the first project's facts")
    ap.add_argument("--index", action="store_true", help="print a Markdown table of all briefs' checks")
    args = ap.parse_args(argv)

    if args.index:
        print(index())
        return 0

    if args.check:
        project = args.projects[0]
        facts = fact_sheet(json.loads((ROOT / project / "outputs/tables/summary.json").read_text()))
        rep = check(Path(args.check).read_text(), facts, CONTEXT.get(project, ""))
        print(json.dumps(rep.counts()), *[f"  UNGROUNDED {c.text}" for c in rep.ungrounded], sep="\n")
        return 1 if rep.ungrounded else 0

    for project in args.projects:
        out_dir = ROOT / project / "outputs" / "briefs"
        results = []
        for audience in AUDIENCES:
            res = draft(project, audience, args.model, args.host)
            write(res, out_dir)
            results.append({k: v for k, v in res.items() if k != "text_cited"})
            print(f"{project} / {audience}: {res['final']} after {len(res['attempts'])} attempt(s); "
                  f"{len(res['removed_sentences'])} sentence(s) removed", flush=True)
        (out_dir / "grounding.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
