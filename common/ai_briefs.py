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
4. A number can be right and the sentence around it wrong ("NC had the largest
   drop" when another state's was larger). So a second pass reads each
   sentence against the project's write-up (its README, which a person
   checked) and the fact sheet, and removes any claim the sources do not
   support. This check is itself done by a language model, so it reduces
   errors rather than ruling them out; every removal is logged.

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
    python -m common.ai_briefs 07-test-and-control-experiments --model qwen2.5:14b
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

You are given the project's write-up and a numbered fact sheet. The write-up says what the numbers mean; follow its interpretation exactly. Do not reinterpret a statistic, do not compare items unless the write-up or the numbers make the comparison, and do not add recommendations the write-up does not make.

Rules:
- Use only numbers from the fact sheet. Every number you write must be followed by the fact it came from in square brackets, e.g. "fell 88% [F3]" or "$21.3B [F9]".
- You may convert a fact to a percentage (0.568 -> 57%), round it, drop a minus sign when the verb says the direction ("fell 88%"), or show it in thousands, millions or billions.
- Do not add, subtract, multiply or average facts. Do not write any number that is not in the fact sheet, including dates and counts, unless it appears in the project description.
- Prefer short sentences and ordinary words. No hype ("game-changing", "unlock", "leverage"). No emojis.
- Do not start with a title or heading of your own; start with the one-sentence headline in bold.
- Output Markdown only. No preamble."""

JUDGE = """You check a short brief against its sources: a project write-up and a fact sheet.

The write-up's findings, recommendations and limitations all count as sources. A sentence is supported when its meaning matches something the sources say, even if the wording, order or emphasis differs, or it combines two points the sources make separately.

Flag a sentence only when it clearly does one of these:
- states something the sources do not say at all;
- contradicts the sources, or interprets a statistic differently from the write-up;
- attaches a number to the wrong thing (the wrong state, channel, plan or year);
- claims a ranking ("largest", "highest", "best") that the numbers do not show.

When in doubt, treat the sentence as supported.

Reply with JSON only: {"unsupported": [{"n": <sentence number>, "why": "<short reason>"}]}. Use an empty list if every sentence is supported."""


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


def writeup(project: str, limit: int = 16000) -> str:
    """The project README without images, links, term glossary or set-up steps: the human-checked
    interpretation of the numbers, including its recommendations and limitations."""
    text = (ROOT / project / "README.md").read_text()
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)\n?", "", text)          # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)          # links -> their text
    text = re.sub(r"\*\*Terms used below\*\*.*?(?=\n---|\n## )", "", text, flags=re.S)
    text = text.split("## Reproduce")[0]
    return re.sub(r"\n{3,}", "\n\n", text)[:limit]


# ---------------------------------------------------------------- checking

NUM = re.compile(
    r"(?P<cur>\$)?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s?(?P<unit>%|pp\b|percentage points?|points?|bn\b|billion|B\b|million|mn\b|M\b|k\b|K\b|thousand)?"
    r"(?P<cite>(?:\s?\[F\d+\])+)?")
CITE = re.compile(r"\[F(\d+)\]")
CITE_GROUP = re.compile(r"\s?\[F\d+(?:\s*,\s*F?\d+)*\]")
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
        # otherwise exempt: years, small counts written as numerals, and numbers stated in the write-up
        if status == "ungrounded" and ((unit is None and not cur and value == int(value)
                                        and (1990 <= value <= 2040 or value <= 12) and not cited)
                                       or raw.replace(",", "") in ctx_numbers):
            status = "exempt"
        rep.checks.append(NumberCheck(m.group(0).strip(), value, cited, status, hit))
    return rep


def strip_citations(text: str) -> str:
    return CITE_GROUP.sub("", text)


def split_citation_groups(text: str) -> str:
    """'[F4, F6]' -> '[F4][F6]', so each cited fact is checked."""
    return CITE_GROUP.sub(lambda m: " " + "".join(f"[F{n}]" for n in re.findall(r"\d+", m.group(0))), text)


HEADING = re.compile(r"^\s*(#+\s.*|(?:[-*]\s*)?\*\*[^*]*[^*.!?]\*\*:?|[*_]?[A-Z][^.!?]{0,40}:[*_]?)\s*$")
# '# x', a bold label that does not end a sentence, or a short label ending in a colon ("What to do:")


def is_heading(line: str) -> bool:
    return bool(HEADING.match(line))


def tidy(text: str) -> str:
    """Remove headings left with nothing under them, and runs of blank lines."""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        if is_heading(line):
            rest = [l for l in lines[i + 1:] if l.strip()]
            nxt = rest[0] if rest else ""
            if not rest or (is_heading(nxt) and len(nxt) - len(nxt.lstrip()) <= len(line) - len(line.lstrip())):
                continue
        out.append(line)
    out = [l[1:] if re.match(r"^\*[^\s*]", l) and l.count("*") % 2 else l for l in out]  # orphaned italic marker
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


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


def sentences(text: str) -> list[tuple[int, int, str]]:
    """(line index, position, sentence) for every sentence with content, skipping headings."""
    out = []
    for li, line in enumerate(text.splitlines()):
        body = line.strip()
        if not body or is_heading(body):
            continue
        for pi, sent in enumerate(re.split(r"(?<=[.!?])\s+(?=[A-Z*(])", body)):
            if len(re.sub(r"[^A-Za-z]", "", sent)) > 12:
                out.append((li, pi, sent.strip()))
    return out


def judge(text: str, facts: list[Fact], wu: str, model: str, host: str) -> list[dict]:
    sents = sentences(text)
    listing = "\n".join(f"{n}. {strip_citations(s)}" for n, (_, _, s) in enumerate(sents, 1))
    msg = (f"Project write-up:\n{wu}\n\nFact sheet:\n{render_facts(facts)}\n\nBrief, one sentence per line:\n{listing}")
    raw = ollama_chat([{"role": "system", "content": JUDGE}, {"role": "user", "content": msg}], model, host, fmt="json",
                      temperature=0.0)
    try:
        flagged = json.loads(raw).get("unsupported", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    out = []
    for f in flagged:
        try:
            n = int(f.get("n"))
        except (TypeError, ValueError, AttributeError):
            continue
        if 1 <= n <= len(sents):
            li, pi, s = sents[n - 1]
            out.append({"line": li, "part": pi, "sentence": s, "why": str(f.get("why", ""))[:200]})
    return out


def remove_sentences(text: str, flagged: list[dict]) -> str:
    lines = text.splitlines()
    for li in sorted({f["line"] for f in flagged}, reverse=True):
        drop = {f["part"] for f in flagged if f["line"] == li}
        line = lines[li]
        lead = re.match(r"^(\s*(?:[-*]|\d+\.)?\s*)", line).group(1)
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z*(])", line.strip())
        # sentences() numbers parts of the stripped line, bullet marker included in the first part
        kept = [p for i, p in enumerate(parts) if i not in drop]
        if not kept or all(re.fullmatch(r"[-*]?\s*(\*\*[^*]+\*\*:?)?\s*", p) for p in kept):
            lines[li] = None
        else:
            new = " ".join(kept)
            lines[li] = new if new.lstrip().startswith(("-", "*")) or not lead.strip() else lead + new.lstrip("-* ")
    return "\n".join(l for l in lines if l is not None)


# ---------------------------------------------------------------- generation

def ollama_chat(messages: list[dict], model: str, host: str, fmt: str | None = None, temperature: float = 0.2) -> str:
    import requests

    body = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_ctx": 16384, "seed": 7}}
    if fmt:
        body["format"] = fmt
    r = requests.post(f"{host}/api/chat", json=body, timeout=3600)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def draft(project: str, audience: str, model: str, host: str, attempts: int = 3) -> dict:
    summary = json.loads((ROOT / project / "outputs" / "tables" / "summary.json").read_text())
    facts = fact_sheet(summary)
    ctx = CONTEXT[project]
    wu = writeup(project)
    user = (f"Project description:\n{ctx}\n\nProject write-up:\n{wu}\n\nFact sheet (from summary.json):\n"
            f"{render_facts(facts)}\n\nTask: {AUDIENCES[audience]['ask']}")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    history = []
    text, rep = "", Report()
    for attempt in range(1, attempts + 1):
        text = split_citation_groups(ollama_chat(messages, model, host))
        rep = check(text, facts, ctx + " " + wu)
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
    flagged = judge(text, facts, wu, model, host)
    first_flags = list(flagged)
    if flagged:
        # one repair round: the drafter sees the objections and rewrites the whole brief
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": (
                         "A reviewer found these sentences unsupported by the write-up:\n"
                         + "\n".join(f"- \"{strip_citations(f['sentence'])}\" ({f['why']})" for f in flagged)
                         + "\nRewrite the whole brief in the same structure. Fix or drop those claims, keep every "
                           "section filled using only what the write-up says, and keep the [F] citations.")}]
        text = split_citation_groups(ollama_chat(messages, model, host))
        rep = check(text, facts, ctx + " " + wu)
        history.append({"attempt": len(history) + 1, "after_review": True, **rep.counts(),
                        "ungrounded": [c.text for c in rep.ungrounded]})
        if rep.ungrounded:
            text, more = drop_ungrounded_sentences(text, rep)
            removed += more
        flagged = judge(text, facts, wu, model, host)
    text = tidy(remove_sentences(text, flagged))
    rep = check(text, facts, ctx + " " + wu)
    return {"project": project, "audience": audience, "model": model, "facts": len(facts),
            "attempts": history, "removed_sentences": removed, "claims_removed": flagged,
            "claims_flagged_first_review": first_flags, "final": rep.counts(),
            "checks": [c.__dict__ for c in rep.checks], "text_cited": text}


def write(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    a = result["audience"]
    c = result["final"]
    n_ok = c["grounded"] + c["uncited"] + c["exempt"]
    head = (f"*Drafted by the open-weights model {result['model']} on {dt.date.today().isoformat()} from this "
            f"project's results. {n_ok} of {n_ok + c['ungrounded']} numbers traced to the results; "
            f"{len(result['removed_sentences']) + len(result['claims_removed'])} unsupported sentences removed. "
            f"Audit trail: [{a}.cited.md]({a}.cited.md), [grounding.json](grounding.json).*\n\n")
    (out_dir / f"{a}.md").write_text(f"# {AUDIENCES[a]['title']}\n\n" + head + strip_citations(result["text_cited"]) + "\n")
    (out_dir / f"{a}.cited.md").write_text(result["text_cited"] + "\n")


def index() -> str:
    """A table of every brief and how its checks went, for the repository README."""
    rows = ["| Project | Brief | Numbers traced to results | Unsupported numbers left | Sentences removed (numbers / claims) | Drafts |",
            "|---|---|---|---|---|---|"]
    tot = {"ok": 0, "bad": 0, "rn": 0, "rc": 0}
    names = {"brand_lead": "Brand lead", "analytics_review": "Analytics reviewer"}
    for project in sorted(CONTEXT):
        g = ROOT / project / "outputs" / "briefs" / "grounding.json"
        if not g.exists():
            continue
        for r in json.loads(g.read_text()):
            c = r["final"]
            ok = c["grounded"] + c["uncited"] + c["exempt"]
            rn, rc = len(r["removed_sentences"]), len(r.get("claims_removed", []))
            tot["ok"] += ok
            tot["bad"] += c["ungrounded"]
            tot["rn"] += rn
            tot["rc"] += rc
            rows.append(f"| {project[:2]} | [{names[r['audience']]}]({project}/outputs/briefs/{r['audience']}.md) | "
                        f"{ok} | {c['ungrounded']} | {rn} / {rc} | {len(r['attempts'])} |")
    rows.append(f"| | **All** | **{tot['ok']}** | **{tot['bad']}** | **{tot['rn']} / {tot['rc']}** | |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("projects", nargs="*", default=sorted(CONTEXT))
    ap.add_argument("--model", default=os.environ.get("BRIEF_MODEL", "qwen2.5:14b"))
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
