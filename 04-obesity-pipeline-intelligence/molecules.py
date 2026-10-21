"""Reference list of obesity drug candidates: aliases, sponsor and mechanism.

ClinicalTrials.gov stores interventions as free text ('LY3437943', 'Retatrutide',
'Tirzepatide 15 mg'...). This table maps the names and code numbers sponsors use
to one molecule, so trials can be grouped by mechanism.

Sources: company pipeline pages and press releases, ClinicalTrials.gov records.
Unmatched interventions are reported separately rather than guessed.
"""

import re

# mechanism labels, used as plot categories
GLP1 = "GLP-1 agonist (injectable)"
GLP1_ORAL_SM = "Oral small-molecule GLP-1"
GLP1_ORAL_PEP = "Oral peptide GLP-1"
DUAL_GIP = "GIP / GLP-1 dual agonist"
DUAL_GCG = "GLP-1 / glucagon dual agonist"
TRIPLE = "GIP / GLP-1 / glucagon triple agonist"
AMYLIN = "Amylin-based (alone or with GLP-1)"
GIPR_ANT = "GLP-1 agonist + GIP-receptor antagonist"
MUSCLE = "Muscle-preserving (activin / myostatin)"
MC4R = "MC4R agonist"
OLDER = "Older oral agents"

MOLECULES = [
    # name,                  aliases (lower-case substrings),                         sponsor,                       mechanism
    ("semaglutide (inj.)",   ["semaglutide", "nn9536", "wegovy", "ozempic"],           "Novo Nordisk",                GLP1),
    ("oral semaglutide",     ["oral semaglutide", "rybelsus", "nn9932"],               "Novo Nordisk",                GLP1_ORAL_PEP),
    ("liraglutide",          ["liraglutide", "saxenda", "victoza"],                    "Novo Nordisk",                GLP1),
    ("ecnoglutide",          ["ecnoglutide", "xw003"],                                 "Sciwind Biosciences",         GLP1),
    ("efpeglenatide",        ["efpeglenatide"],                                        "Hanmi",                       GLP1),
    ("MET-097i",             ["met-097", "met097"],                                    "Metsera / Pfizer",            GLP1),
    ("orforglipron",         ["orforglipron", "ly3502970"],                            "Eli Lilly",                   GLP1_ORAL_SM),
    ("danuglipron",          ["danuglipron", "pf-06882961"],                           "Pfizer (discontinued)",       GLP1_ORAL_SM),
    ("aleniglipron",         ["aleniglipron", "gsbr-1290"],                            "Structure Therapeutics",      GLP1_ORAL_SM),
    ("CT-996",               ["ct-996"],                                               "Roche",                       GLP1_ORAL_SM),
    ("AZD5004",              ["azd5004", "ecc5004"],                                   "AstraZeneca",                 GLP1_ORAL_SM),
    ("HRS-7535",             ["hrs-7535", "hrs7535"],                                  "Hengrui / Kailera",           GLP1_ORAL_SM),
    ("tirzepatide",          ["tirzepatide", "ly3298176", "mounjaro", "zepbound"],     "Eli Lilly",                   DUAL_GIP),
    ("VK2735",               ["vk2735"],                                               "Viking Therapeutics",         DUAL_GIP),
    ("CT-388",               ["ct-388"],                                               "Roche",                       DUAL_GIP),
    ("HRS9531",              ["hrs9531", "hrs-9531", "kai-9531"],                      "Hengrui / Kailera",           DUAL_GIP),
    ("survodutide",          ["survodutide", "bi 456906", "bi456906"],                 "Boehringer Ingelheim",        DUAL_GCG),
    ("mazdutide",            ["mazdutide", "ibi362", "ly3305677"],                     "Innovent / Eli Lilly",        DUAL_GCG),
    ("pemvidutide",          ["pemvidutide", "alt-801"],                               "Altimmune",                   DUAL_GCG),
    ("efinopegdutide",       ["efinopegdutide", "mk-6024"],                            "Merck",                       DUAL_GCG),
    ("dapiglutide",          ["dapiglutide"],                                          "Zealand Pharma",              DUAL_GCG),
    ("retatrutide",          ["retatrutide", "ly3437943"],                             "Eli Lilly",                   TRIPLE),
    ("cagrilintide",         ["cagrilintide", "cagrisema", "nn9838"],                  "Novo Nordisk",                AMYLIN),
    ("amycretin",            ["amycretin"],                                            "Novo Nordisk",                AMYLIN),
    ("petrelintide",         ["petrelintide", "zp8396"],                               "Zealand Pharma / Roche",      AMYLIN),
    ("eloralintide",         ["eloralintide", "ly3841136"],                            "Eli Lilly",                   AMYLIN),
    ("MET-233i",             ["met-233", "met233"],                                    "Metsera / Pfizer",            AMYLIN),
    ("maridebart cafraglutide", ["maridebart", "amg 133", "amg133", "maritide"],       "Amgen",                       GIPR_ANT),
    ("bimagrumab",           ["bimagrumab", "bym338", "ly3986300"],                    "Eli Lilly",                   MUSCLE),
    ("apitegromab",          ["apitegromab", "srk-015"],                               "Scholar Rock",                MUSCLE),
    ("trevogrumab / garetosmab", ["trevogrumab", "garetosmab", "regn1033", "regn2477"], "Regeneron",                  MUSCLE),
    ("setmelanotide",        ["setmelanotide", "imcivree", "rm-493"],                  "Rhythm Pharmaceuticals",      MC4R),
    ("phentermine/topiramate", ["qsymia", "phentermine"],                              "Vivus",                       OLDER),
    ("naltrexone/bupropion", ["contrave", "naltrexone"],                               "Currax",                      OLDER),
    ("orlistat",             ["orlistat", "xenical"],                                  "various",                     OLDER),
]

# Order matters: 'oral semaglutide' must be tested before 'semaglutide'.
MOLECULES.sort(key=lambda m: -max(len(a) for a in m[1]))


def classify(text: str) -> list[tuple[str, str, str]]:
    """Return every (molecule, sponsor, mechanism) whose alias appears in text."""
    t = (text or "").lower()
    hits = [(name, sponsor, mech) for name, aliases, sponsor, mech in MOLECULES
            if any(a in t for a in aliases)]
    names = {h[0] for h in hits}
    # 'oral semaglutide' and 'CagriSema' both contain 'semaglutide'
    if "oral semaglutide" in names or ("cagrilintide" in names and "cagrisema" in t):
        if not re.search(r"(?<!oral )(?<!cagri)semaglutide(?! *\(oral)", t.replace("cagrisema", "")):
            hits = [h for h in hits if h[0] != "semaglutide (inj.)"]
    return hits
