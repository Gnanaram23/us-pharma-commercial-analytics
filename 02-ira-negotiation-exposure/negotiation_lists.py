"""Drugs selected for Medicare price negotiation, with published prices.

Sources
-------
IPAY 2026 (first cycle): CMS fact sheet, Aug 15 2024, "Medicare Drug Price
  Negotiation Program: Negotiated Prices for Initial Price Applicability Year 2026".
  Net savings: ~$6B (22%) had prices applied in 2023, on $56.2B 2023 gross Part D cost.
IPAY 2027 (second cycle): CMS press release, Nov 25 2025. Net savings ~$12B (44%)
  on 2024 spending; $42.5B 2024 gross Part D cost.
IPAY 2028 (third cycle, Part B and Part D): CMS fact sheet, Jan 2026. Prices
  due by Nov 30 2026 and not yet published when this analysis was written.
List prices and maximum fair prices (MFP) are per 30-day supply as published by
CMS (list price year: 2023 for IPAY 2026, 2024 for IPAY 2027).

`match` holds lower-case prefixes of the CMS 'Brnd_Name' values that belong to
each selected drug (CMS groups all dosage forms of one active moiety).
"""

SELECTED = [
    # cycle, drug,                    match prefixes,                                   list $,   MFP $,   condition
    (2026, "Eliquis",                 ["eliquis"],                                         521,     231,  "Blood clots"),
    (2026, "Jardiance",               ["jardiance"],                                       573,     197,  "Diabetes, heart failure"),
    (2026, "Xarelto",                 ["xarelto"],                                         517,     197,  "Blood clots"),
    (2026, "Januvia",                 ["januvia"],                                         527,     113,  "Diabetes"),
    (2026, "Farxiga",                 ["farxiga"],                                         556,   178.5,  "Diabetes, heart failure, CKD"),
    (2026, "Entresto",                ["entresto"],                                        628,     295,  "Heart failure"),
    (2026, "Enbrel",                  ["enbrel"],                                         7106,    2355,  "Rheumatoid arthritis, psoriasis"),
    (2026, "Imbruvica",               ["imbruvica"],                                     14934,    9319,  "Blood cancers"),
    (2026, "Stelara",                 ["stelara"],                                       13836,    4695,  "Psoriasis, Crohn's, UC"),
    (2026, "NovoLog / Fiasp",         ["novolog", "fiasp"],                                495,     119,  "Diabetes"),
    (2027, "Ozempic / Rybelsus / Wegovy", ["ozempic", "rybelsus", "wegovy"],               959,     274,  "Diabetes, CV risk, obesity"),
    (2027, "Trelegy Ellipta",         ["trelegy"],                                         654,     175,  "Asthma, COPD"),
    (2027, "Xtandi",                  ["xtandi"],                                        13480,    7004,  "Prostate cancer"),
    (2027, "Pomalyst",                ["pomalyst"],                                      21744,    8650,  "Multiple myeloma"),
    (2027, "Ofev",                    ["ofev"],                                          12622,    6350,  "Pulmonary fibrosis"),
    (2027, "Ibrance",                 ["ibrance"],                                       15741,    7871,  "Breast cancer"),
    (2027, "Linzess",                 ["linzess"],                                         539,     136,  "IBS-C, constipation"),
    (2027, "Calquence",               ["calquence"],                                     14228,    8600,  "Blood cancers"),
    (2027, "Austedo / Austedo XR",    ["austedo"],                                        6623,    4093,  "Huntington's chorea, TD"),
    (2027, "Breo Ellipta",            ["breo"],                                            397,      67,  "Asthma, COPD"),
    (2027, "Tradjenta",               ["tradjenta"],                                       488,      78,  "Diabetes"),
    (2027, "Xifaxan",                 ["xifaxan"],                                        2696,    1000,  "Hepatic encephalopathy, IBS-D"),
    (2027, "Vraylar",                 ["vraylar"],                                        1376,     770,  "Schizophrenia, bipolar, MDD"),
    (2027, "Janumet / Janumet XR",    ["janumet"],                                         526,      80,  "Diabetes"),
    (2027, "Otezla",                  ["otezla"],                                         4722,    1650,  "Psoriasis, psoriatic arthritis"),
    (2028, "Trulicity",               ["trulicity"],                                      None,    None,  "Diabetes"),
    (2028, "Biktarvy",                ["biktarvy"],                                       None,    None,  "HIV"),
    (2028, "Orencia",                 ["orencia"],                                        None,    None,  "RA, psoriatic arthritis"),
    (2028, "Cosentyx",                ["cosentyx"],                                       None,    None,  "Psoriasis, psoriatic arthritis"),
    (2028, "Erleada",                 ["erleada"],                                        None,    None,  "Prostate cancer"),
    (2028, "Kisqali",                 ["kisqali"],                                        None,    None,  "Breast cancer"),
    (2028, "Entyvio",                 ["entyvio"],                                        None,    None,  "Crohn's, UC"),
    (2028, "Verzenio",                ["verzenio"],                                       None,    None,  "Breast cancer"),
    (2028, "Botox",                   ["botox"],                                          None,    None,  "Migraine, spasticity"),
    (2028, "Lenvima",                 ["lenvima"],                                        None,    None,  "Thyroid, liver, kidney cancers"),
    (2028, "Xolair",                  ["xolair"],                                         None,    None,  "Asthma, urticaria"),
    (2028, "Rexulti",                 ["rexulti"],                                        None,    None,  "Depression, schizophrenia"),
    (2028, "Xeljanz",                 ["xeljanz"],                                        None,    None,  "RA, PsA, UC"),
    (2028, "Anoro Ellipta",           ["anoro"],                                          None,    None,  "COPD"),
    (2028, "Cimzia",                  ["cimzia"],                                         None,    None,  "Crohn's, psoriasis, RA"),
]

CMS_CYCLE_SAVINGS = {
    # cycle: (net savings $, savings %, base year, gross Part D cost in base year)
    2026: (6.0e9, 0.22, 2023, 56.2e9),
    2027: (12.0e9, 0.44, 2024, 42.5e9),
}
