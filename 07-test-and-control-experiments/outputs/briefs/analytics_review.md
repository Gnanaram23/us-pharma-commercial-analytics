# Brief for an analytics reviewer

*Drafted by the open-weights model qwen2.5:14b on 2026-11-11 from this project's results. 20 of 20 numbers traced to the results; 0 unsupported sentences removed. Audit trail: [analytics_review.cited.md](analytics_review.cited.md), [grounding.json](grounding.json).*

> **Review note.** This draft passed the automatic checks but a review found problems:
> - "Use recovered to 57% of the pre-gap level" is wrong: it recovered to 57% of the synthetic counterfactual, the level continuous coverage would have reached.
> - "DiD was adjusted for each state's pre-trend" describes the pilot design in section 5, not the coverage-removal estimates.


**Estimated:** North Carolina's GLP-1 use for obesity fell 88% after coverage ended in 2025 Q4, from 165.1 to 18.7 prescriptions per 10,000 Medicaid prescriptions. California, Pennsylvania, and New Hampshire saw drops of 75–93% in their first quarter without coverage. When coverage resumed in North Carolina, use recovered to 57% of the pre-gap level in 2026 Q1.

**How:** The analysis used synthetic control and difference-in-differences (DiD) methods. Synthetic control was fitted on log rates with the pre-period mean removed, allowing for accurate matching even in states with high GLP-1 use. DiD was adjusted for each state's pre-trend.

**How far to trust it:** Placebo tests showed no spurious drops larger than 34% or gains larger than 15%, validating the real impact of coverage changes. The 95% confidence interval for the spillover effect on diabetes GLP-1 use ranged from −5.9% to +2.4%, indicating no evidence of substitution. The synthetic control method tracked the pre-period within 3.3% log RMSE, confirming its accuracy. However, the analysis relies on suppressed counts for small states, and the newest quarter's data may be revised upwards.
