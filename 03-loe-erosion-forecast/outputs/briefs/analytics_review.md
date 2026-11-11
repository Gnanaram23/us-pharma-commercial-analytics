# Brief for an analytics reviewer

*Drafted by the open-weights model qwen2.5:14b on 2026-11-11 from this project's results. 13 of 13 numbers traced to the results; 1 unsupported sentences removed. Audit trail: [analytics_review.cited.md](analytics_review.cited.md), [grounding.json](grounding.json).*

**Loss of Exclusivity: How Fast Brands Erode, and What Happens to Eliquis in 2028**

Some brands retain 64%–78%. Eliquis, made free to Medicaid in 2026, could retain 1.9 million prescriptions in 2029.

**How:** 
- Used Medicaid State Drug Utilization Data (2017–Q1 2026).
- Fit brand share erosion curves with a Bayesian hierarchical model.
- Validated models on Brilinta and Entresto.

**How far to trust it:** 
- Mean absolute error for Entresto was 6 points, for Brilinta 19 points.
- Conformal prediction intervals had average half-widths of 26–32 points.
- Known weaknesses: Medicaid data may not reflect commercial insurance behavior [project write-up].
