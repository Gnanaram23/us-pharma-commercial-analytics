# Brief for an analytics reviewer

*Drafted by the open-weights model qwen2.5:14b on 2026-11-11 from this project's results. 14 of 14 numbers traced to the results; 0 unsupported sentences removed. Audit trail: [analytics_review.cited.md](analytics_review.cited.md), [grounding.json](grounding.json).*

**Estimated:** Medicare Part D GLP-1 gross spend grew 40% annually to $27.9B in 2024. The opportunity to generate extra claims by moving under-adopting primary care prescribers to the median adoption of their peers is significant.

**How:** Segmented 231,338 prescribers into four groups based on potential and adoption. Used gradient-boosted trees to predict top growth contributors, validated with cross-validation. Fitted three growth curves to GLP-1 claims from 2018 to 2024, tested on 2023–2024 data.

**How far to trust it:** Model reached 10% more of next year's growth for the same number of calls compared to ranking by current volume. Cross-validation showed modest improvement over current volume ranking. Growth curves under-forecast by 20–25% due to accelerating growth in 2023. Known limitations include CMS's small-count suppression and focus on Medicare only.
