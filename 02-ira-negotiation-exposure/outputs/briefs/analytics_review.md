# Brief for an analytics reviewer

*Drafted by the open-weights model qwen2.5:14b on 2026-11-11 from this project's results. 13 of 13 numbers traced to the results; 0 unsupported sentences removed. Audit trail: [analytics_review.cited.md](analytics_review.cited.md), [grounding.json](grounding.json).*

**Estimated:** The 40 drugs selected so far took $126B of Medicare spend in 2024, and their negotiated prices cost manufacturers an estimated $21–25B a year in net revenue.

**How:** The analysis used CMS Medicare Part D and Part B spending data from 2020 to 2024. It also incorporated CMS's own net-savings estimates for the first two negotiation cycles. The net revenue impact was calculated by applying the published list-price cuts to gross spend and adjusting for existing rebates.

**How far to trust it:** The back-test showed the screen correctly identified 11 of 15 actual selections in both the 2027 and 2028 cycles. However, the model's performance was mixed, with the hybrid approach (rules + logistic regression) performing best by finding 10 of the 15 actual picks. The analysis relies on Medicare gross spend before rebates, and the net impact estimates are based on CMS's aggregate estimates. Known limitations include hand-coded orphan status and biosimilar presence, which may not be exhaustive.
