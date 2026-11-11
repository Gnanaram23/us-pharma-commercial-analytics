# Brief for an analytics reviewer

*Drafted by the open-weights model qwen2.5:14b on 2026-11-11 from this project's results. 4 of 4 numbers traced to the results; 2 unsupported sentences removed. Audit trail: [analytics_review.cited.md](analytics_review.cited.md), [grounding.json](grounding.json).*

> **Review note.** This draft passed the automatic checks but a review found problems:
> - After the claim check, the 'Estimated' section is missing, so this brief does not state the main results.


**How:**
- Simulated 10,000 futures to estimate the number of approvals.

**How far to trust it:**
- Validation showed text classifier accuracy was 15%, below the 25% from always guessing the most common mechanism.
- The model leans optimistic due to the unusually high success rates in obesity drugs.
- ClinicalTrials.gov dates are estimates and often slip.
- About 40% of industry trials use code numbers that could not be confidently mapped.
