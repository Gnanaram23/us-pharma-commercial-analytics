# Brief for an analytics reviewer

*Drafted by the open-weights model qwen2.5:14b on 2026-11-11 from this project's results. 13 of 13 numbers traced to the results; 0 unsupported sentences removed. Audit trail: [analytics_review.cited.md](analytics_review.cited.md), [grounding.json](grounding.json).*

**Estimated:** In 2024, 1.1 million promotional payments totaling $54.4M reached 154,019 HCPs. Novo Nordisk and Lilly reached 57% and 49%, respectively, of top-decile GLP-1 prescribers. Only 28% of bottom-half prescribers received any payment.

**How:** Real data from CMS Open Payments and Medicare Part D claims were joined to measure reach and effect. A simulation with 20,000 prescribers over 52 weeks tested attribution methods and next-best-action policies.

**How far to trust it:** The simulation's known true effects validate methods. Attribution methods misallocate about 30% of credit. Email targeting on uplift captures 78% of email's total effect. The contextual bandit policy outperforms fixed rules by 14%. Limitations include Open Payments' lower-bound reach and potential targeting biases.
