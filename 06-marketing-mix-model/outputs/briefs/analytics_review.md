# Brief for an analytics reviewer

*Drafted by the open-weights model qwen2.5:14b on 2026-11-11 from this project's results. 12 of 12 numbers traced to the results; 2 unsupported sentences removed. Audit trail: [analytics_review.cited.md](analytics_review.cited.md), [grounding.json](grounding.json).*

**Estimated:** The model recovers the field force's return almost exactly, with a true ROI of 1.37 and an estimated ROI of 1.39. Other channels' returns are mixed: TV's true ROI of 0.69 is estimated at 0.61, while social's true ROI of 1.31 is estimated at 1.21. The model underestimates HCP digital's ROI, estimating 2.40 compared to the true 2.95.

**How:** The Bayesian model uses adstock carry-over, saturation, and regional baselines. It fits with an in-sample R² of 98% and a 0.9% MAPE. It forecasts the last 13 weeks with a 0.7% MAPE. Lift tests calibrate social and search, tightening their intervals.

**How far to trust it:** The model's in-sample fit is strong, but it overpromises on budget reallocation. The model underestimates TV's true response and overestimates speaker programs' marginal return.
