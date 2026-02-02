Autonomous data exploration (Agentic)

Goal: Agent explores a dataset like a strong analyst: checks quality, finds patterns, proposes features, suggests models, asks clarifying questions.

Autonomous data exploration is an agent that runs an EDA playbook: schema checks, missingness, distributions, outliers, and key relationships with the target. 
It generates plots and summary tables via tools, not by guessing. Then it proposes hypotheses and next steps, like which features to engineer or which model family to try, while documenting 
each step and keeping execution in a sandbox for safety.

Architecture:

Dataset / table(s)
   ->
Planner: exploration plan (EDA checklist)
   ->
Tools: 1-schema profiling (types, missingness), 2-stats tests (corr, chi-square), 3-anomaly detection (outliers, drift), 4-visualization tool (plots), 5-SQL/pandas execution sandbox
    ->
Insight miner: findings + hypotheses
    ->
Recommendation: next analyses + candidate models + feature ideas
    ->
Stop condition: budget/time/novelty threshold


