# Tier-0 length-confound check — hint-following hinted runs

Joined runs: 19852  (base=7084, H=9743, S=3025)

## Raw (uncontrolled): acknowledgment rate and CoT length per model

| model | n | ack rate | median tokens | mean tokens |
|---|---|---|---|---|
| base | 7084 | 0.705 | 1402 | 2226 |
| H | 9743 | 0.757 | 2112 | 2938 |
| S | 3025 | 0.865 | 1633 | 2257 |

## (B) Stratified: acknowledgment rate by model WITHIN length tercile

Terciles computed within each (dataset,hint) cell (matched relative length,
same dataset), then pooled. If DPO > base in every tercile, length is not the cause.

| length tercile | base ack | H ack | S ack | H−base | S−base |
|---|---|---|---|---|---|
| short | 0.564 | 0.585 | 0.699 | +0.021 | +0.135 |
| mid | 0.784 | 0.800 | 0.902 | +0.017 | +0.119 |
| long | 0.836 | 0.855 | 0.942 | +0.018 | +0.106 |

Median tokens per tercile (context): short=901, mid=1666, long=3400

## (A) Logistic regression: ack ~ model + z(log tokens) + dataset FE + hint FE

Odds ratios vs base, controlling for length + dataset + hint (cluster-robust on dataset×instance):

| term | coef (log-odds) | odds ratio | 95% CI (OR) | p |
|---|---|---|---|---|
| H (harmless-DPO) | +0.112 | 1.12 | [0.99, 1.27] | 7.58e-02 |
| S (anti-syc-DPO) | +0.085 | 1.09 | [0.93, 1.27] | 2.83e-01 |
| +1 SD log(tokens) | +1.033 | 2.81 | [2.37, 3.33] | 2.86e-32 |
