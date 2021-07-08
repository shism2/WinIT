# tsi-research
Time Series Interpretability Research

Generate spike comparison:
```
python -m FIT.evaluation.baselines --data simulation_spike --explainer ifit --train_gen --N 10 --delay 0 --train
```
Results:
```
Saving file to  ./output//simulation_spike/ifit_test_importance_scores_0.pkl
-----------------------------------------------
Important features within each timestep metrics
Ranked accuracy: 0.935672514619883
Imp ft at least 1 std > than unimp ft: 0.9005847953216374
KL div between imp ft and unimp ft in imp ts: inf
-----------------------------------------------
auc: 0.9270861324561059  aupr: 0.2664588488268385
```

# Test:
Time delay = 2
Windows = [1, 2, 4, 8, 16]

evals = [[0.5315586944617352, 0.0010457909597314484],
        [],
        [],
        [],
        []]

Time delay = 4
Windows = [1, 2, 4, 8, 16]


# Recording

```
time python -m FIT.evaluation.baselines --data simulation_spike --explainer ifit --train_gen --N 2 --delay 2 --train --cv 1

-----------------------------------------------
Important features within each timestep metrics
Ranked accuracy: 0.9166666666666666
Imp ft at least 1 std > than unimp ft: 0.21153846153846154
KL div between imp ft and unimp ft in imp ts: inf
-----------------------------------------------
auc: 0.908414671449626  aupr: 0.008165649188572025

real    3m3.373s
user    33m31.012s
sys     0m36.947s
```
```
time python -m FIT.evaluation.baselines --data simulation_spike --explainer ifit --train_gen --N 2 --delay 2 --train --cv 2

-----------------------------------------------
Important features within each timestep metrics
Ranked accuracy: 0.9358974358974359
Imp ft at least 1 std > than unimp ft: 0.9358974358974359
KL div between imp ft and unimp ft in imp ts: inf
-----------------------------------------------
auc: 0.9584807063064761  aupr: 0.6466087566541691

real    3m7.271s
user    34m20.948s
sys     0m38.593s
```

```
time python -m FIT.evaluation.baselines --data simulation_spike --explainer ifit --train_gen --N 2 --delay 2 --train --cv 3

-----------------------------------------------
Important features within each timestep metrics
Ranked accuracy: 0.8717948717948718
Imp ft at least 1 std > than unimp ft: 0.391025641025641
KL div between imp ft and unimp ft in imp ts: inf
-----------------------------------------------
auc: 0.778824250666142  aupr: 0.026271528329866144

real    2m28.111s
user    32m40.002s
sys     0m36.312s
```

```
time python -m FIT.evaluation.baselines --data simulation_spike --explainer ifit --train_gen --N 2 --delay 2 --train --cv 4

-----------------------------------------------
Important features within each timestep metrics
Ranked accuracy: 0.9102564102564102
Imp ft at least 1 std > than unimp ft: 0.4230769230769231
KL div between imp ft and unimp ft in imp ts: 0.22519961594744067
-----------------------------------------------
auc: 0.9303699689977533  aupr: 0.01973027072324309

real    3m15.925s
user    33m31.230s
sys     0m36.372s
```

AUROC (Explanation)