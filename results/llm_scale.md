# LLM miqyas təcrübəsi (n=500)

| model            | family      |   shots |   params_B |   macro_f1 |   accuracy |   F1_neu |   valid_json_% |   sec/item |
|:-----------------|:------------|--------:|-----------:|-----------:|-----------:|---------:|---------------:|-----------:|
| deepseek-r1:1.5b | deepseek-r1 |       0 |      1.500 |      0.368 |      0.510 |    0.048 |         99.200 |     10.648 |
| deepseek-r1:1.5b | deepseek-r1 |       3 |      1.500 |      0.237 |      0.246 |    0.050 |         97.200 |      4.468 |

Müqayisə (eyni 500): xlm-roberta-base_weighted = 0.608, tfidf_logreg = 0.573, baseline_random = 0.326