# az-sentiment: Azərbaycan dilində tətbiq rəylərinin sentiment analizi

Üç yanaşmanın eyni, təmizlənmiş test datasında müqayisəsi:

| Səviyyə | Yanaşma | Skript |
|---|---|---|
| 0 | Baseline (çoxluq sinfi, təsadüfi) | `02_baselines.py` |
| 1 | TF-IDF (söz + simvol n-qram) + Logistic Regression | `03_tfidf.py` |
| 2 | Transformer fine-tune (XLM-RoBERTa / mBERT) | `04_transformer.py` |
| 3 | LLM zero-/few-shot (Ollama, deepseek-r1:1.5b) | `05_llm.py` |

**Dataset:** [`hajili/azerbaijani_review_sentiment_classification`](https://huggingface.co/datasets/hajili/azerbaijani_review_sentiment_classification) (MIT), Google Play rəyləri, 1–5 ulduz.
Label: 1–2 → negative, 3 → neutral, 4–5 → positive.

**Əsas metrika: macro-F1.** Data balanssızdır (73% positive), hamıya "positive" deyən model 73% accuracy alır.

## Data problemləri (01_prepare_data.py tapır)

- Orijinal test mətnlərinin ~53%-i train-də də var (leakage) → train+test birləşdirilir, dublikat silinir, yenidən 80/10/10 stratified bölünür.
- 159k sətrin yalnız ~78k-sı unikal mətndir ("əla" 17k dəfə).
- Ulduz ≠ mətn: bəzi şikayətlər 5★ ilə gəlir (label səs-küyü), 655 mətn dublikatlarda fərqli label daşıyır.

## Colab

`colab.ipynb`: kodu GitHub-dan clone edir və ya zip yükləyir, bütün addımları T4 GPU-da işlədir, `results.zip` endirir.

## Quraşdırma (lokal)

```bash
uv sync                      # Python 3.12, torch CUDA 12.8
ollama pull deepseek-r1:1.5b # LLM hissəsi üçün
```

## İşə salma (ardıcıl)

```bash
uv run scripts/01_prepare_data.py                         # data + EDA -> data/processed, results/data_stats.json
uv run scripts/02_baselines.py
uv run scripts/03_tfidf.py                                # grid, leakage təcrübəsi, learning curve
uv run scripts/04_transformer.py --weighted               # xlm-roberta-base (~GPU)
uv run scripts/04_transformer.py                          # çəkisiz variant (müqayisə üçün)
uv run scripts/05_llm.py --shots 0
uv run scripts/05_llm.py --shots 3
uv run scripts/06_compare.py                              # results/comparison.md
uv run scripts/07_error_analysis.py --model xlm-roberta-base_weighted --other tfidf_logreg
```

Windows-da Azərbaycan hərfləri konsolda xəta verərsə: `$env:PYTHONIOENCODING="utf-8"`.

## Nəticələr

`results/` qovluğu:
- `metrics/*.json`: hər modelin accuracy, macro-F1, sinif üzrə precision/recall/F1
- `figures/`: EDA, confusion matrix-lər, learning curve, train/val loss əyriləri, müqayisə
- `comparison.md`: yekun cədvəl
- `errors_*.md`: səhv təhlili, `label_noise_candidates_*.csv`: əl ilə yoxlanacaq nümunələr
