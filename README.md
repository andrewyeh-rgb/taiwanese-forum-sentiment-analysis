# Sentiment Analysis on Taiwanese Online Forums with Fine-Tuned BERT

Undergraduate graduation research project — Computer Science & Engineering, Yuan Ze University (Mar. 2024 – Dec. 2024)

## Overview

Social media and online forums have become a primary channel for people to express opinions and emotions, and large-scale sentiment analysis of this content is valuable for business, government, and academic research. However, Taiwan has a distinct online culture, and general-purpose Chinese sentiment resources do not transfer well to local forums such as **Dcard** and **PTT**.

This project fine-tunes a pre-trained **BERT** model (`bert-base-chinese`) for **binary sentiment classification (positive / negative)** of Traditional Chinese forum comments, and investigates a key research question:

> **Does a small, self-collected, domain-matched training set outperform a much larger general-purpose sentiment dataset when the target domain is Taiwanese online forums?**

To answer this, the same model architecture and training pipeline were used to fine-tune three separate models on three different training sets, and all three were evaluated on an identical held-out test set of real Dcard comments.

## Methodology

1. **Data collection**
   - Scraped comments from Dcard and PTT.
   - Segmented text with **CKIP Transformers** (Academia Sinica).
   - Weakly labeled sentiment polarity using the **NTUSD** (National Taiwan University Sentiment Dictionary) and its extension **ANTUSD**.
   - Held out ~150–250 hand-checked Dcard comments as a common **test set** shared by all three experiments.
2. **Model**: `bert-base-chinese` fine-tuned with Hugging Face `transformers.Trainer` for binary sequence classification.
3. **Three training configurations, same architecture & test set:**

   | Training set | Description |
   |---|---|
   | **ANTUSD** | Dictionary-weighted labels applied to general text — large but domain-mismatched |
   | **JD Binary** | Public Chinese e-commerce review sentiment dataset — large, general domain |
   | **Dcard (self-collected)** | Small, hand-curated corpus scraped from Taiwanese forums — matches test-set domain/register |

4. **Evaluation metrics**: Accuracy, F1 score, AUC (via `evaluate` — `accuracy`, `f1`, `roc_auc`).

## Results

| Training Set | Loss | Accuracy | F1 Score | AUC |
|---|---|---|---|---|
| ANTUSD | 2.1998 | 51.82% | 0.3354 | 0.6683 |
| JD Binary | 1.3762 | 62.16% | 0.5787 | 0.7318 |
| **Dcard (self-collected)** | **0.4366** | **81.64%** | **0.8293** | **0.8907** |

**Key finding:** the model trained on the small, self-collected Dcard dataset dramatically outperformed both larger baselines across every metric, confirming that domain-matched training data is more valuable than raw dataset size for sentiment analysis on Taiwan-specific forum language and slang.

## Repository Structure

```
├── notebooks/
│   ├── train_antusd.ipynb        # Fine-tuning on ANTUSD-weighted labels
│   ├── train_jd_binary.ipynb     # Fine-tuning on public JD Binary dataset
│   └── train_dcard_custom.ipynb  # Fine-tuning on self-collected Dcard corpus
└── README.md
```

Each notebook follows the same pipeline: load & tokenize data → load `bert-base-chinese` → configure `TrainingArguments` → train → evaluate on the shared Dcard test/validation set.

## Tech Stack

- **Model / NLP:** PyTorch, Hugging Face `transformers`, `datasets`, `evaluate`, BERT (`bert-base-chinese`), CKIP Transformers
- **Data collection:** Python web scraping (Dcard, PTT)
- **Lexicons:** NTUSD, ANTUSD (Academia Sinica)
- **Environment:** Google Colab (GPU)

## Future Work

- Expand the self-collected dataset to reduce class imbalance and improve generalization.
- Extend from binary to multi-class / fine-grained sentiment (e.g., positive / neutral / negative / mixed).
- Explore larger or Taiwan-specific pre-trained language models.
- Package the best-performing model behind a lightweight inference API/demo.

## Author

**Andrew Yunteng Yeh** — B.S. Computer Science & Engineering, Yuan Ze University
[GitHub](https://github.com/andrewyeh-rgb)
