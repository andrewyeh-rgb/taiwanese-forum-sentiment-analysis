# Data

All CSVs below live in `raw/` and match the filenames used by the original Colab notebooks
(except the self-collected Dcard train set, which is renamed for clarity).

## Files in this repository

| File | Role | Rows | Columns | Labels |
|---|---|---:|---|---|
| `raw/dcard_self_collected.csv` | Domain-matched Dcard training set (self-collected) | 6,595 | `review`, `label` | `0`: 2,842 / `1`: 3,753 |
| `raw/updated_comments_small.csv` | Shared held-out Dcard validation / test set | 658 | `review`, `label` | `0`: 282 / `1`: 376 |
| `raw/online_shopping_10_cats.csv` | JD Binary public shopping-review training set | 62,774 | `cat`, `label`, `review` | `0`: 31,046 / `1`: 31,728 |
| `raw/processed_opinion_word_utf8.csv` | ANTUSD / lexicon-derived training rows | 28,798 | `review`, `label` | `0`: 11,278 / `1`: 17,520 |

**Label convention:** `0` = negative, `1` = positive.

## Mapping to original Colab paths

| Original Colab path | File in this repo |
|---|---|
| `/content/drive/MyDrive/CSV/updated_comments.csv` | `raw/dcard_self_collected.csv` |
| `/content/drive/MyDrive/CSV/updated_comments_small.csv` | `raw/updated_comments_small.csv` |
| `/content/drive/MyDrive/CSV/online_shopping_10_cats.csv` | `raw/online_shopping_10_cats.csv` |
| `/content/drive/MyDrive/CSV/processed_opinion_word_utf8.csv` | `raw/processed_opinion_word_utf8.csv` |

## Notes

- `dcard_self_collected.csv` is the domain-matched corpus that performed best in the three-way experiment.
- `online_shopping_10_cats.csv` is a public Chinese e-commerce review dataset (Simplified Chinese categories + reviews).
- `processed_opinion_word_utf8.csv` comes from dictionary-based weak labeling (NTUSD / ANTUSD pipeline).
- Forum text may include informal slang; review content before redistributing outside academic / portfolio use.
