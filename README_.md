# DS605 Lab 6: Feature Extraction and ML with Image and Text Data

Name : Raj Udaybhai Ruparel
Enrollment Number : 202618065


Converts raw images and emails into numeric features, then classifies them with traditional ML (no CNNs, no pretrained embeddings).

- **Image data:** Asphalt Crack Dataset, 400 images (200 Cracks / 200 NonCracks)
- **Text data:** Email Spam Classification Dataset, 5,172 emails
- **Run:** `pip install -r requirements.txt`, then run `lab6.ipynb` top to bottom

## Repo structure

```
lab6.ipynb        # Parts A, B, C
data/             # asphalt images + emails.csv
outputs/          # feature tables, plots, result tables
requirements.txt
```

## Part A: Image features

**Pipeline:** read (OpenCV) -> inspect -> resize 128x128 -> grayscale -> NumPy intensity stats + Canny edges -> 1 row per image.

- All 400 images were 448x448, 3-channel, none unreadable.
- Resizing is therefore for **compute reduction** (200,704 -> 16,384 pixels), not standardisation.
- **12 features:** mean, std (contrast), min, max, median, range, p10, p90, dark ratio, bright ratio, edge count, edge density.
- **Classifiers:** Logistic Regression, SVM (RBF), Random Forest, KNN. Stratified 80/20 split, scaler fit on train only.
- **Best model:** Random Forest (Acc 0.925, Recall 0.950, F1 0.927).

**Observations**
- Cracks have ~2.3x higher edge density (0.260 vs 0.115) and much darker minimum intensity (41.7 vs 89.8).
- Bright-end features (p90, max intensity) carried almost no signal.
- Recall matters most here: a missed crack costs more than a false alarm.

Outputs: `canny_samples.png`, `feature_distributions.png`, `confusion_matrices.png`, `feature_importance.png`, `image_features.csv`, `image_model_results.csv`.

## Part B: Text vectorization

**Dataset note:** the Kaggle CSV is already vectorised (3,000 word-count columns, no raw text). Raw text was rebuilt by repeating each word by its count. Word order is lost, which does not affect bag-of-words models. Because of this, n-grams were not possible.

- Class balance: `<fill from B2>`
- Duplicate texts dropped: `<fill from B4>`
- Split before vectorising; `CountVectorizer` fit on train only (no leakage).
- **Models:** Multinomial Naive Bayes, Logistic Regression, LinearSVC.
- Baseline vocabulary: **2,974 features**; mean F1 **0.939**; mean train time **942 ms**.

Outputs: `text_results_original.csv`, `text_confusion_matrices.png`.

## Part C: Improving the representation

### Image: dark-pixel threshold 60 -> 100
**Why:** with threshold 60, `dark_ratio` was tiny in both classes (0.0071 vs 0.0001), so it separated them poorly.

| | Original (<60) | Improved (<100) |
|---|---|---|
| Features | 12 | 12 |
| Best model | Random Forest | Random Forest |
| Best Accuracy | 0.925 | 0.938 |
| Best Recall | 0.950 | 0.950 |
| Best F1 | 0.927 | 0.938 |
| Mean F1 (4 models) | 0.923 | 0.926 |
| Mean train time | 0.0497 s | 0.0489 s |

Small gain at zero extra cost. The test set has only 80 images, so one image is ~1.25%. The improvement is **within noise** and should not be over-claimed.

Outputs: `image_before_vs_after_metrics.png`, `dark_ratio_before_vs_after.png`, `image_delta_table.csv`.

### Text: stop-word removal + vocabulary cap (top 500)
**Why:** fewer, more informative words should cut compute with little loss.

| | Original | Improved |
|---|---|---|
| Features | 2,974 | 500 |
| Mean F1 | 0.939 | 0.919 |
| Mean train time | 942 ms | 511 ms |

- Features cut by **83%**, train time by **~46%**, at a cost of **~0.02 F1**.
- Spam signal is spread across many words, so a hard cap discards some of it.

Outputs: `text_results_variants.csv`, `text_tradeoff.png`.

### Trade-off discussion
- **Dimensionality vs performance:** fewer features gave faster, simpler models but lower F1 in text. More features are not always needed, but cutting too hard loses signal.
- **Computation:** the text saving is real (about half the training time). In the image task, 12 features already train in milliseconds, so there is nothing to save.
- **Representation matters:** a small change in image feature definition (threshold) changed how well a feature separated the classes, with no change in model.

## Limitations
- Small image test set (80); results vary with the split.
- Timing differences are milliseconds and depend on the machine.
- Text was reconstructed from counts, so word order and n-grams were unavailable.
- Single train/test split, no cross-validation.

Final summary table: `outputs/final_summary.csv`.