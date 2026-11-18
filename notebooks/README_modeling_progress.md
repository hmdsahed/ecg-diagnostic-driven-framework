# ECG Diagnostic-Driven Hybrid Machine Learning Framework

## Project Overview

This project investigates whether statistical and signal-level characteristics of ECG recordings can help guide the selection of suitable machine-learning strategies for heartbeat classification.

The study uses the **MIT-BIH Arrhythmia Database** and follows a diagnostic-driven approach. Rather than only comparing classifiers, it first examines recording-level and heartbeat-level properties, then evaluates how different modelling strategies respond to those properties.

**Central research question**

> Can statistical and signal-level characteristics help us select a more suitable modelling strategy for different ECG recording conditions?

The diagnostic-driven selection component has not yet been implemented or experimentally validated. The experiments completed so far establish the data preparation, diagnostic analysis, and initial modelling baselines.

## Research Objectives

1. Characterize statistical and signal-level properties of ECG recordings.
2. Investigate how these characteristics relate to the performance of different machine-learning approaches.
3. Compare traditional machine learning, deep learning, and hybrid approaches for heartbeat classification.
4. Develop and evaluate diagnostic-driven model selection against fixed-model baselines.
5. Examine robustness and generalization across subjects and recording conditions.

## Dataset

The project uses the **MIT-BIH Arrhythmia Database**.

- Sampling frequency: 360 Hz
- ECG channels: two per recording
- Recordings inspected: 48
- Subjects represented: 47
- Annotations: expert beat and rhythm annotations
- Data access: WFDB Python package and PhysioNet

Recordings `201` and `202` belong to the same subject. They are assigned the same subject identifier when constructing subject-independent splits.

## Project Structure

| Notebook | Purpose | Status |
|---|---|---|
| `01_Data_Inspection_and_Statistical_Analysis.ipynb` | Dataset inspection, annotation analysis, recording-level statistics and diagnostics | Completed |
| `02_Hybrid_Model_Selection.ipynb` | Heartbeat segmentation, preprocessing investigation, evaluation design and baseline modelling | In progress |

Processed files currently include:

```text
ECG_Research/
└── processed_data/
    ├── X_ecg_raw.npy
    ├── heartbeat_metadata.csv
    ├── X_ecg_raw_sensitivity.npy
    └── heartbeat_metadata_sensitivity.csv
```

The local folder layout may differ depending on the machine and repository setup.

---

# Part I — Recording-Level Statistical Analysis

## 1. Initial Dataset Inspection

The first notebook inspected the MIT-BIH recordings, signal channels, sampling frequency, signal dimensions, and annotation files.

The work included:

- Loading ECG recordings and annotations with WFDB.
- Plotting ECG signal examples.
- Inspecting annotation symbols.
- Counting annotation types.
- Examining heartbeat-class distributions.
- Comparing class distributions across recordings.

A provisional AAMI-style grouping was used:

```python
aami_mapping = {
    "N": ["N", "L", "R", "e", "j"],
    "S": ["A", "a", "J", "S"],
    "V": ["V", "E"],
    "F": ["F"],
    "Q": ["/", "f", "Q"]
}
```

This mapping supported exploratory analysis. The final manuscript must verify and document the annotation mapping and inclusion rules against the selected benchmark protocol and authoritative references.

## 2. Annotation Distribution

Across the 48 inspected recordings:

- Total annotations: 112,647
- Candidate heartbeat annotations: 109,494
- Other annotation events: 3,153
- Unique annotation symbols: 23

The provisional five-class distribution was:

| AAMI group | Count | Percentage |
|---|---:|---:|
| N | 90,631 | 82.77% |
| S | 2,781 | 2.54% |
| V | 7,236 | 6.61% |
| F | 803 | 0.73% |
| Q | 8,043 | 7.35% |
| **Total** | **109,494** | **100%** |

The Q group was dominated by paced beats:

- `/` paced beats: 7,028
- `f` fusion of paced and normal beats: 982
- `Q` unclassifiable beats: 33

Q is therefore heterogeneous. Its treatment in the final classification task requires an explicit protocol-based decision.

## 3. Stationarity Analysis

ADF and KPSS tests were applied to all 48 recordings.

At a significance level of 0.05:

- ADF rejected the unit-root null hypothesis in all 48 recordings.
- KPSS rejected the stationarity null hypothesis in 19 recordings.
- KPSS did not reject the stationarity null hypothesis in 29 recordings.

For exploratory comparisons, recordings were divided into:

- `KPSS_rejected`: 19 recordings
- `KPSS_not_rejected`: 29 recordings

These are groups defined by statistical test outcomes. They are **not clinical diagnoses** and should not be treated as definitive labels of physiological stationarity or nonstationarity.

## 4. Rolling Statistics and Recording-Level Features

Rolling mean and rolling variance were examined to characterize local signal variation. Recording-level statistical features were combined into:

```text
mitbih_integrated_diagnostic_features.csv
```

The analysis included descriptive statistics, Spearman correlations, pairwise significance testing, Benjamini–Hochberg false-discovery-rate correction, feature redundancy screening, Mann–Whitney U tests, rank-biserial effect sizes, boxplots, IQR-based outlier checks, and sensitivity analysis.

### Main exploratory group-comparison findings

Three rolling-statistic features differed significantly between the two KPSS groups after BH-FDR correction:

| Feature | Adjusted p-value | Rank-biserial effect size |
|---|---:|---:|
| `std_rolling_variance` | 0.00086 | -0.62613 |
| `max_rolling_variance` | 0.00051 | -0.68058 |
| `max_rolling_mean` | 0.00051 | -0.66606 |

The median values of all three features were higher in the `KPSS_not_rejected` group. The differences remained significant after the feature-specific outlier sensitivity checks.

These findings show differences between the statistical-test groups in this dataset. They do **not** yet show that the features predict model performance or select a better classifier.

---

# Part II — Heartbeat Dataset Preparation

## 5. Recording Selection and Classification Target

A conventional 44-record DS1/DS2-style recording set was inspected. The supplied lists contained 22 recordings each, while records `102`, `104`, `107`, and `217` were not included.

A subject-overlap issue was identified: recordings `201` and `202` belong to the same subject but appeared in opposite groups in the inspected split. Therefore, that split cannot be described as fully subject-independent without resolving the overlap.

The current heartbeat modelling dataset was constructed from the 44 selected recordings. A four-class target (`N`, `S`, `V`, `F`) was used provisionally, with Q excluded from this experiment.

This is a **provisional modelling decision**, not a final claim that four classes are the only appropriate benchmark. The final paper must explain the selected record set, annotation mapping, and handling of Q beats with reference to the chosen evaluation protocol.

## 6. Subject-Level Class Distribution

At the annotation stage, the selected 44-record set contained:

| Class | Beat count |
|---|---:|
| N | 90,125 |
| S | 2,781 |
| V | 7,009 |
| F | 803 |
| **Total** | **100,718** |

Subject coverage:

- N: 43 subjects
- S: 31 subjects
- V: 32 subjects
- F: 16 subjects

The F class was especially concentrated. Subjects 208 and 213 contributed 734 of the 802 F beats in the later segmented dataset (about 91.5%). This concentration matters when interpreting subject-independent evaluation.

Randomly splitting individual heartbeats could place beats from the same person in both training and test data. The modelling work therefore uses subject identifiers for splitting.

## 7. Subject-Independent Evaluation Design

A Leave-One-Subject-Out (LOSO) structure was checked using 43 subjects:

- Folds: 43
- Training subjects per fold: 42
- Held-out test subjects per fold: 1

This provides a subject-independent outer evaluation structure. For a final LOSO experiment, hyperparameter tuning and model selection must take place within the training subjects of each outer fold, using grouped inner validation.

A five-fold `StratifiedGroupKFold` experiment based on each subject's dominant class was also examined. That particular strategy was unsuitable because 42 subjects had N as their dominant class and only one had S as the dominant class. It produced a highly uneven distribution of F beats across folds. This rejects that specific dominant-class stratification setup; it does not establish that all grouped five-fold evaluation is unsuitable.

## 8. Heartbeat Segmentation

The selected ECG channel was MLII (the first channel in the loaded recordings).

For each candidate heartbeat annotation:

- 90 samples before the annotated location were extracted.
- 90 samples after the annotated location were extracted.
- Each segment contains 180 samples.
- At 360 Hz, this corresponds to a 0.5-second window.

Segments extending beyond recording boundaries were excluded.

The segmented dataset contained:

- ECG matrix `X_ecg`: `(100693, 180)`
- Metadata `heartbeat_data`: `(100693, 6)`
- Boundary segments excluded: 25

After boundary exclusion:

| Class | Segmented beats |
|---|---:|
| N | 90,102 |
| S | 2,781 |
| V | 7,008 |
| F | 802 |
| **Total** | **100,693** |

Saved files:

```text
X_ecg_raw.npy
heartbeat_metadata.csv
```

The ECG matrix has 100,693 rows (heartbeats) and 180 columns (samples per heartbeat).

---

# Part III — Signal Quality and Preprocessing Investigation

## 9. Initial Segment-Level Statistics

Five example segments from each class were visualized. The waveforms varied within classes, and visual inspection did not establish clear class separation.

Exploratory segment-level statistics included mean amplitude, standard deviation, minimum, maximum, and peak-to-peak amplitude. The class-level results showed differences in average variability, but substantial overlap remained. These descriptive statistics do not establish reliable class separability.

## 10. Within-Window Baseline Difference

An exploratory proxy was calculated by comparing the mean of the first 30 samples with the mean of the last 30 samples of each heartbeat window.

This was named **within-window baseline difference**. It is not a direct measurement of baseline wander: the first and last portions of a short window may contain parts of neighbouring cardiac activity.

## 11. Three Signal Representations

Three versions were created without overwriting the original raw data.

### P0 — Raw ECG

```python
X_raw = X_ecg.copy()
```

### P1 — Per-beat mean-centred ECG

```python
X_mean_centered = X_ecg - X_ecg.mean(axis=1, keepdims=True)
```

### P2 — Edge-corrected ECG

The mean of the first and last 30 samples was subtracted from each segment:

```python
edge_baseline = np.concatenate(
    [X_ecg[:, :30], X_ecg[:, -30:]],
    axis=1
).mean(axis=1, keepdims=True)

X_edge_corrected = X_ecg - edge_baseline
```

These are candidate signal representations. Their usefulness must be determined by model evaluation rather than assumed.

## 12. Possible Saturation in Recording 116

Amplitude diagnostics identified unusually large values in some segments, particularly in recording 116.

The full-recording comparison showed:

- Recording 100 had a relatively narrow amplitude range.
- Recording 116 reached approximately -5.12 mV and 5.115 mV.
- Recording 116 contained repeated samples at these extreme values, including consecutive runs.
- The longest observed runs were 255 samples at the minimum and 265 samples at the maximum.

These observations are consistent with possible clipping or ADC saturation in recording 116. They should be described cautiously until signal scaling and recording metadata have been fully checked.

Six extracted heartbeat windows in recording 116 overlapped the identified saturation episodes. All six were N-class beats.

A sensitivity dataset was created with those six windows excluded:

```text
X_ecg_raw_sensitivity.npy
heartbeat_metadata_sensitivity.csv
```

It contains 100,687 heartbeat segments. The six excluded beats represent approximately 0.006% of the original segmented dataset. The original dataset remains preserved.

---

# Part IV — Initial Machine-Learning Experiments

## 13. Dataset Verification

The saved data were reloaded and checked:

- Raw ECG matrix: `(100693, 180)`
- Metadata: `(100693, 6)`
- Subjects: 43
- Recordings: 44
- Missing metadata values: none reported
- NaN or infinite ECG values: none reported

The class counts were verified against the saved metadata.

## 14. Initial 80:20 Subject-Independent Holdout

An initial holdout split was created with `GroupShuffleSplit`, using subject ID as the grouping variable.

- Training heartbeats: 80,250
- Test heartbeats: 20,443
- Training subjects: 34
- Test subjects: 9
- Subject overlap: none

Training class counts:

| Class | Training |
|---|---:|
| N | 73,298 |
| S | 2,592 |
| V | 3,944 |
| F | 416 |

Test class counts:

| Class | Test |
|---|---:|
| N | 16,804 |
| S | 189 |
| V | 3,064 |
| F | 386 |

The test F beats came mainly from subject 208, with a smaller number from subject 223.

This holdout has been used for multiple exploratory model comparisons. It should therefore not be described as an untouched final test set in the final paper.

## 15. Random Forest on Raw ECG Segments

The first baseline used the 180 raw ECG samples as input features.

```python
RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
```

### 80:20 subject-independent holdout

| Metric | Result |
|---|---:|
| Balanced accuracy | 0.3106 |
| Macro F1 | 0.3124 |
| Accuracy | Approximately 0.81 |

The model mainly recognized N and V. It did not correctly identify S or F in this test split. This demonstrates why accuracy alone is inadequate for this imbalanced task.

## 16. Engineered Heartbeat Features

Sixteen summary features were extracted from each 180-sample heartbeat:

1. Mean
2. Standard deviation
3. Minimum
4. Maximum
5. Range
6. Root mean square (RMS)
7. Energy
8. Skewness
9. Kurtosis
10. 25th percentile
11. Median
12. 75th percentile
13. Derivative mean
14. Derivative standard deviation
15. Derivative maximum
16. Derivative minimum

The feature matrix had shape `(100693, 16)`. These features summarize amplitude distribution and local waveform variation. They do not yet include explicit RR-interval features or full heartbeat sequence context.

## 17. Random Forest on Engineered Features

The same 80:20 subject split was used.

| Metric | Raw ECG RF | Engineered-feature RF |
|---|---:|---:|
| Balanced accuracy | 0.3106 | 0.3686 |
| Macro F1 | 0.3124 | 0.3637 |
| Accuracy | ~0.81 | ~0.84 |

The engineered-feature model improved the aggregate metrics in this split, particularly V-class recognition. It still did not correctly identify S or F.

The leading impurity-based feature importances included standard deviation, kurtosis, maximum amplitude, skewness, and derivative standard deviation. These importances are descriptive model outputs; they do not establish causality or independent feature value.

## 18. Random Forest LOSO Evaluation

Raw ECG and engineered-feature Random Forest models were evaluated using Leave-One-Subject-Out cross-validation.

| Model | Balanced accuracy | Macro F1 | Evaluation time |
|---|---:|---:|---:|
| RF — raw ECG | 0.4211 | 0.3933 | 1,724.87 s |
| RF — engineered features | 0.4320 | 0.4062 | 705.03 s |

The engineered-feature model produced a modest improvement in balanced accuracy and macro F1 and required less evaluation time in this experiment.

Both models had very poor S and F recall. In raw ECG LOSO, only 1 of 2,781 S beats and 1 of 802 F beats were correctly classified. The engineered-feature LOSO experiment also correctly classified only 1 S beat and 1 F beat.

The models performed substantially better on N and V than on S and F.

## 19. Random Oversampling

Random oversampling was tested to investigate whether balancing training class counts could improve minority-class performance. Oversampling was applied to training data only; test data were not oversampled.

### 80:20 holdout experiment

Before oversampling, the engineered-feature training set contained:

- N: 73,298
- V: 3,944
- S: 2,592
- F: 416

After random oversampling, each class had 73,298 training examples.

| Metric | Engineered RF | Engineered RF + oversampling |
|---|---:|---:|
| Balanced accuracy | 0.3686 | 0.4005 |
| Macro F1 | 0.3637 | 0.3873 |
| Accuracy | ~0.84 | 0.8577 |

V-class recall increased, but S and F remained unrecognized in the test set.

### LOSO experiment

Oversampling was performed separately inside each training fold. The held-out subject was not oversampled.

| Model | Balanced accuracy | Macro F1 | Evaluation time |
|---|---:|---:|---:|
| RF — engineered features | 0.4320 | 0.4062 | 705.03 s |
| RF — engineered features + oversampling | 0.4412 | 0.4043 | 2,332.91 s |

Oversampling increased V recall but did not solve S- and F-class recognition. Macro F1 decreased slightly compared with the non-oversampled engineered-feature LOSO model.

These results suggest that duplicating minority examples alone does not provide the model with new subject-level variation.

---

# Part V — Current Findings and Interpretation

## 20. Preliminary Observations

The experiments completed so far show:

1. The MIT-BIH recordings show differing statistical test outcomes under ADF and KPSS.
2. Some rolling-statistic features differ between the two KPSS-defined recording groups.
3. The heartbeat dataset is strongly imbalanced, and minority classes are concentrated in a limited number of subjects.
4. The engineered-feature Random Forest performed modestly better than the raw-waveform Random Forest in the experiments completed so far.
5. Random oversampling improved V-class recall in some experiments but did not solve S- and F-class recognition.
6. Subject-independent evaluation reveals a substantial generalization challenge for minority classes.
7. Possible saturation in recording 116 affects a very small number of extracted heartbeat windows; a separate sensitivity dataset has been preserved.

## 21. What Has Not Yet Been Demonstrated

The current results do **not** establish that:

- The diagnostic-driven selection strategy improves classification.
- Any one preprocessing method is universally best.
- Recording-level stationarity statistics cause differences in model performance.
- The four-class target is definitively preferable to a five-class target.
- The current Random Forest baselines are the best possible models.
- The model generalizes reliably to all minority-class subjects.

The diagnostic-driven selection mechanism still needs to be designed, implemented, and tested against fixed-model baselines.

---

# Part VI — Next Research Steps

## 22. Immediate Next Steps

1. Create and inspect a consolidated comparison table for the completed Random Forest LOSO experiments.
2. Train an XGBoost baseline using the same engineered features and subject-independent split.
3. If useful, evaluate XGBoost under LOSO with grouped evaluation.
4. Investigate class-sensitive learning methods carefully, particularly for S and F.
5. Consider adding meaningful heartbeat morphology and RR-interval features, while defining whether the intended use is offline or real-time classification.
6. Develop a 1D-CNN baseline using raw 180-sample heartbeat segments.
7. Consider a feature-level hybrid model combining waveform-derived and engineered features.
8. Compare raw, mean-centred, and edge-corrected signal representations under the same evaluation protocol.
9. Develop the diagnostic-driven model-selection rule using training data only.
10. Compare diagnostic-driven selection with fixed-model baselines under subject-independent evaluation.

## 23. Evaluation Requirements for the Final Study

The final evaluation should:

- Keep subjects separate between training and test data.
- Perform preprocessing, scaling, feature selection, oversampling, and tuning using training data only.
- Use grouped inner validation for tuning within each outer LOSO training partition if LOSO is used for final evaluation.
- Report macro F1 and balanced accuracy, not accuracy alone.
- Include per-class precision, recall, and F1.
- Include confusion matrices.
- Report uncertainty intervals where appropriate.
- Consider paired statistical comparisons and effect sizes for model comparisons.
- Account for correlation among heartbeats from the same subject.
- Clearly state annotation mapping, recording selection, and treatment of paced or unclassifiable beats.
- Report sensitivity analyses involving the six flagged recording-116 heartbeat segments.

---

## Reproducibility Notes

- Environment: local Anaconda/Jupyter environment on macOS.
- WFDB version reported during data preparation: 4.3.1.
- Main libraries include NumPy, pandas, SciPy, scikit-learn, WFDB, and imbalanced-learn.
- Random state in the initial Random Forest experiments: 42.
- Initial holdout: `GroupShuffleSplit(test_size=0.20, random_state=42)`.
- LOSO evaluation used subject identifiers as groups.
- Oversampling was applied only to training data, including separately within each LOSO training fold.

For a reproducible release, record exact package versions, dataset source/download details, final annotation mapping, and all model parameters.

## Current Project Status

| Component | Status |
|---|---|
| MIT-BIH data inspection | Completed |
| Recording-level statistical analysis | Completed |
| ADF/KPSS and rolling-statistic analysis | Completed |
| Integrated diagnostic feature analysis | Completed |
| Provisional annotation grouping | Exploratory work completed; final protocol verification pending |
| Subject mapping and split inspection | Completed |
| Heartbeat segmentation | Completed |
| Signal representation investigation | Completed |
| Recording-116 saturation investigation | Preliminary investigation completed |
| Sensitivity dataset creation | Completed |
| Initial subject-independent holdout | Completed |
| Random Forest raw ECG baseline | Completed |
| Random Forest engineered-feature baseline | Completed |
| Random oversampling experiments | Completed |
| XGBoost baseline | Not yet started |
| 1D-CNN baseline | Not yet started |
| Hybrid model | Not yet started |
| Diagnostic-driven selection mechanism | Not yet implemented |
| Final nested/grouped evaluation | Not yet completed |
| Full manuscript preparation | Not yet started |

## Important Research Caution

All numerical results in this README are the current exploratory results from the working notebooks. Check them against the saved notebook outputs before publication. The final paper should distinguish exploratory analysis, model-development experiments, and confirmatory evaluation. In particular, the current 80:20 holdout has been reused during model development and should not be presented as an untouched final test set.
