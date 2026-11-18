# Diagnostic-Driven Hybrid Machine Learning for ECG Analysis

**Authors:** Haris Md Sahed, Fatih Ozaydin

## 1. Project Overview

This project investigates a diagnostic-driven framework for selecting appropriate machine learning strategies for ECG classification under nonstationary and imbalanced data conditions.

The main idea is to examine the statistical and signal-level characteristics of ECG data before selecting modelling strategies. Rather than applying a single machine learning approach to every dataset, this framework aims to use diagnostic information to guide the selection of suitable modelling techniques.

The initial experiments focus on ECG arrhythmia analysis using the MIT-BIH Arrhythmia Database.

## 2. Research Objectives

The main objectives of this research are:

1. **ECG Data Diagnostics:** Investigate the statistical properties and signal characteristics of ECG recordings using time-series and signal-level diagnostic methods.
2. **Stationarity Analysis:** Examine ECG signal behaviour using the Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests.
3. **Class Imbalance Analysis:** Identify and quantify class imbalance in ECG arrhythmia datasets.
4. **Condition Mapping:** Develop a diagnostic-based approach to characterize different data conditions, including statistical behaviour and class imbalance.
5. **Adaptive Strategy Selection:** Investigate how diagnostic results can guide the selection of appropriate machine learning and hybrid modelling strategies.
6. **Performance Evaluation:** Compare different modelling strategies to determine whether diagnostic-driven selection provides improvements over fixed modelling approaches.
7. **Generalization:** Investigate whether the proposed diagnostic-driven approach can be adapted to other ECG datasets and potentially different types of biomedical data.

## 3. Proposed Methodology

The proposed framework follows five main stages:

### Stage 1: Data Diagnostics

Examine ECG recordings using:

- Augmented Dickey-Fuller (ADF) test
- KPSS test
- Rolling mean
- Rolling variance
- Signal-level statistical characteristics
- Dataset quality and distribution analysis

### Stage 2: Condition Mapping

Characterize ECG recordings based on their diagnostic results, including:

- Stationarity test outcomes
- Local signal variability
- Statistical feature distributions
- Class distribution
- Class imbalance severity

### Stage 3: Hybrid Strategy Selection

Investigate suitable machine learning strategies based on the identified data conditions.

Potential strategies include:

- Standard machine learning baselines
- Class imbalance handling
- Signal-aware modelling
- Combined adaptive strategies

### Stage 4: Model Training

Train selected machine learning models using the diagnostic information and appropriate experimental configurations.

### Stage 5: Performance Comparison

Evaluate and compare the selected strategies using:

- Macro F1-score
- Per-class precision and recall
- Balanced accuracy
- Confusion matrix

The final objective is to determine whether diagnostic-driven strategy selection offers advantages over applying a fixed modelling strategy.

## 4. Datasets

### MIT-BIH Arrhythmia Database

**Status:** Initial data inspection and statistical analysis completed.

- Source: [PhysioNet – MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/)
- 48 half-hour ECG recordings
- Two ECG channels per recording
- Expert-annotated heartbeats and rhythm information
- Sampling frequency: 360 Hz

The MIT-BIH database is being used for the initial feasibility experiments and development of the proposed diagnostic-driven approach.

### St Petersburg INCART 12-lead Arrhythmia Database

**Status:** Planned for subsequent investigation.

- Source: [PhysioNet – INCART Database](https://physionet.org/content/incartdb/)
- 75 annotated ECG recordings
- 12-lead ECG signals

This dataset will be considered for evaluating the framework on another ECG dataset.

## 5. Current Experimental Progress

The initial feasibility experiment has been conducted using the MIT-BIH Arrhythmia Database.

### Experiment 1: Data Inspection and Statistical Analysis

| Experiment | Current status |
|---|---|
| Dataset loading and inspection | Completed |
| ECG signal visualization | Completed |
| Annotation analysis | Completed |
| AAMI class grouping | Preliminary |
| Recording-level class distribution | Completed |
| ADF analysis across 48 recordings | Completed |
| KPSS analysis across 48 recordings | Completed |
| Rolling mean and variance analysis | Completed |
| Rolling statistical feature extraction | Completed |
| Spearman correlation analysis | Completed |
| Feature redundancy screening | Completed |
| Mann–Whitney U group comparisons | Completed |
| Effect-size analysis | Completed |
| Multiple-testing correction (BH-FDR) | Completed |
| IQR-based outlier detection | Completed |
| Sensitivity analysis | Completed |

### 5.1 Preliminary Findings

**1. Stationarity Analysis**

ADF and KPSS tests were applied to the first ECG channel of all 48 recordings.

At a significance level of 0.05:

- ADF rejected the unit-root null hypothesis in all 48 recordings.
- KPSS rejected the stationarity null hypothesis in 19 recordings.
- KPSS did not reject the stationarity null hypothesis in 29 recordings.

These findings indicate that the two tests do not provide identical conclusions for all recordings.

The KPSS outcomes were subsequently used to define two statistical comparison groups for further exploratory analysis.

**2. Class Imbalance Analysis**

A provisional AAMI grouping was applied to the annotated heartbeat symbols.

| Class | Description | Percentage |
|---|---|---:|
| N | Normal and bundle branch block beats | 82.77% |
| S | Supraventricular ectopic beats | 2.54% |
| V | Ventricular ectopic beats | 6.61% |
| F | Fusion beats | 0.73% |
| Q | Unknown and selected paced categories | 7.35% |

**Note:** The annotation mapping is preliminary and will be verified against the selected AAMI evaluation protocol before model development.

**3. Rolling Statistical Analysis**

Rolling mean and rolling variance were investigated to characterize local variation in ECG recordings.

Recording-level statistical features were extracted and compared between the two KPSS outcome groups.

Spearman correlation analysis identified strong relationships among several rolling statistical features, indicating potential feature redundancy.

**4. Comparative Statistical Analysis**

Mann–Whitney U tests were performed to compare rolling statistical features between the two KPSS groups.

After Benjamini–Hochberg false discovery rate (BH-FDR) correction across the nine rolling features, three features showed statistically significant differences:

- Standard deviation of rolling variance (`std_rolling_variance`)
- Maximum rolling variance (`max_rolling_variance`)
- Maximum rolling mean (`max_rolling_mean`)

For all three features, the median values were higher in the KPSS non-rejected group than in the KPSS rejected group.

**5. Outlier and Sensitivity Analysis**

Potential outliers were identified using the interquartile range (IQR) method.

A sensitivity analysis was conducted by repeating the group comparisons after excluding the identified potential outliers from the relevant feature-specific comparisons.

All three previously significant features remained statistically significant after sensitivity analysis and BH-FDR correction. Their effect sizes also remained similar to the original results.

These findings suggest that the observed group differences were not driven solely by the identified potential outliers.

**Important interpretation:** The KPSS groups represent statistical test outcomes, not clinical diagnostic categories. These exploratory comparisons do not establish clinical significance or predictive performance. Further experiments are required to determine whether the identified features are useful for machine learning strategy selection.

## 6. Technologies and Tools

The experiments currently use:

- Python 3.12
- Jupyter Notebook
- NumPy
- Pandas
- SciPy
- Matplotlib
- Seaborn
- WFDB
- Statsmodels
- Scikit-learn

## 7. Repository Structure

## 7. Repository Structure

The repository is organized to separate research notebooks, reusable implementation code, datasets, experimental configurations, and results.

```text
ecg-diagnostic-driven-framework/
│
├── notebooks/
│   ├── 01_Data_Inspection_and_Statistical_Analysis.ipynb
│   ├── 02_Hybrid_Model_Selection.ipynb
│   └── 03_Reusable_Generalization_Framework.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   └── evaluation/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│   ├── statistical_analysis/
│   ├── model_comparison/
│   └── diagnostic_selection/
│
├── configs/
│
├── .gitignore
├── requirements.txt
└── README.md
```

### Directory Description

- `notebooks/`: Research notebooks documenting the experimental workflow.
- `src/preprocessing/`: Reusable ECG preprocessing functions.
- `src/features/`: ECG feature extraction implementations.
- `src/models/`: Traditional, deep learning, and hybrid model implementations.
- `src/evaluation/`: Model evaluation and statistical comparison functions.
- `data/raw/`: Original ECG recordings and annotations.
- `data/processed/`: Preprocessed signals and extracted features.
- `results/statistical_analysis/`: Statistical analysis outputs.
- `results/model_comparison/`: Model training and comparison results.
- `results/diagnostic_selection/`: Diagnostic-driven selection results.
- `configs/`: Experimental configurations and parameters.

Large datasets and model checkpoints will be managed separately and will not be committed to GitHub.

The notebooks are organized according to the research workflow:

1. Data inspection and statistical analysis
2. Hybrid model selection and experimental design
3. Development of a reusable dataset inspection and generalization framework

## 8. Future Work

The planned research activities include:

1. Investigate additional ECG signal characteristics.
2. Verify annotation mapping and establish a consistent evaluation protocol.
3. Develop a diagnostic-based condition mapping procedure.
4. Identify suitable hybrid machine learning strategies.
5. Define a controlled experimental design.
6. Conduct model training and comparative evaluation.
7. Evaluate the framework using the INCART dataset.
8. Develop reusable code for automated dataset inspection and diagnostic reporting.
9. Investigate the reproducibility and generalizability of the findings.

## 9. Research Status

**Current phase:** Preliminary data diagnostics completed; hybrid model selection and experimental design are the next stages.

The first experiment has established an initial statistical foundation for investigating diagnostic-driven machine learning strategies.

However, no predictive model training or comparative model evaluation has been completed at this stage.

This repository is under active development. The proposed framework and experimental findings are preliminary and may be updated as the research progresses.
### Notebook 02 --- Hybrid Model Selection and Experimental Design
Notebook: 02_Hybrid_Model_Selection.ipynb

Status: Initial heartbeat dataset preparation and signal-quality
investigation completed. Model training has not started.
Recording and subject selection
The initial experiment uses 44 recordings drawn from the conventional
DS1 and DS2 recording lists. Four recordings---102, 104, 107, and
217---are not included in those lists.

There is an important subject-level consideration: recordings 201 and
202 belong to the same subject but appear in different conventional
groups. Therefore, the conventional DS1/DS2 assignment is not
automatically subject-independent under the subject mapping used here.

The current dataset contains 44 recordings from 43 unique subjects.
Subject identifiers are retained so that evaluation can keep all
recordings from the same subject together.

The exact benchmark protocol and annotation inclusion rules still
require verification before benchmark claims are made.
Provisional four-class heartbeat dataset
A provisional AAMI mapping was used:

  AAMI class   Annotation symbols
  N            N, L, R, e, j
S            A, a, J, S
V            V, E
F            F
Q            /, f, Q

The initial modelling experiment uses N, S, V, and F. The Q class was
excluded provisionally because it combines paced and unclassifiable
categories and is dominated by paced beats. This is an experimental
scope decision, not a claim that Q is unimportant. The final mapping and
inclusion/exclusion rules must be verified and justified.

The 44-recording annotation extraction produced:

  Class         Annotated beats   Percentage
  N                      90,125      89.483%
S                       2,781       2.761%
V                       7,009       6.959%
F                         803       0.797%
Total         100,718     100%
Heartbeat segmentation
The first ECG channel (MLII) was used for the initial segmentation. Each
heartbeat segment contains 90 samples before and 90 samples after its
annotation position.
- Sampling frequency: 360 Hz
- Segment length: 180 samples
- Approximate segment duration: 0.5 seconds
- Initial annotations: 100,718
- Complete segments extracted: 100,693
- Boundary segments excluded: 25
The resulting signal matrix has shape (100693, 180). Each row
represents one annotated heartbeat. Its recording, subject, annotation
position, original symbol, and AAMI class are stored in a metadata
table.
Preliminary waveform and segment statistics
Representative segments from all four classes were visualized. Waveform
morphology and amplitude varied across classes, but substantial overlap
was also observed.

The following descriptive features were calculated for each segment:
mean amplitude, standard deviation, minimum amplitude, maximum
amplitude, and peak-to-peak amplitude.

  Class     Mean standard deviation   Mean peak-to-peak
  F                          0.6240              2.9952
N                          0.3462              2.0059
S                          0.2335              1.5100
V                          0.6320              2.5922

These values are descriptive only. Their distributions overlap and do
not establish reliable class separability.
Signal representation comparison
Three signal representations were created without overwriting the
original segments:
- P0 --- Raw: Original ECG segments.
- P1 --- Mean-centered: The mean of each heartbeat segment was
  subtracted.
- P2 --- Edge-corrected: The average of the first and last 30
  samples was subtracted from the segment.
The edge-based measure is exploratory. Because the segment edges may
contain neighbouring cardiac activity, it should not be interpreted as a
direct measurement of baseline wander.

  Statistic                       P0: Raw   P1: Mean-centered   P2: Edge-corrected
  Overall mean                   -0.32471             0.00000              0.03531
Overall standard deviation      0.50314             0.40066              0.42166
Minimum                        -5.12000            -3.37314             -3.94950
Maximum                         5.11500             6.76528              5.82442

The preprocessing operations changed the amplitude distributions. In
particular, mean-centering and edge correction sometimes increased
maximum absolute amplitude. Their value must therefore be assessed
experimentally rather than assumed.
Investigation of recording 116
Recording 116 showed unusual repeated extreme amplitudes compared with
recording 100.

  Statistic                   Recording 100   Recording 116
  Minimum amplitude (mV)             -2.715          -5.120
Maximum amplitude (mV)              1.435           5.115
Mean amplitude (mV)               -0.3063         -0.9363
Standard deviation (mV)            0.1932          0.6641

Recording 116 contained 663 samples at -5.120 mV and 265 samples at
5.115 mV. Consecutive extreme-value episodes lasted up to 255 samples at
the minimum and 265 samples at the maximum. This provides evidence
consistent with possible ADC saturation or clipping; the final
signal-quality interpretation should remain appropriately cautious.

Six extracted heartbeat segments in recording 116 overlapped these
identified extreme-value regions. All six were class N.
Sensitivity dataset
The original dataset was retained as the primary dataset. A separate
sensitivity dataset was created after excluding the six flagged
heartbeat segments.

  Dataset                 Heartbeat segments
  Primary dataset                    100,693
Sensitivity dataset                100,687
Difference                               6

The class distribution changed negligibly after excluding the six
segments:

  Class     Primary   Sensitivity   Primary %   Sensitivity %
  N          90,102        90,096    89.48189        89.48126
S           2,781         2,781     2.76186         2.76202
V           7,008         7,008     6.95977         6.96018
F             802           802     0.79648         0.79653

The largest percentage-point change was 0.00063 for class N. The
sensitivity dataset will allow us to investigate whether the flagged
segments affect model results without materially changing class
proportions.
Saved local data files
The following files were saved locally in
ECG_Research/processed_data/:

  File                                   Description
  X_ecg_raw.npy                        Primary raw ECG heartbeat segments
heartbeat_metadata.csv               Primary dataset metadata
X_ecg_raw_sensitivity.npy            Sensitivity ECG segments
heartbeat_metadata_sensitivity.csv   Sensitivity dataset metadata

These are local experimental outputs. Large datasets and model
checkpoints should not be committed to GitHub unless there is a clear
reason and suitable data-distribution permission.
Evaluation considerations
The dataset contains 43 unique subjects. Heartbeats from one subject
must not be split between training and testing sets.

A Leave-One-Subject-Out (LOSO) structure was inspected, resulting in 43
outer folds, with one subject held out in each fold. This was a
structural split check; no model training or evaluation has yet been
performed.

Model tuning, feature selection, scaling, and resampling must be carried
out using only the training subjects within each outer fold. A nested or
otherwise properly grouped validation procedure will be needed.

The four-class dataset is highly imbalanced, particularly for F.
Accuracy alone will not be sufficient. Macro-F1, balanced accuracy,
per-class precision/recall/F1, confusion matrices, and suitable
precision-recall analyses will be considered.
Current status of Notebook 02
  Activity                            Status
  Recording and subject               Completed
identification                      

  Provisional AAMI mapping and class  Completed provisionally;
selection                           verification pending

  Heartbeat annotation extraction     Completed

  Heartbeat segmentation              Completed

  Preliminary waveform inspection     Completed

  Segment-level descriptive           Completed
statistics                          

  Signal representation comparison    Completed

  Recording 116 signal-quality        Completed provisionally
investigation                       

  Sensitivity dataset creation        Completed

  Dataset saving                      Completed

  Engineered feature extraction for   Not started
model training                      

  Traditional ML model training       Not started

  Deep-learning model training        Not started

  Hybrid model development            Not started

  Diagnostic-driven model selection   Not started
experiment                          
Final subject-independent model     Not started
evaluation
6. Technologies and Tools
The experiments currently use:
- Python
- Jupyter Notebook
- NumPy
- Pandas
- SciPy
- Matplotlib
- Seaborn
- WFDB
- Statsmodels
- Scikit-learn
Additional modelling libraries may be added when the model experiments
begin.

## 10. Citation and Data Acknowledgement

The MIT-BIH Arrhythmia Database is provided by PhysioNet.

Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*. 2001;20(3):45–50.

Dataset: https://physionet.org/content/mitdb/

---

**Repository:** [ecg-diagnostic-driven-framework](https://github.com/hmdsahed/ecg-diagnostic-driven-framework)
