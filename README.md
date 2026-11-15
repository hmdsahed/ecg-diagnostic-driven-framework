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

```text
ecg-diagnostic-driven-framework/
│
├── 01_Data_Inspection_and_Statistical_Analysis.ipynb
├── 02_Hybrid_Model_Selection.ipynb
├── 03_Reusable_Generalization_Framework.ipynb
│
├── .gitignore
├── README.md
│
├── data/          # Local datasets (not tracked by Git)
├── results/       # Experimental outputs
└── src/           # Future reusable implementation
```

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

## 10. Citation and Data Acknowledgement

The MIT-BIH Arrhythmia Database is provided by PhysioNet.

Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*. 2001;20(3):45–50.

Dataset: https://physionet.org/content/mitdb/

---

**Repository:** [ecg-diagnostic-driven-framework](https://github.com/hmdsahed/ecg-diagnostic-driven-framework)
