Corrected Kaggle Dataset Summary

Context

This dataset is a pre-processed and restructured version of the dataset originally featured in Andrzejak et al. (2001), commonly used for epilepsy and tumor-related EEG signal analysis.

Original Study Overview

Reference: Andrzejak RG, Lehnertz K, Rieke C, Mormann F, David P, Elger CE (2001). "Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state," Physical Review E, 64, 061907.

Dataset Description:

The original study dataset consists of EEG recordings from 10 subjects total:

5 healthy volunteers: EEG recorded from the scalp surface in two states—eyes open and eyes closed.

5 epilepsy patients: Intracranial EEG recorded from within and outside the epileptogenic zone during seizure-free intervals and during seizures.

Each subject provided multiple EEG segments, each of 23.6 seconds duration. The EEG data was sampled at 173.61 Hz, resulting in 4097 data points per segment.

Content:

Each EEG segment was further divided into chunks, with each chunk containing 178 data points corresponding to approximately 1 second of EEG recording. The final dataset thus consists of segmented EEG data organized into multiple classes defined by brain state and recording conditions:

Class A: EEG from healthy subjects, eyes open

Class B: EEG from healthy subjects, eyes closed

Class C: EEG from epilepsy patients, recorded from brain regions opposite the tumor location during seizure-free intervals

Class D: EEG from epilepsy patients, recorded from the epileptogenic zone during seizure-free intervals

Class E: EEG from epilepsy patients, recorded during epileptic seizures

The primary response variable (y) is in column 179, and the explanatory variables (X1 to X178) are EEG amplitude measurements over 1 second.

Classification Task

The main motivation for restructuring the data was to simplify analysis and facilitate binary or multi-class classification tasks such as:

Seizure detection (class 1 vs. classes 2-5)

Healthy vs. pathological conditions (classes 4 & 5 vs. classes 1-3)

Dataset Source

The dataset is originally from the study by Andrzejak et al. (2001), with this simplified, restructured version collected from the UCI Machine Learning Repository.
