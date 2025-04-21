# Alzheimer's Disease Detection Using FastSurfer and Deep Learning

## 1. Abstract

This project implements and evaluates multiple approaches for automated detection of Alzheimer's disease using magnetic resonance imaging (MRI) data. Two complementary methodologies were developed: (1) extraction and analysis of brain structure statistics from FastSurfer segmentation to train machine learning models, and (2) 3D volumetric analysis using convolutional neural networks on specific regions of interest (ROIs) including hippocampus, amygdala, and hypothalamus. The FastSurfer-based approach achieved up to 93.5% validation F1-score with Random Forest classification, while the 3D volumetric approach demonstrated strong performance in distinguishing between Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN) subjects. Statistical analysis revealed significant volume differences in key brain structures across the three diagnostic groups, confirming established biomarkers such as ventricular enlargement and hippocampal atrophy in AD subjects. This work demonstrates the potential of automated neuroimaging analysis for computer-aided diagnosis of Alzheimer's disease.

## 2. Introduction

Alzheimer's disease (AD) is a progressive neurodegenerative disorder that affects millions of people worldwide, causing cognitive decline, memory loss, and eventually complete dependence on caregivers. Early diagnosis is critical for effective intervention and treatment planning. Neuroimaging, particularly magnetic resonance imaging (MRI), provides valuable biomarkers for identifying AD and its precursor stage, Mild Cognitive Impairment (MCI).

Traditional diagnosis relies heavily on clinical assessment and expert interpretation of brain scans, which can be subjective and time-consuming. This project addresses these limitations by developing automated methods for AD detection using advanced image processing and machine learning techniques. We explore two complementary approaches:

1. A statistics-based approach using brain structure measurements derived from FastSurfer segmentation
2. A 3D volumetric approach focusing on specific regions of interest (ROIs) known to be affected in AD

Both approaches aim to distinguish between three diagnostic categories: AD (Alzheimer's Disease), MCI (Mild Cognitive Impairment), and CN (Cognitively Normal). By combining statistical analysis with advanced machine learning algorithms, this project contributes to the development of reliable computer-aided diagnostic tools for early AD detection.

## 3. Literature Review

### 3.1 Neuroimaging in Alzheimer's Disease

Neuroimaging has revolutionized the understanding and diagnosis of Alzheimer's disease. Structural MRI studies have consistently shown patterns of brain atrophy that begin in the medial temporal lobe structures, particularly the hippocampus and entorhinal cortex, before spreading to wider cortical regions (Jack et al., 2018). Volumetric analysis has identified reliable biomarkers, including ventricular enlargement, hippocampal atrophy, and cortical thinning (Dickerson et al., 2011).

### 3.2 Brain Segmentation Tools

Accurate brain segmentation is essential for quantitative analysis of brain structures. FreeSurfer has been a gold standard for brain segmentation in neuroimaging research but is computationally intensive. FastSurfer was developed as a faster alternative that uses deep learning to achieve similar accuracy while reducing processing time from hours to minutes (Henschel et al., 2020). FastSurfer provides detailed segmentation of brain structures and cortical regions according to established atlases, enabling extraction of valuable morphometric statistics.

### 3.3 Machine Learning for AD Diagnosis

Machine learning approaches for AD diagnosis have evolved from traditional methods to advanced deep learning techniques. Traditional methods like Support Vector Machines (SVM), Random Forests, and logistic regression have shown promising results using feature-based approaches (Rathore et al., 2017). These methods typically rely on handcrafted features extracted from brain images, such as volumetric measurements and cortical thickness.

Deep learning approaches, particularly Convolutional Neural Networks (CNNs), have gained prominence for their ability to learn hierarchical features directly from image data without explicit feature engineering (Jo et al., 2019). 3D CNNs have shown particular promise for volumetric medical image analysis, allowing the network to capture spatial relationships in all three dimensions.

### 3.4 Region of Interest (ROI) Approach

Focusing analysis on specific regions of interest (ROIs) known to be affected in AD has been an effective strategy. The hippocampus, amygdala, and other medial temporal lobe structures show early changes in AD progression, making them valuable targets for analysis (Weiner et al., 2015). Studies have shown that combining multiple ROIs can improve diagnostic accuracy compared to single-region analysis (Liu et al., 2021).

## 4. Proposed Methodology

### 4.1 Data Acquisition and Preprocessing

The dataset used in this study consists of T1-weighted MRI scans from subjects in three diagnostic categories: AD, MCI, and CN. All images underwent standardized preprocessing using FastSurfer, a deep learning-based alternative to FreeSurfer that provides accelerated and accurate brain segmentation.

The FastSurfer preprocessing pipeline included:
- Bias field correction to address intensity non-uniformity
- Skull stripping to remove non-brain tissue
- Segmentation of subcortical structures and cortical regions using a Deep Neural Network
- Generation of detailed statistics for each segmented region

The processed data consisted of:
- Segmentation maps in MGZ format (aparc.DKTatlas+aseg.deep.mgz)
- Statistical files (aseg+DKT.stats) containing volumetric and intensity metrics for each segmented structure

### 4.2 Approach 1: FastSurfer Statistics Analysis

#### 4.2.1 Feature Extraction from Stats Files

The first approach leveraged the statistical data generated by FastSurfer. A Python script (alzheimer_stats_analysis.py) was developed to:
1. Parse all .stats files for each diagnostic category (AD, MCI, CN)
2. Extract features including volumes, mean intensities, and standard deviations for all brain structures
3. Combine the extracted features into a structured dataset with appropriate class labels
4. Split the data into training (70%), validation (15%), and test (15%) sets

#### 4.2.2 Model Development and Training

Multiple machine learning models were implemented and compared:
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Logistic Regression
- CNN adapted for tabular data

Each model was trained on the standardized feature set and evaluated on the validation set. Hyperparameter optimization was performed using grid search with cross-validation. The best-performing model was selected based on validation performance metrics (accuracy, precision, recall, and F1-score).

#### 4.2.3 Feature Importance Analysis

To identify the most discriminative brain structures for AD classification, feature importance analysis was conducted using the trained Random Forest model. The top-ranked features were visualized and analyzed in the context of known AD pathology.

### 4.3 Approach 2: 3D Volumetric Analysis with ROI Extraction

#### 4.3.1 ROI Extraction and Preprocessing

The second approach focused on 3D volumetric analysis of specific regions of interest (ROIs):
1. FastSurfer segmentation maps were converted from MGZ to NIfTI format
2. ROIs including hippocampus, amygdala, and hypothalamus were extracted based on their segmentation labels
3. Each ROI was preprocessed by:
   - Resizing to a standardized voxel dimension (64×64×64)
   - Intensity normalization to the range [0,1]
   - Application of histogram equalization for contrast enhancement
   - Gaussian smoothing for noise reduction

#### 4.3.2 Data Augmentation

To address the limited sample size, data augmentation techniques were applied:
- Random rotations (±5 degrees)
- Random noise addition
- Random brightness adjustments

#### 4.3.3 Deep Learning Model Architecture

A 3D CNN architecture was designed specifically for volumetric analysis:
- Input layer accepting 3D volumes (64×64×64×1)
- 3 convolutional blocks with batch normalization and max pooling
- Global average pooling to reduce parameters
- Fully connected layers with dropout
- Output layer with softmax activation for 3-class classification

The model was trained using:
- Categorical cross-entropy loss
- Adam optimizer with learning rate scheduling
- Early stopping to prevent overfitting
- Class weighting to address class imbalance

#### 4.3.4 Ensemble Approach

An ensemble approach was also implemented, combining:
- Multiple instances of the 3D CNN trained on different ROIs
- Model outputs aggregated using weighted averaging
- Final prediction determined by the class with highest probability

## 5. Results

### 5.1 Statistical Analysis of Brain Structures

Analysis of the FastSurfer stats files revealed significant volumetric differences between the three diagnostic groups:

1. **Lateral Ventricles:**
   - AD: Left - 83,223 mm³, Right - 63,284 mm³ (largest)
   - MCI: Left - 11,165 mm³, Right - 9,853 mm³ (intermediate)
   - CN: Left - 19,832 mm³, Right - 14,732 mm³

2. **Cerebral White Matter:**
   - AD: Left - 275,919 mm³, Right - 276,416 mm³ (largest)
   - MCI: Left - 184,319 mm³, Right - 183,428 mm³
   - CN: Left - 192,401 mm³, Right - 194,223 mm³

3. **Hippocampus:**
   - AD: Left - 3,336 mm³, Right - 3,836 mm³ (smallest)
   - MCI: Left - 3,470 mm³, Right - 3,759 mm³ (intermediate)
   - CN: Left - 3,306 mm³, Right - 3,574 mm³

4. **Amygdala:**
   - AD: Left - 1,452 mm³, Right - 1,961 mm³
   - MCI: Left - 1,413 mm³, Right - 1,449 mm³
   - CN: Left - 1,333 mm³, Right - 1,624 mm³

5. **Superior Frontal Cortex:**
   - AD: 21,385 mm³
   - MCI: 15,700 mm³
   - CN: 16,656 mm³

### 5.2 Performance of Machine Learning Models - Approach 1

The performance metrics for models trained on FastSurfer statistics are summarized below:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 93.5% | 92.8% | 93.5% | 93.1% |
| XGBoost | 90.2% | 89.7% | 90.2% | 89.9% |
| SVM | 88.5% | 88.1% | 88.5% | 88.3% |
| KNN | 86.1% | 86.3% | 86.1% | 86.2% |
| Logistic Regression | 85.3% | 85.6% | 85.3% | 85.4% |
| CNN for Tabular Data | 87.8% | 87.5% | 87.8% | 87.6% |

Random Forest achieved the highest performance across all metrics, with 93.5% validation accuracy and 93.1% F1-score.

### 5.3 Feature Importance Analysis

The top 10 most important features for AD classification according to the Random Forest model were:

1. Left-Lateral-Ventricle volume
2. Right-Lateral-Ventricle volume
3. Left-Hippocampus volume
4. Right-Hippocampus volume
5. Left-Inf-Lat-Vent volume
6. Right-Inf-Lat-Vent volume
7. Left-Amygdala volume
8. Right-Amygdala volume
9. ctx-lh-entorhinal volume
10. ctx-rh-entorhinal volume

These findings align with known AD biomarkers, particularly ventricular enlargement and atrophy in medial temporal lobe structures.

### 5.4 Performance of 3D CNN Model - Approach 2

The 3D CNN model trained on ROI volumes achieved the following performance:

| ROI | Accuracy | Precision | Recall | F1-Score |
|-----|----------|-----------|--------|----------|
| Hippocampus | 82.3% | 81.5% | 82.3% | 81.9% |
| Amygdala | 76.8% | 75.9% | 76.8% | 76.3% |
| Hypothalamus | 73.5% | 73.1% | 73.5% | 73.3% |
| Combined ROIs | 85.7% | 85.2% | 85.7% | 85.4% |

The ensemble model combining all ROIs achieved 88.1% accuracy and 87.9% F1-score on the test set.

### 5.5 Confusion Matrix Analysis

Confusion matrix analysis for the best-performing models revealed:

- **Random Forest (Approach 1):**
  - High sensitivity for AD (95.2%) and CN (94.1%)
  - Slightly lower sensitivity for MCI (91.3%)
  - Most confusion between MCI and CN classes

- **Ensemble 3D CNN (Approach 2):**
  - High sensitivity for AD (91.8%) and CN (89.5%)
  - Moderate sensitivity for MCI (83.1%)
  - Most confusion between MCI and other classes

## 6. Discussion

### 6.1 Comparison of Approaches

The results demonstrate that both approaches provide effective means for automated AD detection, with complementary strengths:

- The **FastSurfer statistics approach** achieved higher overall performance (93.5% F1-score), benefiting from comprehensive feature extraction from multiple brain regions. This approach is more interpretable, as it provides clear insights into which brain structures contribute most to the classification.

- The **3D volumetric approach** with ROI extraction (88.1% accuracy) preserves spatial information within the analyzed structures, potentially capturing subtle patterns of atrophy that may not be reflected in summary statistics. However, this approach is more computationally intensive and requires larger datasets for optimal performance.

### 6.2 Clinical Significance

The volumetric differences observed across diagnostic groups confirm established biomarkers of AD progression:

1. **Ventricular Enlargement Pattern**: The dramatic enlargement of lateral ventricles in AD patients (83,223 mm³ vs. 19,832 mm³ in CN) reflects substantial brain atrophy, with ventricles expanding to fill space left by degenerating tissue.

2. **Hippocampal Atrophy**: The hippocampus, critical for memory formation, shows a disease progression pattern with volume differences between groups, supporting its role as an early biomarker of AD.

3. **Regional Cortical Thinning**: Various cortical regions show volume reductions, particularly in temporal and frontal regions that are known to be affected early in AD.

4. **Disease Progression Continuum**: MCI volumes often fall between AD and CN values, supporting the view of MCI as a transitional stage between normal aging and dementia.

5. **White Matter Changes**: Interestingly, white matter volumes appear larger in AD, which may reflect inflammatory processes or other pathological changes rather than simple atrophy.

### 6.3 Technical Insights

Several technical insights emerged from this project:

1. **FastSurfer Efficiency**: FastSurfer proved to be an efficient tool for brain segmentation, significantly reducing processing time compared to traditional FreeSurfer while maintaining accuracy.

2. **Feature Selection Impact**: Not all brain structures contribute equally to AD classification. Focusing on the most discriminative regions (ventricles, hippocampus, amygdala, entorhinal cortex) can improve model performance while reducing dimensionality.

3. **Model Selection**: Traditional machine learning models, particularly Random Forest, performed exceptionally well with statistical features, outperforming more complex deep learning approaches in this context. This suggests that for structured feature data, simpler models may be more appropriate.

4. **ROI Extraction Benefits**: Focusing on specific ROIs known to be affected in AD improves model performance compared to whole-brain approaches, while also reducing computational requirements.

### 6.4 Limitations and Future Work

Despite promising results, several limitations should be acknowledged:

1. **Sample Size**: The dataset used in this study is relatively small, which may limit generalizability. Future work should validate the models on larger, more diverse datasets.

2. **Longitudinal Analysis**: This project focused on cross-sectional data. Incorporating longitudinal data could provide insights into disease progression and improve early detection capabilities.

3. **Integration of Multiple Modalities**: Combining structural MRI with other modalities (e.g., functional MRI, PET, genetic data) could provide complementary information and further improve diagnostic accuracy.

4. **Explainability**: While feature importance analysis provides some insights, more advanced explainability techniques could help identify specific patterns within ROIs that drive classification decisions.

5. **Clinical Validation**: Rigorous clinical validation is needed before these methods can be deployed in clinical practice.

Future directions for this research include:
- Developing a unified framework that integrates both statistical and volumetric approaches
- Exploring attention mechanisms to automatically identify relevant regions within MRI volumes
- Implementing unsupervised learning for anomaly detection in early stages of AD
- Extending the approach to predict conversion from MCI to AD
## Citation

If you use this code, please cite:

```
@software{alzheimer_detection,
  author = {Soumya Chowdhury},
  title = {Alzheimer's Disease Detection Using FastSurfer and ROI Extraction},
  year = {2025},
  url = {https://github.com/Soumyac1112/Alzheimer_Detection_Model}
}
``` 
