# Systematic Corrections to Paper Misrepresentations in SLR

## Date: 2025-12-31
## Document: access.tex

---

# PART I: Temperature Network Paper - Final Refinements

## Date: 2025-12-31
## Document: access.tex

### Summary of Inaccuracies Identified and Corrected

This section summarizes the final refinements made to address overstatements and omissions regarding the Temperature Network paper (Zhu et al., 2021) and performance reporting for Zhou et al. (2022).

---

## 1. ✅ CORRECTED: Performance Reporting for Zhou et al. (2022)

**Location:** Section V.A.2 (line 538)  
**Correction:** 
- Corrected **5-shot accuracy** for 3-way tasks on ISIC 2019 to **75.67%**.
- Clarified that **70.12%** refers to the **3-shot setting**.

---

## 2. ✅ CORRECTED: Performance Metric and Dataset Attribution (Zhu et al.)

**Location:** Section III.B.1 (line 370) and Section V.A.1 (line 535)  
**Correction:** 
- Corrected **Dermnet performance reporting**: The paper achieves a **3.32% improvement in 5-way 5-shot accuracy** over the second-best GNN baseline (the 63.37% figure previously cited refers to Stanford Dogs).
- Clarified that **52.39%** refers to **5-way 1-shot accuracy on miniImageNet**.
- Specified that for **Dermnet**, query samples were reduced to **5 per category** because the **smallest category contains only 10 images**, making the standard 15-query protocol impossible for 5-shot evaluation (not due to long-tail constraints).

**Rationale:** Precise reporting of metrics and experimental conditions is essential for scientific integrity and comparison fairness.

---

# PART IV: FAA-Net Paper Corrections (Lee et al., 2023)

## Date: 2025-12-31
## Document: access.tex

### Summary of Inaccuracies Identified and Corrected

This section summarizes the final refinements made to address the representation of the FAA-Net paper.

---

## 1. ✅ FINAL REFINEMENT: Methodology and Categorization

**Location:** Table 3 (line 313) and Section IV.C.3 (line 465)  
**Correction:** 
- Recategorized as **Multi-Task (Attention-Based FSL)** to reflect its use of few-shot components without implementing optimization-based meta-learning.
- Updated N-way from "3-class" to **"Fixed 3-class"** to remove any implication of episodic N-way sampling.
- Clarified that **RAC storage sizes (3, 5, 7)** are internal similarity-matching parameters, not episodic shot counts.
- Replaced "meta-learning principles" with **"attention-based few-shot inspired components."**

---

## 2. ✅ CORRECTED: Clinical Context and Scope Justification

**Location:** Section IV.C.3 (line 465)  
**Correction:** 
- Explicitly stated that the targeted diseases (**Rosacea, Dermatitis**) are common inflammatory conditions.
- Added a **justification for inclusion**: The paper is included for its methodological merit in adapting few-shot concepts to **small clinical datasets** where episodic training is not feasible, addressing a key technical challenge in localized clinical studies.

---

## 3. ✅ ARCHITECTURAL ACCURACY: RAC & AFS Blocks

**Location:** Section IV.C.3 (line 465)  
**Correction:** 
- Corrected component naming to **Recyclable Attention Collection (RAC)** and **Amplifying Focused Similarity (AFS) Blocks.**
- Added technical details: RAC is built via **CAMs during pre-training**, and AFS blocks utilize a **Siamese-network triplet input mechanism (P1, P2, P3).**

---

## 4. ✅ IMAGING SYSTEM PRECISION

**Location:** Section IV.C.3 (line 465)  
**Correction:** 
- Specified capture under **370 nm UV excitation** with specific optical filters.

# PART V: FEGGNN Paper Corrections (Noman et al., 2025)

## Date: 2025-12-31
## Document: access.tex

### Summary of Inaccuracies Identified and Corrected

This section summarizes the corrections made to address inaccuracies in the reporting of the FEGGNN paper.

---

## 1. ✅ CORRECTED: Authorship and Publication Details

**Location:** Table 3 (line 302) and Bibliography (line 730)  
**Correction:** 
- Corrected the first author's name to **Abdulrahman Noman** (previously "M. Noman").
- Updated the journal name to the full title: **Computers in Biology and Medicine** (previously abbreviated).

---

## 2. ✅ CORRECTED: Methodology and Evaluation Context

**Location:** Table 3 (line 302) and Section IV.C.1 (line 459)  
**Correction:** 
- Recategorized the methodology as a **Unified Graph-Based Framework** (previously "Hybrid").
- Specified the evaluation settings: **2-way 1-shot and 5-shot** on SD-198 and Derm7pt.
- Clarified the role of **Grad-CAM as an analysis tool** rather than a primary architectural contribution.

# PART VI: SCAN Paper Corrections (Li et al., 2025)

## Date: 2025-12-31
## Document: access.tex

### Summary of Inaccuracies Identified and Corrected

This section summarizes the corrections made to address inaccuracies in the reporting of the SCAN paper.

---

## 1. ✅ CORRECTED: Authorship and Algorithm Naming

**Location:** Table 3 (line 300), Section I (line 73), and Bibliography (line 728)  
**Correction:** 
- Updated first author to **Shuhan Li** (previously abbreviated or "Li (S)").
- Expanded full author list in bibliography: **S. Li, X. Li, X. Xu, and K. T. Cheng**.
- Removed the misleading "Dynamic" prefix from the algorithm name (**Subcluster-Aware Network**).

---

## 2. ✅ CORRECTED: Methodology and Architecture

**Location:** Table 3 (line 300) and Section III.F (line 439)  
**Correction:** 
- Clarified that SCAN is a **dual-branch transfer-learning framework** with subcluster awareness.
- Added mention of key architectural components: **three memory banks** (feature, class centroid, cluster centroid) and **purity loss**.
- Corrected the clustering description: It uses **unsupervised K-means** (initialized once) rather than real-time "dynamic" clustering.
- Specified evaluated backbones: **Conv4/6, ResNet18/34, and WRN-28-10**.

---

## 3. ✅ CORRECTED: Performance and XAI

**Location:** Table 3 (line 300) and Section III.F (line 439)  
**Correction:** 
- Updated performance gains: Corrected to **~6.5% improvement on SD-198** (2-way 5-shot) and **~3.75% on Derm7pt**.
- Updated XAI status to **"Y"** in Table 3, citing the use of **t-SNE visualizations** and qualitative subcluster interpretation.

# PART VII: CD-FSS Paper Corrections (Yixin Wang et al., 2022)

## Date: 2025-12-31
## Document: access.tex

### Summary of Inaccuracies Identified and Corrected

This section summarizes the corrections made to address inaccuracies and omissions in the reporting of the CD-FSS paper.

---

## 1. ✅ CORRECTED: Authorship and Algorithm Naming

**Location:** Table 3 (line 314) and Section III.G (line 473)  
**Correction:** 
- Updated first author name to **Yixin Wang** (previously abbreviated as "Wang (Y)").
- Ensured consistent naming of the first author in narrative text.

---

## 2. ✅ CORRECTED: Methodology and Evaluation Nuance

**Location:** Section III.G (line 473)  
**Correction:** 
- Clarified the **alternating meta-training** schema: Defined **generic learning** (natural images) and **specific learning** (medical images) phases and their mutual promotion.
- Explicitly mentioned that the model was tested on **unseen rare diseases** (melanomas, common nevi, atypical nevi) in the **PH2 dataset**.
- Refined the performance comparison: Clarified that the **93.03% DSC** outperforms a supervised baseline (90.62%) that was limited to **seen classes** within PH2, highlighting the cross-domain transfer capability.

# PART VIII: SS-DCN Paper Corrections (Fu et al., 2024)

## Date: 2025-12-31
## Document: access.tex

### Summary of Inaccuracies Identified and Corrected

This section summarizes the corrections made to address fatal inaccuracies in the reporting of the SS-DCN paper.

---

## 1. ✅ CORRECTED: Backbone and Methodology Categorization

**Location:** Table 3 (line 307) and Section III.E (line 437)  
**Correction:** 
- Specified the backbone details: **Conv4 architecture** including **BatchNorm, LeakyReLU, and MaxPooling** within each of the four 64-channel blocks.
- Clarified that **ResNet50** was only used for baseline comparisons (Meta-Rep) and not as the primary SS-DCN architecture.
- Recategorized the method as a **Unified Framework (SSL+EDC)** rather than a generic "Hybrid" model.

---

## 2. ✅ CORRECTED: Distribution Calibration (EDC) Mechanism

**Location:** Section III.E (line 437)  
**Correction:** 
- Redefined **Enhanced Distribution Calibration (EDC)** as a **sample generation strategy** rather than just a statistical shift.
- Documented the full technical sequence: **Yeo-Johnson transform** for Gaussianization, **top-$k$ base class selection**, and synthetic sample generation from calibrated distributions.

---

## 3. ✅ CORRECTED: Evaluation and Performance Context

**Location:** Section III.E (line 437) and Section III.I (line 535)  
**Correction:** 
- Corrected the validation claim: Acknowledged the **multi-dataset evaluation** across **ISIC2018, Derm7pt, and SD-198**.
- Added architectural context: Specified that the **90.43% accuracy** for 2-way 5-shot on SD-198 is achieved using the **Conv4 backbone**.
- Corrected the experimental settings: Updated K-shot values to **1, 5, 10** (previously 1, 3, 5).
- **Removed general-domain result (miniImageNet)** from the dermatology-specific trend narrative (Zhu et al., 2021) to avoid misleading claims about clinical progress.

**Rationale:** These edits provide an accurate technical representation of the SS-DCN unified framework and ensure that comparative performance and progress metrics are strictly relevant to the dermatological domain.
