# Rooftop Solar Potential Estimation

An end-to-end deep learning pipeline for **automated rooftop segmentation** and **solar potential estimation** from high-resolution aerial imagery.  

Uses HRNet + OCR with boundary-aware modules to produce accurate rooftop masks, estimate usable area, and compute annual energy generation with PVGIS data.

---

## Features

- HRNet + OCR backbone with boundary refinement
- Two-stage training: RID pretraining + Indian fine-tuning
- Post-processing with CRF and edge filtering
- Usable area estimation in m²
- Solar energy yield estimation via PVGIS

---

## Project Structure
data_preparation/ # Tiling images, mask creation
model/ # HRNet + OCR + boundary modules
training/ # Training scripts and configs
evaluation/ # Metrics, CRF post-processing
solar_estimation/ # Area calculation, PVGIS API
Indian_dataset/ # Custom tiled dataset
RID_dataset/ # Public dataset for pretraining


---

## 🔗 References

- HRNet [1], OCR [2], SolarNet [3], SolarNet+ [4]
- PVGIS: [https://ec.europa.eu/jrc/en/pvgis](https://ec.europa.eu/jrc/en/pvgis)

---

## 📜 Selected Papers

[1] Wang et al., *Deep High-Resolution Representation Learning*, IEEE TPAMI, 2021.  
[2] Yuan et al., *Object-Contextual Representations*, ECCV, 2020.  
[3] Malakar et al., *SolarNet*, IEEE TGRS, 2019.  
[4] Malakar et al., *SolarNet+*, RSE, 2022.

---

*Feel free to clone, use, or adapt this project.*


