# Rooftop Solar Potential Estimation

An end-to-end deep learning pipeline for **automated rooftop segmentation** and **solar potential estimation** from high-resolution aerial imagery.  

Uses HRNet + OCR with boundary-aware modules to produce accurate rooftop masks, estimate usable area, and compute annual energy generation with PVGIS data.

---

## 🌟 Features

- HRNet + OCR backbone with boundary refinement
- Two-stage training: RID pretraining + Indian fine-tuning
- Post-processing with CRF and edge filtering
- Usable area estimation in m²
- Solar energy yield estimation via PVGIS

---

## 📂 Project Structure

