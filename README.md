# 📡 Range Prediction from CT using Deep Learning & Physics

## 🚀 Overview
This project predicts **Water Equivalent Thickness (WET)** from CT volumes using a hybrid approach combining:
- Deep Learning (U-Net)
- Physics-based modeling
- Residual learning

---

## 🧱 Project Structure
```
.
├── dataset_range.py
├── physics.py
├── unet.py
├── train_range_model.py
├── pinn_wet_deepxde.py
├── *.pt
```

---

## ⚙️ Pipeline

### 1. Data Loading
- Sparse CT data → reconstructed into 3D volumes (128×128×128)

### 2. Physics Modeling
HU → Relative Electron Density → SPR → WET

### 3. Baseline
Simple geometric depth-based WET approximation.

### 4. Deep Learning
Residual learning:
WET_pred = WET_baseline + UNet(CT)

---

## 🧠 Model
- 2D U-Net
- Encoder-decoder with skip connections
- Predicts residual correction

---

## 🏋️ Training
Run:
```
python train_range_model.py
```

Key parameters:
- epochs: 40
- batch size: 8
- learning rate: 1e-3

---

## 📊 Output
- Metric: MAE (converted to cm)

---

## 🔮 Future Work
- 3D models
- Multi-axis prediction
- Better physics integration

---

## 📦 Requirements
```
pip install torch numpy deepxde
```
