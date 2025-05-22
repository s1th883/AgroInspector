# 🌱 AgroInspector

AgroInspector is a deep learning-based Streamlit web application that assists farmers, researchers, and policymakers in identifying crop legality and plant diseases from images. It combines computer vision and intelligent classification models in a seamless pipeline to ensure agricultural monitoring and decision support.

---

## 🚀 Features

- ✅ Detects whether a crop is **Legal or Illegal**
- 🌾 If Legal:
  - Identifies if the crop is **Healthy or Diseased**
  - If Healthy: Classifies the **crop type**
  - If Diseased: Identifies the **specific disease**
- 📦 Built with custom-trained CNN models
- ⚡ Real-time prediction with Streamlit interface

---

## 🧠 Model Pipeline

1. **Illegal vs Legal Classifier**
2. **Illegal Crop Type Classifier** (if illegal)
3. **Healthy vs Diseased Classifier** (if legal)
4. **Healthy Crop Type Classifier** (if healthy)
5. **Disease Classifier** (if diseased)

---

## 🛠️ Tech Stack

- Python, PyTorch
- Streamlit
- OpenCV, NumPy, Pandas
- Custom CNN architectures
- Trained using GPU-enabled Google Colab

---

## 📁 File Structure

Agroinspector-Web/
├── app.py
├── requirements.txt
├── AgroInspectorApp/
│ ├── *.pth (model weights)
│ └── helper functions

---

## 🧪 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
