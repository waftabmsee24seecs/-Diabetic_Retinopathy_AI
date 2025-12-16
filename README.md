# 🩺 VisionAI – Retina & Diabetic Retinopathy Detection

Welcome to **VisionAI**, an intelligent healthcare application designed to detect **retinal images** and analyze them for **Diabetic Retinopathy (DR)** in real-time.  
The project leverages **deep learning models** and provides an **interactive Streamlit interface** for easy usage, testing, and performance evaluation.

---

## 🌟 Project Overview

VisionAI assists healthcare professionals and researchers by automating retina and DR detection.

### Application Workflow
1. Provide retinal images via:
   - 📷 **Camera capture**
   - 📁 **File upload** (JPG, JPEG, PNG)
2. Automatically detect whether the image is a **retina image**
3. Analyze the retina for **Diabetic Retinopathy**
4. Display **real-time results** with confidence scores

The application features a **modern horizontal UI**, **healthcare-themed design**, and **interactive Streamlit controls**.

---

## 🧠 Key Features

- Dual-step detection pipeline (Retina → DR)
- Real-time inference with confidence metrics
- Streamlit-based interactive UI
- Horizontal layout for improved visualization
- Jetson Nano / NVIDIA GPU compatible
- Portable Conda environment (USB-friendly)

---

## 🗂️ Repository Structure & Branch Workflow

### Branch Strategy
- **main** — Stable, production-ready application
- **development** — Active development & feature upgrades
- **code-and-simulation** — Experimental models and simulations

### Directory Structure
```
- Diabetic_Retinopathy_AI/
│── Reto2.0/
│   └── src/
│       └── codes/
│           ├── EvalPerfomence_App.py
│           ├── requirements.txt
│           └── ...
│── env/                # Conda environment (created locally)
│── README.md
```

---

## 💻 Installation & Setup (Conda Environment)

### Clone the repository
```bash
git clone https://github.com/waftabmsee24seecs/-Diabetic_Retinopathy_AI.git
cd -Diabetic_Retinopathy_AI
```

### Create a Conda environment
```bash
conda create -p env python=3.10 -y
```

### Activate the environment
```bash
conda activate ./env
```

### Navigate to source code
```bash
cd src/codes/
```

### Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Application
```bash
conda activate ./env
streamlit run Streamapp.py
```

📌 Open your browser at:  
**http://localhost:8501**

---

## 🛠 Requirements

- Python 3.10
- Linux Ubuntu ≥ 18.04
- Webcam (for live capture)
- Streamlit
- OpenCV
- TensorFlow ≥ 2.20
- NumPy
- Pillow
- Pandas (optional)
- CUDA-enabled GPU (optional, recommended for Jetson)

---

## 📌 Notes

- Conda environment is created **inside the project directory**
- Always activate the environment before running Streamlit
- Tested on **Linux / NVIDIA Jetson platforms (CPU Based For Jetson Nano 4Gb Kit)**

---

## 👥 Group Members

| Name           | Role                                   |
|----------------|----------------------------------------|
| Ammar Khan     | Embedded Systems / Hardware Integration |
| Manahil Sheikh | Algorithm Design & Simulation           |
| Waleed Aftad   | Project Lifecycle & Documentation       |

---

## 📜 License

This project is intended for **academic and research purposes**.