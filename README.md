# 🩺 VisionAI – Retina & Diabetic Retinopathy Detection

Welcome to **VisionAI**, an intelligent healthcare application designed to detect **retinal images** and analyze them for **Diabetic Retinopathy (DR)** in real-time. This project leverages **deep learning models** for accurate retina and DR detection and provides an **interactive Streamlit interface** for easy usage.  

---

## 🗂️ Repository Structure & Branch Workflow

This repository follows a clean and stable branching strategy to support development, testing, and long-term maintenance of the VisionAI project.

- **main** — Stable, production-ready version of the VisionAI app.  
- **development** — Active development branch for upcoming features and improvements.  
- **code-and-simulation** — Experimental branch for model testing, simulations, and research trials.


## 🌟 Project Overview

VisionAI is designed to assist healthcare professionals and researchers by automating retina and DR detection. The app workflow is simple:

1. **Provide retinal images** via:  
   - **Camera capture** (live image)  
   - **File upload** (JPG, JPEG, PNG)
2. **Automatically detect** whether the image is a retina image.  
3. **Analyze the retina** for Diabetic Retinopathy if a retina is detected.  
4. **Display results** with confidence scores in real-time.  

The application features a **modern, horizontal layout**, interactive **blue-themed buttons**, and a clean **healthcare design**.  

---

## 🧠 Features

- **Dual-step detection pipeline**: Retina detection → DR detection (if retina confirmed)  
- **Real-time results** with confidence metrics  
- **Interactive UI** built with Streamlit  
- **Horizontal layout** for better visualization  
- **Jetson Nano compatible** (CPU/GPU aware)  
- **Portable virtual environment setup** via USB  

---

## 👥 Group Members

| Name             | Role                                      |
|-----------------|-------------------------------------------|
| Ammar            | Embedded System / Hardware Integration    |
| Manahil Sheikh   | Algorithm Design & Simulation             |
| Waleed Aftad     | Project Life Cycle & Documentation        |

---

## Requirements

- Python 3.8
- Linux Ubuntu >= 18 
- Web Cam
- Streamlit
- OpenCV 
- TensorFlow or PyTorch
- NumPy
- Pandas (optional, if used in preprocessing)
- Pillow (for image handling)


## 💻 Installation & Setup

#### Clone the repository:

    git clone https://github.com/waftabmsee24seecs/-Diabetic_Retinopathy_AI.git
    cd VisionAI

#### Create and activate a Python virtual environment:

    cd embeddedcodes/
    python3.8 -m venv venv
    source ./venv/bin/activate

#### Install dependencies from requirements.txt:

    pip install --upgrade pip
    python -m pip install -r requirements.txt

#### Running the Application

    source ./venv/bin/activate
    streamlit run Steramapp.py

**Open your browser at: http://localhost:8501**
