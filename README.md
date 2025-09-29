# Hybrid CNN–Transformer Model for Ovarian Cyst Classification

This project implements a lightweight hybrid deep learning model combining Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) for the classification of ovarian cysts in ultrasound images. The goal is to improve both accuracy and interpretability for clinical use.

---

## 📁 Dataset

The dataset contains 2D grayscale ultrasound images of ovarian cysts labeled as either:

- **1 = Complex Cyst** → *Malignant*
- **2 = Simple Cyst** → *Benign*

> **Source**: Acharya UR, Akter A, Chowriappa P, Dua S, Raghavendra U, et al. (2018).  
> *Use of nonlinear features for automated characterization of suspicious ovarian tumors using ultrasound images in fuzzy forest framework.*  
> *International Journal of Fuzzy Systems, 20(5), 1385–1402.*  
> [DOI: 10.1007/s40815-018-0477-2](https://doi.org/10.1007/s40815-018-0477-2)

Please cite the dataset if using for research purposes.

---

## 🧪 Project Objectives

- To analyze the performance of deep learning models on ovarian cyst classification
- To design and train a hybrid CNN–Transformer model
- To apply explainable AI (XAI) tools such as Grad-CAM and attention visualization
- To evaluate performance using accuracy, precision, recall, F1-score, and interpretability

---

 ## ⚙️ Project Structure 
 
 <pre>
   ├── ovarian_cysts/ # Dataset directory
   │ ├── 1 (###).jpg # Complex cyst (malignant)
   │ └── 2 (###).jpg # Simple cyst (benign)
   ├── cnn_transformer_model.ipynb # Google Colab notebook for training
   ├── requirements.txt # Required libraries 
   ├── model/ # Saved model and weights 
   ├── outputs/ # Sample predictions and attention maps 
   └── README.md # Project documentation 
 </pre>

 ---

## 🔧 Tools & Libraries

- Python 3.10+
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib / Seaborn
- scikit-learn
- Grad-CAM
- Google Colab

---

## 🚀 How to Run (Google Colab)

1. Upload the `ovarian_cysts` folder to your Google Colab environment.
2. Open and run the notebook:
   - `cnn_transformer_model.ipynb`
3. Adjust paths and batch sizes if needed.
4. Model training starts after dataset preprocessing.
5. Outputs (metrics, sample predictions, and Grad-CAM maps) will be saved in `outputs/`.

---

## 📊 Performance Metrics

Metrics tracked:
- Training and validation accuracy/loss
- Confusion matrix
- Precision, recall, F1-score
- Grad-CAM visualizations
- Attention heatmaps from Transformer blocks

---

## 📎 Citation

If you use this project, please cite:

```bibtex
@article{acharya2018ovarian,
  title={Use of nonlinear features for automated characterization of suspicious ovarian tumors using ultrasound images in fuzzy forest framework},
  author={Acharya, U Rajendra and Akter, Anika and Chowriappa, Priyadarsini and Dua, Sumeet and Raghavendra, U and Koh, Jeong-Eun and Tan, Jun-Hu and Leong, Seng-Soon and Vijayananthan, A and Hagiwara, Yuki},
  journal={International Journal of Fuzzy Systems},
  volume={20},
  number={5},
  pages={1385--1402},
  year={2018},
  publisher={Springer}
}

 

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/blswXyO9)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=20098761&assignment_repo_type=AssignmentRepo)
