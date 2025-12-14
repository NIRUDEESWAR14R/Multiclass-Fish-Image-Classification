# 🐟 Multiclass Fish Image Classification

## 📌 Project Overview
The **Multiclass Fish Image Classification** project focuses on classifying fish images into multiple categories using deep learning techniques. A baseline CNN model and a transfer learning approach using **MobileNetV2** were implemented, evaluated, and compared to select the best-performing model. The final model was deployed using **Streamlit** for real-time predictions.

---

## 🎯 Problem Statement
To classify fish images into one of **11 different fish categories** using deep learning models. The project involves model training, evaluation, comparison, and deployment as a web application.

---

## 🏢 Business Use Cases
- Automated fish identification in fisheries
- Marine biology education tools
- Seafood industry product categorization
- AI-powered image recognition systems

---

## 🧠 Skills & Technologies Used
- Python  
- Deep Learning  
- TensorFlow / Keras  
- Convolutional Neural Networks (CNN)  
- Transfer Learning (MobileNetV2)  
- ImageDataGenerator  
- Model Evaluation & Visualization  
- Streamlit (Deployment)

---

## 🗂️ Dataset Description
The dataset consists of fish images organized into training, validation, and test sets.

dataset/

├── train/

├── val/

└── test/


- Each folder contains **11 subfolders**, one per fish category
- Each class has **at least 100 images**
- Data loaded using `ImageDataGenerator`

---

## 🐠 Fish Categories (11 Classes)
1. animal fish  
2. animal fish bass  
3. fish sea_food black_sea_sprat  
4. fish sea_food gilt_head_bream  
5. fish sea_food hourse_mackerel  
6. fish sea_food red_mullet  
7. fish sea_food red_sea_bream  
8. fish sea_food sea_bass  
9. fish sea_food shrimp  
10. fish sea_food striped_red_mullet  
11. fish sea_food trout  

---

## 🔄 Project Workflow
1. Data preprocessing and augmentation  
2. CNN baseline model training  
3. Transfer learning with MobileNetV2  
4. Model evaluation using test data  
5. Confusion matrix and classification report  
6. Model comparison and selection  
7. Streamlit deployment  

---

## 🏗️ Models Implemented

### 🔹 CNN (From Scratch)
- Custom CNN architecture
- Used as a baseline model for comparison

### 🔹 MobileNetV2 (Transfer Learning)
- Pre-trained on ImageNet
- Base layers frozen
- Custom classifier head added
- Selected as the final deployment model

---

## 📊 Model Comparison

| Model | Test Accuracy | F1-score | Training Time | Selected |
|------|--------------|----------|---------------|----------|
| CNN (Scratch) | ~98% | ~0.98 | High | ❌ |
| MobileNetV2 (Transfer Learning) | **99.40%** | **0.99** | Low | ✅ |

---

## 📈 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

**Observation:**  
MobileNetV2 achieved higher accuracy, faster convergence, and improved minority-class performance compared to the CNN baseline.

---

## 🚀 Model Deployment (Streamlit)

A Streamlit web application was built to:
- Upload a fish image
- Predict the fish category
- Display confidence score

### ▶️ Run the Streamlit App
```bash
streamlit run src/app.py

---

## 📁 Project Structure

---

## 📌 Conclusion

- CNN served as a strong baseline model
- MobileNetV2 achieved 99.4% test accuracy
- Transfer learning improved generalization and efficiency
- MobileNetV2 selected for real-time deployment using Streamlit

---

## 🔮 Future Enhancements

- Address class imbalance using class weighting
- Add Grad-CAM visualizations
- Support external (internet) images
- Cloud deployment (AWS / GCP / Azure)

---


# 👨‍💻 Developer
 
  Nirudeeswar R
 
 📍 Chennai
 
 🎓 B.Tech CSE
 
 📧 nirudeeswarr14@gmail.com
