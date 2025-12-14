# Project Progress Logger

## 29 July 2025 🤝

- First meeting with mentor Jinesha Kothari  
  - Discussed project expectations, deliverables, and timelines  
  - Reviewed and validated our approach to the problem statement  

---

## 30 July 2025 🧠

- Team deep-dive into convolutional neural networks  
  - Studied CNN architecture, layers, and feature-extraction concepts  
- First build of the pneumonia-detection model  
  - Encountered longer training times due to high epoch counts  

---

## 31 July 2025 🖥️

- Frontend development alongside model training  
  - Set up file upload, preview, and analysis UI scaffold  
- Monitored training progress and outlined next integration steps  

---

## 1 August 2025 📈

- Model performance analysis and debugging

- Discovered that the initial low accuracy was due to a class imbalance issue, leading to a "lazy" classifier.

- Implemented class weighting to correct the model's bias towards the majority class.
  
- Achieved a validation accuracy of 56%

---

## 2 August 2025 💻

- Backend development on the Flask server

  - Successfully integrated the model into the backend. The server can now load the saved .h5 model file.

  - Developed API endpoints to handle file uploads and return real-time predictions.

- Frontend development for seamless user experience

  - Connected the file upload and preview UI to the Flask backend.

  - Designed and implemented a results dashboard to display the model's prediction, confidence score

- The project is now a fully functional, end-to-end application.

---

## 3 August 2025 🛠️

- Integrated Generative AI for personalized health guidance

  - Successfully connected our model's prediction results to the Gemini-pro API.

  - The system now automatically generates personalized, non-diagnostic health and wellness advice for a user based on whether their X-ray was           classified as Pneumonia, Normal, or an Anomaly.

  - This feature adds a valuable layer of user-centricity to the application.

---

## 4 August 2025 🛠️

- Defined a new roadmap for further model improvement
  
  - Switched to a combined training approach, fine-tuning the top layers of the ResNet50 model from the start.
  
  - Achieved a significant jump in validation accuracy from 56% to 81.25%. This is a major breakthrough, and the model is now performing as expected.

  - The goal is to push the validation accuracy beyond 81.25%.

- Outlined a plan to systematically tune hyperparameters, including learning rate and unfreezing more layers in the base model.

- Began integrating explainable AI (Grad-CAM) to visualize model decisions, which will be a key feature for transparency.

---

## 5 August 2025 🚀

- Final model training and evaluation

  - Completed the final combined training phase, achieving a significant boost in performance.

  - Final test set results were highly encouraging, with a Test Accuracy of 84.94%, a well-balanced Test Precision of 88.34%, and a Test Recall of      87.44%.

  - The model is now performing at a high level and is ready for final deployment.

- Database integration

  - Began integrating a database into the backend of the project.

  - The database will be used to store user information, prediction history, and model performance metrics.

  - This adds a crucial layer of persistence and allows us to track the application's usage and effectiveness over time.

---

## 6 August 2025 🐞

- Initial Anomaly Detector Evaluation

  - Successfully ran the anomaly detector, confirming the model could load and produce reconstruction errors.

  - Discovered a key inconsistency: the model confidently predicted "Pneumonia" but failed to flag the same image as an          anomaly, revealing a lack of sensitivity.

  - Concluded that the initial autoencoder architecture was too simple and that the anomaly threshold was not optimal.

- Anomalies Detected

  - Observed that the model was incorrectly identifying watermarks and text overlays as anomalies in some images.

  - Outlined a plan to retrain the autoencoder on a cleaner dataset and correct the logic for anomaly localization.









