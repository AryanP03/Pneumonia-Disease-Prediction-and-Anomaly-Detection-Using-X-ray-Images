# **Pneumonia Disease Prediction and Anomaly Detection**

## **Project Overview**

This project presents a web-based application designed for the **prediction of pneumonia** and **simulated anomaly detection** from chest X-ray images. Leveraging a deep learning model, the application aims to provide a preliminary assessment of X-ray scans and offer personalized health guidance using generative AI.

Users can upload their chest X-ray images through a simple web interface. The backend processes these images using a pre-trained TensorFlow model to classify them as either "Pneumonia Detected" or "Normal." For transparency, the system visually highlights areas on the X-ray, and for positive cases, it provides general health recommendations powered by Google's Gemini API.

## **Key Features**

* **Pneumonia Prediction**: Utilizes a **ResNet50-based deep learning model** trained on chest X-ray datasets to classify images as "Pneumonia Detected" or "Normal" with a confidence score.  
* **Simulated Anomaly Visualization**: For demonstration purposes, the application visually highlights **random areas** on the X-ray image when pneumonia is detected, simulating where a more advanced anomaly detection system might focus.  
* **Explainable AI (XAI) Concept**: The visual highlighting demonstrates the *concept* of explainable AI by showing regions of interest, even if the current anomaly detection is simulated.  
* **Personalized Health Guidance (Generative AI)**: Integrates the **Google Gemini API** to generate general, empathetic, and helpful health guidance for individuals whose X-rays are classified as "Pneumonia Detected." This guidance focuses on well-being and next steps, such as consulting a healthcare professional.  
* **User-Friendly Web Interface**: A clean and intuitive HTML/CSS/JavaScript frontend allows for easy image uploads and clear display of analysis results and AI-generated guidance.

## **Technologies Used**

* **Backend**:  
  * **Python 3.x**: Core programming language.  
  * **Flask**: Web framework for handling API requests and serving the frontend.  
  * **TensorFlow/Keras**: For building, training, and deploying the deep learning classification model (specifically, a fine-tuned **ResNet50** architecture).  
  * **OpenCV (cv2)**: For image preprocessing (resizing, color conversion) before model inference.  
  * **NumPy**: For numerical operations on image data.  
  * **Google Generative AI (genai)**: For integrating the **Gemini 1.5 Flash** model to generate health guidance.  
* **Frontend**:  
  * **HTML5**: Structure of the web page.  
  * **CSS3**: Styling for a modern and responsive user interface.  
  * **JavaScript**: Client-side logic for image upload handling, dynamic content display, and communication with the Flask backend.

## **Installation Guide**

To set up and run this project locally, follow these steps:

1. **Clone the repository:**  
   git clone https://github.com/eeshchaturvedi/Pneumonia-Disease-Prediction-and-Anomaly-Detection-Using-X-ray-Images 
   cd your-repository-name

2. **Create a Python Virtual Environment (Recommended):**  
   python \-m venv venv  
   \# On macOS/Linux:  
   source venv/bin/activate  
   \# On Windows:  
   venv\\Scripts\\activate

3. Install Python Dependencies:  
   You'll need to create a requirements.txt file in your project's root directory. This file should list all the Python libraries used.  
   \# Example content for requirements.txt:  
   Flask  
   tensorflow  
   opencv-python  
   numpy  
   google-generativeai

   Then, install them using pip:  
   pip install \-r requirements.txt

4. Place Your Trained Model:  
   Ensure your trained deep learning model file, named pneumonia\_resnet\_best\_model\_1.h5, is placed in the same directory as your app.py file.  
5. Configure Google Gemini API Key:  
   Your app.py currently has the API key hardcoded. For better security and manageability, it's highly recommended to use environment variables.  
   * Option A (Recommended \- Environment Variable):  
     Set the GOOGLE\_API\_KEY environment variable before running app.py:  
     \# On macOS/Linux:  
     export GOOGLE\_API\_KEY='YOUR\_API\_KEY\_HERE'  
     \# On Windows (Command Prompt):  
     set GOOGLE\_API\_KEY='YOUR\_API\_KEY\_HERE'  
     \# On Windows (PowerShell):  
     $env:GOOGLE\_API\_KEY='YOUR\_API\_KEY\_HERE'

     Then, modify app.py to read from the environment variable:  
     \# In app.py  
     import os \# Add this import at the top of app.py

     API\_KEY \= os.getenv('GOOGLE\_API\_KEY')  
     if not API\_KEY:  
         print("--- WARNING: GOOGLE\_API\_KEY environment variable not found. Guidance feature will be disabled. \---")  
         genai\_model \= None  
     else:  
         genai.configure(api\_key=API\_KEY)  
         genai\_model \= genai.GenerativeModel('gemini-1.5-flash-latest')  
         print("--- Generative AI Model configured successfully \---")

   * Option B (Current \- Hardcoded):  
     If you choose to keep it hardcoded for local testing, ensure your API key is correctly placed in app.py:
     
     *(**Security Warning**: Hardcoding API keys is not recommended for production environments.)*

## **Usage**

After completing the installation steps:

1. Start the Flask Backend Server:  
   Open your terminal, navigate to the project's root directory (where app.py is located), and run:  
   python app.py

   You should see output indicating that the Flask development server is running, typically on http://127.0.0.1:5000/.  
2. Access the Web Application:  
   Open your web browser and go to http://127.0.0.1:5000/.  
3. Upload an X-ray Image:  
   On the web page, fill in the patient information (Name, Age, Gender) and then use the "Drag and drop your X-ray image here, or click to browse" area to upload a chest X-ray image (JPG, PNG, or DICOM formats are supported, though DICOM handling might require additional libraries not explicitly shown in app.py's cv2.imread).  
4. Analyze and View Results:  
   Click the "Analyze X-ray" button. The application will send the image to the backend for prediction. Once complete, the results will be displayed, including:  
   * The prediction (Pneumonia Detected or Normal).  
   * A confidence score.  
   * The uploaded X-ray image with simulated anomaly highlights (red circles).  
   * AI-generated health guidance (if pneumonia is detected).

## **Important Considerations**

* **Simulated Anomaly Detection**: The current implementation of anomaly detection in app.py generates random circles on the image for visualization purposes. For a real-world application, this would require a dedicated anomaly detection model (e.g., an autoencoder or a more sophisticated XAI technique like Grad-CAM applied to the classification model's output) to highlight actual areas of concern.  
* **Health Guidance Disclaimer**: The AI-generated health guidance is for general informational purposes only and should **not** be considered medical advice. Users should always consult with a qualified healthcare professional for diagnosis and treatment.  
* **Model Training**: The provided training script snippet (training\_model.py) outlines the approach for training the classification model using ResNet50 and ImageDataGenerator. Ensure your pneumonia\_resnet\_best\_model\_1.h5 model is properly trained and saved.

## **Project Structure**

.  
├── app.py                      \# Flask backend application  
├── index.html                  \# Frontend web interface  
├── pneumonia\_resnet\_best\_model\_1.h5 \# Your trained TensorFlow model file  
├── requirements.txt            \# Python dependencies list  
└── README.md                   \# This README file

## **Contributing**

Contributions are welcome\! If you'd like to improve this project, please consider:

* Implementing a more robust, model-based anomaly detection or XAI visualization.  
* Enhancing the UI/UX of the web application.  
* Adding more comprehensive error handling.  
* Expanding the generative AI's capabilities (while maintaining medical disclaimers).

Please fork the repository, create a new branch, make your changes, and open a pull request.

## **License**

This project is open-source. Please add a LICENSE file to your repository to specify the terms under which your project is distributed (e.g., MIT, Apache 2.0).
