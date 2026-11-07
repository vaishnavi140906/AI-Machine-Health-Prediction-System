🏭 AI-Based Machine Health Monitoring and Failure Prediction System

📌 Project Overview

An AI-based predictive maintenance application that uses deep learning to analyze industrial machine sensor parameters and predict potential machine failures.

The system estimates failure probability and identifies machine health conditions to support preventive maintenance and reduce unexpected downtime.

🎯 Objectives

- Monitor industrial machine conditions.
- Analyze sensor parameters using deep learning.
- Predict potential machine failures.
- Estimate failure probability.
- Support predictive maintenance decisions.

⚙️ Technologies Used

- Programming: Python
- Deep Learning: TensorFlow, Keras
- Machine Learning: Scikit-learn, NumPy, Pandas
- Backend: Flask
- Frontend: HTML, CSS, JavaScript
- Development Tools: Google Colab, VS Code, GitHub

📊 Input Parameters

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

🧠 Model Details

- Deep learning-based machine failure prediction.
- Updated V2 trained model.
- StandardScaler for data preprocessing.
- Failure probability estimation.
- Configurable failure threshold: 0.65.

Current Model: "machine_failure_model_v2.keras"

Current Scaler: "machine_scaler_v2.pkl"

🌟 Key Features

- Machine health monitoring.
- Failure risk prediction.
- Sensor-based analysis.
- Health condition classification.
- Prediction history storage.
- Predictive maintenance support.

🚀 Installation and Execution

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL

2. Navigate to the project folder

cd AI_Machine_Predictor

3. Install dependencies

pip install -r requirements.txt

4. Run the application

python app.py

5. Open in browser

http://127.0.0.1:5000

📁 Project Structure

AI_Machine_Predictor/
│
├── app.py
├── static/
├── templates/
├── failure_threshold.txt
├── machine_failure_model_v2.keras
├── machine_failure_model.keras
├── machine_scaler_v2.pkl
├── machine_scaler.pkl
├── prediction_history.json
├── requirements.txt
├── README.md
└── .gitignore

🔮 Future Enhancements

- Real-time IoT sensor integration.
- Advanced anomaly detection.
- Live machine monitoring.
- Automated maintenance recommendations.
- Improved prediction accuracy.

👩‍💻 Developer

Vaishnavi M.

B.Tech – Artificial Intelligence and Data Science

Prathyusha Engineering College
