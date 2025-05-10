# AI-Driven Fake Profile Detection System
This project is a comprehensive Fake Profile Detection system designed as a major academic project. It leverages advanced machine learning models and multi-modal analysis techniques to identify fake social media profiles with high accuracy. The system integrates text, image, and behavioral analysis to provide robust detection capabilities. The project includes a user-friendly dashboard for real-time monitoring, model performance visualization, and risk scoring, making it a practical tool for social media platforms and researchers.

## Overview
This project implements a comprehensive AI-driven system to detect fake profiles using multi-modal analysis:
- Text-Based Analysis (NLP on username, bio, posts, comments, sentiment)
- Image-Based Analysis (Computer Vision and Deep Learning for profile and post images, deepfake detection)
- Behavior-Based Analysis (Pattern recognition on account activity, follower ratio, posting frequency, network analysis)

The system includes data preprocessing, feature engineering, model training/testing with multiple ML models, model optimization, and a real-time interactive dashboard built with Streamlit.

## Folder Structure
- `data/`: Dataset and preprocessing scripts
- `analysis/`: Modules for text, image, and behavior analysis
- `models/`: Model training and evaluation scripts
- `dashboard/`: Streamlit dashboard app and visualization scripts
- `utils/`: Helper functions
- `config/`: Configuration files
- `main.py`: Main entry point for training and evaluation
- `requirements.txt`: Python dependencies

## Setup and Run
1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```

## Features
- Multi-modal fake profile detection
- Model comparison and optimization
- Real-time dashboard with visualizations and explainability
- Export results to PDF/CSV

# Fake Profile Detector Project

## How to Run the Project

To run the project successfully, follow these steps:

1. **Start the API Server**

Open a terminal and run the following command to start the API server:


This will start the Flask API server on `http://localhost:5000`.

2. **Start the Dashboard Application**

In a separate terminal, run the following command to start the dashboard app:


This will start the Dash dashboard on `http://127.0.0.1:8050`.

3. **Access the Dashboard**

Open your web browser and navigate to:

http://127.0.0.1:8050


You should see the full dashboard with all visualizations and functionalities.

---

## Notes

- Make sure both servers are running concurrently for the dashboard to communicate with the API.
- If you encounter any issues, check the terminal outputs for errors.
- Clear your browser cache or use incognito mode if you face rendering issues.
- The project requires Python packages listed in `requirements.txt`. Install them using: 

pip install -r requirements.txt


---

For any questions or contributions, feel free to open an issue or submit a pull request. Enjoy exploring the Fake Profile Detector! or (mail me - shreyasbabukv@gmail.com)
