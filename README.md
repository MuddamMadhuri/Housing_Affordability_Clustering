📌 Project Overview
This project implements an end-to-end integration between a data analysis pipeline and a web application to enable dynamic, model-driven policy recommendations.
Previously, the system suffered from a mismatch:
The analysis pipeline (run_analysis.py) dynamically retrains clustering models.
The web application (app.py) relied on hardcoded cluster-to-policy mappings.
This integration ensures that whenever the model is retrained, the application automatically reflects the updated cluster definitions and policy recommendations.

🎯 Objective
Remove hardcoded policy mappings from the application.
Generate machine-readable cluster definitions during model training.
Dynamically load cluster labels and policy recommendations in the web app.
Maintain application stability with fallback logic.

🏗️ System Architecture
run_analysis.py
     │
     ├── Model Training
     ├── Cluster Statistics
     └── cluster_config.json  ─────────▶  app.py
                                              │
                                              ├── Load Dynamic Config
                                              ├── Predict Cluster
                                              └── Return Policy Recommendation

📂 Repository Structure
├── run_analysis.py          # Data analysis & model training
├── app.py                   # Web application (API layer)
├── cluster_config.json      # Auto-generated cluster definitions
├── test_restore.py          # Automated tests
├── requirements.txt
└── README.md

🔄 Implementation Details
1️⃣ Analysis Component (run_analysis.py)
Purpose
Generate cluster definitions and corresponding policy recommendations during model training.

Changes Implemented
Modified Phase 5 of the pipeline to export cluster metadata.

Automatically generates:
cluster_config.json

Output Format
{
  "0": {
    "label": "Middle Income, Low Burden",
    "policy": "Moderate subsidy and targeted tax relief"
  },
  "1": {
    "label": "High Income, High Burden",
    "policy": "Progressive taxation and investment incentives"
  }
}
Cluster labels and policies are derived from aggregate metrics such as income and burden.
File is regenerated every time the model is retrained.

2️⃣ Application Component (app.py)

Purpose
Consume dynamic cluster configuration at runtime.

Changes Implemented
Updated load_resources() to read cluster_config.json
Removed dependency on hardcoded cluster_policy_map
Implemented fallback logic:
If cluster_config.json is missing or invalid, the application uses default mappings to prevent runtime failures.

⚠️ Important Note (User Review Required)
Hardcoded policy mappings have been removed.

Risk
If run_analysis.py is not executed before starting the application, cluster_config.json will not exist.
Mitigation
The application includes fallback defaults.
Running the analysis pipeline before app startup is strongly recommended.

✅ Verification Plan
🔁 Automated Testing
python run_analysis.py
python test_restore.py


📌 Note
Some tests assume fixed cluster IDs (e.g., Cluster 0 = Middle Income).
If retraining changes cluster assignments:
Update test expectations, or
Use a fixed random seed for reproducibility.

🧪 Manual Verification
Step 1: Run Analysis Pipeline
python run_analysis.py


✔ Verify cluster_config.json is created.

Step 2: Start Application
python app.py


✔ Health Check:
GET /health

Step 3: Test Prediction API
POST /api/predict


✔ Confirm the returned policy matches the newly generated cluster configuration.
📦 Benefits of This Integration
🔄 Automatic updates after retraining
🧠 Single source of truth for cluster-policy mapping
🛡️ Reduced risk of configuration drift
🚀 Production-ready design
🔧 Easier maintenance and extensibility

🛠️ Setup Instructions
# Install dependencies
pip install -r requirements.txt

# Run analysis pipeline (must be done first)
python run_analysis.py

# Start the web application
python app.py

📌 Future Improvements
Versioning of cluster_config.json
Schema validation for configuration file
UI visualization for cluster definitions
CI/CD integration for automated retraining

👩‍💻 Author & Credits
Developed as part of an end-to-end ML system integration project focusing on robust deployment and maintainability.
