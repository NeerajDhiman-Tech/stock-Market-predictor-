Stock Project Template
----------------------
This is a ready-to-run template for your stock prediction project.

What is included:
- train_model.py : script to train RandomForest models for each CSV in datasets/
- app.py : Streamlit demo app to upload CSV and get prediction using trained models
- requirements.txt : pip install -r requirements.txt
- datasets/ : place your CSV files here (I included a small sample)
- models/ : trained models will be saved here after running train_model.py

Quick start:
1. Put your CSV files (one or more) into the datasets/ folder.
2. Create a Python virtualenv, activate it, and install requirements.
3. Run: python train_model.py  (this will create models/)
4. Run: streamlit run app.py
