Using following commands to lunch the streamlit application
```bash
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
python prepare_data.py --data-path heapo_data --out artifacts
python train_models.py --artifacts artifacts
streamlit run app.py
```