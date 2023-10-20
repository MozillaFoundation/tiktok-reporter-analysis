Example use:

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

# Train
python -m tiktok-reporter-analysis train.py

# Run analysis
python -m tiktok-reporter-analysis data/training_data/screen_recordings/RecordIt-1693292574.mp4
```
Setup dev environment

```
pip install -r requirements-dev.txt
pre-commit install
```

Run in docker
```
docker build -t tiktok-reporter-analysis .
docker run --rm -it --gpus all -v ${PWD}:/app tiktok-reporter-analysis PYTHON CMD HERE
```
