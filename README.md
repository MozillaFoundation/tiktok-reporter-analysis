Example use:

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python extract_frames.py training_data/screen_recordings/RecordIt-1693292574.mp4 frames
python train.py frames training_data/labels.txt checkpoints/
```
Setup dev environment

```
pip install -r requirements-dev.txt
pre-commit install
```

Run in docker
```
docker build -t tiktok-reporter-analysis
docker run --rm -it --gpus all -v ${PWD}:/app tiktok-reporter-analysis PYTHON CMD HERE
```