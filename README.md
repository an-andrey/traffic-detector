# SAVMO (Save Me Officer!)

# Introduction

SAVMO is a software suite built in 24hrs as part of the MAIS HACKS 2025 hackathon. The goal of SAVMO is to detect if the driver crashed, based on live dashcam footage. Once a crash is detected, SAVMO will generate a summary of the crash, and transfer it (along with key frames from the crash) to the SAVMO dashboard.

The dashboard is ideally to be used by emergency responders, or could be sent out to close relatives of the owner of the dashcam in order to let them know that a crash occured.

# Replication

In order to run SAVMO on your own, clone this repo, and download all required libraries using `requirements.txt`. Here are the main elements of the project:

- `website/`: this holds the Flask application, that you should with `python app.py`.

The website looks something like this: ![website overview of SAVO](README_images/website.png)

- `model_weights.py` : this holds the weights for our model.

- `main.py` :
  Takes in 2 arguments: the video file path and the weights. Once you got the website open, run main.py with a demo video, which will give you a live preview of the video, with an overlay of whether our model detects a crash. If a crash is detected - a report is sent out which should appear on the website once the page is refreshed.

Here is how the live feed looks:
![live feed of SAVO](README_images/crash.png)

# Model Information

We fine-tuned the ResNet18 model based on the [Car Crash Dataset](https://www.kaggle.com/datasets/asefjamilajwad/car-crash-dataset-ccd/data). Due to our time constraints, we trained the model on randomized frames, but a future implementation could be to use a sequential model to use context of the entire video.

After fine-tuning, we got the following confusion matrix, when testing on 20% of our 75 thousand images:

| Truth / Prediction | No Crash | Crash | Accuracy |
| :----------------- | :------: | ----: | -------: |
| No Crash           |  10838   |   319 |      97% |
| Crash              |   317    |  3526 |      91% |

Which is a satisfactory performance, but could be improved in the future.
