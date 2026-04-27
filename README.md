# AI-Generated-Image-Detector
A deep learning project that classifies images as **Real** or **AI-Generated** using transfer learning with EfficientNet and provides visual explanations via Grad-CAM. The system includes a **Streamlit web app** for interactive predictions.

## Overview

* Binary image classification: Real vs AI-generated
* Transfer learning using EfficientNet
* Visual explainability with Grad-CAM
* Interactive web app using Streamlit

## Features

* Upload image and get prediction instantly
* Confidence score for each class
* Grad-CAM heatmap visualization
* Clean and modular project structure
* Training + evaluation pipeline

##  Project Structure

<img width="380" height="668" alt="image" src="https://github.com/user-attachments/assets/68f47400-1091-4455-9f77-0f739682ac43" />


## Dataset

* Dataset used: CIFAKE (Kaggle)
* Contains real and AI-generated images
* Organize dataset as:
  
<img width="365" height="197" alt="image" src="https://github.com/user-attachments/assets/0e9bdead-3225-48cb-b87d-cb7eee04c05b" />


##  Training

```
cd src
python train.py
```

##  Evaluation

```
python evaluate.py
```

## Run the App

```
cd ..
streamlit run app.py
```


---

##  Results

| Metric    | Value |
| --------- | ----- |
| Accuracy  | XX%   |
| Precision | XX%   |
| Recall    | XX%   |
| F1 Score  | XX%   |


## Grad-CAM (Explainability)

* Highlights important regions in the image
* Helps understand model decisions
* Useful for detecting fake patterns

## Tech Stack

* Python
* PyTorch
* OpenCV
* NumPy / Pandas
* Matplotlib / Seaborn
* Streamlit

## Author

* Name: Adarsh Kumar Mishra
* College: DIT University, Dehradun
* Course: B.Tech CSE

## License

This project is for academic purposes.
