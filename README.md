# AI-based-Cattle-Analysis-and-Tracking-Kattlescan-ATC-
 AI-based cattle analysis system using CNN (MobileNetV2) with OpenCV preprocessing and achieved improved classification accuracy using transfer learning.

# AI based Cattle Analysis and Tracking

This project focuses on building a system that can analyze cattle images and classify breeds using deep learning. The aim is to automate cattle identification and improve accuracy compared to traditional machine learning methods.

## Overview

The system takes an input image of cattle and processes it using computer vision techniques. After preprocessing, the image is passed through a convolutional neural network model for classification.

## Features

Image based cattle breed classification  
Use of CNN model MobileNetV2  
Image preprocessing using OpenCV  
Improved accuracy using transfer learning  
Basic analysis and scoring of results  

## Tech Stack

Python  
TensorFlow and Keras  
OpenCV  
NumPy  

## Working

1. Input image is provided  
2. Image preprocessing is applied such as blur and edge detection  
3. Features are extracted from the image  
4. The processed image is passed into the CNN model  
5. The model predicts the cattle breed  

## Results

The earlier model using Random Forest achieved very low accuracy.  
The CNN model improved the accuracy significantly to around fifty percent.

## Project Structure

project  
 main.py  
 model.py  
 preprocessing.py  
 requirements.txt  
 README.md  

## How to Run

Install required libraries  

pip install -r requirements.txt  

Run the program  

python main.py  

## Future Scope

Increase dataset size  
Improve model accuracy  
Deploy as a web application  

## Author

Aliviya Saha,Abesha Mitra
