# ML Project: Sber Auto Subscription

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Model](#model)
7. [Results](#results)

## Introduction
A brief description of the project, its purpose, and goals. Explain what problem it solves and its importance.

## Project Structure
- **data/skillbox_diploma_main_dataset_sberautopodpiska** # Contains two datasets: information about sessions and information about hits
- **notebooks/** # Jupyter notebooks for data preprocessing, EDA, feature engineering, and modeling
- **model/** # Directory for model files:
- - **pipeline.py** # Steps for model creation
- - **model.pkl** # Final trained model
- **main.py** # FastAPI app configuration
- **requirements.txt** # Python dependencies
- **README.md** Project documentation

## Installation
1. Clone the repository
    ```sh
    git@github.com:serverdaun/ml_project_sber.git
    cd ml_project_sber
    ```
2. Create and activate a virtual environment
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install the required dependencies
    ```sh
    pip install -r requirements.txt
    ```
