# ML Project: Sber Auto Subscription

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)


## Introduction
A brief description of the project, its purpose, and goals. Explain what problem it solves and its importance.

## Project Structure
- **data/skillbox_diploma_main_dataset_sberautopodpiska** # Contains two datasets: information about sessions and information about hits
- **notebooks/** # Jupyter notebooks for data preprocessing, EDA, feature engineering, and modeling
- **model/** # Directory for model files:
  - **pipeline.py** # Steps for model creation
  - **model.pkl** # Final trained model
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

## Usage
1. Start the FastAPI application to serve the model and make predictions through a REST API.
   ```sh
   uvicorn main:app --reload
   ```
2. Once the server is running, you can use tools like 'curl' or Postman to make requests to the API. Below is the list
of the calls.
- GET
  - 'http://127.0.0.1:8000/status' to get the status of the app
  - 'http://127.0.0.1:8000/version' to get model metadata 
- POST
  - http://127.0.0.1:8000/predict to make predictions with json format data in body

3. Example data for API
    ```sh
   {
    "session_id": "9055434745589932991.1637753792.1637753792",
    "client_id": "2108382700.1637753791",
    "visit_date": "2021-11-24",
    "visit_time": "14:36:32",
    "visit_nuber": 1,
    "utm_source": "ZpYIoDJMcFzVoPFsHGJL",
    "utm_medium": "banner",
    "utm_campaign": "LEoPHuyFvzoNfnzGgfcd",
    "utm_adcontent": "vCIpmpaGBnIQhyYNkXqp",
    "utm_keyword": "puhZPIYqKXeFPaUviSjo",
    "device_category": "mobile",
    "device_os": "Android",
    "device_brand": "Huawei",
    "device_model": "example_model",
    "device_screen_resolution": "360x720",
    "device_browser": "Chrome",
    "geo_country": "Russia",
    "geo_city": "Zlatoust"
   }
    ```