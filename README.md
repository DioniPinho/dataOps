# Fetal Health Prediction API

## Project Description
This project provides a RESTful API for predicting fetal health based on a set of medical measurements. The API allows users to submit data about a fetal health case and receive a prediction of the fetal health status.

## Features
- Get API health status
- Submit fetal health data and receive a prediction

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
```bash
  git clone https://github.com/dioni/dataOps/fetal_health.git
```
2. Navigate to the project directory:
```bash
  cd fetal_health
```
3. Install the required dependencies:

```bash
    pip install -r requirements.txt
```
4. Start the API server:

```bash
    python main.py
```

## Usage
The API provides the following endpoints:

1. **GET /** - Check the health status of the API.
- Response: `{"status": "healthy"}`

2. **POST /predict** - Submit fetal health data and receive a prediction.
- Request Body:
  ```json
  {
    "measurements": [
      {"name": "feature1", "value": 10.5},
      {"name": "feature2", "value": 7.2},
      {"name": "feature3", "value": 3.8}
    ]
  }
- Response:
  ```json
  {
    "prediction": 0
  }

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For any questions or feedback, please contact the project maintainer at [dioni@ionsolutions.cloud](mailto:dioni@ionsolutions.cloud).
