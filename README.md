# Stress Detection API

Stress Level Detect - project for determining stress levels based on the analysis of emotions from facial images.

## Project description

This project provides an API for determining the level of stress (Low, Medium, High) based on the analysis of facial emotions in an image. The system uses the DeepFace model to recognise emotions and a proprietary neural network to predict stress levels based on the emotional vector.

## Features

- API for uploading images and receiving stress level predictions
- Support for uploading images in file format
- Ability to transfer a predetermined vector of emotions for prediction
- deployed on the platform on railway.com

## Project structure

```
SLD_server_main/
├── main.py              # The main file of application
├── model_loader.py      # Module for loading a neural network model
├── utils.py             # Additional functions for image and emotion processing
├── client_example.py    # The client for API testing
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker file for creating a container
├── render.yaml          # Configuration for deployment on railway.com
├── LICENSE              # Project licence (Apache 2.0)
├── README.md            # Documentation with description and instructions
├── static/              # Static resources (images, HTML pages)
└── models/              # Folder where models are loaded
```

## Requirements
- Python 3.9+
- FastAPI
- DeepFace
- PyTorch
- OpenCV
- Other dependencies are specified in requirements.txt

## Setting up and launching

### Local launch

1. Clone the repository:
   ```
   git clone https://github.com/cracery/SLD_server_main.git
   cd SLD_server_main
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Launch server:
   ```
   uvicorn main:app --reload
   ```

### Running with Docker

1. Build the Docker image:
   ```
   docker build -t SLD_server_main .
   ```

2. Start the container:
   ```
   docker run -p 8000:8000 SLD_server_main
   ```

## Deploying Railway.com

1. Create a new web service and specify the repository URL.
2. Select ‘Docker’ as the environment.
3. Add the necessary environment variables:
   - `MODEL_BASE_URL`: URL to download the models.
4. Click ‘Create Web Service’.

## API Endpoints

### `POST /predict/image`

Predicts stress levels based on the uploaded image

**Options**:
- `file`: Image file

**Answer**:
```json
{
  "stress_level": "Low",
  "probabilities": {
    "Low": 0.7,
    "Middle": 0.2,
    "High": 0.1
  },
  "confidence": 0.7
}
```
### `GET /healthcheck`

Checks the API status.

**Answer**:
```json
{
  "status": "ok",
  "model_loaded": true
}
```
## License

MIT
