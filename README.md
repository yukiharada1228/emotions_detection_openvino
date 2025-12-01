# Face Detection

A real-time face detection application using OpenVINO.

## Features

- Real-time face detection
- Configurable number of faces to detect (default: 1)
- Adjustable confidence threshold (default: 0.8)
- Detects faces starting from the nearest one
- Camera port configurable via environment variables

## Requirements

- Python 3.12 or higher
- OpenVINO 2024.6.0 or higher
- OpenCV 4.11.0 or higher

## Installation

### Using uv (Recommended)

```bash
git clone https://github.com/yukiharada1228/face_detection.git
cd face_detection
uv sync
```

## Setup

1. Copy `.env.example` to create a `.env` file:

```bash
cp .env.example .env
```

2. Configure the camera port in the `.env` file (optional):

```env
CAMERA_PORT=0  # Camera port number (default: 0)
```

If the `.env` file does not exist, camera port 0 will be used by default.

## Usage

### Basic Usage

```bash
# Using uv
uv run main.py

# Or run directly
python main.py
```

### Options

```bash
# Specify the number of faces to detect (confidence uses default 0.8)
uv run main.py [number_of_faces]

# Specify both number of faces and confidence threshold
uv run main.py [number_of_faces] [confidence]

# Examples
uv run main.py          # Default (1 face, confidence 0.8)
uv run main.py 2        # Detect 2 faces (confidence 0.8)
uv run main.py 1 0.9    # Detect 1 face (confidence 0.9)
```

### Controls

- **ESC key**: Exit the application

## Project Structure

```
face_detection/
├── face_detection/
│   ├── __init__.py
│   ├── config.py          # Configuration file (model path, camera port, etc.)
│   └── face_detect.py     # Main face detection logic
├── main.py                 # Entry point
├── pyproject.toml          # Project settings and dependencies
├── .env.example            # Environment variables template
└── README.md
```

## Tech Stack

- **OpenVINO**: Intel's inference engine (using API from 2024.6.0 onwards)
- **OpenCV**: Image processing and camera capture
- **python-dotenv**: Environment variable management

## Model

This project uses the `face-detection-retail-0005` model. It will be automatically downloaded on first run.

## License

See the LICENCE file for details.

## Author

- Yuki Harada
- yukiharada1228@gmail.com
