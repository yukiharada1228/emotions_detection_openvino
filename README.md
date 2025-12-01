# Emotions Detection

A real-time face detection and emotion recognition application using OpenVINO.

## Features

- Real-time face detection
- Real-time emotion recognition (detects 5 types of emotions)
- Configurable number of faces to detect (default: 1)
- Adjustable confidence threshold for emotion recognition (default: 0.2)
- Detects faces starting from the nearest one
- Camera port configurable via environment variables

## Detectable Emotions

- **Neutral** (Gray): Neutral
- **Happy** (Yellow): Happy
- **Sad** (Blue): Sad
- **Surprise** (Green): Surprise
- **Anger** (Red): Anger

## Requirements

- Python 3.12 or higher
- OpenVINO 2024.6.0 or higher
- OpenCV 4.11.0 or higher

## Installation

### Using uv (Recommended)

```bash
git clone https://github.com/yukiharada1228/emotions_detection_openvino.git
cd emotions_detection_openvino
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
# Specify the number of faces to detect (confidence uses default 0.2)
uv run main.py [number_of_faces]

# Specify both number of faces and confidence threshold
uv run main.py [number_of_faces] [confidence]

# Examples
uv run main.py          # Default (1 face, confidence 0.2)
uv run main.py 2        # Detect 2 faces (confidence 0.2)
uv run main.py 1 0.5    # Detect 1 face (confidence 0.5)
```

### Controls

- **ESC key**: Exit the application

## Project Structure

```
emotions_detection_openvino/
├── emotions_detection/
│   ├── __init__.py
│   ├── config.py              # Configuration file (model path, camera port, etc.)
│   └── emotions_detection.py  # Main emotion recognition logic
├── face_detection_openvino/   # Face detection submodule
│   └── face_detection/
│       ├── config.py
│       └── face_detect.py
├── main.py                     # Entry point
├── pyproject.toml              # Project settings and dependencies
├── .env.example                # Environment variables template
└── README.md
```

## Tech Stack

- **OpenVINO**: Intel's inference engine (using API from 2024.6.0 onwards)
- **OpenCV**: Image processing and camera capture
- **python-dotenv**: Environment variable management

## Models

This project uses the following models:

- **Face Detection**: `face-detection-retail-0005` - Automatically downloaded on first run
- **Emotion Recognition**: `emotions-recognition-retail-0003` - Automatically downloaded on first run

## License

See the LICENCE file for details.

## Author

- Yuki Harada
- yukiharada1228@gmail.com
