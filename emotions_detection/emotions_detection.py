import logging

import cv2 as cv
import numpy as np
from openvino import Core

logger = logging.getLogger(__name__)


INDEX_X = 1
INDEX_Y = 0
RESIZE_X = 64
RESIZE_Y = 64

# Emotion labels definition
EMOTION_LABELS = ["Neutral", "Happy", "Sad", "Surprise", "Anger"]
EMOTION_COLORS = {
    "Neutral": (128, 128, 128),  # Gray
    "Happy": (0, 255, 255),      # Yellow
    "Sad": (255, 0, 0),          # Blue
    "Surprise": (0, 255, 0),     # Green
    "Anger": (0, 0, 255),        # Red
}


class EmotionsRecognition:
    def __init__(self, model_path):
        core = Core()
        model = core.read_model(model_path + ".xml", model_path + ".bin")
        self.compiled_model = core.compile_model(model=model, device_name="CPU")
        self.input_layer = self.compiled_model.input(0)
        self.input_size = self.input_layer.shape
        self.output_layer = self.compiled_model.output(0)
        output_size = self.output_layer.shape
        logger.debug({"input_size": self.input_size, "output_size": output_size})
        self.frame = None
        self.faces = None
        self.probs = []  # List of all emotion probabilities for each face
        self.emotions = []  # Most probable emotion for each face

    def recognize(self, frame, faces):
        self.probs = []
        self.emotions = []
        for face in faces:
            xmin = int(face["xmin"] * frame.shape[INDEX_X])
            ymin = int(face["ymin"] * frame.shape[INDEX_Y])
            xmax = int(face["xmax"] * frame.shape[INDEX_X])
            ymax = int(face["ymax"] * frame.shape[INDEX_Y])
            frame_face = frame[ymin:ymax, xmin:xmax]
            frame_face = cv.resize(frame_face, (RESIZE_X, RESIZE_Y))
            frame_face = frame_face.transpose((2, 0, 1))
            frame_face = np.expand_dims(frame_face, axis=0)
            result = self.compiled_model({self.input_layer: frame_face})
            output_data = np.squeeze(result[self.output_layer])
            # Save all emotion probabilities
            self.probs.append(output_data)
            # Get index of the most probable emotion
            emotion_idx = np.argmax(output_data)
            self.emotions.append(EMOTION_LABELS[emotion_idx])
        logger.debug(
            {
                "action": "recognize",
                "probs": self.probs,
                "emotions": self.emotions,
            }
        )
        self.frame = frame
        self.faces = faces
        return self.probs

    def draw(self, conf=0.2):
        output_image = self.frame.copy()

        for idx, face in enumerate(self.faces):
            xmin = int(face["xmin"] * output_image.shape[INDEX_X])
            ymin = int(face["ymin"] * output_image.shape[INDEX_Y])
            xmax = int(face["xmax"] * output_image.shape[INDEX_X])
            ymax = int(face["ymax"] * output_image.shape[INDEX_Y])
            
            # Get the most probable emotion
            emotion = self.emotions[idx]
            max_prob = np.max(self.probs[idx])
            
            # Use emotion color if probability is above threshold, otherwise use gray
            if max_prob >= conf:
                color = EMOTION_COLORS[emotion]
            else:
                color = EMOTION_COLORS["Neutral"]
            
            # Draw face rectangle
            cv.rectangle(
                output_image,
                (xmin, ymin),
                (xmax, ymax),
                color=color,
                thickness=3,
            )
            
            # Display emotion label and probability
            label_text = f"{emotion}: {max_prob:.2f}"
            label_size, _ = cv.getTextSize(label_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = ymin - 10 if ymin - 10 > 10 else ymin + 30
            
            # Draw label background
            cv.rectangle(
                output_image,
                (xmin, label_y - label_size[1] - 5),
                (xmin + label_size[0], label_y + 5),
                color=color,
                thickness=-1,
            )
            # Draw label text
            cv.putText(
                output_image,
                label_text,
                (xmin, label_y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        logger.debug(
            {
                "action": "draw",
                "frame.shape": self.frame.shape,
                "output_image.shape": output_image.shape,
            }
        )
        return output_image