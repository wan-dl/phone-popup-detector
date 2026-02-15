import cv2
import torch

class PopupDetector:
    def __init__(self, model_path):
        # Load the YOLO v8 model
        self.model = torch.hub.load('ultralytics/yolov8', 'custom', path=model_path)

    def detect(self, image):
        # Convert image to format suitable for detection
        results = self.model(image)
        return results

    def show_results(self, results):
        # Show results on image
        results.show()