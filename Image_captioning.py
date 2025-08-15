import sys
import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

# Dummy Captioning Model (Replace with transformer or LSTM-based model for real use)
def generate_caption(features):
    return "A person standing in front of a camera."

class ImageCaptionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Image Captioning App")
        self.setGeometry(300, 100, 800, 600)

        self.label = QLabel("Image will appear here")
        self.label.setAlignment(Qt.AlignCenter)

        self.caption_label = QLabel("Caption: ")
        self.caption_label.setWordWrap(True)
        self.caption_label.setAlignment(Qt.AlignCenter)

        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.caption_label)
        layout.addWidget(self.load_btn)
        self.setLayout(layout)

        # Load pre-trained CNN model for feature extraction
        self.model = resnet50(pretrained=True)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg)")
        if file_name:
            image = Image.open(file_name).convert('RGB')
            self.display_image(image)
            caption = self.get_caption(image)
            self.caption_label.setText(f"Caption: {caption}")

    def display_image(self, image):
        image = image.resize((400, 300))
        qimage = QPixmap.fromImage(ImageQt(image))
        self.label.setPixmap(qimage)

    def get_caption(self, image):
        input_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            features = self.model(input_tensor)
        caption = generate_caption(features)
        return caption

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageCaptionApp()
    window.show()
    sys.exit(app.exec_())
