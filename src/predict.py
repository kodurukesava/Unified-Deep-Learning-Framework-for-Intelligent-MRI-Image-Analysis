import os

import torch
import torch.nn.functional as F

from PIL import Image

from torchvision import transforms

from models.model_architecture import (
    create_resnet50
)

from config import (
    MODEL_PATH,
    IMG_SIZE
)


class MRIPredictor:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "Using device:",
            self.device
        )

        if not os.path.exists(
            MODEL_PATH
        ):

            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        print(
            "Loading MRI model..."
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device
        )

        self.class_names = checkpoint[
            "class_names"
        ]

        num_classes = len(
            self.class_names
        )

        print(
            "Classes:",
            self.class_names
        )

        self.model = create_resnet50(
            num_classes
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        self.transform = transforms.Compose([

            transforms.Resize(
                (IMG_SIZE, IMG_SIZE)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],

                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

        print(
            "MRI model loaded successfully."
        )

    def preprocess_image(
        self,
        image_path
    ):

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(
            image
        )

        image = image.unsqueeze(
            0
        )

        return image.to(
            self.device
        )

    def predict(
        self,
        image_path
    ):

        image = self.preprocess_image(
            image_path
        )

        with torch.no_grad():

            outputs = self.model(
                image
            )

            probabilities = F.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

        predicted_index = predicted.item()

        confidence_value = (
            confidence.item()
        )

        predicted_class = self.class_names[
            predicted_index
        ]

        probability_dict = {}

        for index, probability in enumerate(
            probabilities[0]
        ):

            probability_dict[
                self.class_names[index]
            ] = round(
                probability.item() * 100,
                2
            )

        return {

            "prediction":
                predicted_class,

            "confidence":
                round(
                    confidence_value * 100,
                    2
                ),

            "probabilities":
                probability_dict
        }