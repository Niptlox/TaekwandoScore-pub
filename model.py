import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import mediapipe as mp
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Define the PoseClassifier class using CNN layers
class PoseClassifierCNN(nn.Module):
    def __init__(self, num_keypoints, num_classes):
        super(PoseClassifierCNN, self).__init__()
        input_dim = num_keypoints * 2

        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.flattened_size = self._get_flattened_size(input_dim)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _get_flattened_size(self, input_dim):
        dummy_input = torch.randn(1, 1, input_dim)
        size = self.cnn_layers(dummy_input).size()
        return size[1] * size[2]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn_layers(x)
        x = self.classifier(x)
        return x


class TaekwondoPoseClassifier:
    def __init__(self, num_keypoints=33):
        self.num_keypoints = num_keypoints
        self.pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.model = None
        self.label_to_id = {}
        self.id_to_label = {}
        self.num_classes = 0

    def get_skeletons_and_labels_from_coco(self, annotation_file, image_dir):
        """
        Извлекает скелетные точки и соответствующие метки классов из файла аннотаций COCO.
        Выполняет обнаружение позы на изображениях.
        """
        skeletons = []
        labels = []
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') and f != '_annotations.coco.json']

        # Load the COCO annotation data
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        # Create dictionaries for mapping
        image_id_to_annotations = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in image_id_to_annotations:
                image_id_to_annotations[image_id] = []
            image_id_to_annotations[image_id].append(annotation)

        category_id_to_name = {category['id']: category['name'] for category in coco_data['categories']}
        filename_to_image_id = {os.path.basename(image['file_name']): image['id'] for image in coco_data['images']}


        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            try:
                img = Image.open(img_path)
                img_rgb = img.convert('RGB')
                img_np = np.array(img_rgb)

                results = self.pose.process(img_np)

                if results.pose_landmarks:
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y])
                    skeletons.append(landmarks)

                    # Get the corresponding label from COCO data
                    image_id = filename_to_image_id.get(img_file)
                    label = "unknown" # Default label
                    if image_id is not None:
                        annotations = image_id_to_annotations.get(image_id, [])
                        if annotations:
                             first_annotation_category_id = annotations[0]['category_id']
                             label = category_id_to_name.get(first_annotation_category_id, "unknown")
                    labels.append(label)

            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

        return skeletons, labels

    def extract_features(self, skeletons):
        """Извлекает признаки из скелетных точек."""
        extracted_features = []
        for skeleton in skeletons:
            if skeleton is not None:
                flattened_skeleton = [coord for point in skeleton for coord in point]
                extracted_features.append(flattened_skeleton)
        return extracted_features

    def prepare_data(self, skeletons, labels):
        """Подготавливает данные для обучения."""
        extracted_features = self.extract_features(skeletons)

        # Convert string labels to numerical labels
        self.unique_labels = sorted(list(set(labels)))
        self.label_to_id = {label: i for i, label in enumerate(self.unique_labels)}
        numeric_labels = [self.label_to_id[label] for label in labels]
        self.num_classes = len(self.unique_labels)
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}


        if extracted_features:
            data_tensor = torch.tensor(extracted_features, dtype=torch.float32)
            labels_tensor = torch.tensor(numeric_labels, dtype=torch.long)
            return data_tensor, labels_tensor
        else:
            return torch.empty(0, self.num_keypoints * 2), torch.empty(0, dtype=torch.long)


    def build_model(self):
        """Строит модель классификации."""
        # num_keypoints here refers to the number of coordinate pairs (x, y), which is half the input dimension
        num_keypoints_coords = self.num_keypoints # Assuming extract_features returns 2*num_keypoints
        self.model = PoseClassifierCNN(num_keypoints_coords, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, data_tensor, labels_tensor, num_epochs=1500):
        """Обучает модель."""
        if self.model is None:
            print("Model not built. Call build_model() first.")
            return

        self.model.train()
        print("Starting training...")
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(data_tensor)
            loss = self.criterion(outputs, labels_tensor)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"Эпоха [{epoch+1}/{num_epochs}], Потеря: {loss.item():.4f}")

        print("Обучение завершено.")

    def evaluate(self, data_tensor, labels_tensor):
        """Оценивает производительность модели."""
        if self.model is None:
            print("Model not built or trained.")
            return

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data_tensor)
            _, predicted_labels = torch.max(outputs, 1)

            accuracy = accuracy_score(labels_tensor.numpy(), predicted_labels.numpy())
            print(f"\nAccuracy on the dataset: {accuracy:.4f}")

            precision, recall, f1_score, _ = precision_recall_fscore_support(
                labels_tensor.numpy(),
                predicted_labels.numpy(),
                labels=range(self.num_classes),
                zero_division=0
            )

            print("\nMetrics per class:")
            for i, label_name in enumerate(self.unique_labels):
                 if i < len(precision):
                    print(f"  Class '{label_name}' (ID {i}):")
                    print(f"    Precision: {precision[i]:.4f}")
                    print(f"    Recall:    {recall[i]:.4f}")
                    print(f"    F1-score:  {f1_score[i]:.4f}")
                 else:
                    print(f"  Class ID {i} not present in predictions.")


            precision_macro, recall_macro, f1_score_macro, _ = precision_recall_fscore_support(
                labels_tensor.numpy(),
                predicted_labels.numpy(),
                average='macro',
                zero_division=0
            )
            print("\nAveraged Metrics (Macro):")
            print(f"  Precision (Macro): {precision_macro:.4f}")
            print(f"  Recall (Macro):    {recall_macro:.4f}")
            print(f"  F1-score (Macro):  {f1_score_macro:.4f}")

            precision_weighted, recall_weighted, f1_score_weighted, _ = precision_recall_fscore_support(
                labels_tensor.numpy(),
                predicted_labels.numpy(),
                average='weighted',
                zero_division=0
            )
            print("\nAveraged Metrics (Weighted):")
            print(f"  Precision (Weighted): {precision_weighted:.4f}")
            print(f"  Recall (Weighted):    {recall_weighted:.4f}")
            print(f"  F1-score (Weighted):  {f1_score_weighted:.4f}")


    def predict(self, image_path):
        """Выполняет предсказание для нового изображения."""
        if self.model is None:
            print("Model not built or trained.")
            return None

        try:
            new_image = Image.open(image_path)
            img_rgb = new_image.convert('RGB')
            img_np = np.array(img_rgb)

            results = self.pose.process(img_np)

            new_skeleton = None
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y])
                new_skeleton = landmarks
            else:
                print("No pose detected on the image.")
                return None

            new_features = []
            if new_skeleton:
                new_features = [coord for point in new_skeleton for coord in point]
            else:
                 print("No features extracted.")
                 return None

            if new_features:
                new_features_tensor = torch.tensor(new_features, dtype=torch.float32).unsqueeze(0)

                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(new_features_tensor)
                    _, predicted_class_id = torch.max(outputs, 1)
                    predicted_class_id = predicted_class_id.item()

                    if hasattr(self, 'id_to_label'):
                         predicted_label = self.id_to_label.get(predicted_class_id, "unknown")
                         return predicted_label
                    else:
                        print("Label mapping not available.")
                        return predicted_class_id

            else:
                print("Classification skipped due to empty features.")
                return None

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

# Example Usage:
# classifier = TaekwondoPoseClassifier()

# Load and prepare data for training
# train_skeletons, train_labels = classifier.get_skeletons_and_labels_from_coco('/content/train/_annotations.coco.json', '/content/train/')
# train_data_tensor, train_labels_tensor = classifier.prepare_data(train_skeletons, train_labels)

# Build and train the model
# classifier.build_model()
# if train_data_tensor.shape[0] > 0:
#     classifier.train(train_data_tensor, train_labels_tensor)

# Load and prepare data for validation
# valid_skeletons, valid_labels = classifier.get_skeletons_and_labels_from_coco('/content/valid/_annotations.coco.json', '/content/valid/')
# valid_data_tensor, valid_labels_tensor = classifier.prepare_data(valid_skeletons, valid_labels)

# Evaluate on validation data
# if valid_data_tensor.shape[0] > 0:
#     classifier.evaluate(valid_data_tensor, valid_labels_tensor)

# Make a prediction on a new image
# predicted_class = classifier.predict('/content/test/IMG_1986_jpg.rf.cbf82c03d42156f01b5a85d407a375ef.jpg')
# if predicted_class:
#     print(f"Predicted class for the new image: {predicted_class}")



# Создание экземпляра класса
classifier = TaekwondoPoseClassifier(num_keypoints=33) # MediaPipe Pose gives 33 keypoints by default

# Загрузка и подготовка данных для обучения
print("Loading and preparing training data...")
train_annotation_file = '/content/train/_annotations.coco.json'
train_image_dir = '/content/train/'
train_skeletons, train_labels = classifier.get_skeletons_and_labels_from_coco(train_annotation_file, train_image_dir)
train_data_tensor, train_labels_tensor = classifier.prepare_data(train_skeletons, train_labels)

# Построение модели
print("\nBuilding the model...")
classifier.build_model()

# Обучение модели
if train_data_tensor.shape[0] > 0:
    print("\nStarting model training...")
    classifier.train(train_data_tensor, train_labels_tensor, num_epochs=600) # Using 100 epochs for example

# Загрузка и подготовка данных для валидации
print("\nLoading and preparing validation data...")
valid_annotation_file = '/content/valid/_annotations.coco.json'
valid_image_dir = '/content/valid/'
valid_skeletons, valid_labels = classifier.get_skeletons_and_labels_from_coco(valid_annotation_file, valid_image_dir)
valid_data_tensor, valid_labels_tensor = classifier.prepare_data(valid_skeletons, valid_labels)


# Оценка модели на валидационных данных
if valid_data_tensor.shape[0] > 0:
    print("\nEvaluating model on validation data...")
    classifier.evaluate(valid_data_tensor, valid_labels_tensor)
else:
    print("\nNo valid validation data to evaluate on.")


# Предсказание на новом изображении
print("\nMaking a prediction on a new image...")
new_image_path = '/content/test/IMG_1986_jpg.rf.cbf82c03d42156f01b5a85d407a375ef.jpg' # Example image from test set
predicted_class = classifier.predict(new_image_path)

if predicted_class:
    print(f"\nPredicted class for {new_image_path}: {predicted_class}")
else:
    print(f"\nCould not make a prediction for {new_image_path}")