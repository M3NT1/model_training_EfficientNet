import os
import sys
import warnings
import tensorflow as tf

# Figyelmeztetések és információs üzenetek elnyomása
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add meg itt az OpenCV elérési útját
opencv_path = "/Users/kasnyiklaszlo/PycharmProjects/model_training_EfficientNet/.venv/lib/python3.12/site-packages"
sys.path.append(opencv_path)


def check_dependencies():
    required_packages = ['tensorflow', 'matplotlib', 'scipy', 'cv2']
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"Hiányzó függőségek: {', '.join(missing_packages)}")
        print("Kérlek, telepítsd ezeket a csomagokat a 'pip install' paranccsal.")
        sys.exit(1)


def main():
    check_dependencies()

    from data_loader import create_data_generators
    from model import create_model
    from train import train_model
    from utils import plot_training_history
    from face_detection import run_face_recognition

    data_dir = "/Users/kasnyiklaszlo/PycharmProjects/model_training/data"
    img_size = (300, 300)  # EfficientNetB3 ajánlott bemeneti mérete
    batch_size = 32
    epochs = 100

    train_generator, validation_generator, num_train_samples, num_val_samples, num_classes = create_data_generators(
        data_dir, img_size, batch_size)

    model = create_model(num_classes)

    history = train_model(model, train_generator, validation_generator, num_train_samples, num_val_samples, batch_size,
                          epochs)

    if history:
        plot_training_history(history)
        print("Training completed. Model saved as 'model_efficientnet_b3.keras'")

        # Run face recognition
        class_names = list(train_generator.class_indices.keys())
        run_face_recognition("model_efficientnet_b3.keras", class_names)
    else:
        print("Training was not completed due to an error.")


if __name__ == "__main__":
    main()
