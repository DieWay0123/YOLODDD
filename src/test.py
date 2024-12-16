from ultralytics import YOLO

import os


TEST_TYPE = "mouth"

MODELS_ROOT_DIR = "models"
MODEL_FILENAME = "model-20241216-131506.pt"
MODEL_PATH = os.path.join(MODELS_ROOT_DIR, TEST_TYPE, MODEL_FILENAME)

TEST_IMAGES_ROOT_DIR = "test_images"

TEST_IMAGE_OPEN_FILENAME = "open_1.jpeg"
TEST_IMAGE_CLOSE_FILENAME = "close_1.jpeg"

TEST_IMAGE_OPEN_PATH = os.path.join(TEST_IMAGES_ROOT_DIR, TEST_TYPE, TEST_IMAGE_OPEN_FILENAME)
TEST_IMAGE_CLOSE_PATH = os.path.join(TEST_IMAGES_ROOT_DIR, TEST_TYPE, TEST_IMAGE_CLOSE_FILENAME)


model = YOLO(MODEL_PATH)
model.info(detailed=True)

results = model.predict([TEST_IMAGE_CLOSE_PATH, TEST_IMAGE_OPEN_PATH])

for result in results:
    result.show()