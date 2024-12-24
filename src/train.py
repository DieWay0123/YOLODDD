from ultralytics import YOLO
from datetime import datetime, timezone, timedelta

import os


WEIGHTS_ROOT_DIR = "weights"
WEIGHT_FILENAME = "yolo11s-cls.pt"
PRETRAINED_WEIGHT_PATH = os.path.join(WEIGHTS_ROOT_DIR, WEIGHT_FILENAME)

TRAIN_TYPE = "mouth"

DATASETS_ROOT_DIR = "datasets"
DATASETS_PATH = os.path.join(DATASETS_ROOT_DIR, TRAIN_TYPE)

IMG_SIZE = 96
EPOCHS = 10
BATCH_SIZE = 80
OPTIMIZER = "sgd"
LEARNING_RATE = 0.001

MODELS_ROOT_DIR = "models"
MODELS_EXPORT_DIR = os.path.join(MODELS_ROOT_DIR, TRAIN_TYPE)


model = YOLO(model=PRETRAINED_WEIGHT_PATH, verbose=True)
model.cuda()

results = model.train(data=DATASETS_PATH, epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE, optimizer=OPTIMIZER, lr0=LEARNING_RATE)

metrics = model.val(imgsz=IMG_SIZE)
print(f"Best accuracy: {metrics.top1}")

model_filename = "model-" + datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S") + ".pt"
model_path = os.path.join(MODELS_EXPORT_DIR, model_filename)
model.save(model_path)