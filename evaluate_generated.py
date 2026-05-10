# ==================== Basic Setup ====================
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import mindspore as ms
from mindspore import nn

# ==================== Path Settings ====================
DATA_ROOT = "./datasets/generate"
MODEL_PATH = "./infer/emnist_classifier_2.ckpt"

SAVE_DIR = "./infer/eval_result"
os.makedirs(SAVE_DIR, exist_ok=True)

ACC_TXT = os.path.join(SAVE_DIR, "class_accuracy.txt")
CM_PATH = os.path.join(SAVE_DIR, "confusion_matrix.png")
CM_CSV = os.path.join(SAVE_DIR, "confusion_matrix.csv")

# ==================== MindSpore Environment ====================
ms.set_context(mode=ms.PYNATIVE_MODE)
ms.set_device("CPU")

# ==================== Class Mapping ====================
chars = list("abcdefghijklmnopqrstuvwxyz")
char_to_idx = {c: i for i, c in enumerate(chars)}

# ==================== Classification Model ====================
class EMNISTClassifier(nn.Cell):
    def __init__(self):
        super().__init__()

        self.conv = nn.SequentialCell([
            nn.Conv2d(1, 32, 3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        ])

        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(64 * 7 * 7, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(128, 26)

    def construct(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

# ==================== Image Preprocessing ====================
def load_image(path):
    img = Image.open(path).convert("L")
    img = img.resize((28, 28))

    img = np.array(img).astype(np.float32)

    # Rotate and flip to match training data
    img = np.rot90(img, k=-1)
    img = np.fliplr(img)

    # Standard normalization for EMNIST
    img = img / 255.0
    img = (img - 0.5) / 0.5

    img = img[np.newaxis, np.newaxis, :, :]
    return ms.Tensor(img)

# ==================== Load Model ====================
net = EMNISTClassifier()
ms.load_checkpoint(MODEL_PATH, net)
net.set_train(False)

# ==================== Confusion Matrix ====================
conf_matrix = np.zeros((26, 26), dtype=np.int32)
class_acc = {}

# ==================== Inference ====================
print("Starting evaluation...")

for cls in chars:
    folder = os.path.join(DATA_ROOT, cls)
    files = os.listdir(folder)

    correct = 0
    total = 0
    true_idx = char_to_idx[cls]

    print(f"\n[CLASS {cls}]")

    for f in tqdm(files):
        path = os.path.join(folder, f)

        try:
            x = load_image(path)
            logits = net(x)
            pred = int(np.argmax(logits.asnumpy(), axis=1)[0])

            conf_matrix[true_idx][pred] += 1

            if pred == true_idx:
                correct += 1
            total += 1
        except:
            continue

    acc = correct / total if total > 0 else 0
    class_acc[cls] = acc
    print(f"{cls} accuracy = {acc:.4f}")

# ==================== Save Class Accuracy ====================
with open(ACC_TXT, "w", encoding="utf-8") as f:
    for k, v in class_acc.items():
        f.write(f"{k}: {v:.6f}\n")

print("\nclass accuracy saved")

# ==================== Save Confusion Matrix CSV ====================
np.savetxt(CM_CSV, conf_matrix, fmt="%d", delimiter=",")
print("confusion matrix saved (csv)")

# ==================== Plot and Save Confusion Matrix ====================
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, cmap="Blues")
plt.colorbar()

plt.xticks(range(26), chars)
plt.yticks(range(26), chars)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (EMNIST a-z)")

plt.tight_layout()
plt.savefig(CM_PATH)
plt.close()

print("confusion matrix image saved")

print("\nEvaluation completed!")
print("accuracy txt:", ACC_TXT)
print("confusion matrix:", CM_PATH)