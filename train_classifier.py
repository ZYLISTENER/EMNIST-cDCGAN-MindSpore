# ==================== Basic Setup ====================
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import mindspore as ms
from mindspore import nn, Tensor, ops
import mindspore.dataset as ds

# ==================== Path Settings ====================
SAVE_DIR = "./infer"

MODEL_PATH = os.path.join(SAVE_DIR, "emnist_classifier_2.ckpt")
STATE_PATH = os.path.join(SAVE_DIR, "train_state_2.npz")
CURVE_PATH = os.path.join(SAVE_DIR, "train_curve_2.png")

os.makedirs(SAVE_DIR, exist_ok=True)

# ==================== MindSpore Environment ====================
ms.set_context(mode=ms.PYNATIVE_MODE)
ms.set_device("CPU")

# ==================== Dataset Paths ====================
DATA_ROOT = "./datasets/emnist/raw"

TRAIN_IMG = os.path.join(DATA_ROOT, "emnist-byclass-train-images-idx3-ubyte")
TRAIN_LBL = os.path.join(DATA_ROOT, "emnist-byclass-train-labels-idx1-ubyte")

TEST_IMG = os.path.join(DATA_ROOT, "emnist-byclass-test-images-idx3-ubyte")
TEST_LBL = os.path.join(DATA_ROOT, "emnist-byclass-test-labels-idx1-ubyte")

NUM_CLASSES = 26
BATCH_SIZE = 32
EPOCHS = 20

# ==================== Training Parameters ====================
LR = 0.0001
CLIP_NORM = 5.0
PATIENCE = 2
MIN_DELTA = 0.001

# ==================== Data Loading ====================
def read_images(path):
    with open(path, 'rb') as f:
        _, n, r, c = struct.unpack('>IIII', f.read(16))
        return np.fromfile(f, dtype=np.uint8).reshape(n, r, c)

def read_labels(path):
    with open(path, 'rb') as f:
        _, n = struct.unpack('>II', f.read(8))
        return np.fromfile(f, dtype=np.uint8)

def load_data(img_path, lbl_path):
    imgs = read_images(img_path)
    lbls = read_labels(lbl_path)

    # Select lowercase letters only (labels 36~61)
    mask = (lbls >= 36) & (lbls < 62)
    imgs = imgs[mask]
    lbls = lbls[mask] - 36

    imgs = imgs.astype(np.float32) / 255.0
    imgs = (imgs - 0.5) / 0.5
    imgs = imgs[:, np.newaxis, :, :]
    lbls = lbls.astype(np.int32)

    print(f"[DATA] samples = {len(lbls)}")
    return imgs, lbls

# ==================== Dataset Loader ====================
def create_dataset(imgs, lbls):
    dataset = ds.NumpySlicesDataset(
        {"image": imgs, "label": lbls},
        shuffle=True
    )
    return dataset.batch(BATCH_SIZE, drop_remainder=True)

# ==================== CNN Classifier ====================
class EMNISTClassifier(nn.Cell):
    def __init__(self):
        super().__init__()

        self.conv = nn.SequentialCell([
            nn.Conv2d(1, 32, 3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, 3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        ])

        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(64 * 7 * 7, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(128, NUM_CLASSES)

    def construct(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

# ==================== Checkpoint ====================
def load_state():
    if os.path.exists(STATE_PATH):
        data = np.load(STATE_PATH, allow_pickle=True)
        print("[RESUME] loaded")
        return int(data["epoch"]), list(data["loss"]), list(data["acc"])
    return 0, [], []

def save_state(epoch, loss, acc):
    np.savez(STATE_PATH, epoch=epoch, loss=loss, acc=acc)

# ==================== Training ====================
def train():
    print("Loading data...")

    train_imgs, train_lbls = load_data(TRAIN_IMG, TRAIN_LBL)
    test_imgs, test_lbls = load_data(TEST_IMG, TEST_LBL)

    train_ds = create_dataset(train_imgs, train_lbls)
    test_ds = create_dataset(test_imgs, test_lbls)

    net = EMNISTClassifier()

    loss_fn = nn.SoftmaxCrossEntropyWithLogits(
        sparse=True,
        reduction='mean'
    )

    optimizer = nn.Adam(
        net.trainable_params(),
        learning_rate=LR,
        eps=1e-8
    )

    def forward_fn(x, y):
        logits = net(x)
        loss = loss_fn(logits, y)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)

    start_epoch, loss_curve, acc_curve = load_state()

    best_acc = 0.0
    wait = 0

    print("Start training...")

    for epoch in range(start_epoch, EPOCHS):

        net.set_train(True)

        total_loss = 0.0
        total_acc = 0.0
        step = 0

        print(f"\n========== Epoch {epoch+1} ==========")

        pbar = tqdm(
            train_ds.create_tuple_iterator(),
            total=int(len(train_imgs)//BATCH_SIZE),
            desc=f"Epoch {epoch+1}"
        )

        for x, y in pbar:

            loss, grads = grad_fn(x, y)

            grads = ops.clip_by_global_norm(grads, CLIP_NORM)
            optimizer(grads)

            logits = net(x)
            pred = np.argmax(logits.asnumpy(), axis=1)

            loss_value = float(np.mean(loss.asnumpy()))

            if np.isnan(loss_value):
                print("NaN detected")
                return

            total_loss += loss_value
            total_acc += (pred == y.asnumpy()).mean()

            pbar.set_postfix(loss=loss_value)
            step += 1

        avg_loss = total_loss / step
        avg_acc = total_acc / step

        loss_curve.append(avg_loss)
        acc_curve.append(avg_acc)

        print(f"\n[EPOCH {epoch+1}] loss={avg_loss:.4f} acc={avg_acc:.4f}")

        # ==================== Test ====================
        net.set_train(False)

        acc_list = []
        for x, y in test_ds.create_tuple_iterator():
            logits = net(x)
            pred = np.argmax(logits.asnumpy(), axis=1)
            acc_list.append((pred == y.asnumpy()).mean())

        test_acc = float(np.mean(acc_list))
        print(f"[TEST] acc={test_acc:.4f}")

        # ==================== Early Stop ====================
        if test_acc > best_acc + MIN_DELTA:
            best_acc = test_acc
            wait = 0
        else:
            wait += 1

        print(f"[EARLY STOP] best={best_acc:.4f}, wait={wait}/{PATIENCE}")

        if wait >= PATIENCE:
            print("\nEarly stopping")
            break

        ms.save_checkpoint(net, MODEL_PATH)
        save_state(epoch + 1, loss_curve, acc_curve)

    # ==================== Plot Curves ====================
    plt.figure()

    plt.subplot(2,1,1)
    plt.plot(loss_curve)
    plt.title("Loss")

    plt.subplot(2,1,2)
    plt.plot(acc_curve)
    plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig(CURVE_PATH)
    plt.close()

    print("\nTraining finished")
    print("Model:", MODEL_PATH)
    print("Curve:", CURVE_PATH)

# ==================== Main ====================
if __name__ == "__main__":
    train()