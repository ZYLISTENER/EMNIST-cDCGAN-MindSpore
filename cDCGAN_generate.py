# ==================== Configuration ====================
import os
import numpy as np
from PIL import Image

import mindspore as ms
from mindspore import nn, ops, Tensor

# ==================== Path Settings ====================
MODEL_PATH = "./models/generator_epoch38.ckpt"
SAVE_ROOT = "./datasets/generate"

NUM_CLASSES = 26
LATENT_DIM = 100
NUM_PER_CLASS = 10000
BATCH_SIZE = 128

os.makedirs(SAVE_ROOT, exist_ok=True)

# ==================== MindSpore Environment ====================
ms.set_context(mode=ms.PYNATIVE_MODE)
ms.set_device("CPU")

# ==================== Generator Network ====================
class ConvGenerator(nn.Cell):
    def __init__(self, latent_dim, num_classes, feature_dim=64):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.fc = nn.Dense(latent_dim * 2, 7 * 7 * feature_dim * 4)

        self.bn1 = nn.BatchNorm2d(feature_dim * 4)
        self.relu = nn.ReLU()

        self.deconv1 = nn.Conv2dTranspose(
            in_channels=feature_dim * 4,
            out_channels=feature_dim * 2,
            kernel_size=4,
            stride=2,
            pad_mode='pad',
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(feature_dim * 2)

        self.deconv2 = nn.Conv2dTranspose(
            in_channels=feature_dim * 2,
            out_channels=feature_dim,
            kernel_size=4,
            stride=2,
            pad_mode='pad',
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(feature_dim)

        self.deconv3 = nn.Conv2dTranspose(
            in_channels=feature_dim,
            out_channels=1,
            kernel_size=3,
            stride=1,
            pad_mode='pad',
            padding=1
        )

        self.tanh = nn.Tanh()

    def construct(self, noise, labels):
        emb = self.label_emb(labels)
        x = ops.concat((noise, emb), axis=1)

        x = self.fc(x)
        x = x.view(x.shape[0], -1, 7, 7)

        x = self.relu(self.bn1(x))
        x = self.relu(self.bn2(self.deconv1(x)))
        x = self.relu(self.bn3(self.deconv2(x)))
        x = self.deconv3(x)

        return self.tanh(x)

# ==================== Model Loading ====================
generator = ConvGenerator(LATENT_DIM, NUM_CLASSES)

param_dict = ms.load_checkpoint(MODEL_PATH)
ms.load_param_into_net(generator, param_dict)

generator.set_train(False)

print("Model loaded successfully")

# ==================== Letter Labels ====================
letters = [chr(ord('a') + i) for i in range(26)]

# ==================== Image Post-processing ====================
def postprocess(img):
    img = (img + 1) / 2.0
    img = np.rot90(img, k=-1)
    img = np.fliplr(img)
    img = (img * 255).astype(np.uint8)
    return img

# ==================== Batch Generation ====================
def generate_batch(label_id, batch_size):
    noise = Tensor(
        np.random.normal(0, 1, (batch_size, LATENT_DIM)),
        ms.float32
    )
    labels = Tensor([label_id] * batch_size, ms.int32)

    imgs = generator(noise, labels).asnumpy()
    return imgs

# ==================== Save Image ====================
def save_image(img, path):
    Image.fromarray(img).save(path)

# ==================== Main Generation Pipeline ====================
for idx, ch in enumerate(letters):

    print(f"\nGenerating class: {ch}")

    folder = os.path.join(SAVE_ROOT, ch)
    os.makedirs(folder, exist_ok=True)

    count = 0

    while count < NUM_PER_CLASS:
        cur_bs = min(BATCH_SIZE, NUM_PER_CLASS - count)

        imgs = generate_batch(idx, cur_bs)

        for i in range(cur_bs):
            img = imgs[i, 0]
            img = postprocess(img)

            path = os.path.join(folder, f"{count:05d}.png")
            save_image(img, path)

            count += 1

        print(f"{ch}: {count}/{NUM_PER_CLASS}")

print("\nAll classes generated successfully!")