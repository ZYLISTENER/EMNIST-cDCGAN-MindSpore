"""
Conditional DCGAN for 26 lowercase letters (a-z) generation
Output images are rotated 90° clockwise and flipped left-right
Resume training from checkpoint supported
"""

import os
import glob
import re
import struct
import zipfile
import gzip
import requests
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import Normal
import mindspore.dataset as ds

# ==================== Configuration ====================
ms.set_context(mode=ms.PYNATIVE_MODE)
ms.set_device("CPU")           # Change to GPU if available

BATCH_SIZE = 128
LATENT_DIM = 100
NUM_CLASSES = 26
TOTAL_EPOCHS = 200
LEARNING_RATE = 0.0002
BETAS = (0.5, 0.999)
IMG_SIZE = 28
CHANNELS = 1

DATA_ROOT = "./datasets/emnist"
RAW_DIR = os.path.join(DATA_ROOT, "raw")
OUTPUT_DIR = "./outputs"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

# ==================== Data Loading ====================
def download_file(url, save_path):
    if os.path.exists(save_path):
        return
    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(save_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as pbar:
        for chunk in response.iter_content(1024):
            f.write(chunk)
            pbar.update(len(chunk))

def ensure_data_files():
    train_img = os.path.join(RAW_DIR, 'emnist-byclass-train-images-idx3-ubyte')
    train_lbl = os.path.join(RAW_DIR, 'emnist-byclass-train-labels-idx1-ubyte')
    if os.path.exists(train_img) and os.path.exists(train_lbl):
        return
    zip_path = os.path.join(RAW_DIR, 'gzip.zip')
    if not os.path.exists(zip_path):
        url = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
        download_file(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(RAW_DIR)
    for f in os.listdir(RAW_DIR):
        if f.endswith('.gz'):
            gz_path = os.path.join(RAW_DIR, f)
            out_path = gz_path[:-3]
            if not os.path.exists(out_path):
                with gzip.open(gz_path, 'rb') as gz_f, open(out_path, 'wb') as out_f:
                    out_f.write(gz_f.read())

def read_idx_file(filepath):
    with open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        if magic == 2051:
            rows, cols = struct.unpack('>II', f.read(8))
            data = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        elif magic == 2049:
            data = np.fromfile(f, dtype=np.uint8)
        else:
            raise ValueError(f"Unknown magic {magic}")
    return data

def load_lowercase_letters(batch_size):
    ensure_data_files()
    train_img = os.path.join(RAW_DIR, 'emnist-byclass-train-images-idx3-ubyte')
    train_lbl = os.path.join(RAW_DIR, 'emnist-byclass-train-labels-idx1-ubyte')
    images = read_idx_file(train_img)
    labels = read_idx_file(train_lbl)

    lower_start, lower_end = 36, 62
    mask = (labels >= lower_start) & (labels < lower_end)
    images_lower = images[mask]
    labels_mapped = labels[mask] - lower_start

    class EMNISTDataset:
        def __init__(self, imgs, lbls):
            self.imgs = imgs
            self.lbls = lbls
        def __getitem__(self, idx):
            img = self.imgs[idx].astype(np.float32) / 127.5 - 1.0
            img = img[np.newaxis, ...]
            lbl = self.lbls[idx].astype(np.int32)
            return img, lbl
        def __len__(self):
            return len(self.lbls)

    dataset = ds.GeneratorDataset(EMNISTDataset(images_lower, labels_mapped),
                                  column_names=["image", "label"], shuffle=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

train_dataset = load_lowercase_letters(BATCH_SIZE)
data_size = train_dataset.get_dataset_size()

# ==================== Generator ====================
class ConvGenerator(nn.Cell):
    def __init__(self, latent_dim, num_classes, img_size=28, channels=1, feature_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.fc = nn.Dense(latent_dim + latent_dim, 7*7*feature_dim*4)
        self.bn1 = nn.BatchNorm2d(feature_dim*4)
        self.relu = nn.ReLU()
        self.deconv1 = nn.Conv2dTranspose(feature_dim*4, feature_dim*2, 4, stride=2, padding=1, pad_mode='pad')
        self.bn2 = nn.BatchNorm2d(feature_dim*2)
        self.deconv2 = nn.Conv2dTranspose(feature_dim*2, feature_dim, 4, stride=2, padding=1, pad_mode='pad')
        self.bn3 = nn.BatchNorm2d(feature_dim)
        self.deconv3 = nn.Conv2dTranspose(feature_dim, channels, 3, stride=1, padding=1, pad_mode='pad')
        self.tanh = nn.Tanh()

    def construct(self, noise, labels):
        label_emb = self.label_emb(labels)
        x = ops.concat((noise, label_emb), axis=1)
        x = self.fc(x)
        x = x.view(x.shape[0], -1, 7, 7)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.deconv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.deconv3(x)
        return self.tanh(x)

# ==================== Discriminator ====================
class ConvDiscriminator(nn.Cell):
    def __init__(self, num_classes, img_size=28, channels=1, feature_dim=64):
        super().__init__()
        self.img_size = img_size
        self.label_emb = nn.Embedding(num_classes, img_size*img_size)
        self.conv1 = nn.Conv2d(channels + 1, feature_dim, 4, stride=2, padding=1, pad_mode='pad')
        self.leaky = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(feature_dim, feature_dim*2, 4, stride=2, padding=1, pad_mode='pad')
        self.bn2 = nn.BatchNorm2d(feature_dim*2)
        self.conv3 = nn.Conv2d(feature_dim*2, feature_dim*4, 4, stride=2, padding=1, pad_mode='pad')
        self.bn3 = nn.BatchNorm2d(feature_dim*4)
        self.fc = nn.Dense(feature_dim*4 * 3 * 3, 1)
        self.sigmoid = nn.Sigmoid()

    def construct(self, img, labels):
        batch = img.shape[0]
        label_emb = self.label_emb(labels)
        label_map = label_emb.view(batch, 1, self.img_size, self.img_size)
        x = ops.concat((img, label_map), axis=1)
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.bn2(self.conv2(x)))
        x = self.leaky(self.bn3(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return self.sigmoid(x)

# ==================== Model Init & Resume ====================
generator = ConvGenerator(LATENT_DIM, NUM_CLASSES, IMG_SIZE, CHANNELS)
discriminator = ConvDiscriminator(NUM_CLASSES, IMG_SIZE, CHANNELS)

def init_weights(net):
    for _, param in net.parameters_and_names():
        if 'weight' in param.name and param.ndim > 1:
            param.set_data(ms.common.initializer.initializer(Normal(0.02), param.shape, param.dtype))

init_weights(generator)
init_weights(discriminator)

def get_latest_epoch(model_dir, prefix="generator_epoch"):
    pattern = os.path.join(model_dir, f"{prefix}*.ckpt")
    files = glob.glob(pattern)
    epochs = []
    for f in files:
        basename = os.path.basename(f)
        match = re.search(r'epoch(\d+)\.ckpt', basename)
        if match:
            epochs.append(int(match.group(1)))
    return max(epochs) if epochs else 0

start_epoch = get_latest_epoch(MODEL_DIR, "generator_epoch")
if start_epoch > 0:
    gen_ckpt = os.path.join(MODEL_DIR, f"generator_epoch{start_epoch}.ckpt")
    disc_ckpt = os.path.join(MODEL_DIR, f"discriminator_epoch{start_epoch}.ckpt")
    ms.load_param_into_net(generator, ms.load_checkpoint(gen_ckpt))
    ms.load_param_into_net(discriminator, ms.load_checkpoint(disc_ckpt))

# ==================== Loss & Optimizers ====================
adversarial_loss = nn.BCELoss(reduction='mean')
g_optimizer = nn.Adam(generator.trainable_params(), learning_rate=LEARNING_RATE, beta1=BETAS[0], beta2=BETAS[1])
d_optimizer = nn.Adam(discriminator.trainable_params(), learning_rate=LEARNING_RATE, beta1=BETAS[0], beta2=BETAS[1])

# ==================== Training Step ====================
def train_step(real_imgs, real_labels):
    batch_size = real_imgs.shape[0]
    real = Tensor(np.ones((batch_size, 1), dtype=np.float32), ms.float32)
    fake = Tensor(np.zeros((batch_size, 1), dtype=np.float32), ms.float32)
    random_labels = Tensor(np.random.randint(0, NUM_CLASSES, size=(batch_size,)), dtype=ms.int32)

    def d_forward(real_imgs, real_labels, noise, cond_labels):
        fake_imgs = generator(noise, cond_labels)
        real_out = discriminator(real_imgs, real_labels)
        fake_out = discriminator(fake_imgs, cond_labels)
        d_loss_real = adversarial_loss(real_out, real)
        d_loss_fake = adversarial_loss(fake_out, fake)
        return (d_loss_real + d_loss_fake) / 2

    def g_forward(noise, cond_labels):
        fake_imgs = generator(noise, cond_labels)
        fake_out = discriminator(fake_imgs, cond_labels)
        return adversarial_loss(fake_out, real)

    noise = Tensor(np.random.normal(0, 1, (batch_size, LATENT_DIM)), ms.float32)
    d_loss, d_grads = ops.value_and_grad(d_forward, grad_position=None, weights=discriminator.trainable_params())(
        real_imgs, real_labels, noise, random_labels)
    d_optimizer(d_grads)

    noise = Tensor(np.random.normal(0, 1, (batch_size, LATENT_DIM)), ms.float32)
    g_loss, g_grads = ops.value_and_grad(g_forward, grad_position=None, weights=generator.trainable_params())(
        noise, random_labels)
    g_optimizer(g_grads)

    return d_loss.asnumpy(), g_loss.asnumpy()

# ==================== Image Transform ====================
def transform_output(img):
    img = np.rot90(img, k=-1)
    img = np.fliplr(img)
    return img

# ==================== Training Loop ====================
fixed_noise = Tensor(np.random.normal(0, 1, (NUM_CLASSES, LATENT_DIM)), ms.float32)
fixed_labels = Tensor(np.arange(NUM_CLASSES), dtype=ms.int32)

for epoch in range(start_epoch, TOTAL_EPOCHS):
    d_loss_epoch = 0
    g_loss_epoch = 0
    num_batches = 0

    pbar = tqdm(train_dataset.create_tuple_iterator(), desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}", total=data_size)
    for imgs, lbls in pbar:
        d_loss, g_loss = train_step(imgs, lbls)
        d_loss_epoch += d_loss
        g_loss_epoch += g_loss
        num_batches += 1
        pbar.set_postfix({"D": f"{d_loss:.4f}", "G": f"{g_loss:.4f}"})

    avg_d = d_loss_epoch / num_batches
    avg_g = g_loss_epoch / num_batches

    generator.set_train(False)
    gen_imgs = generator(fixed_noise, fixed_labels).asnumpy()
    generator.set_train(True)

    fig, axes = plt.subplots(2, 13, figsize=(18, 4))
    plt.subplots_adjust(hspace=0.3, bottom=0.1)
    for i in range(NUM_CLASSES):
        row, col = i // 13, i % 13
        img = (gen_imgs[i, 0] + 1) / 2
        img = transform_output(img)
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_xlabel(chr(ord('a') + i), fontsize=9)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

    plt.suptitle(f"Generated Lowercase Letters - Epoch {epoch+1}", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"epoch_{epoch+1:03d}_all_letters.png"))
    plt.close()

    ms.save_checkpoint(generator, os.path.join(MODEL_DIR, f"generator_epoch{epoch+1}.ckpt"))
    ms.save_checkpoint(discriminator, os.path.join(MODEL_DIR, f"discriminator_epoch{epoch+1}.ckpt"))

    print(f"Epoch {epoch+1:3d}/{TOTAL_EPOCHS} | D_loss: {avg_d:.4f} | G_loss: {avg_g:.4f}")

ms.save_checkpoint(generator, os.path.join(MODEL_DIR, "generator_final.ckpt"))
ms.save_checkpoint(discriminator, os.path.join(MODEL_DIR, "discriminator_final.ckpt"))