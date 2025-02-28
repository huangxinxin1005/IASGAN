import argparse
import os
import numpy as np
import tensorflow as tf
from keras import layers, Model, optimizers
import matplotlib.pyplot as plt
import yaml
import segmentation_models as sm
import keras.backend as K
from dataIO.data import IonoDataManager

cfgs = yaml.load(open('your config path', 'r'), Loader=yaml.BaseLoader)

parser = argparse.ArgumentParser(description='Keras IASGAN Training')
parser.add_argument('--batchSize', type=int, default=1, help='Training Batchsize')
parser.add_argument('--niter', type=int, default=200000, help='Total Number of Training Iterations')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
parser.add_argument('--seed', type=int, default=42, help='Random Seed')
parser.add_argument('--outpath', default='your path', help='Output Path')
opt = parser.parse_args()

tf.random.set_seed(opt.seed)
os.makedirs(opt.outpath, exist_ok=True)
os.makedirs(os.path.join(opt.outpath, "inputs"), exist_ok=True)
os.makedirs(os.path.join(opt.outpath, "labels"), exist_ok=True)
os.makedirs(os.path.join(opt.outpath, "results"), exist_ok=True)

class NetD(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.conv1 = tf.keras.Sequential([
            layers.Conv2D(64, 4, strides=2, padding='same'),
            layers.LeakyReLU(0.2)
        ])

        self.conv2 = tf.keras.Sequential([
            layers.Conv2D(128, 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])

        self.conv3 = tf.keras.Sequential([
            layers.Conv2D(256, 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])

        self.conv4 = tf.keras.Sequential([
            layers.Conv2D(512, 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])

        self.conv5 = tf.keras.Sequential([
            layers.Conv2D(512, 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])

        self.conv6 = tf.keras.Sequential([
            layers.Conv2D(512, 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        out1 = self.conv1(inputs)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)

        features = [
            tf.reshape(inputs, [batch_size, -1]),
            1 * tf.reshape(out1, [batch_size, -1]),
            2 * tf.reshape(out2, [batch_size, -1]),
            2 * tf.reshape(out3, [batch_size, -1]),
            2 * tf.reshape(out4, [batch_size, -1]),
            2 * tf.reshape(out5, [batch_size, -1]),
            4 * tf.reshape(out6, [batch_size, -1])
        ]

        return tf.concat(features, axis=1)

def build_generator():
    return sm.FPN(
        backbone_name='resnet50',
        encoder_weights=None,
        input_shape=(None, None, 3),
        classes=3,
        activation='sigmoid'
    )

def binary_focal_loss(gt, pr, gamma=2.0, alpha=0.25):
    gt = tf.cast(gt, tf.float32)
    pr = tf.cast(pr, tf.float32)

    pr = K.clip(pr, K.epsilon(), 1.0 - K.epsilon())

    loss_1 = - gt * (alpha * K.pow((1 - pr), gamma) * K.log(pr))
    loss_0 = - (1 - gt) * ((1 - alpha) * K.pow(pr, gamma) * K.log(1 - pr))
    loss = K.mean(loss_0 + loss_1)
    return loss

class LossCalculator:
    @staticmethod
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
        return 1 - tf.reduce_mean(numerator / (denominator + 1e-7))

    @staticmethod
    def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
        return binary_focal_loss(y_true, y_pred, gamma=gamma, alpha=alpha)

class IASGANTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.data_manager = IonoDataManager(cfgs)

        self.netG = build_generator()
        self.netD = NetD()

        self.optimizerG = optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta1)
        self.optimizerD = optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta1)

        self.loss_history = []

    def train_step(self, inputs, labels):
        with tf.GradientTape(persistent=True) as tape:

            gen_outputs = self.netG(inputs, training=True)

            real_output = self.netD(labels, training=True)
            fake_output = self.netD(gen_outputs, training=True)

            loss_D = -tf.reduce_mean(tf.abs(fake_output - real_output))

            #loss_G_focal = LossCalculator.focal_loss(labels, gen_outputs)
            loss_G_focal = sm.losses.binary_focal_loss(tf.cast(labels, tf.float32), tf.cast(gen_outputs, tf.float32))
            #loss_G_dice = LossCalculator.dice_loss(labels, gen_outputs)
            loss_G_adv = tf.reduce_mean(tf.abs(fake_output - real_output))
            loss_G = 50 * loss_G_focal + loss_G_adv

        grads_D = tape.gradient(loss_D, self.netD.trainable_variables)
        self.optimizerD.apply_gradients(zip(grads_D, self.netD.trainable_variables))

        grads_G = tape.gradient(loss_G, self.netG.trainable_variables)
        self.optimizerG.apply_gradients(zip(grads_G, self.netG.trainable_variables))

        for var in self.netD.trainable_variables:
            var.assign(tf.clip_by_value(var, -0.05, 0.05))

        return loss_G, loss_D

    def save_images(self, epoch, inputs, labels, outputs):

        base_dir = self.opt.outpath
        dirs = {
            'inputs': os.path.join(base_dir, 'inputs'),
            'labels': os.path.join(base_dir, 'labels'),
            'outputs': os.path.join(base_dir, 'outputs')
        }

        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        def save_plot(data, path, epoch_idx):
            img = data[0].numpy()

            plt.figure()
            plt.imshow(img, origin='lower', cmap='viridis')
            plt.xlim([0, 250])
            plt.ylim([0, 250])
            plt.xlabel('Critical Frequency (MHz)', fontsize=10)
            plt.ylabel('Virtual Height (km)', fontsize=10)
            plt.savefig(f"{path}/epoch_{epoch_idx}.png", dpi=300, bbox_inches='tight')
            plt.close()

        save_plot(inputs, dirs['inputs'], epoch)
        save_plot(labels, dirs['labels'], epoch)
        save_plot(outputs, dirs['outputs'], epoch)


    def train(self):
        for epoch in range(self.opt.niter):

            x, y = self.data_manager.get_train_batch(epoch)
            x_tensor = tf.convert_to_tensor(x)
            y_tensor = tf.convert_to_tensor(y)

            loss_G, loss_D = self.train_step(x_tensor, y_tensor)

            self.loss_history.append([loss_G.numpy(), loss_D.numpy()])

            if epoch % 100 == 0:
                print(f"Epoch {epoch:06d} | G Loss: {loss_G:.4f} | D Loss: {loss_D:.4f}")

            if epoch % 500 == 0:
                gen_output = self.netG(x_tensor)
                self.save_images(epoch, x_tensor, y_tensor, gen_output)

            if epoch % 2000 == 0:
                self.netG.save_weights(f"{self.opt.outpath}/netG_epoch_{epoch}.h5")
                #self.netG.save(f"{self.opt.outpath}/netG_epoch_{epoch}.h5")
               
                np.save(f"{self.opt.outpath}/loss_history.npy", np.array(self.loss_history))

        self.netG.save_weights(f"{self.opt.outpath}/netG_final.h5")
        #self.netG.save_weights(f"{self.opt.outpath}/netG_final.h5")
       
if __name__ == "__main__":
    trainer = IASGANTrainer(opt)
    trainer.train()