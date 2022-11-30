import project_utils
import env_createProject

import os 
import numpy as np
import time

import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, Reshape, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Conv2DTranspose

import matplotlib.pyplot as plt

import tqdm
from tqdm import tqdm
from joblib import Parallel, delayed

from skimage import io, img_as_ubyte
from skimage.transform import resize

# ------------------------------------------ Arguments -----------------------------------------

import argparse
parser = argparse.ArgumentParser()

# Dirs
parser.add_argument("--main", help="Main path to save plots, videos and weights", required=True)
parser.add_argument("--project", help="Main path to save plots, videos and weights", required=True)
parser.add_argument("--data", help="Path on there are train images", required=True)

# Train stats
parser.add_argument("--imgSize", help="Set the image size. BIGGER SIZES REQUIRES MUCH MORE COMPUTING POWER", default=64)
parser.add_argument("--batch", help="batch size used on training", default=64)
parser.add_argument("--epochs", help="Train duration on epochs", default=2000)
parser.add_argument("--resetMetrics", help="How many epochs to reset loss metrics", default=200)
parser.add_argument("--checkpoints", help="How many epochs to save a checkpoint", default=400)

# plotting train
parser.add_argument("--createPlots", help="Create plots that show information about metrics on that epoch and generates _n random images with actual weights", action="store_true")
parser.add_argument("--plotEpochsEvery", help="How many epochs to save a plot, more epochs can be more representative but it will a performance drop on training", default=16)
parser.add_argument("--createVideo", help="Create video with all saved images", action="store_true")

args = parser.parse_args()

# ---------------------------------- Setting initial variables ---------------------------------

import warnings
warnings.filterwarnings('ignore')

main_dir = args.main
IMAGES = args.data   # Dir where are all the images

WIDTH = project_utils.real_img_size(int(args.imgSize))
HEIGHT = WIDTH
LATENTsize = 20

# Creating project folders
PROJECT, SAVE_RESULTS, SAVE_WEIGHTS, SAVE_RESULTS_VIDEO = env_createProject.create_dirs(main_dir, project_name=args.project)

# --------------------------- Parallelized load and data processing ----------------------------

data_train = []

def image_set(IMAGES, Images_DIR, data_train):
    image = IMAGES + '/' + Images_DIR
    image = io.imread(image)
    image = resize(image, (WIDTH, HEIGHT))
    data_train.append(img_as_ubyte(image))

_1 = Parallel(n_jobs=-1, backend='threading')(delayed(image_set)(IMAGES, Images_DIR, data_train) for Images_DIR in tqdm(os.listdir(IMAGES)))

data_train = np.array([data_train])
print("Created array with data, shape: {}\n".format(data_train.shape))

images = data_train.reshape(-1, WIDTH, HEIGHT, 3)
print("Reshaped data to use in train: {}\n-- DATA IS READY --".format(images.shape))


# --------------------------- Creating the main Class, it will have: ---------------------------
# 
#  - Generator architecture
#  - Discriminator architecture
#  - Train loop
#  - Results plot


class GAN():
    def __init__(self):
        self.img_shape = (WIDTH, HEIGHT, 3)
        
        self.noise_size = LATENTsize

        optimizer = Adam(0.0002,0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        self.combined = Sequential()
        self.combined.add(self.generator)
        self.combined.add(self.discriminator)
        
        self.discriminator.trainable = False
        
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        self.combined.summary()


    # ---------------------------------------------------------------------------------------------------------------------    
    # - Creating the generator, a big number of kernel fiters in convolutional layers allow the network to generate very 
    #   detailed images based any kind of image data.
    # ---------------------------------------------------------------------------------------------------------------------

    def build_generator(self):
        epsilon = 0.00001 # Small float added to variance to avoid dividing by zero in the BatchNorm layers.
        noise_shape = (self.noise_size,)
        base = int(WIDTH/16)  # 16 = 2^(number of convolution layers) => 16 = 2^4
        
        model = Sequential()
        
        model.add(Dense(base*base*512, activation='linear', input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((base, base, 512)))
        
        model.add(Conv2DTranspose(512, kernel_size=[4,4], strides=[2,2], padding="same",
                                  kernel_initializer= keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2DTranspose(256, kernel_size=[4,4], strides=[2,2], padding="same",
                                  kernel_initializer= keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2DTranspose(128, kernel_size=[4,4], strides=[2,2], padding="same",
                                  kernel_initializer= keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2DTranspose(64, kernel_size=[4,4], strides=[2,2], padding="same",
                                  kernel_initializer= keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2DTranspose(3, kernel_size=[4,4], strides=[1,1], padding="same",
                                  kernel_initializer= keras.initializers.TruncatedNormal(stddev=0.02)))

        # Standard activation for the generator of a GAN
        model.add(Activation("tanh"))
        
        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    # ---------------------------------------------------------------------------------------------------------------------
    # - The magic of DCGANs is the discriminator, it has to be capable to predict with most accuracy when image is false 
    #   to force the generator to improve the performance and the complexity of the images
    # ---------------------------------------------------------------------------------------------------------------------

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(192, (3,3), padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Conv2D(192, (3,3), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3,3)))
        model.add(Dropout(0.2))

        model.add(Conv2D(160, (3,3), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Conv2D(160, (3,3), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3,3)))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        
        model.summary()
        
        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    # ---------------------------------------------------------------------------------------------------------------------
    # - Train loop, on the loop, we have to set the correct metrics to improve the generator and discriminator at same time 
    # ---------------------------------------------------------------------------------------------------------------------

    def train(self, epochs=2000, batch_size=128, metrics_update=50, save_images=100, save_model=2000, createPlots=False):
        # Setting global variables and lists on save the loss to plot them by the time on loop
        global epoch
        global start

        global D_accuracyList
        global D_lossList
        global G_lossList
        
        D_accuracyList = []
        D_lossList = []
        G_lossList = []

        start = time.time()

        # Setting the train set, batch size and discriminator and generator losses
        X_train = (images.astype(np.float32) - 127.5) / 127.5

        half_batch = int(batch_size / 2)
        
        mean_d_loss=[0,0]
        mean_g_loss=0

        # Starting the train loop
        for epoch in range(epochs):
            # ------------------------------------------ TRAINING PHASE ------------------------------------------ 

            # Select "half_batch" random images from our train set 
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # Create the random noise vector and it's generated image
            noise = np.random.normal(0, 1, (half_batch, self.noise_size))
            gen_imgs = self.generator.predict(noise)

            # Training the discriminator
            # The loss of the discriminator is the mean of the losses while training on authentic and fake images
            d_loss = 0.5 * np.add(self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1))),
                                  self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1))))

            # Training the generator
            for _ in range(2):
                noise = np.random.normal(0, 1, (batch_size, self.noise_size))

                valid_y = np.array([1] * batch_size)
                g_loss = self.combined.train_on_batch(noise, valid_y)
            
            mean_d_loss[0] += d_loss[0]
            mean_d_loss[1] += d_loss[1]
            mean_g_loss += g_loss
            
            # ----------------------------------- SAVES AT THE END OF THE EPOCH ----------------------------------

            # We print the losses and accuracy of the networks every 200 batches mainly to make sure the accuracy 
            # of the discriminator is not stable at around 50% or 100% (which would mean the discriminator performs 
            # not well enough or too well)
            if epoch % metrics_update == 0:
                print ("%d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" % (epoch, mean_d_loss[0]/metrics_update, 100*mean_d_loss[1]/metrics_update, mean_g_loss/metrics_update))
                mean_d_loss=[0,0]
                mean_g_loss=0
            
            # Calling the plot functions if we want to see the results of training every _n epochs:
            # In first 150 epochs, save _n images each 4 epochs, after that we only save each _n epochs
            if createPlots:

                if epoch < 150:
                    if epoch % 4 == 0:
                        self.save_images(epoch)
                else:
                    if epoch % save_images == 0:
                        self.save_images(epoch)
            
            # We save the architecture of the model, the weights and the state of the optimizer
            # This way we can restart the training exactly where we stopped
            if epoch % save_model == 0:
                self.generator.save(SAVE_WEIGHTS + "/generator_%d" % epoch)
                self.discriminator.save(SAVE_WEIGHTS + "/discriminator_%d" % epoch)


            # ------+------ Global variables to results plot ------+------
            global D_loss
            global D_acc
            global G_loss

            D_loss = mean_d_loss[0]/metrics_update
            D_acc = 100*mean_d_loss[1]/metrics_update
            G_loss = mean_g_loss/metrics_update

            if len(D_accuracyList) > 100:
                D_accuracyList.pop(0)
                D_lossList.pop(0)
                G_lossList.pop(0)

            D_accuracyList.append(D_loss)
            D_lossList.append(D_acc)
            G_lossList.append(G_loss)

    # ---------------------------------------------------------------------------------------------------------------------
    # Saving 16 randomly generated images to have a representation of the spectrum of images created by the generator
    # ---------------------------------------------------------------------------------------------------------------------

    def save_images(self, epoch):
        noise = np.random.normal(0, 1, (16, self.noise_size))
        gen_imgs = self.generator.predict(noise)
        
        gen_imgs = 0.5 * gen_imgs + 0.5   # Rescale from [-1,1] into [0,1]

        fig = plt.figure(constrained_layout=True, figsize=(6, 15))
        plt.suptitle("\nResults at Epoch: {}\nRTX 3060 12GB total time: {}s\n".format(epoch, round(time.time() - start, 1)), fontsize=20, family="sans-serif")

        subfigs = fig.subfigures(2, 1)
        for outerind, subfig in enumerate(subfigs.flat):

            if outerind == 0:
                subfig.suptitle("GAN inference on 16 random latent spaces", fontsize=18, family="sans-serif")
                axs = subfig.subplots(4, 4)
                for i in range(4):
                    for j in range(4):
                        axs[i,j].imshow(gen_imgs[4*i+j])
                        axs[i,j].axis('off')

            else:
                subfig.suptitle("\nMetrics on last 100 epochs\n", fontsize=18, family="sans-serif")
                axs = subfig.subplots(3, 1)
                
                axs[0].plot([i for i in range(epoch-len(D_accuracyList), epoch)], D_accuracyList)
                axs[0].set_xticks(np.arange(int(epoch/10)*10-90, int(epoch/10)*10+10, 10))
                axs[0].set_yticks(np.arange(0, 1.1, 0.1))
                axs[0].set_title("Discriminator Accuracy", fontsize=13, family="sans-serif")

                axs[1].plot([i for i in range(epoch-len(D_lossList), epoch)], D_lossList)
                axs[1].set_xticks(np.arange(int(epoch/10)*10-90, int(epoch/10)*10+10, 10))
                axs[1].set_yticks(np.arange(0, 100, 10))
                axs[1].set_title("\nDiscriminator Loss", fontsize=13, family="sans-serif")

                axs[2].plot([i for i in range(epoch-len(G_lossList), epoch)], G_lossList)
                axs[2].set_xticks(np.arange(int(epoch/10)*10-90, int(epoch/10)*10+10, 10))
                axs[2].set_yticks(np.arange(0, 10, 1))
                axs[2].set_title("\nGenerator Loss", fontsize=13, family="sans-serif")
        
        fig.savefig(SAVE_RESULTS + "/Faces_{}.png".format(epoch))
        plt.clf()
        plt.close()


# ------------------------------------ DCGAN model training ------------------------------------

gan = GAN()
gan.train(epochs = int(args.epochs), 
          batch_size = int(args.batch), 
          metrics_update = int(args.resetMetrics),  
          save_model = int(args.checkpoints),
          createPlots = args.createPlots,
          save_images = int(args.plotEpochsEvery))


# ---------- We can create a video to see the evolution of the model during training -----------

if args.createVideo:
    project_utils.create_video(SAVE_RESULTS, project_dir=SAVE_RESULTS_VIDEO, epochs=args.batch, video_name="DCGAN_training", fps=24)