# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 09:18:14 2021

@author: dylan
"""

# example of loading the cifar10 dataset
from matplotlib import pyplot as plt
from numpy.random import randn
import random
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model
import glob
from numpy import expand_dims
from numpy import ones
from numpy import zeros
from numpy import vstack
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from numpy.random import rand
import h5py
from numpy.random import randint		
from PIL import Image
from keras.models import load_model
import threading
import os
import time
import progressbar

# load the images into memory
#(trainX, trainy), (testX, testy) = load_data()

size = 128
n_batch = 60 # change this to reduce strain on memory but increase training time 
num_epochs = 200
latent_dim = 100
n_samples = 49
path = 'landscape_data.npy'

widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]
    
try:
    trainX = np.load(path)
except FileNotFoundError:
    trainX = []

try:
    if trainX.shape[1] != size: # resize images!!!!!
        array = []
        for i in range(0, trainX.shape[0]):
            img = Image.fromarray(trainX[i])
            img = img.resize((size, size))
            array.append(np.asarray(img))
        np.save('data2.npy', array)
        trainX = np.load('data2.npy')
except (AttributeError):
    print('%s does not exist!' % path)

def define_discriminator(in_shape=(size, size, 3)):
    model = Sequential()
    
    # normal quality
    model.add(Conv2D(64, (3, 3), strides=(2,2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    
    # higher quality
    model.add(Conv2D(128, (3, 3), strides=(2,2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    
    # higher quality
    model.add(Conv2D(128, (3, 3), strides=(2,2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    
    # higher quality
    model.add(Conv2D(256, (3, 3), strides=(2,2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    
    if size > 32:
        # higher quality
        model.add(Conv2D(256, (3, 3), strides=(2,2), padding='same', input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))
    
    if size > 64:
        # higher quality
        model.add(Conv2D(256, (3, 3), strides=(2,2), padding='same', input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))
        
    if size > 128:
        # higher quality
        model.add(Conv2D(512, (3, 3), strides=(2,2), padding='same', input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))
    
    # framework
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def load_real_samples():
    X = trainX.astype('float32')
    X = (X - 127.5) / 127.5    
    return X

def get_real_examples(data, n_samples):
    ix = randint(0, data.shape[0], n_samples)
    X = data[ix]
    y = ones((n_samples, 1))
    return X, y

def get_fake_examples(n_samples):
    X = rand(size * size * 3 * n_samples)
    X = (X * 2) - 1
    X = X.reshape(n_samples, size, size, 3)
    y = zeros((n_samples, 1))
    return X, y

#def train(model, data, n_iter=20, n_batch=128):
#    half_batch = int(n_batch / 2)
#    for i in range(n_iter):
#        X_real, y_real = get_real_examples(data, half_batch)
#        _, real_acc = model.train_on_batch(X_real, y_real)
#        X_fake, y_fake = get_fake_examples(half_batch)
#        _, fake_acc = model.train_on_batch(X_fake, y_fake)
#        print('>%d real=%.0f%% fake=%.0f%%' % (i + 1, real_acc * 100, fake_acc * 100))

def define_generator(latent_dim):
    model = Sequential()
    
    n_nodes = 256 * 4 * 4
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))
    
    # upsample to 8x8
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    if size > 32:
        # upsample to 64x64
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        
    if size > 64:
        # upsample to 128x128
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        
    if size > 128:
        # upsample to 128x128
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
    
    
    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return model

def generate_latent_points(latent_dim, n_samples):
    X_input = randn(latent_dim * n_samples)
    x_input = X_input.reshape(n_samples, latent_dim)
    return x_input

def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y = zeros((n_samples, 1))
    return X, y

def define_gan(g_model, d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def evaluate(epoch, g_model, d_model, gan, data, latent_dim, n_samples=150):
    d_model.trainable = False
    X_real, y_real = get_real_examples(data, n_samples)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(X_fake, y_fake, verbose=0)
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    save_plot(X_fake, epoch)
    g_filename = 'Models/generator_model.h5'
    g_model.save(g_filename)
    d_filename = 'Models/discriminator_model.h5'
    d_model.save(d_filename)
    gan_filename = 'Models/gan_model.h5'
    gan.save(gan_filename)
    return acc_real, acc_fake

    
def save_plot(examples, epoch, n=2):
	# scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
	# plot images
    for i in range(n * n):
		# define subplot
        plt.subplot(n, n, 1 + i)
		# turn off axis
        plt.axis('off')
		# plot raw pixel data
        plt.imshow(examples[i])
    # save plot to file
    filename = 'Generated Images/Images/generated_plot_e%03d.png' % (epoch+1)
    plt.savefig(filename)
    plt.close()    
   
def plotSummary(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
    # loss
    plt.subplot(2, 1, 1)
    plt.plot(d1_hist, label='d-loss-real')
    plt.plot(d2_hist, label='d-loss-fake')
    plt.plot(g_hist, label='gen-loss')
    plt.legend()
    
    # discriminator acc
    plt.subplot(2, 1, 2)
    plt.plot(a1_hist, label='acc-real')
    plt.plot(a2_hist, label='acc-fake')
    plt.legend()
    plt.savefig('Results/plot_line_plot_loss.png')
    plt.close()
    
# train the composite model
#def train_gan(gan_model, latent_dim, n_epochs=200, n_batch=128):
	# manually enumerate epochs
#	for i in range(n_epochs):
		# prepare points in latent space as input for the generator
#		x_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
#		y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
#		gan_model.train_on_batch(x_gan, y_gan)

def train(g_model, d_model, gan_model, data, latent_dim, n_epochs=num_epochs, n_batch=n_batch):
    bat_per_epo = int(data.shape[0] / n_batch)
    half_batch  = int(n_batch / 2)
    d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            X_real, y_real = get_real_examples(data, half_batch)
			# update discriminator model weights
            d_loss1, d_acc1 = d_model.train_on_batch(X_real, y_real)
			# generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update discriminator model weights
            d_loss2, d_acc2 = d_model.train_on_batch(X_fake, y_fake)
			# prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan) 
			# summarize loss on this batch
            print('>epoch: %d, batch: %d/%d, d_loss_real = %.3f, d_loss_fake = %.3f, gan_loss = %.3f' % (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
            if ((j + 1) % bat_per_epo) == 0:
                d1_hist.append(d_loss1)
                d2_hist.append(d_loss2)
                g_hist.append(g_loss)
                d_acc1, d_acc2 = evaluate(i, g_model, d_model, gan_model, data, latent_dim)
                a1_hist.append(d_acc1)
                a2_hist.append(d_acc2)
                plotSummary(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)

def loadModel(samples):
    # load model
    model = load_model('Models/generator_model_128.h5')
    # all 0s
    vector = np.asarray([[random.random() for _ in range(latent_dim)]])
    vector = generate_latent_points(latent_dim, samples)
    # generate image
    X = model.predict(vector)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot the result
    d = 'Generated Images/Finished Samples'
    loc = os.listdir(d)
    for i, f in enumerate(loc):
        try:
            os.rename(os.path.join(d, f), os.path.join(d, 'sample%d.png' % i))
        except:
            print('')
    for i in range(0, X.shape[0]):
        plt.figure(figsize=(20, 20))
        plt.axis('off')
        plt.imshow(X[i, :, :])
        plt.savefig('Generated Images/Finished Samples/sample%d.png' % len(os.listdir(d)), bbox_inches='tight', pad_inches = 0)
    
def images_to_array():
    bar = progressbar.ProgressBar(maxval=200, widgets=widgets).start()
    progress = 0
    image_path = 'landscape'
    data_path = 'landscape_data.npy'
    npy_array = []
    error_array = 0
    folder_path = glob.glob('%s/*.jpg' % image_path)
    for name in folder_path:
        try:
            image = Image.open(name)
            image = image.resize((size, size))
            npy = np.asarray(image)
            if (npy.shape == (128, 128, 3)):
                npy_array.append(npy)
            else:
                #print('%s is not the right shape!' % name)
                error_array += 1
        except:
            print('%s does not exist!' % name)
        progress += 200/len(folder_path)
        bar.update(progress)
    
    np.save(data_path, npy_array)
    print('There are now %d images in %s! %d images were deleted due to error. :(' % (np.load(data_path).shape[0], data_path, error_array))

def merge_with_data(data1, data2):
    d1 = np.load(data1)
    d2 = np.load(data2)

    d3 = np.append(d1, d2, 0)
    np.save(data1, d3)
    print('There are now %d images in %s! %d images were added!' % (np.load(data1).shape[0], data1, d2.shape[0]))

g_model = define_generator(latent_dim)
d_model = define_discriminator()
#g_model = load_model('Models/generator_model.h5')
#d_model = load_model('Models/discriminator_model.h5')
#gan_model = load_model('Models/gan_model.h5')
gan_model = define_gan(g_model, d_model)

try:
    data = load_real_samples()
except AttributeError:
    print('%s does not exist!' % path)

train(g_model, d_model, gan_model, data, latent_dim)
#loadModel(50)
#print('There are now %d images in %s!' % (np.load(path).shape[0], path))
#images_to_array()
#merge_with_data('data.npy', 'landscape_data.npy')

#g_model = load_model('Models/generator_model.h5')

