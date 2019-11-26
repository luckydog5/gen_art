import os 
import matplotlib.pyplot as plt
import numpy as np 

from keras import initializers
from keras.datasets import cifar10,mnist
from keras.initializers import RandomNormal
from keras.layers import (Conv2D,BatchNormalization,Conv2DTranspose,Dense,
                          Dropout,Flatten,Input,Reshape,UpSampling2D,ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU

from keras.models import Model,Sequential
from keras.optimizers import Adam 
from PIL import Image,ImageDraw 

np.random.seed(1337)

class Config(object):

    noise_dim = 100
    batch_size = 16
    steps_per_epoch = 312 # 50000 / 16 
    epochs = 800
    save_path = 'dcgan-images'
    img_rows,img_cols,channels = 32,32,3

def gen():

    config = Config()
    (x_train,y_train),(x_test,y_test) = cifar10.load_data()

    # Normalize to between -1 and 1 
    x_train = (x_train.astype(np.float32)-127.5)/127.5 

    # Reshape and only save cat images

    x_train = x_train[np.where(y_train==0)[0]].reshape(-1,config.img_rows,config.img_cols,config.channels)
    return x_train 


def create_generator():
    config = Config()
    generator = Sequential()
    d = 4 
    generator.add(Dense(d*d*256,kernel_initializer=RandomNormal(0,0.02),input_dim=config.noise_dim))
    generator.add(LeakyReLU(0.2))
    # 4*4*256
    generator.add(Reshape((d,d,256)))
    # 8*8*128
    generator.add(Conv2DTranspose(128,(4,4),strides=2,padding='same',kernel_initializer=RandomNormal(0,0.02)))
    generator.add(LeakyReLU(0.2))

    # 16*16*128 
    generator.add(Conv2DTranspose(128,(4,4),strides=2,padding='same',kernel_initializer=RandomNormal(0,0.02)))
    generator.add(LeakyReLU(0.2))

    # 32*32*128 
    generator.add(Conv2DTranspose(128,(4,4),strides=2,padding='same',kernel_initializer=RandomNormal(0,0.02)))
    generator.add(LeakyReLU(0.2))

    # 32*32*3 
    generator.add(Conv2D(config.channels,(3,3),padding='same',activation='tanh',kernel_initializer=RandomNormal(0,0.02)))
    optimizer = Adam(0.0002,0.5)
    generator.compile(loss='binary_crossentropy',optimizer=optimizer)

    return generator 

def create_descriminator():

    config = Config()
    discriminator = Sequential()

    discriminator.add(Conv2D(64,(3,3),padding='same',kernel_initializer=RandomNormal(0,0.02),input_shape=(config.img_cols,config.img_rows,config.channels)))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(128,(3,3),strides=2,padding='same',kernel_initializer=RandomNormal(0,0.02)))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(128,(3,3),strides=2,padding='same',kernel_initializer=RandomNormal(0,0.02)))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(256,(3,3),strides=2,padding='same',kernel_initializer=RandomNormal(0,0.02)))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Flatten())
    discriminator.add(Dropout(0.4))
    discriminator.add(Dense(1,activation='sigmoid',input_shape=(config.img_cols,config.img_rows,config.channels)))
    optimizer = Adam(0.0002,0.5)
    discriminator.compile(loss='binary_crossentropy',optimizer=optimizer)
    return discriminator
def show_images(generator,noise,epoch=None):

    # Display images, and save them if the epoch number is specified
    config = Config() 
    if not os.path.isdir(config.save_path):
        os.mkdir(config.save_path)

    generated_images = generator.predict(noise)
    plt.figure(figsize=(10,10))
    for i,image in enumerate(generated_images):
        plt.subplot(10,10,i+1)
        if config.channels == 1:
            plt.imshow(image.reshape((config.img_rows,config.img_cols)),cmap='gray')
        else:
            plt.imshow(image.reshape((config.img_rows,config.img_cols,config.channels)))
        plt.axis('off')
    plt.tight_layout()
    if epoch != None:
        plt.savefig(f'{config.save_path}/gan-images_epoch-{epoch}.png')

def train():
    config = Config()
    x_train = gen()
    discriminator = create_descriminator()
    generator = create_generator()
    discriminator.trainable = False 
    gan_input = Input(shape=(config.noise_dim,))
    fake_image = generator(gan_input)

    gan_output = discriminator(fake_image)

    gan = Model(inputs=gan_input,outputs=gan_output)
    optimizer = Adam(0.0002,0.5)
    gan.compile(loss='binary_crossentropy',optimizer=optimizer)
    # Constant noise for viewing how the GAN progresses
    static_noise = np.random.normal(0,1,size=(100,config.noise_dim))
    for epoch in range(config.epochs):
        for batch in range(config.steps_per_epoch):
            noise = np.random.normal(0,1,size=(config.batch_size,config.noise_dim))
            real_x = x_train[np.random.randint(0,x_train.shape[0],size=config.batch_size)]

            fake_x = generator.predict(noise)

            x = np.concatenate((real_x,fake_x))
            disc_y = np.zeros(2*config.batch_size)
            disc_y[:config.batch_size] = 0.9

            d_loss = discriminator.train_on_batch(x,disc_y)

            y_gen = np.ones(config.batch_size)

            g_loss = gan.train_on_batch(noise,y_gen)
        print("Epoch {} \t Discriminator Loss: {} \t\t Generator Loss: {}".format(epoch,d_loss,g_loss))
        if epoch % 10 == 0:
            show_images(generator,static_noise,epoch)
    discriminator.save('dcdiscriminator.h5')
    generator.save('dcgenerator.h5')
def img_to_gif(save_path):
    frames = []
    image_names = os.listdir(save_path)
    for image in sorted(image_names,key=lambda name: int(''.join(i for i in name if i.isdigit()))):
        frames.append(Image.open(save_path + '/' + image))
    frames[0].save('gan_training.gif',format='GIF',append_images=frames[1:],save_all=True,duration=80,loop=0)



if __name__ == '__main__':

    train()

