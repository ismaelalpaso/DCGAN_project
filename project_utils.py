import cv2
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from natsort import natsorted 


# Adjust the args.imgSize of train.py to fit in accepted image sizes to train the model
def real_img_size(value):
    img_size = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 
                272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]

    for x in range(0, 32):
        if value > 512 or value < 16:
            raise ValueError("The value has to be between 16 and 512")

        if value < img_size[x]:
            if value > img_size[x]-8:
                return img_size[x]
            else:
                return img_size[x-1]


# Load the model given a folder with the model.pb, keras_metadata.pb, and the subfolders /assets and /variables
def model_load(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


# Generate a image given the latent space, model, and path to save the image
def latentSpace2image(latent_space, model, image_path):
    generated_img = model.predict(latent_space)
    generated_img = 0.5 * generated_img + 0.5

    cv2.imwrite(image_path, generated_img)

# Create a plt.figure with the generated image given the latent space, model, and path to save the image
def latentSpace2imgJpg(latent_space, model, image_path):
    gen_imgs = model.predict(latent_space)
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig = plt.figure(constrained_layout=True, figsize=(1.55, 1.55))
    plt.imshow(gen_imgs[0])
    plt.axis('off')
    plt.savefig(image_path)


# Create a video with saved plots created on model training to see the progress
def create_video(images_dir, project_dir, epochs, video_name="DCGAN_training", fps=24):
    frameSize = (600, 1500)

    out = cv2.VideoWriter(project_dir + '{}{}_{}Fps.mp4'.format(video_name, epochs, fps), cv2.VideoWriter_fourcc(*'MJPG'), fps,         frameSize)
                          # Video folder + name.mp4                                       # Encoding technique             # Framerate  # Size (Width_xHeight)

    for filename in natsorted(glob.glob(images_dir + '/*.png')):
        img = cv2.imread(filename)
        out.write(img)

    out.release()