# DCGAN_project

That repository contains the necessary to train a DCGAN with Keras api, and play with the noise vector to generate custom images usin a Tkinter GUI. Feel free to download and play!

## train.py script. 
It contains the `GAN class` and functions to train and create plots during train to see the evolution of model, it creates plots of random noise vectors to generate images and shows the losses of generator and discriminator and discriminator accuracy. For example, in the video you can see a train on 3000 epchs using https://www.kaggle.com/datasets/andy8744/ganyu-genshin-impact-anime-faces-gan-training to train the model:


https://user-images.githubusercontent.com/114246096/204888785-17bdad04-cd5a-4fe8-9926-5ab87dc94094.mp4


To only train a model use that args: 

```sh
cd clone_fold
python3 train.py --main project_path --project project_name --data train_data_dir --imgSize --batch 256 --resetMetrics --epochs 10001 --checkpoints 2000 
```

If you want to create plots add the follow args:

```sh
--createPlots --plotEpochsEvery epochs_to_save_plot 
```

If you want to create a video automatically with the plots:

```sh
--createVideo
```

## interface.py script 
It creates a GUI with tkinter that allows play with latent space (noise vector) to generate customized images with the models that you've created, the default latent vector size is 20.

To call it you can write:

```sh
cd clone_fold
python3 interface.py --model model_fold --memmory fold_to_save_genImages
```

On the window there are two images, at right, the default generated setting all values of noise vector to 0, at left the generated image after changing the values of latent space. With default script and vector size you've around 200^20 combinations!

![image](https://user-images.githubusercontent.com/114246096/204952093-beb2aaf0-908a-4910-be73-5576fc21a9a8.png)
