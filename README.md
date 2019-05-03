# DCGAN with pytorch

The goal of this project is to completely re-implement original DCGAN and to replicate some of experiments presented in [_**Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks**_](https://arxiv.org/abs/1511.06434)



### How to run

To train model from scratch, use following command.

```
python main.py --dataset [CelebA, LSUN, STL10]
```

To train model following existing checkpoint, use following command.

```
python main.py --dataset [CelebA, LSUN, STL10] --model_path MODEL_PATH
```



To generate new images using the trained model, use following command. (image_save_path is optional)

```
python main.py --test --model_path MODEL_PATH (--image_save_path IMAGE_SAVE_PATH)
```



### Results

Under training...
