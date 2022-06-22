{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "AI_cuoi_ky",
      "provenance": [],
      "collapsed_sections": [],
      "private_outputs": true,
      "mount_file_id": "1f1LigfkN4nPMZvhQFfixpAFZC9xBfLXF",
      "authorship_tag": "ABX9TyO+hIm6dd3fAae9FaXlV5NJ",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU",
    "gpuClass": "standard"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/TruongLieu/Hand_Language/blob/main/train_model.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras.preprocessing.image import load_img, img_to_array\n",
        "from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D\n",
        "from tensorflow.keras.optimizers import Adam, SGD\n",
        "from tensorflow.keras.utils import to_categorical\n",
        "from sklearn.model_selection import train_test_split\n",
        "from keras.models import Sequential\n",
        "from keras.utils import np_utils\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "import glob"
      ],
      "metadata": {
        "id": "ohhhyPD2vMOl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "fist = list()\n",
        "for img in glob.glob('/content/drive/MyDrive/dataset/fist/*.jpg'):\n",
        "  fist.append(img_to_array(load_img(img,target_size=(150,150))))\n",
        "L = list()\n",
        "for img in glob.glob('/content/drive/MyDrive/dataset/L/*.jpg'):\n",
        "  L.append(img_to_array(load_img(img,target_size=(150,150))))\n",
        "okay = list()\n",
        "for img in glob.glob('/content/drive/MyDrive/dataset/okay/*.jpg'):\n",
        "  okay.append(img_to_array(load_img(img,target_size=(150,150))))\n",
        "palm = list()\n",
        "for img in glob.glob('/content/drive/MyDrive/dataset/palm/*.jpg'):\n",
        "  palm.append(img_to_array(load_img(img,target_size=(150,150))))\n",
        "peace = list()\n",
        "for img in glob.glob('/content/drive/MyDrive/AIdataset/peace/*.jpg'):\n",
        "  peace.append(img_to_array(load_img(img,target_size=(150,150))))"
      ],
      "metadata": {
        "id": "UEGrKEiHvUAM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "input_dataset = np.array(fist + L + okay + palm + peace)\n",
        "label_dataset = np.array([0]*len(fist) + [1]*len(L) + [2]*len(okay) + [3]*len(palm) + [4]*len(peace))"
      ],
      "metadata": {
        "id": "KiHE0GF05dxQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(input_dataset.shape)\n",
        "print(label_dataset.shape)"
      ],
      "metadata": {
        "id": "cy6oB3Kj56hN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x_train, x_test, y_train, y_test = train_test_split(input_dataset,label_dataset\n",
        "                                                    ,train_size=0.8, test_size=0.2, random_state= 0)"
      ],
      "metadata": {
        "id": "VhLNKRVF6JKN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(y_train.shape)\n",
        "print(x_train.shape)"
      ],
      "metadata": {
        "id": "-b_VJo4D-knK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x_train = x_train.astype('float32')\n",
        "x_test = x_test.astype('float32')\n",
        "x_train /= 255\n",
        "x_test/= 255"
      ],
      "metadata": {
        "id": "GhB4D0Ot-oOk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y_train = to_categorical(y_train,5)\n",
        "y_test = to_categorical(y_test,5)"
      ],
      "metadata": {
        "id": "W7FFAZln-wfY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "Model = Sequential()\n",
        "Model.add(Conv2D(32,(3,3),activation = 'relu', kernel_initializer= 'he_uniform', padding = 'same',input_shape = (150,150,3)))\n",
        "Model.add(Conv2D(32,(3,3),activation = 'relu', kernel_initializer= 'he_uniform',padding = 'same'))\n",
        "Model.add(MaxPooling2D((2,2)))\n",
        "Model.add(Conv2D(64,(3,3),activation = 'relu', kernel_initializer= 'he_uniform', padding = 'same'))\n",
        "Model.add(Conv2D(64,(3,3),activation = 'relu', kernel_initializer= 'he_uniform',padding = 'same'))\n",
        "Model.add(MaxPooling2D((2,2)))\n",
        "Model.add(Conv2D(128,(3,3),activation = 'relu', kernel_initializer= 'he_uniform', padding = 'same'))\n",
        "Model.add(Conv2D(128,(3,3),activation = 'relu', kernel_initializer= 'he_uniform',padding = 'same'))\n",
        "Model.add(MaxPooling2D((2,2)))\n",
        "Model.add(Flatten())\n",
        "Model.add(Dense(128,activation= 'relu',kernel_initializer='he_uniform'))\n",
        "Model.add(Dense(5,activation = 'softmax'))\n",
        "opt = SGD(lr = 0.01,momentum = 0.9)\n",
        "Model.compile(optimizer= opt,loss = 'categorical_crossentropy', metrics=['accuracy'])\n",
        "history = Model.fit(x_train,y_train,epochs=50,batch_size = 64,validation_data=(x_test,y_test), verbose=1)"
      ],
      "metadata": {
        "id": "BxjxVhY9-1JL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "Model.save('/content/drive/MyDrive/dataset/mymodel.h5')"
      ],
      "metadata": {
        "id": "E4mRlKs1_D5_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "3"
      ],
      "metadata": {
        "id": "hSF_Vn029isD"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}