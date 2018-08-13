import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist


class BasicClassification(object):
    CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def __init__(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

    def trian(self):
        # 构建深度学习网络
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(256, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        # 确定学习参数
        model.compile(
            optimizer=tf.train.AdadeltaOptimizer(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # 开始训练
        model.fit(self.train_images, self.train_labels, epochs=5)

        return model

    def test(self, model):
        test_loss, test_acc = model.evaluate(self.test_images, self.test_labels)

        print('Test accuracy:', test_loss, test_acc)

    def test_show(self):
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid('off')
            plt.imshow(self.train_images[i], cmap=plt.cm.binary)
            plt.xlabel(self.CLASS_NAMES[self.train_labels[i]])
        plt.show()

    @classmethod
    def show_picture(cls, image):
        plt.figure()
        plt.imshow(image)
        plt.colorbar()
        plt.gca().grid(False)
        plt.show()


def main():
    basic = BasicClassification()
    model = basic.trian()
    basic.test(model)


if __name__ == "__main__":
    main()
