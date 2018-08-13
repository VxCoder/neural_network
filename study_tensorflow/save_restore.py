import numpy as np
import tensorflow as tf

from tensorflow import keras
from matplotlib import pyplot as plt


class SaveRestore(object):
    CHECKPOINT_PATH = "training/cp.ckpt"

    def __init__(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = tf.keras.datasets.mnist.load_data()
        self.train_images = self.train_images.reshape(-1, 28 * 28) / 255.0
        self.test_images = self.test_images.reshape(-1, 28 * 28) / 255.0

    def create_model(self):
        model = tf.keras.models.Sequential([
            keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784, )),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])

        return model

    def train(self, model):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(self.CHECKPOINT_PATH,
                                                         save_weights_only=True,
                                                         verbose=1)

        model.fit(self.train_images, self.train_labels,  epochs=10,
                  callbacks=[cp_callback])

    def test(self, model):
        model.load_weights(self.CHECKPOINT_PATH)
        loss, acc = model.evaluate(self.test_images, self.test_labels)
        print("Restored model, loss: {:5.2f}% accuracy: {:5.2f}%".format(100 * loss, 100 * acc))

    def predict(self, model, image):
        image = image.reshape(1, 784)
        return np.argmax(model.predict(image))

    @classmethod
    def show_image(cls, image):
        plt.figure()
        plt.imshow(image)
        plt.show()


def main():
    save_restore = SaveRestore()
    model = save_restore.create_model()
    # save_restore.train(model)
    # save_restore.test(model)
    model.load_weights(save_restore.CHECKPOINT_PATH)
    save_restore.show_image(save_restore.test_images[0].reshape(28, 28))
    save_restore.predict(model, save_restore.test_images[0])


if __name__ == "__main__":
    main()
