import tensorflow as tf


class PreProcessor():
    def __init__(self):
        pass

    def __augment_layer(self):

        img_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomRotation(factor=(-0.2, 0.2)),
                tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
                tf.keras.layers.RandomFlip("horizontal"),
            ],
            name="img_augmentation",
        )

        return img_augmentation

    def rescale(self):
        rescale_layer = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1. / 255)
        ])

        return rescale_layer

    def augment(self):
        data_augmentation = self.__augment_layer()
        return data_augmentation

    def augment_gpu(self):
        pass

    def augment_cpu(self, ds, shuffle=False, augment=False):
        AUTOTUNE = tf.data.AUTOTUNE

        # # Resize and rescale all datasets.
        # rescale_layer = self.rescale()
        # ds = ds.map(lambda x, y: (rescale_layer(x), y),
        #             num_parallel_calls=AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(100)

        # Batch all datasets.
        # ds = ds.batch(batch_size)

        # Use data augmentation only on the training set.
        if augment:
            data_augmentation = self.__augment_layer()

            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=AUTOTUNE).cache()

        ds = ds.cache()
        # Use buffered prefetching on all datasets.
        return ds.prefetch(buffer_size=AUTOTUNE)
