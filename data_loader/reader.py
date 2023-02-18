from base.base_data_loader import BaseDataLoader
import tensorflow as tf
from utils import img_process


class FruitsDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(FruitsDataLoader, self).__init__(config)

        self.img_size = self.__get_img_size()
        self.__autotune = tf.data.AUTOTUNE

    def __get_img_size(self):
        img_size = self.config.data_loader.img_size
        height, width, _ = img_process.get_image_size(img_size)
        return height, width

    def get_train_data(self):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.data_loader.data_dir,
            validation_split=self.config.trainer.validation_split,
            subset="training",
            seed=12344,
            batch_size=self.config.trainer.batch_size,
            image_size=self.img_size
        )
        train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=self.__autotune)

        return train_ds

    def get_val_data(self):
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.data_loader.data_dir,
            validation_split=self.config.trainer.validation_split,
            subset="validation",
            seed=12344,
            batch_size=self.config.trainer.batch_size,
            image_size=self.img_size

        )

        val_ds = val_ds.cache().prefetch(buffer_size=self.__autotune)

        return val_ds

    def get_test_data(self):
        # val_batches = tf.data.experimental.cardinality(val_ds)
        # test_dataset = val_ds.take(1)
        # val_ds = val_ds.skip(1)
        pass

    def get_train_val_data(self):
        train_data = self.get_train_data()
        val_data = self.get_val_data()

        return train_data, val_data
