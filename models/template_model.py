from base.base_model import BaseModel
import tensorflow as tf
from preprocessor.preprocessor import PreProcessor
from utils import img_process


class OptimizerException(Exception):
    pass


class TemplateModel(BaseModel):
    def __init__(self, config):
        super(TemplateModel, self).__init__(config)

        self.prep = PreProcessor()
        self.build_model()
        self.init_saver()

    @staticmethod
    def __get_optimizer_class(requested_opt: str):
        requested_opt = requested_opt.lower()

        opts = {
            "adam": tf.keras.optimizers.Adam,
            "rmsprop": tf.keras.optimizers.RMSprop,
            "sgd": tf.keras.optimizers.SGD,
            "nadam": tf.keras.optimizers.Nadam
        }

        opt_class = opts.get(requested_opt)
        if opt_class is not None:
            return opt_class
        else:
            raise OptimizerException("Optimizer Unknown... Use only adam, nadam, rmsprop, sgd... ")

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        data_augmentation = self.prep.augment()
        inputs     = tf.keras.Input(shape = img_process.get_image_size(self.config.data_loader.img_size))
        x          = data_augmentation(inputs)
        # preprocess = tf.keras.applications.densenet.preprocess_input
        # x          = preprocess(x)
        base_model = tf.keras.applications.EfficientNetB4(include_top=False, input_tensor=x)
        base_model.trainable = False

        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
        x = tf.keras.layers.Dense(1000, activation="selu", kernel_initializer="lecun_normal")(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(500, activation="selu", kernel_initializer="lecun_normal")(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Dense(500, activation="selu", kernel_initializer="lecun_normal")(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Dropout(0.3)(x)
        # x = tf.keras.layers.Dense(1000, activation="leaky_relu")(x)
        # x = tf.keras.layers.Dense(2000, activation="selu", kernel_initializer="lecun_normal")(x)
        #
        # top_dropout_rate = 0.5
        # x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax", name="pred")(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EfficientNet")

        opt_str = self.config.model.optimizer
        opt = self.__get_optimizer_class(opt_str)

        lr = self.config.model.learning_rate
        # if self.config.model.dynamic_lr:
        #     print("using dynamic learning_rate scheduler")
        #     lr =  tf.keras.optimizers.schedules.ExponentialDecay(
        #         initial_learning_rate=self.config.model.learning_rate,
        #         decay_steps=10000,
        #         decay_rate=0.9
        #     )

        self.model.compile(loss="sparse_categorical_crossentropy", optimizer=opt(learning_rate=lr),
                           metrics=["accuracy"])

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        pass
