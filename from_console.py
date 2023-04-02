from data_loader.reader import FruitsDataLoader
from models.template_model import TemplateModel
from trainers.fruit_model_trainer import FruitModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from evaluater import evaluate
import numpy as np
import tensorflow as tf


config_path = "configs/conv_mnist_from_config.json"
config = process_config(config_path)

# create the experiments dirs
create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

print('Create the data generator.')
data_loader = FruitsDataLoader(config)
train_data  = data_loader.get_train_data()
valid_data  = data_loader.get_val_data()
test_data   = data_loader.get_test_data()


print('Create the model.')
model = TemplateModel(config)

print('Create the trainer')

trainer     = FruitModelTrainer(model.model, (train_data, valid_data), config)

print('Start training the model.')
trainer.train()

model.model.evaluate(valid_data)

evaluate.plot_acc_vs_valacc(trainer)

evaluate.plot_loss_vs_valloss(trainer)

image_batch, label_batch = data_loader.get_val_data().as_numpy_iterator().next()
predictions = np.argmax(trainer.model.predict_on_batch(image_batch), axis=1)

import matplotlib.pyplot as plt

class_names = ["best" ,"poor"]

fig, ax = plt.subplots(nrows=3, ncols=4)
plt.tight_layout()

for image, label in test_data:
    pred_proba  = model.model.predict(image)
    prediction  = np.argmax( pred_proba, axis=1 )
    best_indx   = np.max(pred_proba, axis=1)

    batch_size  = image.shape[0]
    img_indx    = 0
    for row in range(3):
        for col in range(4):
            ax[row, col].imshow(tf.cast( image[img_indx], tf.uint8))
            title = f"True {class_names[label[img_indx]]} Pred {class_names[prediction[img_indx]]} - {best_indx[img_indx]:.3f}"
            ax[row, col].set_title(title)
            ax[row, col].axis("off")
            # ax[row, col].title(f"{label[img_indx]}")
            img_indx += 1

plt.show()

def make_batch_predictions(images):
    pred = model.model.predict_on_batch(images)
    return np.argmax(pred, axis=1)



val_ds = data_loader.get_val_data()

i = 1
for images, true_labels in val_ds:
    pred_labels = make_batch_predictions(images)
    for indx, (tl, pl) in enumerate(zip(true_labels, pred_labels)):
        if tl != pl:
            print(tl, pl)

            ax = plt.subplot(5, 5, i)
            title = f"True {class_names[tl]} Pred {class_names[pl]}"
            plt.title(title)
            plt.imshow(images[indx].numpy().astype("uint8"))
            plt.axis("off")
            i += 1

            # plot_image_with_labels(images[indx], tl, pl)

true_labelss = list()
pred_labels = list()

for images, true_labels in test_data:
    true_labelss.extend(true_labels)
    pred_labels.extend(make_batch_predictions(images))

from sklearn import metrics

print(metrics.classification_report(true_labelss, pred_labels))

metrics.confusion_matrix(true_labelss, pred_labels)

precision = metrics.precision_score(true_labelss, pred_labels, average=None)
recall    = metrics.recall_score(true_labelss, pred_labels, average=None)
f1        = metrics.f1_score(true_labelss, pred_labels, average=None)
species = ("Best",  "Poor")
penguin_means = {
    'Precision': (precision[0], precision[1]),
    'Recall': (recall[0], recall[1]),
    'F1Score': (f1[0], f1[1])
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0


model.model.save("current_model_binary_classV3_regularizedV1.h5")

tf.keras.models.load_model("current_model_binary_classV3_regularizedV1.h5")