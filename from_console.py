from data_loader.reader import FruitsDataLoader
from models.template_model import TemplateModel
from trainers.fruit_model_trainer import FruitModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from evaluater import evaluate
import numpy as np

config_path = "configs/conv_mnist_from_config.json"
config = process_config(config_path)

# create the experiments dirs
create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

print('Create the data generator.')
data_loader = FruitsDataLoader(config)

print('Create the model.')
model = TemplateModel(config)

print('Create the trainer')

trainer = FruitModelTrainer(model.model, data_loader.get_train_val_data(), config)

print('Start training the model.')
trainer.train()

model.model.evaluate(data_loader.get_val_data())
evaluate.plot_acc_vs_valacc(trainer)

evaluate.plot_loss_vs_valloss(trainer)

image_batch, label_batch = data_loader.get_val_data().as_numpy_iterator().next()
predictions = np.argmax(trainer.model.predict_on_batch(image_batch), axis=1)

import matplotlib.pyplot as plt

class_names = ["best", "normal" ,"poor"]
plt.figure(figsize=(10, 10))
for i in range(9):
    w = i
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[w].astype("uint8"))
    title = f"True {class_names[label_batch[w]]} Pred {class_names[predictions[w]]}"
    plt.title(title)
    plt.axis("off")


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

for images, true_labels in val_ds:
    true_labelss.extend(true_labels)
    pred_labels.extend(make_batch_predictions(images))

from sklearn import metrics

print(metrics.classification_report(true_labelss, pred_labels))

metrics.confusion_matrix(true_labelss, pred_labels)

precision = metrics.precision_score(true_labelss, pred_labels, average=None)
recall    = metrics.recall_score(true_labelss, pred_labels, average=None)
f1        = metrics.f1_score(true_labelss, pred_labels, average=None)
species = ("Best", "Normal", "Poor")
penguin_means = {
    'Precision': (precision[0], precision[1], precision[2]),
    'Recall': (recall[0], recall[1], recall[2]),
    'F1Score': (f1[0], f1[1], f1[2]),
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(constrained_layout=True)

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_title('Precision Recall F1Score of 3 classes')
ax.set_xticks(x + width, species)
ax.legend()
ax.set_ylim(0, 1)

plt.show()

