
import os, pathlib, PIL
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model

class AlexNet(Model):
  def __init__(self, data_shape=(224, 224, 3), num_classes=1000):
    super(AlexNet, self).__init__()
    self.data_augmentation = keras.Sequential(
      [
        layers.experimental.preprocessing.RandomFlip(
          "horizontal", 
          input_shape=data_shape),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
      ]
    )
    
    self.rescaling = layers.experimental.preprocessing.Rescaling(1./255)
    
    # layer 1
    self.conv1 = layers.Conv2D(
      filters=96,
      kernel_size=(11, 11),
      strides=4,
      padding="valid",
      activation='relu',
      input_shape= data_shape,
      kernel_initializer='GlorotNormal')
    self.pool1 = layers.MaxPooling2D(
      pool_size=(3, 3),
      strides=2,
      padding="valid")
    self.norm1 = tf.keras.layers.BatchNormalization()
    
    # layer 2
    self.conv2 = layers.Conv2D(
      filters=256,
      kernel_size=(5, 5),
      strides=1,
      padding="same",
      activation='relu',
      kernel_initializer='GlorotNormal')
    self.pool2 = layers.MaxPooling2D(
      pool_size=(3, 3),
      strides=2,
      padding="valid")
    self.norm2 = tf.keras.layers.BatchNormalization()
    
    # layer 3
    self.conv3 = layers.Conv2D(
      filters=384,
      kernel_size=(3, 3),
      strides=1,
      padding="same",
      activation='relu',
      kernel_initializer='GlorotNormal')
        
    # layer 4
    self.conv4 = layers.Conv2D(
      filters=384,
      kernel_size=(3, 3),
      strides=1,
      padding="same",
      activation='relu',
      kernel_initializer='GlorotNormal')
    
    # layer 5
    self.conv5 = layers.Conv2D(
      filters=256,
      kernel_size=(3, 3),
      strides=1,
      padding="same",
      activation='relu',
      kernel_initializer='GlorotNormal')
    self.pool5 = layers.MaxPooling2D(
      pool_size=(3, 3),
      strides=2,
      padding="valid")
    self.norm5 = tf.keras.layers.BatchNormalization()
        
    # layer 6
    self.flatten6 = tf.keras.layers.Flatten()
    self.d6 = tf.keras.layers.Dense(
      units=4096,
      activation='relu')
    self.drop6 = tf.keras.layers.Dropout(rate=0.5)
    
    # layer 7
    self.d7 = tf.keras.layers.Dense(
      units=4096,
      activation='relu')
    self.drop7 = tf.keras.layers.Dropout(rate=0.5)
    
    # layer 8
    self.d8 = tf.keras.layers.Dense(
      units=num_classes,
      activation='softmax')
      
    self.build((None,) + data_shape)

  def call(self, x):
    x = self.data_augmentation(x)
    x = self.rescaling(x)
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.norm1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.norm2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.pool5(x)
    x = self.norm5(x)
    x = self.flatten6(x)
    x = self.d6(x)
    x = self.drop6(x)
    x = self.d7(x)
    x = self.drop7(x)
    x = self.d8(x)
    return x


class AlexNetWork():
  def __init__(self, args):
    # dataset
    data_dir = pathlib.Path(args.dataset_path)
    self.image_height = args.image_height
    self.image_width = args.image_width
    data_shape = (args.image_height, args.image_width, 3)
    batch_size = args.batchsize
    
    pretrain_model_path_or_dir = args.pre_train_model_path_dir
    
    # create model
    self.model = AlexNet(
        data_shape = data_shape,
        num_classes=args.classes)
        
    if os.path.exists(pretrain_model_path_or_dir):
      if args.use_whole_network_model:
        dir = pretrain_model_path_or_dir
        self.model = keras.models.load_model(dir)
        print("Whole network load from {} dir".format(dir))
      else:
        path = pretrain_model_path_or_dir
        self.model.load_weights(path)
        print("Network model load from {}".format(path))
    
    # Optimization
    self.learning_rate = args.lr
    self.epochs = args.epochs
    
    if args.opt_type == 'Adam':
      self.optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.lr)
    elif args.opt_type == 'SGD':
      self.optimizer = tf.keras.optimizers.SGD(
        learning_rate=args.lr,
        momentum=args.momentum)
    elif args.opt_type == 'Adadelta':
      self.optimizer = tf.keras.optimizers.Adadelta(
        learning_rate=args.lr)
    elif args.opt_type == 'Adamax':
      self.optimizer = tf.keras.optimizers.Adamax(
        learning_rate=args.lr)
    elif args.opt_type == 'Ftrl':
      self.optimizer = tf.keras.optimizers.Ftrl(
        learning_rate=args.lr)
    elif args.opt_type == 'Nadam':
      self.optimizer = tf.keras.optimizers.Nadam(
        learning_rate=args.lr)
    else:
      self.optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=args.lr)
        
    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # get the data set
    image_count = 0
    image_count += len(list(data_dir.glob('*/*.jpg')))
    image_count += len(list(data_dir.glob('*/*.JPEG')))
    print("image number:", image_count)
    
    # train dataset
    self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(args.image_height, args.image_width),
      batch_size=batch_size)
    self.class_names = self.train_ds.class_names
    self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
      
    # valid/test dataset
    self.test_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(args.image_height, args.image_width),
      batch_size=batch_size)
    self.test_ds = self.test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    self.test_loss = tf.keras.metrics.Mean(name='valid_loss')
    self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='vaild_accuracy')
  
  @tf.function
  def train_step(self, images, labels):
    with tf.GradientTape() as tape:
      predictions = self.model(images)
      loss = self.loss_object(labels, predictions)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    
    self.train_loss(loss)
    self.train_accuracy(labels, predictions)
  # [end train_step]
    
  @tf.function
  def test_step(self, images, labels):
    predictions = self.model(images)
    t_loss = self.loss_object(labels, predictions)

    self.test_loss(t_loss)
    self.test_accuracy(labels, predictions)
  # [end test_step]
    
  def train(self):
    # Model summary
    self.model.summary()
    
    for epoch in range(self.epochs):
    
      self.train_loss.reset_states()
      self.train_accuracy.reset_states()
      self.test_loss.reset_states()
      self.test_accuracy.reset_states()
      
      try:
        with tqdm(self.train_ds, ncols=80) as t:
          for images, labels in t:
            self.train_step(images, labels)
            template = '[Train\t Epoch {}] Loss: {:.4f}, Accuracy: {:.4f}'
            template = template.format(epoch+1, self.train_loss.result(), self.train_accuracy.result()*100)
            t.set_description(desc=template)
      except KeyboardInterrupt:
        t.close()
        raise

      try:
        with tqdm(self.test_ds, ncols=80) as t:
          for test_images, test_labels in t:
            self.test_step(test_images, test_labels)
            template = '[Test\t Epoch {}] Loss: {:.4f}, Accuracy: {:.4f}'
            template = template.format(epoch+1, self.test_loss.result(), self.test_accuracy.result()*100)
            t.set_description(desc=template)
      except KeyboardInterrupt:
        t.close()
        raise
  # [end train]
        
  def saveModel(self, path_or_dir, mode='save_weight'):
    if mode == 'save_weight':
      path = path_or_dir
      self.model.save_weights(path)
      print("Network model save to {}".format(path))
    elif mode == 'whole_network':
      dir = path_or_dir
      self.model.save(dir)
      print("Whole network save to {} dir".format(dir))
  # [end saveModel]
  
  def test(self, args):
    if not os.path.exists(args.test_image):
      return
      
    image_path = args.test_image
      
    img = keras.preprocessing.image.load_img(
      image_path, target_size=(
        args.image_height,
        args.image_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = self.model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
        
    import numpy as np
    print("{} most likely belongs to {} with a {:.2f} percent confidence.".format(image_path, self.class_names[np.argmax(score)], 100 * np.max(score)))
  # [end test]
    
    
    
    
    