import tensorflow as tf
from typing import Any


class ModelLoader:
    imgHeight: int
    imgWidth: int
    imgDepth: int

    def __init__(self, imgHeight: int = 224, imgWidth: int = 224, imgDepth: int = 3):
        self.imgHeight = imgHeight
        self.imgWidth = imgWidth
        self.imgDepth = imgDepth

    def loadMobileNetV1(
        self,
        train_ds: tf.data.Dataset,
        freezeBaseModel: bool = False,
        initWeightsRandom: bool = True,
    ) -> tf.keras.Model:
        """
        Loads MobileNetV1 with custom top

        Args:
            train_ds (tf.data.Dataset): training dataset
            freezeBaseModel (bool, optional): whether to freeze the weights of the base model. Defaults to False.
            initWeightsRandom (bool, optional): initialize weights random if True or load imagenet weights if False. Defaults to True.

        Returns:
            tf.keras.Model
        """
        weights = None
        if not initWeightsRandom:
            weights = "imagenet"
        baseModel: tf.keras.Model = tf.keras.applications.MobileNet(
            input_shape=(self.imgHeight, self.imgWidth, self.imgDepth),
            include_top=False,
            weights=weights,  # type: ignore
        )
        if freezeBaseModel:
            baseModel.trainable = False

        model = self.__buildModel(
            baseModel, train_ds, tf.keras.applications.mobilenet.preprocess_input
        )

        return model

    def loadMobileNetV1WithTop(self, initWeightsRandom: bool = True) -> tf.keras.Model:
        """
        Loads MobileNetV1 with the default top

        Args:
            initWeightsRandom (bool, optional): initialize weights random if True or load imagenet weights if False. Defaults to True.

        Returns:
            tf.keras.Model
        """
        weights = None
        if not initWeightsRandom:
            weights = "imagenet"
        return tf.keras.applications.MobileNet(
            input_shape=(self.imgHeight, self.imgWidth, self.imgDepth),
            include_top=True,
            classes=1,
            weights=weights,  # type: ignore
            classifier_activation="softmax",
        )

    def loadMobileNetV2(
        self,
        train_ds: tf.data.Dataset,
        freezeBaseModel: bool = False,
        initWeightsRandom: bool = True,
    ) -> tf.keras.Model:
        """
        Loads MobileNetV2 with custom top

        Args:
            train_ds (tf.data.Dataset): training dataset
            freezeBaseModel (bool, optional): whether to freeze the weights of the base model. Defaults to False.
            initWeightsRandom (bool, optional): initialize weights random if True or load imagenet weights if False. Defaults to True.

        Returns:
            tf.keras.Model:
        """
        weights = None
        if not initWeightsRandom:
            weights = "imagenet"
        baseModel: tf.keras.Model = tf.keras.applications.MobileNetV2(
            input_shape=(self.imgHeight, self.imgWidth, self.imgDepth),
            include_top=False,
            weights=weights,  # type: ignore
        )
        if freezeBaseModel:
            baseModel.trainable = False

        model = self.__buildModel(
            baseModel, train_ds, tf.keras.applications.mobilenet_v2.preprocess_input
        )

        return model

    def loadTrainedModel(self, modelPath: str) -> tf.keras.Model:
        return tf.keras.models.load_model(modelPath)

    def __getGlobalAverageLayer(self) -> tf.keras.layers.GlobalAveragePooling2D:
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        return global_average_layer

    def __getPredictionLayer(self) -> tf.keras.layers.Dense:
        prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid")
        return prediction_layer

    def __buildModel(
        self, baseModel: tf.keras.Model, train_ds: tf.data.Dataset, preprocess
    ) -> tf.keras.Model:
        global_average_layer = self.__getGlobalAverageLayer()

        inputs = tf.keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgDepth))
        x = preprocess(inputs)
        x = baseModel(x)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = self.__getPredictionLayer()(x)
        model = tf.keras.Model(inputs, outputs)
        return model
