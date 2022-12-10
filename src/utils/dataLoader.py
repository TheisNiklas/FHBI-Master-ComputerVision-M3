import tensorflow as tf


class DataLoader:
    # parameters
    imageHeight: int
    imageWidth: int
    trainSplit: float
    valSplit: float
    testSplit: float
    shuffle: bool
    seed: int

    def __init__(
        self,
        imageHeight: int = 224,
        imageWidth: int = 224,
        trainSplit: float = 0.70,
        valSplit: float = 0.15,
        testSplit: float = 0.15,
        shuffle: bool = True,
        seed: int = 123,
    ):
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        self.trainSplit = trainSplit
        self.valSplit = valSplit
        self.testSplit = testSplit
        self.shuffle = shuffle
        self.seed = seed

    def loadDatasets(
        self, dataDir: str, batchSize: int
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Loads a dataset and split it into train, validation and test datasets

        Args:
            dataDir (str): folder the dataset should be created from
            batchSize (int): batch size for the datasets

        Returns:
            tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: datasets in the order train, validation, test
        """
        ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
            dataDir,
            seed=123,
            shuffle=False,
            image_size=(self.imageHeight, self.imageWidth),
            batch_size=1,
            labels="inferred",
            
        )  # type: ignore
        ds, dsSize = self.__unbatchDataset(ds)
        return self.__createDatasets(ds, dsSize, batchSize)

    def __unbatchDataset(self, ds: tf.data.Dataset) -> tuple[tf.data.Dataset, int]:
        dsSize = ds.__len__().numpy()
        ds = ds.unbatch()
        return (ds, dsSize)

    def __createDatasets(
        self, ds: tf.data.Dataset, dsSize: int, batchSize: int
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Split the dataset into train, validation und test. Create batches and init prefetching.

        Args:
            ds (tf.data.Dataset): dataset to split
            dsSize (int): size of the dataset
            batchSize (int): batch size for created datasets

        Returns:
            tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: datasets in the order train, validation, test
        """
        assert (self.trainSplit + self.testSplit + self.valSplit) == 1

        if self.shuffle:
            ds = ds.shuffle(2 * dsSize, seed=self.seed)

        trainSize = int(self.trainSplit * dsSize)
        valSize = int(self.valSplit * dsSize)

        train_ds = ds.take(trainSize)
        train_ds = train_ds.batch(batchSize)
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = ds.skip(trainSize).take(valSize)
        val_ds = val_ds.batch(batchSize)
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = ds.skip(trainSize).skip(valSize)
        test_ds = test_ds.batch(batchSize)
        test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, val_ds, test_ds
