import numpy as np


class CrossValidation:

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.parts = dict()

    def splitDataIntoParts(self, parts_amount):
        dataArray = self.mergeIntoTable(self.features, self.labels)
        self.randomizeOrder(dataArray)
        arrayParts = self.splitArray(dataArray, parts_amount)
        self.saveDataToDict(arrayParts)

    def mergeIntoTable(self, vec1, vec2):
        return np.c_[vec1, vec2]

    def randomizeOrder(self, array):
        return np.random.shuffle(array)

    def splitArray(self, array, part_amount):
        return np.array_split(array, part_amount)

    def saveDataToDict(self, dataArray):
        n = len(dataArray)
        name = "part{}"
        for i in range(n):
            part_name = name.format(i+1)
            features = dataArray[i][:, :-1]
            labels = dataArray[i][:, -1]

            self.parts[part_name] = (features, labels)

    def iterate(self):
        data_parts = sorted(self.parts.keys())

        for part in data_parts:
            test_features = self.parts[part][0]
            test_labels = self.parts[part][1]

            learning_features = np.array([]).reshape((0,2))
            learning_labels = np.array([]).reshape((0,1))
            for key in self.parts.keys():
                if key == part:
                    continue
                learning_features = np.append(learning_features, self.parts[key][0], axis=0)
                learning_labels = np.append(learning_labels, self.parts[key][1])

            yield learning_features, test_features, learning_labels, test_labels
