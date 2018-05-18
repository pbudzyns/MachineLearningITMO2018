import json
from pandas import DataFrame
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.svm import SVC


def load_data(filename):
    data = {'id': [], 'ingredients': [], 'cuisine': []}
    with open(filename, 'r') as f:
        recipes = json.load(f)
        for recipe in recipes:
            data['id'].append(recipe['id'])
            data['ingredients'].append(recipe['ingredients'])
            data['cuisine'].append(recipe.get('cuisine', ''))

    return DataFrame(data)


if __name__ == '__main__':

    train_cookbook = load_data('data/train.json')
    test_cookbook = load_data('data/test.json')
    # print(train_cookbook)

    x = np.array(train_cookbook['ingredients'])
    y = np.array(train_cookbook['cuisine'])

    x = np.reshape(x, (len(x), 1))
    y = np.reshape(y, (len(y), 1))


    test_x = np.array(test_cookbook['ingredients'])
    # model = GaussianNB()
    model = SVC()
    model.fit(x, y)
    pred = model.predict(test_x)
    print(pred)