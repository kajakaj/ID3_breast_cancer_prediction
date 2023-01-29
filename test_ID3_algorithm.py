import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from ID3_algorithm import ID3


def get_prediction(tree, instance):
    while isinstance(tree, dict):
        attr_name = next(iter(tree))
        attr_value = instance[attr_name]
        if attr_value not in tree[attr_name]:
            attr_value = random.choice([*tree[attr_name]])
        tree = tree[attr_name][attr_value]

    return tree


def get_ID3_accuracy(ID3_tree, test, class_label):
    correct = 0
    for i in range(len(test)):
        instance = test.iloc[i]
        if get_prediction(ID3_tree, instance) == test[class_label].iloc[i]:
            correct += 1
    accuracy = correct/len(test) * 100
    return accuracy


def get_best_depth(class_label, class_values, train, validation):
    max_accuracy = 0
    for depth in range(1, 10, 1):
        tree = ID3(train, class_label, class_values, depth)
        tree_accur = get_ID3_accuracy(tree, validation, class_label)
        if tree_accur > max_accuracy:
            max_accuracy = tree_accur
            best_depth = depth
    return best_depth


def generate_plot(class_label, class_values, train, validation):
    y_s = []
    x_s = []
    for depth in range(1, 10, 1):
        tree = ID3(train, class_label, class_values, depth)
        tree_accur = get_ID3_accuracy(tree, validation, class_label)
        y_s.append(tree_accur)
        x_s.append(depth)
        plt.plot(x_s, y_s)
        plt.ylabel("accuracy [%]")
        plt.xlabel("depth")
        plt.savefig('plot_id3.png')
        plt.clf()


def main():
    class_label = "irradiat"
    data_headers = ["Class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradiat"]
    train_data = pd.read_csv("data/breast-cancer.data", header=None)
    train_data.columns = data_headers
    train, validation, test = np.split(train_data.sample(frac=1), [int(.6*len(train_data)),int(.8*len(train_data))])
    class_values = train[class_label].unique()
    
    best_depth = get_best_depth("irradiat", class_values, train, validation)
    print("Best depth: {}".format(best_depth))
    tree = ID3(train, class_label, class_values, best_depth)
    print("Accuracy: {:2f}".format(get_ID3_accuracy(tree, test, class_label)))

    generate_plot(class_label, class_values, train, validation)


if __name__ == "__main__":
    main()