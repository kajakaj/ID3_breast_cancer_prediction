import numpy as np

def get_entropy(data, class_label, class_values):
    entropy = 0
    line_count = len(data)

    for val in class_values:
        val_line_count = len(data[data[class_label] == val])
        proportion = val_line_count/line_count
        if proportion != 0: # log2(0) is nan
            entropy -= proportion * np.log2(proportion)

    return entropy


def get_attr_inf_gain(data, class_label, class_values, a_label):
    a_values = data[a_label].unique()
    line_count = len(data)
    avg_inf = 0
    
    for a_val in a_values:
        a_val_line_count = len(data[data[a_label] == a_val])
        a_entropy = get_entropy(data[data[a_label] == a_val], class_label, class_values)
        avg_inf += a_val_line_count/line_count * a_entropy

    return get_entropy(data, class_label, class_values) - avg_inf


def get_best_attr(data, class_label, class_values):
    atributes = data.columns.drop(class_label)
    max_gain = - np.inf
    max_gain_atribute = None

    for a_label in atributes:
        a_gain = get_attr_inf_gain(data, class_label, class_values, a_label)
        if max_gain < a_gain:
            max_gain = a_gain
            max_gain_atribute = a_label

    return max_gain_atribute


def get_subdata(data, a_label, a_value):
    return data[data[a_label] == a_value].reset_index(drop=True)


def ID3(data, class_label, class_values, depth):
    # if depth equals max depth or set of attributes is empty - return most occuring class
    if depth == 0 or len(data) == 0:
        class_values_list = list(data[class_label])
        return max(set(class_values_list), key=class_values_list.count)
    
    # if all atributes are the same - return that value
    if len(data[class_label].unique()) == 1:
        return data[class_label][0]
    
    # in all other cases
    tree = {}
    best_attr = get_best_attr(data, class_label, class_values)
    tree[best_attr] = {}

    for a_val in data[best_attr].unique():
        subdata = get_subdata(data, best_attr, a_val)
        values = subdata[class_label].unique()
        tree[best_attr][a_val] = ID3(subdata, class_label, values, depth-1)

    return tree
