import pandas as pd
import numpy as np
from collections import Counter
from math import log2


def entropy(data):
    label_counts = Counter(data)
    total = len(data)
    return -sum((count / total) * log2(count / total) for count in label_counts.values())


def information_gain(data, attribute, label):
    total_entropy = entropy(data[label])
    values, counts = np.unique(data[attribute], return_counts=True)
    weighted_entropy = sum((counts[i] / sum(counts)) * entropy(data[data[attribute] == values[i]][label])
                           for i in range(len(values)))
    return total_entropy - weighted_entropy


def id3(data, attributes, label):
    if len(np.unique(data[label])) == 1:
        return np.unique(data[label])[0]

    if len(attributes) == 0:
        return Counter(data[label]).most_common(1)[0][0]

    gains = [information_gain(data, attr, label) for attr in attributes]
    best_attr = attributes[np.argmax(gains)]

    tree = {best_attr: {}}

    for value in np.unique(data[best_attr]):
        sub_data = data[data[best_attr] == value]
        subtree = id3(sub_data, [attr for attr in attributes if attr != best_attr], label)
        tree[best_attr][value] = subtree

    return tree


def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    value = instance[attr]
    return predict(tree[attr][value], instance)


def main():
    data = pd.DataFrame({
        '天气': ['晴', '晴', '多云', '雨', '雨', '雨', '多云', '晴', '晴', '雨', '晴', '多云', '多云', '雨'],
        '气温': ['热', '热', '热', '适中', '冷', '冷', '冷', '适中', '冷', '适中', '适中', '适中', '热', '适中'],
        '湿度': ['高', '高', '高', '高', '正常', '正常', '正常', '高', '正常', '正常', '正常', '高', '正常', '高'],
        '风': ['无风', '有风', '无风', '无风', '无风', '有风', '有风', '无风', '无风', '无风', '有风', '有风', '无风',
               '有风'],
        '类别': ['N', 'N', 'P', 'P', 'P', 'N', 'P', 'N', 'P', 'P', 'P', 'P', 'P', 'N']
    })

    attributes = ['天气', '气温', '湿度', '风']
    label = '类别'

    tree = id3(data, attributes, label)
    print("决策树结构: ", tree)

    test_instance = {'天气': '晴', '气温': '冷', '湿度': '高', '风': '无风'}
    prediction = predict(tree, test_instance)
    print("预测结果: ", prediction)


if __name__ == "__main__":
    main()
