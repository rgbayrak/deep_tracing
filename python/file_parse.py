import os

def parse_file(filename):
    data_labels = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data, label = line.split(',')
            data, label = data.strip(), label.strip()
            data_labels[label] = data

    return data_labels