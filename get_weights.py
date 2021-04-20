import os
import re
import argparse
import sys
import json

def get_info_classes(cls_path):
    classes = []
    colors = []
    for line in open(cls_path):
        data = line.split()
        classes.append(data[0])
        colors.append([int(data[1]), int(data[2]), int(data[3])])
    labels = [i for i, cls in enumerate(classes)]
    class2labels = {cls: i for i, cls in enumerate(classes)}
    label2color = {classes.index(cls): colors[classes.index(cls)] for cls in classes}
    label2names = {classes.index(cls): cls for cls in classes}

    return classes, labels, class2labels, label2color, label2names


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def main():

    # python get_weights_randla.py --path_txt /home/miguel/Desktop/data/sets/1_1/txt --path_cls /home/miguel/Desktop/data/sets/classes/1.txt #

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_txt', help='path to the txt dataset folder.')
    parser.add_argument('--path_cls', help='path to the classes file.')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_txt = parsed_args.path_txt
    path_cls = parsed_args.path_cls

    path_out = os.path.join(path_txt, "weights.txt")

    classes, _, _, _, _ = get_info_classes(path_cls)
    classes_dict = {i: 0 for i in classes}
    for root, dirs, files in os.walk(path_txt):  # for each folder
        for file in files:
            if file.endswith('.txt'):
                name = file.split("_")[0]
                if name.endswith('.txt'):
                    break
                len = file_len(os.path.join(root,file))
                classes_dict[name] += len

    with open(path_out, 'w') as file:
        file.write(json.dumps(classes_dict))








if __name__ == "__main__":
    main()

