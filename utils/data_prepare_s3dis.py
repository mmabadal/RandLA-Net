from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import pandas as pd
import os, sys, glob, pickle
import os
import sys
import argparse
from natsort import natsorted
BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply
from helper_tool import DataProcessing as DP

parser = argparse.ArgumentParser()
parser.add_argument('--path_in', help='path to the txt data folder.')
parser.add_argument('--path_out', help='path to save ply folder.')
parser.add_argument('--path_cls', help='path to classes txt.')

parsed_args = parser.parse_args(sys.argv[1:])

path_in = parsed_args.path_in
path_out = parsed_args.path_out
path_cls = parsed_args.path_cls

path_orig = os.path.join(path_out,"original")
path_orig_train = os.path.join(path_orig,"training")
path_orig_val = os.path.join(path_orig,"validation")
path_out_sub = os.path.join(path_out,"sub")

if not os.path.exists(path_out):
    os.mkdir(path_out)

if not os.path.exists(path_orig):
    os.mkdir(path_orig)
    os.mkdir(path_orig_train)
    os.mkdir(path_orig_val)

if not os.path.exists(path_out_sub):
    os.mkdir(path_out_sub)

classes, label_values, class2labels, label2color, label2names = DP.get_info_classes(path_cls)

sub_grid_size = 0.04

def convert_pc2ply(case, split):
    """
    Convert original dataset files to ply file (each line is XYZRGBL).
    We aggregated all the points from each instance in the room.
    :param case: case. e.g. office_2
    :return: None
    """
    data_list = []

    anno_path = os.path.join(path_in, split, case, "annotations")

    for f in glob.glob(join(anno_path, '*.txt')):
        class_name = os.path.basename(f).split('_')[0]
        if class_name not in classes: 
            print("ERROR, " + str(class_name) + " CLASE NAME NOT RECOGNIZED")
            break
        pc = pd.read_csv(f, header=None, delim_whitespace=True).values                # TODO LEER DESDE PLY?
        labels = np.ones((pc.shape[0], 1)) * class2labels[class_name]
        data_list.append(np.concatenate([pc, labels], 1))  # Nx7

    pc_label = np.concatenate(data_list, 0)
    xyz_min = np.amin(pc_label, axis=0)[0:3]
    pc_label[:, 0:3] -= xyz_min

    xyz = pc_label[:, :3].astype(np.float32)
    colors = pc_label[:, 3:6].astype(np.uint8)
    labels = pc_label[:, 6].astype(np.uint8)
    ply_file = os.path.join(path_orig, split, case + ".ply" )
    write_ply(ply_file, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    # save sub_cloud and KDTree file
    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
    sub_colors = sub_colors / 255.0
    sub_ply_file = os.path.join(path_out_sub, case + ".ply" )
    write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    search_tree = KDTree(sub_xyz)
    kd_tree_file = os.path.join(path_out_sub, case + "_KDTree.pkl" )
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
    proj_idx = proj_idx.astype(np.int32)
    proj_save = os.path.join(path_out_sub, case + "_proj.pkl" )
    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx, labels], f)


if __name__ == '__main__':

    for split in ("training", "validation"):
        for folder in natsorted(os.listdir(os.path.join(path_in, split))):
            print("working on case: " + str(folder))
            convert_pc2ply(folder, split)
