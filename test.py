from os.path import join
from RandLANet import Network
from tester import ModelTester
from helper_ply import read_ply
from helper_tool import Config as cfg
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import time, pickle, argparse, glob, os


class DATA:
    def __init__(self, data_path, path_cls, test_name):

        self.path = data_path
        self.test_name = test_name

        self.original = os.path.join(self.path, "original")
        self.sub_folder = os.path.join(self.path, "sub")

        classes, label_values, class2labels, label2color, label2names = DP.get_info_classes(path_cls)
        self.label_values = np.array(label_values)

        self.ignored_classes = []       # TODO TEST
        self.ignored_labels = np.array([class2labels[cls] for i, cls in enumerate(self.ignored_classes)])

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'test': []}
        self.input_colors = {'test': []}
        self.input_labels = {'test': []}
        self.input_names = {'test': []}
        self.input_full_xyz = {'test': []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):
        
        for cloud in os.listdir(os.path.join(self.original, 'test', self.test_name)):  
                
            print(cloud)
            cloud_name = cloud[:-4]

            # Name of the input files
            kd_tree_file = join(self.sub_folder, '{:s}_KDTree.pkl'.format(cloud_name))
            
            sub_ply_file = join(self.sub_folder, '{:s}.ply'.format(cloud_name))
            sub_data = read_ply(sub_ply_file)
            sub_colors = np.vstack((sub_data['red'], sub_data['green'], sub_data['blue'])).T
            sub_labels = sub_data['class']

            full_ply_file = join(self.original, 'test', self.test_name, '{:s}.ply'.format(cloud_name))
            full_data = read_ply(full_ply_file)
            full_xyz = np.vstack((full_data['x'], full_data['y'], full_data['z'])).T



            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees['test'] += [search_tree]
            self.input_colors['test'] += [sub_colors]
            self.input_labels['test'] += [sub_labels]
            self.input_names['test'] += [cloud_name]       
            self.input_full_xyz['test'] += [full_xyz]    

            
            # Test projection indices for testing and labels
            proj_file = join(self.sub_folder, '{:s}_proj.pkl'.format(cloud_name))
            with open(proj_file, 'rb') as f:
                proj_idx, labels = pickle.load(f)
            self.val_proj += [proj_idx]
            self.val_labels += [labels]

    # Generate the input data flow
    def get_batch_gen(self, split):

        num_per_epoch = cfg.val_steps * cfg.val_batch_size

        self.possibility[split] = []
        self.min_possibility[split] = []
        # Random initialize
        for i, tree in enumerate(self.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        def spatially_regular_gen():
            # Generator loop
            for i in range(num_per_epoch):

                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility in the cloud as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get all points within the cloud from tree structure
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # Check if the number of points in the selected cloud is less than the predefined num_points
                if len(points) < cfg.num_points:
                    # Query all points within the cloud
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    # Query the predefined number of points
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                # Shuffle index
                queried_idx = DP.shuffle_idx(queried_idx)
                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = self.input_colors[split][cloud_idx][queried_idx]
                queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx]

                # Update the possibility of the selected points
                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[split][cloud_idx][queried_idx] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

                # up_sampled with replacement
                if len(points) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           queried_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None])
        return gen_func, gen_types, gen_shapes

    @staticmethod
    def get_tf_mapping2():
        # Collect flat inputs
        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
            batch_features = tf.concat([batch_xyz, batch_features], axis=-1)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(cfg.num_layers):
                neighbour_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neighbour_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]

            return input_list

        return tf_map

    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [ign_label for ign_label in self.ignored_labels]
        gen_function_val, gen_types, gen_shapes = self.get_batch_gen('test')
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)

        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping2()

        self.batch_val_data = self.batch_val_data.map(map_func=map_func)

        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_val_data.output_types, self.batch_val_data.output_shapes)
        self.flat_inputs = iter.get_next()
        
        self.val_init_op = iter.make_initializer(self.batch_val_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument('--test_name', type=str, help='path to data')
    parser.add_argument('--path_cls', type=str, help='path to classes')
    parser.add_argument('--run', type=str, default='None', help='run folder path')
    parser.add_argument('--snap', type=str, default='None', help='snapshot number')
    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    test_name = FLAGS.test_name
    data_path = FLAGS.data_path
    path_cls = FLAGS.path_cls
    dataset = DATA(data_path, path_cls, test_name)
    dataset.init_input_pipeline()

    cfg.saving = False
    run = FLAGS.run
    snap = FLAGS.snap
    cfg.train_dir = run

    model = Network(dataset, cfg)

    tester = ModelTester(model, dataset, run, test_name, restore_snap=snap)
    tester.test(model, dataset, run, path_cls, test_name)

