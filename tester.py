from os import makedirs
from os.path import exists, join
from helper_ply import write_ply
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import time
import os


def log_out(out_str, log_f_out):
    log_f_out.write(out_str + '\n')
    log_f_out.flush()
    print(out_str)


class ModelTester:
    def __init__(self, model, dataset, run, test_name="", restore_snap=None):
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        self.Log_file = open(os.path.join(run,'log_test_' + test_name + '.txt'), 'a')

        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())

        # Load trained model
        if restore_snap is not None:

            restore_snap = os.path.join(run, 'snapshots', 'snap-'+str(restore_snap))
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

        self.prob_logits = tf.nn.softmax(model.logits)

        # Initiate global prediction over all test clouds
        self.test_probs = [np.zeros(shape=[l.shape[0], model.config.num_classes], dtype=np.float32)     # TODO QUE AHCE AQUI? SE PUEDE QUITAR? RELACIONADO CON INPUT LABELS DE TEST_S3DIS.PY
                           for l in dataset.input_labels['test']]                         

    def test(self, model, dataset, run, path_cls, test_name="", num_votes=100):

        # Smoothing parameter for votes
        test_smooth = 0.95

        # Initialise iterator with validation/test data
        self.sess.run(dataset.val_init_op)

        # Number of points per class in validation set
        val_proportions = np.zeros(model.config.num_classes, dtype=np.float32)
        i = 0
        for label_val in dataset.label_values:
            if label_val not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in dataset.val_labels])
                i += 1

        # Test saving path
        test_path = os.path.join(run, 'predictions_' + test_name)
        makedirs(test_path) if not exists(test_path) else None

        step_id = 0
        epoch_id = 0
        last_min = -0.5

        while last_min < num_votes:
            try:
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['input_inds'],
                       model.inputs['cloud_inds'],
                       )
                #print("a")
                stacked_probs, stacked_labels, point_idx, cloud_idx = self.sess.run(ops, {model.is_training: False})  # TODO SACA PROBS, se puede sacar que caso es con cloud idx?
                correct = np.sum(np.argmax(stacked_probs, axis=1) == stacked_labels)
                acc = correct / float(np.prod(np.shape(stacked_labels)))
                print('step' + str(step_id) + ' acc:' + str(acc))
                stacked_probs = np.reshape(stacked_probs, [model.config.val_batch_size, model.config.num_points,
                                                           model.config.num_classes])

                for j in range(np.shape(stacked_probs)[0]):  # TODO SHAPE Y CONTENIDO DE STACKED PROBS
                    probs = stacked_probs[j, :, :]
                    p_idx = point_idx[j, :]
                    c_i = cloud_idx[j][0]                   # TODO ?? C_I
                    self.test_probs[c_i][p_idx] = test_smooth * self.test_probs[c_i][p_idx] + (1 - test_smooth) * probs # TODO GUARDA PROBS SMOOTHED
                step_id += 1

            except tf.errors.OutOfRangeError:

                new_min = np.min(dataset.min_possibility['test'])
                log_out('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.Log_file)

                if last_min + 1 < new_min:

                    # Update last_min
                    last_min += 1

                    # Show vote results (On subcloud so it is not the good values here)
                    log_out('\nConfusion on sub clouds', self.Log_file)
                    confusion_list = []

                    num_val = len(dataset.input_labels['test'])

                    for i_test in range(num_val):
                        #print("b")
                        probs = self.test_probs[i_test]                                                         # TODO RECUPERA PROBS
                        preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)                 # TODO RECUPERA LAS PREDS
                        labels = dataset.input_labels['test'][i_test]

                        # Confs
                        confusion_list += [confusion_matrix(labels, preds, dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)

                    # Rescale with the right number of point per class
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = DP.IoU_from_confusions(C)
                    m_IoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * m_IoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    log_out(s + '\n', self.Log_file)

                    if int(np.ceil(new_min)) % 1 == 0:

                        # Project predictions
                        log_out('\nReproject Vote #{:d}'.format(int(np.floor(new_min))), self.Log_file)
                        proj_probs_list = []

                        for i_val in range(num_val):
                            #print("c")
                            # Reproject probs back to the evaluations points
                            proj_idx = dataset.val_proj[i_val]               # TODO SE PUEDE QUITAR LA PARTE DE eval Y SACAR LAS PRED DIRECTAMENTE ASI
                            probs = self.test_probs[i_val][proj_idx, :]
                            proj_probs_list += [probs]

                        # Show vote results
                        log_out('Confusion on full clouds', self.Log_file)
                        confusion_list = []
                        for i_test in range(num_val):
                            #print("d")
                            # Get the predicted labels
                            preds = dataset.label_values[np.argmax(proj_probs_list[i_test], axis=1)].astype(np.uint8)

                            # Confusion
                            labels = dataset.val_labels[i_test]
                            acc = np.sum(preds == labels) / len(labels)
                            log_out(dataset.input_names['test'][i_test] + ' Acc:' + str(acc), self.Log_file)

                            confusion_list += [confusion_matrix(labels, preds, dataset.label_values)]
                            name = dataset.input_names['test'][i_test] + '.ply'             
                            xyz = dataset.input_full_xyz['test'][i_test]
                            pred_colors = DP.labels2colors(preds, path_cls)
                            write_ply(join(test_path, name), (xyz, pred_colors), ['x', 'y', 'z', 'red', 'green', 'blue'])



                        # Regroup confusions
                        C = np.sum(np.stack(confusion_list), axis=0)

                        print(type(C))
                        print(C.shape)
                        print(C)

                        IoUs = DP.IoU_from_confusions(C)
                        m_IoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * m_IoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        log_out('-' * len(s), self.Log_file)
                        log_out(s, self.Log_file)
                        log_out('-' * len(s) + '\n', self.Log_file)

                        acc_global, prec_calsses, rec_classes, acc_classes = DP.metrics_from_confusions(C)
                        for i in range(IoUs):
                            str_acc = list(labels.keys())[list(labels.values()).index(i)] + ' accuracy: ' + str(acc_classes[i])
                            str_prec = list(labels.keys())[list(labels.values()).index(i)] + ' precision: ' + str(prec_calsses[i])
                            str_rec = list(labels.keys())[list(labels.values()).index(i)] + ' recall: ' + str(rec_classes[i])
                            log_out(str_acc + '\n')
                            log_out(str_prec + '\n')
                            log_out(str_rec + '\n\n')

                        print('finished \n')
                        self.sess.close()
                        return

                self.sess.run(dataset.val_init_op)
                epoch_id += 1
                step_id = 0
                continue

        return
