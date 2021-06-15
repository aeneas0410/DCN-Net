
import numpy as np
import os.path
from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold, RepeatedKFold
from architectures.arch_creator import generate_model
from utils.callbacks import generate_callbacks, generate_output_filename
from utils.ioutils import read_dataset, read_ADNI_dataset, read_volume, read_volume_data, save_dataset, \
    save_volume_MICCAI2012, save_volume_ADNI, save_intermediate_volume, save_patches, read_patches, \
    read_dentate_dataset, read_dentate_interposed_dataset, read_interposed_fastigial_dataset, read_tha_dataset_v2, \
    read_interposed_fastigial_dataset_rev, save_volume_dentate, save_volume_interposed_fastigial, \
    save_volume_dentate_interposed, read_tha_dataset, read_tha_dataset_unseen, save_volume_tha, save_volume_tha_unseen, \
    save_volume_tha_v2, read_dentate_interposed_dataset_unseen, save_volume_dentate_interposed_unseen, \
    read_sar_dataset, save_sar_volume, __save_sar_volume, save_sar_volume_2_5D
from utils.image import preprocess_test_image, hist_match, normalize_image, standardize_set, hist_match_set, \
    standardize_volume, find_crop_mask, crop_image
from utils.reconstruction import reconstruct_volume, reconstruct_volume_modified, reconstruct_volume_sar
from utils.training_testing_utils import split_train_val, build_training_set, build_testing_set, build_training_set_4d, \
    build_training_set_sar, build_training_set_sar_2_5D, build_testing_set_2_5D, build_training_set_sar_2_5D_MIMO
from utils.general_utils import pad_both_sides
from utils.mixup_generator import MixupGenerator
from utils.mathutils import compute_statistics, computeDice, dice, measure_cmd, measure_msd
from utils import random_eraser, ioutils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import itertools
from importance_sampling.training import ImportanceTraining
import glob


def run_evaluation_in_dataset(gen_conf, train_conf, test_conf, args):

    gen_conf['args'] = args
    gen_conf['validation_mode'] = args.mode
    num_classes = tuple([int(i) for i in args.num_classes.split(',')])
    if len(num_classes) == 1:
        gen_conf['num_classes'] = num_classes[0]
    else:
        gen_conf['num_classes'] = num_classes
    gen_conf['multi_output'] = int(args.multi_output)
    output_name = args.output_name.split(',')
    if len(num_classes) == 1:
        gen_conf['output_name'] = output_name[0]
    else:
        gen_conf['output_name'] = output_name
    gen_conf['dataset_path'] = args.dataset_path
    train_conf['num_of_gpu'] = int(args.num_of_gpu)
    train_conf['approach'] = args.approach
    train_conf['dataset'] = args.dataset
    test_conf['dataset'] = args.dataset
    train_conf['data_augment'] = int(args.data_augment)
    train_conf['attention_loss'] = int(args.attention_loss)
    train_conf['overlap_penalty_loss'] = int(args.overlap_penalty_loss)
    loss = args.loss.split(',')
    if len(loss) == 1:
        train_conf['loss'] = loss[0]
    else:
        train_conf['loss'] = loss
    train_conf['exclusive_train'] = int(args.exclusive_train)
    train_conf['metric'] = args.metric
    train_conf['lamda'] = tuple([float(i) for i in args.lamda.split(',')])
    activation = args.activation.split(',')
    if len(activation) == 1:
        train_conf['activation'] = activation[0]
    else:
        train_conf['activation'] = activation
    train_conf['preprocess'] = int(args.preprocess_trn)
    test_conf['preprocess'] = int(args.preprocess_tst)
    train_conf['num_k_fold'] = int(args.num_k_fold)
    train_conf['batch_size'] = int(args.batch_size)
    train_conf['num_epochs'] = int(args.num_epochs)
    train_conf['patience'] = int(args.patience)
    train_conf['optimizer'] = args.optimizer
    train_conf['initial_lr'] = float(args.initial_lr)
    train_conf['is_set_random_seed'] = int(args.is_set_random_seed)
    if args.random_seed_num != 'None':
        train_conf['random_seed_num'] = int(args.random_seed_num)
    else:
        train_conf['random_seed_num'] = None
    train_conf['bg_discard_percentage'] = float(args.bg_discard_percentage)
    train_conf['bg_value'] = float(args.bg_value)
    train_conf['importance_sampling'] = int(args.importance_spl)
    train_conf['oversampling'] = int(args.oversampling)
    train_conf['patch_shape'] = tuple([int(i) for i in args.trn_patch_size.split(',')])
    train_conf['output_shape'] = tuple([int(i) for i in args.trn_output_size.split(',')])
    train_conf['extraction_step'] = tuple([int(i) for i in args.trn_step_size.split(',')])
    test_conf['patch_shape'] = tuple([int(i) for i in args.tst_patch_size.split(',')])
    test_conf['output_shape'] = tuple([int(i) for i in args.tst_output_size.split(',')])
    test_conf['extraction_step'] = tuple([int(i) for i in args.tst_step_size.split(',')])
    test_conf['threshold'] = float(args.threshold)
    test_conf['is_measure'] = int(args.is_measure)
    test_conf['is_unseen_case'] = int(args.is_unseen_case)
    train_conf['continue_tr'] = int(args.continue_tr)
    train_conf['is_new_trn_label'] = int(args.is_new_trn_label)
    train_conf['new_label_path'] = args.new_label_path
    gen_conf['dataset_info'][train_conf['dataset']]['folder_names'] = args.folder_names.split(',')
    gen_conf['dataset_info'][train_conf['dataset']]['margin_crop_mask'] = \
        tuple([int(i) for i in args.crop_margin.split(',')])
    gen_conf['dataset_info'][train_conf['dataset']]['image_modality'] = args.image_modality.split(',')
    gen_conf['dataset_info'][train_conf['dataset']]['augment_sar_data'] = tuple([int(i) for i in args.augment_sar_data.split(',')])
    gen_conf['dataset_info'][train_conf['dataset']]['trn_dim'] = args.trn_dim.split(',')
    gen_conf['dataset_info'][train_conf['dataset']]['ensemble_weight'] = tuple([float(i) for i in args.ensemble_weight.split(',')])
    target = args.target.split(',')
    if len(target) == 1:
        gen_conf['dataset_info'][train_conf['dataset']]['target'] = target[0]
    else:
        gen_conf['dataset_info'][train_conf['dataset']]['target'] = target
    exclude_label_num = []
    for i in args.exclude_label_num.split(','):
        if not i == '':
            exclude_label_num.append(int(i))
        else:
            exclude_label_num.append([])
    if len(exclude_label_num) == 1:
        gen_conf['dataset_info'][train_conf['dataset']]['exclude_label_num'] = exclude_label_num[0]
    else:
        gen_conf['dataset_info'][train_conf['dataset']]['exclude_label_num'] = exclude_label_num

    train_conf['GAN']['generator']['g_network'] = args.g_network
    g_num_classes = tuple([int(i) for i in args.g_num_classes.split(',')])
    if len(g_num_classes) == 1:
        train_conf['GAN']['generator']['num_classes'] = g_num_classes[0]
    else:
        train_conf['GAN']['generator']['num_classes'] = g_num_classes
    train_conf['GAN']['generator']['multi_output'] = int(args.g_multi_output)

    g_output_name = args.g_output_name.split(',')
    if len(g_output_name) == 1:
        train_conf['GAN']['generator']['output_name'] = g_output_name[0]
    else:
        train_conf['GAN']['generator']['output_name'] = g_output_name
    train_conf['GAN']['generator']['attention_loss'] = int(args.g_attention_loss)
    train_conf['GAN']['generator']['overlap_penalty_loss'] = int(args.g_overlap_penalty_loss)
    train_conf['GAN']['generator']['lamda'] = tuple([float(i) for i in args.g_lamda.split(',')])
    train_conf['GAN']['generator']['adv_loss_weight'] = float(args.g_adv_loss_weight)
    train_conf['GAN']['generator']['metric'] = args.g_metric
    g_loss = args.g_loss.split(',')
    if len(g_loss) == 1:
        train_conf['GAN']['generator']['loss'] = g_loss[0]
    else:
        train_conf['GAN']['generator']['loss'] = g_loss
    g_activation = args.g_activation.split(',')
    if len(g_activation) == 1:
        train_conf['GAN']['generator']['activation'] = g_activation[0]
    else:
        train_conf['GAN']['generator']['activation'] = g_activation
    train_conf['GAN']['generator']['patch_shape'] = tuple([int(i) for i in args.g_trn_patch_size.split(',')])
    test_conf['GAN']['generator']['patch_shape'] = tuple([int(i) for i in args.g_tst_patch_size.split(',')])
    train_conf['GAN']['generator']['output_shape'] = tuple([int(i) for i in args.g_trn_output_size.split(',')])
    test_conf['GAN']['generator']['output_shape'] = tuple([int(i) for i in args.g_tst_output_size.split(',')])
    train_conf['GAN']['generator']['extraction_step'] = tuple([int(i) for i in args.g_trn_step_size.split(',')])
    test_conf['GAN']['generator']['extraction_step'] = tuple([int(i) for i in args.g_tst_step_size.split(',')])

    # discriminator
    train_conf['GAN']['discriminator']['d_network'] = args.d_network
    d_num_classes = tuple([int(i) for i in args.d_num_classes.split(',')])
    if len(d_num_classes) == 1:
        train_conf['GAN']['discriminator']['num_classes'] = d_num_classes[0]
    else:
        train_conf['GAN']['discriminator']['num_classes'] = d_num_classes
    train_conf['GAN']['discriminator']['metric'] = args.d_metric
    d_loss = args.d_loss.split(',')
    if len(d_loss) == 1:
        train_conf['GAN']['discriminator']['loss'] = d_loss[0]
    else:
        train_conf['GAN']['discriminator']['loss'] = d_loss
    d_activation = args.d_activation.split(',')
    if len(d_activation) == 1:
        train_conf['GAN']['discriminator']['activation'] = d_activation[0]
    else:
        train_conf['GAN']['discriminator']['activation'] = d_activation
    train_conf['GAN']['discriminator']['patch_shape'] = tuple([int(i) for i in args.d_trn_patch_size.split(',')])
    test_conf['GAN']['discriminator']['patch_shape'] = tuple([int(i) for i in args.d_tst_patch_size.split(',')])
    train_conf['GAN']['discriminator']['output_shape'] = tuple([int(i) for i in args.d_trn_output_size.split(',')])
    test_conf['GAN']['discriminator']['output_shape'] = tuple([int(i) for i in args.d_tst_output_size.split(',')])
    train_conf['GAN']['discriminator']['extraction_step'] = tuple([int(i) for i in args.d_trn_step_size.split(',')])
    test_conf['GAN']['discriminator']['extraction_step'] = tuple([int(i) for i in args.d_tst_step_size.split(',')])


    if args.mode is '0': # training + testing
        if train_conf['dataset'] == 'iSeg2017':
            return evaluate_using_loo(gen_conf, train_conf, test_conf)
        if train_conf['dataset'] == 'IBSR18' :
            return evaluate_using_loo(gen_conf, train_conf, test_conf)
        if train_conf['dataset'] == 'MICCAI2012' :
            return evaluate_using_training_testing_split(gen_conf, train_conf)
        if train_conf['dataset'] in ['3T7T', '3T7T_real', '3T7T_total', '3T+7T', '3T7T_multi']:
            return evaluate_using_loo(gen_conf, train_conf, test_conf)
        if train_conf['dataset'] in ['CBCT16', 'CBCT57', 'CT30']:
            return evaluate_using_loo(gen_conf, train_conf, test_conf)
        if train_conf['dataset'] in ['dcn_seg_dl']:
            return evaluate_dentate_interposed_seg(gen_conf, train_conf, test_conf)
            # if 'dentate' in target and not 'interposed' in target:
            #     return evaluate_dentate_seg(gen_conf, train_conf, test_conf)
            # if 'dentate' in target and 'interposed' in target:
            #     return evaluate_dentate_interposed_seg(gen_conf, train_conf, test_conf)
            # if 'interposed' in target or 'fastigial' in target:
            #     return evaluate_interposed_fastigial_seg(gen_conf, train_conf, test_conf)
        if train_conf['dataset'] in ['tha_seg_dl']:
            return evaluate_tha_seg(gen_conf, train_conf, test_conf)
        if train_conf['dataset'] in ['sar']:
            if gen_conf['dataset_info'][train_conf['dataset']]['trn_dim'][0] in ['axial', 'sagittal', 'coronal']:
                return evaluate_sar_prediction_2_5D(gen_conf, train_conf, test_conf)
            else:
                return evaluate_sar_prediction(gen_conf, train_conf, test_conf)

    elif args.mode is '1': # training only
        if train_conf['dataset'] in ['3T7T', '3T7T_real', '3T7T_total', '3T+7T', '3T7T_multi']:
            return evaluate_training_only(gen_conf, train_conf)
        if train_conf['dataset'] in ['CBCT16', 'CBCT57', 'CT30']:
            return evaluate_training_only(gen_conf, train_conf)
        if train_conf['dataset'] in ['dcn_seg_dl']:
            return evaluate_dentate_interposed_seg(gen_conf, train_conf, test_conf)
            # if 'dentate' in target and not 'interposed' in target:
            #     return evaluate_dentate_seg(gen_conf, train_conf, test_conf)
            # if 'dentate' in target and 'interposed' in target:
            #     return evaluate_dentate_interposed_seg(gen_conf, train_conf, test_conf)
            # if 'interposed' in target or 'fastigial' in target:
            #     return evaluate_interposed_fastigial_seg(gen_conf, train_conf, test_conf)
        if train_conf['dataset'] in ['tha_seg_dl']:
            return evaluate_tha_seg(gen_conf, train_conf, test_conf)
        if train_conf['dataset'] in ['sar']:
            if gen_conf['dataset_info'][train_conf['dataset']]['trn_dim'][0] in ['axial', 'sagittal', 'coronal']:
                return evaluate_sar_prediction_2_5D(gen_conf, train_conf, test_conf)
            else:
                return evaluate_sar_prediction(gen_conf, train_conf, test_conf)

    elif args.mode is '2': # testing only
        if test_conf['dataset'] == 'ADNI':
            return evaluate_testing_only(gen_conf, train_conf, test_conf)
        if test_conf['dataset'] in ['dcn_seg_dl']:
            return evaluate_dentate_interposed_seg(gen_conf, train_conf, test_conf)
            # if 'dentate' in target and not 'interposed' in target:
            #     return evaluate_dentate_seg(gen_conf, train_conf, test_conf)
            # if 'dentate' in target and 'interposed' in target:
            #     return evaluate_dentate_interposed_seg(gen_conf, train_conf, test_conf)
            # if 'interposed' in target or 'fastigial' in target:
            #     return evaluate_interposed_fastigial_seg(gen_conf, train_conf, test_conf)
        if train_conf['dataset'] in ['tha_seg_dl']:
            return evaluate_tha_seg(gen_conf, train_conf, test_conf)
        if train_conf['dataset'] in ['sar']:
            if gen_conf['dataset_info'][train_conf['dataset']]['trn_dim'][0] in ['axial', 'sagittal', 'coronal']:
                return evaluate_sar_prediction_2_5D(gen_conf, train_conf, test_conf)
            else:
                return evaluate_sar_prediction(gen_conf, train_conf, test_conf)
    else:
        assert True, "error: invalid mode"


def evaluate_using_training_testing_split(gen_conf, train_conf):
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    num_volumes = dataset_info['num_volumes']
    dimension = train_conf['dimension']
    patch_shape = train_conf['patch_shape']
    output_shape = train_conf['output_shape']
    preprocess = train_conf['preprocess']
    bg_value = train_conf['bg_value']

    modality = dataset_info['image_modality']
    num_modality = len(modality)

    input_data, labels = read_dataset(gen_conf, train_conf)

    model, _, _, mean, std = train_model(gen_conf, train_conf, input_data[:num_volumes[0]],
                                             labels[:num_volumes[0]], 1)

    testing_set = [1003, 1019, 1038, 1107, 1119, 1004, 1023, 1039, 1110, 1122, 1005,
        1024, 1101, 1113, 1125, 1018, 1025, 1104, 1116, 1128]

    test_indexes = range(num_volumes[0], num_volumes[0] + num_volumes[1])

    for idx, test_index in enumerate(test_indexes):
        if preprocess == 1 or preprocess == 3:
            test_vol = standardize_volume(input_data[test_index], num_modality, mean, std)
        else:
            test_vol = input_data[test_index]

        if patch_shape != output_shape :
            pad_size = ()
            for dim in range(dimension) :
                pad_size += (output_shape[dim], )
            test_vol = pad_both_sides(dimension, test_vol, pad_size, bg_value)

        x_test = build_testing_set(gen_conf, train_conf, test_vol)
        rec_vol = test_model(gen_conf, train_conf, x_test, model)

        save_volume_MICCAI2012(gen_conf, train_conf, rec_vol, testing_set[idx])

        del x_test


def evaluate_tha_seg(gen_conf, train_conf, test_conf):
    args = gen_conf['args']
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    root_path = gen_conf['root_path']
    results_path = gen_conf['results_path']
    mode = gen_conf['validation_mode']
    approach = train_conf['approach']
    loss = train_conf['loss']
    preprocess_trn = train_conf['preprocess']
    num_k_fold = train_conf['num_k_fold']
    num_epochs = train_conf['num_epochs']
    num_retrain_init = train_conf['num_retrain']
    preprocess_tst = test_conf['preprocess']
    is_unseen_case = test_conf['is_unseen_case']
    is_measure = test_conf['is_measure']

    folder_names = dataset_info['folder_names']
    modality = dataset_info['image_modality']
    patient_id = dataset_info['patient_id']
    target = dataset_info['target']

    num_data = len(patient_id)
    num_modality = len(modality)

    file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])
    if not os.path.exists(file_output_dir):
        os.makedirs(file_output_dir)

    if mode is '0': # k-fold cross validation
        kfold = KFold(num_k_fold, True, 1)  # StratifiedKFold, RepeatedKFold
        train_test_lst = kfold.split(range(num_data))
    elif mode is '1': # designated training and test set (if test set is None, training only)
        # patient id
        # train_patient_lst = [['ET018', 'ET028', 'ET030', 'MS001', 'P030', 'P040', 'PD050', 'PD061', 'PD074',
        #                    'PD081', 'PD085', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105',
        #                    'SLEEP106', 'SLEEP107', 'SLEEP108', 'SLEEP109'],
        #                      ['ET018', 'ET028', 'ET030', 'MS001', 'P030', 'P040', 'PD050', 'PD061', 'PD074',
        #                    'PD081', 'PD085', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105',
        #                    'SLEEP106', 'SLEEP107', 'SLEEP108', 'SLEEP109']]
        train_patient_lst = [['ET018', 'ET019', 'ET020', 'ET021', 'ET028', 'ET030', 'MS001', 'P030', 'P040', 'PD050',
                              'PD061', 'PD074', 'PD081', 'PD085', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104',
                              'SLEEP105', 'SLEEP106', 'SLEEP107', 'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP112',
                              'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116', 'SLEEP117', 'SLEEP118', 'SLEEP119',
                              'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127',
                              'SLEEP128', 'SLEEP130', 'SLEEP131', 'SLEEP133'],
                             ['ET018', 'ET019', 'ET020', 'ET021', 'ET028', 'ET030', 'MS001', 'P030', 'P040', 'PD050',
                              'PD061', 'PD074', 'PD081', 'PD085', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104',
                              'SLEEP105', 'SLEEP106', 'SLEEP107', 'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP112',
                              'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116', 'SLEEP117', 'SLEEP118', 'SLEEP119',
                              'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127',
                              'SLEEP128', 'SLEEP130', 'SLEEP131', 'SLEEP133']]
        test_patient_lst = [[None], [None]]
        train_test_lst = zip(train_patient_lst, test_patient_lst)  # set test_patient_lst as None for only training
    elif mode is '2': # test only # combine to mode 1
        # patient id

        # # 1st fold training data list
        # train_patient_lst = [['ET018', 'ET019', 'ET028', 'ET030', 'MS001', 'P030', 'P040', 'PD050', 'PD061', 'PD074',
        #                       'PD081', 'PD085', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP105', 'SLEEP106', 'SLEEP107',
        #                       'SLEEP108', 'SLEEP110', 'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116', 'SLEEP117',
        #                       'SLEEP119', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP126', 'SLEEP127', 'SLEEP128',
        #                       'SLEEP131', 'SLEEP133']]

        # 2nd fold training data list
        train_patient_lst = [['ET018', 'ET019', 'ET020', 'ET021', 'ET030', 'MS001', 'P030', 'P040', 'PD050', 'PD061',
                              'PD074', 'PD081', 'PD085', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105',
                              'SLEEP107', 'SLEEP109', 'SLEEP110', 'SLEEP112', 'SLEEP113', 'SLEEP114', 'SLEEP115',
                              'SLEEP116', 'SLEEP118', 'SLEEP120', 'SLEEP125', 'SLEEP126', 'SLEEP127', 'SLEEP130',
                              'SLEEP131', 'SLEEP133']]

        # # 3rd fold training data list
        # train_patient_lst = [['ET018', 'ET019', 'ET020', 'ET021', 'ET028', 'ET030', 'MS001', 'P030', 'P040', 'PD050',
        #                       'PD074', 'PD081', 'PD085', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP106', 'SLEEP108',
        #                       'SLEEP109', 'SLEEP112', 'SLEEP113', 'SLEEP116', 'SLEEP117', 'SLEEP118', 'SLEEP119',
        #                       'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP128',
        #                       'SLEEP130', 'SLEEP131']]
        #
        # # 4th fold training data list
        # train_patient_lst = [['ET018', 'ET020', 'ET021', 'ET028', 'ET030', 'P040', 'PD050', 'PD061', 'PD074', 'PD081',
        #                       'SLEEP101', 'SLEEP102', 'SLEEP104', 'SLEEP105', 'SLEEP106', 'SLEEP107', 'SLEEP108',
        #                       'SLEEP109', 'SLEEP110', 'SLEEP112', 'SLEEP114', 'SLEEP115', 'SLEEP117', 'SLEEP118',
        #                       'SLEEP119', 'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126',
        #                       'SLEEP127', 'SLEEP128', 'SLEEP130', 'SLEEP133']]
        #
        # # 5th fold training data list
        # train_patient_lst = [['ET019', 'ET020', 'ET021', 'ET028', 'MS001', 'P030', 'PD061', 'PD085', 'SLEEP101',
        #                       'SLEEP103', 'SLEEP104', 'SLEEP105', 'SLEEP106', 'SLEEP107', 'SLEEP108', 'SLEEP109',
        #                       'SLEEP110', 'SLEEP112', 'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116', 'SLEEP117',
        #                       'SLEEP118', 'SLEEP119', 'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125',
        #                       'SLEEP127', 'SLEEP128', 'SLEEP130', 'SLEEP131', 'SLEEP133']]

        trn_set_num = 2
        #test_patient_lst = [['PD091']]
        #test_patient_lst = [['PD088', 'PD090']]
        #test_patient_lst = [['PD061_071319']]
        #test_patient_lst = [['PD061_071519', 'PD086_071519']] # 3rd fold
        #test_patient_lst = [['PD074_071519']] # 5th fold
        #test_patient_lst = [['P044_071919', 'PD057_071919']] # 4th fold
        #test_patient_lst = [['PD070_071919']]  # 2nd fold
        test_patient_lst = [['ET023_080519', 'ET024_080519', 'ET025_080519']]  # 2nd fold

        # test_patient_lst = [['ET018', 'ET028', 'ET030', 'MS001', 'P030', 'P040', 'PD050', 'PD061', 'PD074',
        #                    'PD081', 'PD085', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105',
        #                    'SLEEP106', 'SLEEP107', 'SLEEP108', 'SLEEP109'],
        #                     ['ET018', 'ET028', 'ET030', 'MS001', 'P030', 'P040', 'PD050', 'PD061', 'PD074',
        #                    'PD081', 'PD085', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105',
        #                    'SLEEP106', 'SLEEP107', 'SLEEP108', 'SLEEP109']]
        # test_patient_lst = [['ET018', 'ET019', 'ET020', 'ET021', 'ET028', 'ET030', 'MS001', 'P030', 'P040', 'PD050',
        #                       'PD061', 'PD074', 'PD081', 'PD085', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104',
        #                       'SLEEP105', 'SLEEP106', 'SLEEP107', 'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP112',
        #                       'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116', 'SLEEP117', 'SLEEP118', 'SLEEP119',
        #                       'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127',
        #                       'SLEEP128', 'SLEEP130', 'SLEEP131', 'SLEEP133'],
        #                      ['ET018', 'ET019', 'ET020', 'ET021', 'ET028', 'ET030', 'MS001', 'P030', 'P040', 'PD050',
        #                       'PD061', 'PD074', 'PD081', 'PD085', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104',
        #                       'SLEEP105', 'SLEEP106', 'SLEEP107', 'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP112',
        #                       'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116', 'SLEEP117', 'SLEEP118', 'SLEEP119',
        #                       'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127',
        #                       'SLEEP128', 'SLEEP130', 'SLEEP131', 'SLEEP133']]
        train_test_lst = zip(train_patient_lst, test_patient_lst)
    else:
        raise NotImplementedError('mode ' + mode + 'does not exist')

    k = 0
    for train_idx, test_idx in train_test_lst:
        if mode is '0': # k-fold cross validation
            train_patient_lst = [patient_id[i] for i in train_idx]
            test_patient_lst = [patient_id[i] for i in test_idx]

            if k in []:
                k += 1
                continue

            # set training set and test set to skip
            skip_train_lst = ['']
            for skip_patient_id in skip_train_lst:
                if skip_patient_id in train_patient_lst:
                    train_patient_lst.remove(skip_patient_id)

            skip_test_lst = ['']
            for skip_patient_id in skip_test_lst:
                if skip_patient_id in test_patient_lst:
                    test_patient_lst.remove(skip_patient_id)

            if len(test_patient_lst) == 0:
                k += 1
                continue
        else:

            if k in []:
                k += 1
                continue

            if mode is '2':
                k = trn_set_num - 1

            train_patient_lst = train_idx
            test_patient_lst = test_idx

        # prepare for the files to write
        f = prepare_files_to_write(train_patient_lst, test_patient_lst, file_output_dir, approach, loss, k, mode, args)

        # load training images, labels, and test images on the roi
        if is_unseen_case == 0:
            train_img_lst, label_lst, test_img_lst, train_fname_lst, label_fname_lst, test_fname_lst  = \
                read_tha_dataset(gen_conf, train_conf, train_patient_lst, test_patient_lst, preprocess_trn,
                                     preprocess_tst, file_output_dir)
        else:
            is_scaling = False
            is_reg = True # false: manual roi setting
            roi_pos = [[73,88,36], [161,150,90]] # change this roi pos into a original res. space of PD091 if is_reg is False
            train_img_lst, label_lst, test_img_lst, train_fname_lst, label_fname_lst, test_fname_lst, is_low_res = \
                read_tha_dataset_unseen(gen_conf, train_conf, train_patient_lst, test_patient_lst, preprocess_trn,
                                     preprocess_tst, file_output_dir, is_scaling, is_reg, roi_pos)

        # processing smaller size
        # train_img_lst, label_lst, test_img_lst, train_fname_lst, label_fname_lst, test_fname_lst  = \
        #     read_tha_dataset_v2(gen_conf, train_conf, train_patient_lst, test_patient_lst, preprocess_trn,
        #                          preprocess_tst, file_output_dir)

        # train the model
        if mode is not '2':
            cur_epochs = -1
            metric_best = 0
            metric_thres = 0.024 # 0.976 for acc (# 0.975 for 32 patch size; retrain when it was early stopped), 0.024 for loss
            num_retrain = 0
            lim_epochs = 7

            while cur_epochs <= lim_epochs and num_epochs > cur_epochs and metric_best <= metric_thres:
                # (num_epochs > cur_epochs) and current epoch is less than 8 (originally 6)(by val_acc or val_acc_dc)
                if cur_epochs != -1:
                    num_retrain += 1
                    train_conf['num_retrain'] = num_retrain
                    train_conf['continue_tr'] = 1
                    lim_epochs = 5
                else:
                    train_conf['num_retrain'] = num_retrain_init
                    if train_conf['continue_tr'] == 1:
                        print('continuing training with the trained model...')
                model, cur_epochs, metric_best, mean, std = train_model(gen_conf, train_conf, train_img_lst, label_lst,
                                                                   train_fname_lst,label_fname_lst, k+1)
                print ('current epoch (total # of epochs): ' + str(cur_epochs) + '(' + str(num_epochs) + ')')

            if mode is '1' and test_patient_lst[0] is None: # training only
                k += 1
                continue
        else: # test only
            # read the trained model
            case_name = k + 1 # or a designated case: train_cases = [0,1], case_name = train_cases[k]
            model = read_model(gen_conf, train_conf, case_name) # model filename should have 'mode_2_'
            mean = []
            std = []

        # predict the test set
        for test_img, test_patient_id, test_fname in zip(test_img_lst, test_patient_lst, test_fname_lst):
            print('#' + str(k + 1) + ': processing test_patient_id - ' + test_patient_id)

            # preprocess test image
            test_vol = preprocess_test_image(test_img, train_img_lst, num_modality, mean, std, preprocess_tst)

            # inference from the learned model
            rec_vol_crop, prob_vol_crop, test_patches = inference(gen_conf, train_conf, test_conf, test_vol, model)

            # uncrop and save the segmentation result

            if is_unseen_case == 0:
                save_volume_tha(gen_conf, train_conf, test_conf, rec_vol_crop, prob_vol_crop, test_fname,
                                test_patient_id, target, file_output_dir)
                # compute DC
                if is_measure == 1:
                    _ = measure_thalamus(gen_conf, train_conf, test_conf, test_patient_id, target, k, mode)
            else:
                save_volume_tha_unseen(gen_conf, train_conf, test_conf, rec_vol_crop, prob_vol_crop, test_fname,
                                   test_patient_id, target, file_output_dir, is_low_res)

            # processing smaller size
            # save_volume_tha_v2(gen_conf, train_conf, test_conf, rec_vol_crop, prob_vol_crop, test_fname,
            #                     test_patient_id, target[0], file_output_dir)

            del test_patches

        k += 1
        f.close()

    return True


def evaluate_dentate_seg(gen_conf, train_conf, test_conf):
    args = gen_conf['args']
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    root_path = gen_conf['root_path']
    results_path = gen_conf['results_path']
    mode = gen_conf['validation_mode']
    approach = train_conf['approach']
    loss = train_conf['loss']
    preprocess_trn = train_conf['preprocess']
    num_k_fold = train_conf['num_k_fold']
    num_epochs = train_conf['num_epochs']
    num_retrain_init = train_conf['num_retrain']
    preprocess_tst = test_conf['preprocess']

    folder_names = dataset_info['folder_names']
    modality = dataset_info['image_modality']
    patient_id = dataset_info['patient_id']
    target = dataset_info['target']

    num_data = len(patient_id)
    num_modality = len(modality)

    file_output_dir = root_path + results_path + dataset + '/' + folder_names[0]
    if not os.path.exists(file_output_dir):
        os.makedirs(os.path.join(root_path, results_path, dataset, folder_names[0]))

    if mode is '0': # k-fold cross validation
        kfold = KFold(num_k_fold, True, 1)  # StratifiedKFold, RepeatedKFold
        train_test_lst = kfold.split(range(num_data))
    elif mode is '1': # designated training and test set (if test set is None, training only)
        # patient id
        train_patient_lst = [['SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105', 'SLEEP107', 'SLEEP108',
                           'SLEEP109', 'SLEEP110', 'SLEEP112', 'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116',
                           'SLEEP117', 'SLEEP118', 'SLEEP119', 'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124',
                           'SLEEP125', 'SLEEP126', 'SLEEP127', 'SLEEP128', 'SLEEP130', 'SLEEP131', 'SLEEP133',
                           'SLEEP134', 'SLEEP135', 'SLEEP136', 'SLEEP137', 'SLEEP138', 'SLEEP139', 'SLEEP140',
                           'SLEEP142', 'SLEEP143', 'SLEEP144'],
                             ['SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105', 'SLEEP107', 'SLEEP108',
                           'SLEEP109', 'SLEEP110', 'SLEEP112', 'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116',
                           'SLEEP117', 'SLEEP118', 'SLEEP119', 'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124',
                           'SLEEP125', 'SLEEP126', 'SLEEP127', 'SLEEP128', 'SLEEP130', 'SLEEP131', 'SLEEP133',
                           'SLEEP134', 'SLEEP135', 'SLEEP136', 'SLEEP137', 'SLEEP138', 'SLEEP139', 'SLEEP140',
                           'SLEEP142', 'SLEEP143', 'SLEEP144']]
        test_patient_lst = [[None], ['C052', 'C053', 'C056', 'PD062', 'PD063', 'PD078', 'PD079', 'PD080', 'PD081']]
        train_test_lst = zip(train_patient_lst, test_patient_lst)  # set test_patient_lst as None for only training
    elif mode is '2': # test only # combine to mode 1
        # patient id
        train_patient_lst = [[None], [None]]
        test_patient_lst = [['C052', 'C053', 'C056', 'PD062', 'PD063','PD078', 'PD079', 'PD080', 'PD081'],
                            ['C052', 'C053', 'C056', 'PD062', 'PD063','PD078', 'PD079', 'PD080', 'PD081']]
        train_test_lst = zip(train_patient_lst, test_patient_lst)
    else:
        raise NotImplementedError('mode ' + mode + 'does not exist')

    k = 0
    for train_idx, test_idx in train_test_lst:

        if mode is '0': # k-fold cross validation
            train_patient_lst = [patient_id[i] for i in train_idx]
            test_patient_lst = [patient_id[i] for i in test_idx]

            if k in []:
                k += 1
                continue

            # set training set and test set to skip
            skip_train_lst = [''] #['C052', 'C053', 'C056', 'PD062', 'PD063', 'PD078', 'PD079', 'PD080', 'PD081']
            for skip_patient_id in skip_train_lst:
                if skip_patient_id in train_patient_lst:
                    train_patient_lst.remove(skip_patient_id)

            skip_test_lst = ['']
            for skip_patient_id in skip_test_lst:
                if skip_patient_id in test_patient_lst:
                    test_patient_lst.remove(skip_patient_id)

            if len(test_patient_lst) == 0:
                k += 1
                continue
        else:

            if k in []:
                k += 1
                continue

            train_patient_lst = train_idx
            test_patient_lst = test_idx

        # prepare for the files to write
        f = prepare_files_to_write(train_patient_lst, test_patient_lst, file_output_dir, approach, loss, k, mode, args)

        # load training images, labels, and test images on the roi
        train_img_lst, label_lst, test_img_lst, train_fname_lst, label_fname_lst, test_fname_lst  = \
            read_dentate_dataset(gen_conf, train_conf, train_patient_lst, test_patient_lst, preprocess_trn,
                                 preprocess_tst, file_output_dir)

        # train the model
        if mode is not '2':
            cur_epochs = -1
            metric_best = 0
            metric_thres = 0.024  # 0.991 for 16 patch size # 0.997 for 32 patch size; retrain when it was early stopped # 0.024 for loss
            num_retrain = 0
            lim_epochs = 7

            while cur_epochs <= lim_epochs and num_epochs > cur_epochs and metric_best <= metric_thres:
                # (num_epochs > cur_epochs) and current epoch is less than 8 (originally 6) (by val_acc or val_acc_dc)
                if cur_epochs != -1:
                    num_retrain += 1
                    train_conf['num_retrain'] = num_retrain
                    train_conf['continue_tr'] = 1
                    lim_epochs = 5
                else:
                    train_conf['num_retrain'] = num_retrain_init
                    if train_conf['continue_tr'] == 1:
                        print('continuing training with the trained model...')
                model, cur_epochs, metric_best, mean, std = train_model(gen_conf, train_conf, train_img_lst, label_lst,
                                                                   train_fname_lst,label_fname_lst, k+1)
                print ('current epoch (total # of epochs): ' + str(cur_epochs) + '(' + str(num_epochs) + ')')

            if mode is '1' and test_patient_lst[0] is None: # training only
                k += 1
                continue
        else: # test only
            # read the trained model
            case_name = k + 1 # or a designated case: train_cases = [0,1], case_name = train_cases[k]
            model = read_model(gen_conf, train_conf, case_name) # model filename should have 'mode_2_'
            mean = []
            std = []

        # predict the test set
        for test_img, test_patient_id, test_fname in zip(test_img_lst, test_patient_lst, test_fname_lst):
            print('#' + str(k + 1) + ': processing test_patient_id - ' + test_patient_id)

            # preprocess test image
            test_vol = preprocess_test_image(test_img, train_img_lst, num_modality, mean, std, preprocess_tst)

            # inference from the learned model
            rec_vol_crop, prob_vol_crop, test_patches = inference(gen_conf, train_conf, test_conf, test_vol, model)

            # uncrop and save the segmentation result
            save_volume_dentate(gen_conf, train_conf, test_conf, rec_vol_crop, prob_vol_crop, test_fname,
                                test_patient_id, target, file_output_dir)

            # compute DC
            _ = measure_dentate(gen_conf, train_conf, test_conf, test_patient_id, k, mode)

            del test_patches

        k += 1
        f.close()

    return True


def evaluate_dentate_interposed_seg(gen_conf, train_conf, test_conf):
    args = gen_conf['args']
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    root_path = gen_conf['root_path']
    results_path = gen_conf['results_path']
    mode = gen_conf['validation_mode']
    multi_output = gen_conf['multi_output']
    approach = train_conf['approach']
    loss = train_conf['loss']
    preprocess_trn = train_conf['preprocess']
    num_k_fold = train_conf['num_k_fold']
    num_epochs = train_conf['num_epochs']
    num_retrain_init = train_conf['num_retrain']
    preprocess_tst = test_conf['preprocess']
    is_unseen_case = test_conf['is_unseen_case']
    is_measure = test_conf['is_measure']

    target = dataset_info['target']
    # target = 'dentate'
    # target = 'interposed'

    folder_names = dataset_info['folder_names']
    modality = dataset_info['image_modality']
    patient_id = dataset_info['patient_id']

    is_new_trn_label = train_conf['is_new_trn_label']

    num_data = len(patient_id)
    num_modality = len(modality)

    file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])
    if not os.path.exists(file_output_dir):
        os.makedirs(file_output_dir)

    if mode is '0': # k-fold cross validation
        kfold = KFold(num_k_fold, True, 1)  # StratifiedKFold, RepeatedKFold
        train_test_lst = kfold.split(range(num_data))
    elif mode is '1': # designated training and test set (if test set is None, training only)
        # patient id
        # train_patient_lst = [['PD081', 'SLEEP101', 'SLEEP105', 'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP113',
        #                       'SLEEP114', 'SLEEP115', 'SLEEP117', 'SLEEP118', 'SLEEP130', 'SLEEP131', 'SLEEP133',
        #                       'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP107', 'SLEEP112', 'SLEEP116', 'SLEEP119',
        #                       'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127',
        #                       'SLEEP128']]
        # train_patient_lst = [['SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP107', 'SLEEP112', 'SLEEP116', 'SLEEP119',
        #                       'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127',
        #                       'SLEEP128']]

        # # selected 31 cases out of 42 cases (for pre-training)
        train_patient_lst = [['C052', 'C053', 'C056', 'C057', 'C060', 'P042', 'P043', 'PD055', 'PD060', 'PD061',
                             'PD062', 'PD063', 'PD064', 'PD069', 'PD074', 'PD078', 'PD079', 'PD080', 'SLEEP106',
                             'SLEEP134', 'SLEEP135', 'SLEEP136', 'SLEEP137', 'SLEEP138', 'SLEEP139', 'SLEEP140',
                             'SLEEP141', 'SLEEP142', 'SLEEP143', 'SLEEP144', 'SLEEP145']]

        # selected 31 cases out of 42 cases (for pre-training)
        # test_patient_lst = [['C052', 'C053', 'C056', 'C057', 'C060', 'P042', 'P043', 'PD055', 'PD060', 'PD061',
        #                      'PD062', 'PD063', 'PD064', 'PD069', 'PD074', 'PD078', 'PD079', 'PD080', 'SLEEP106',
        #                      'SLEEP134', 'SLEEP135', 'SLEEP136', 'SLEEP137', 'SLEEP138', 'SLEEP139', 'SLEEP140',
        #                      'SLEEP141', 'SLEEP142', 'SLEEP143', 'SLEEP144', 'SLEEP145']]

        #selected 32 cases out of 42 cases (for pre-training)
        # train_patient_lst = [['C052', 'C053', 'C056', 'C057', 'C058', 'C060', 'P042', 'P043', 'PD055', 'PD060', 'PD061',
        #                      'PD062', 'PD063', 'PD064', 'PD069', 'PD074', 'PD078', 'PD079', 'PD080', 'SLEEP106',
        #                      'SLEEP134', 'SLEEP135', 'SLEEP136', 'SLEEP137', 'SLEEP138', 'SLEEP139', 'SLEEP140',
        #                      'SLEEP141', 'SLEEP142', 'SLEEP143', 'SLEEP144', 'SLEEP145']]

        # original 42 cases
        # test_patient_lst = [['C050', 'C052', 'C053', 'C056', 'C057', 'C058', 'C059', 'C060', 'P030', 'P032', 'P035',
        #                       'P038', 'P040', 'P042', 'P043', 'P045', 'PD053', 'PD055', 'PD060', 'PD061', 'PD062',
        #                       'PD063','PD064', 'PD065', 'PD069', 'PD074', 'PD078', 'PD079', 'PD080', 'SLEEP106',
        #                       'SLEEP134', 'SLEEP135', 'SLEEP136', 'SLEEP137', 'SLEEP138', 'SLEEP139', 'SLEEP140',
        #                       'SLEEP141', 'SLEEP142', 'SLEEP143', 'SLEEP144', 'SLEEP145']]

        # # 29 test cases
        # test_patient_lst = [['PD081', 'SLEEP101', 'SLEEP105', 'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP113',
        #                       'SLEEP114', 'SLEEP115', 'SLEEP117', 'SLEEP118', 'SLEEP130', 'SLEEP131', 'SLEEP133',
        #                       'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP107', 'SLEEP112', 'SLEEP116', 'SLEEP119',
        #                       'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127',
        #                       'SLEEP128']]

        test_patient_lst = [[None]]

        # train_patient_lst = [['PD081', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105', 'SLEEP107',
        #                    'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP112', 'SLEEP113', 'SLEEP114', 'SLEEP115',
        #                    'SLEEP116','SLEEP117', 'SLEEP118', 'SLEEP119', 'SLEEP120', 'SLEEP122', 'SLEEP123',
        #                    'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127', 'SLEEP128', 'SLEEP130', 'SLEEP131',
        #                    'SLEEP133'],
        #                      ['PD081', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105', 'SLEEP107',
        #                    'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP112', 'SLEEP113', 'SLEEP114', 'SLEEP115',
        #                    'SLEEP116','SLEEP117', 'SLEEP118', 'SLEEP119', 'SLEEP120', 'SLEEP122', 'SLEEP123',
        #                    'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127', 'SLEEP128', 'SLEEP130', 'SLEEP131',
        #                    'SLEEP133']]
        # test_patient_lst = [[None], ['C050', 'C052', 'C053', 'C056', 'C057', 'C058', 'C059', 'C060', 'P030', 'P032', 'P035',
        #                    'P038', 'P040', 'P042', 'P043', 'P045', 'PD053', 'PD055', 'PD060', 'PD061', 'PD062', 'PD063',
        #                    'PD064', 'PD065', 'PD069', 'PD074', 'PD078', 'PD079', 'PD080', 'SLEEP106', 'SLEEP134',
        #                    'SLEEP135', 'SLEEP136', 'SLEEP137', 'SLEEP138', 'SLEEP139', 'SLEEP140', 'SLEEP141',
        #                    'SLEEP142', 'SLEEP143', 'SLEEP144', 'SLEEP145']]
        train_test_lst = zip(train_patient_lst, test_patient_lst)  # set test_patient_lst as None for only training
    elif mode is '2': # test only # combine to mode 1
        # patient id
        # train_patient_lst = [['PD081', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105', 'SLEEP107',
        #                    'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP112', 'SLEEP113', 'SLEEP114', 'SLEEP115',
        #                    'SLEEP116','SLEEP117', 'SLEEP118', 'SLEEP119', 'SLEEP120', 'SLEEP122', 'SLEEP123',
        #                    'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127', 'SLEEP128', 'SLEEP130', 'SLEEP131',
        #                    'SLEEP133']]

        # 1st fold training data list (5 folds)
        # train_patient_lst = [['PD081', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105', 'SLEEP107',
        #                       'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP112', 'SLEEP113', 'SLEEP114', 'SLEEP115',
        #                       'SLEEP117', 'SLEEP118', 'SLEEP119', 'SLEEP125', 'SLEEP126', 'SLEEP127', 'SLEEP130',
        #                       'SLEEP131', 'SLEEP133']]

        # 2nd fold training data list (5 folds)
        # train_patient_lst = [['PD081', 'SLEEP101', 'SLEEP105', 'SLEEP107', 'SLEEP108', 'SLEEP109', 'SLEEP110',
        #                       'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116', 'SLEEP117', 'SLEEP118', 'SLEEP119',
        #                       'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP127', 'SLEEP128', 'SLEEP130',
        #                       'SLEEP131', 'SLEEP133']]

        # 3rd fold training data list (5 folds)
        # train_patient_lst = [['PD081', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105', 'SLEEP109',
        #                       'SLEEP110', 'SLEEP112', 'SLEEP113', 'SLEEP114', 'SLEEP116', 'SLEEP117', 'SLEEP118',
        #                       'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP128',
        #                       'SLEEP131', 'SLEEP133']]

        # 4th fold training data list (5 folds)
        # train_patient_lst = [['SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105', 'SLEEP107', 'SLEEP108', 'SLEEP109',
        #                       'SLEEP110', 'SLEEP112', 'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116', 'SLEEP119',
        #                       'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127',
        #                       'SLEEP128', 'SLEEP130']]
        #
        # 5th fold training data list (5 folds)
        train_patient_lst = [['PD081', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP107', 'SLEEP108',
                              'SLEEP112', 'SLEEP115', 'SLEEP116', 'SLEEP117', 'SLEEP118', 'SLEEP119', 'SLEEP120',
                              'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127', 'SLEEP128',
                              'SLEEP130', 'SLEEP131', 'SLEEP133']]

        # 1st fold training data list (2 folds)
        # train_patient_lst = [['PD081', 'SLEEP101', 'SLEEP105', 'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP113',
        #                       'SLEEP114', 'SLEEP115', 'SLEEP117', 'SLEEP118', 'SLEEP130', 'SLEEP131', 'SLEEP133']]

        # 2nd fold training data list (2 folds)
        # train_patient_lst = [['SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP107', 'SLEEP112', 'SLEEP116', 'SLEEP119',
        #                       'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127',
        #                       'SLEEP128']]
        # train_patient_lst = [['PD081', 'SLEEP101', 'SLEEP105', 'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP113',
        #                       'SLEEP114', 'SLEEP115', 'SLEEP117', 'SLEEP118', 'SLEEP130', 'SLEEP131', 'SLEEP133',
        #                       'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP107', 'SLEEP112', 'SLEEP116', 'SLEEP119',
        #                       'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127',
        #                       'SLEEP128']]

        trn_set_num = 5

        # test_patient_lst = [['C050', 'C052', 'C053', 'C056', 'C057', 'C058', 'C059', 'C060', 'P030', 'P032', 'P035',
        #                       'P038', 'P040', 'P042', 'P043', 'P045', 'PD053', 'PD055', 'PD060', 'PD061', 'PD062',
        #                       'PD063','PD064', 'PD069', 'PD074', 'PD078', 'PD079', 'PD080', 'SLEEP106',
        #                       'SLEEP134', 'SLEEP135', 'SLEEP136', 'SLEEP137', 'SLEEP138', 'SLEEP139', 'SLEEP140',
        #                       'SLEEP141', 'SLEEP142', 'SLEEP143', 'SLEEP144', 'SLEEP145']]
        # test_patient_lst = [['C050', 'C052', 'C053', 'C056', 'C057', 'C058', 'C059', 'C060', 'P030', 'P032', 'P035',
        #                       'P038', 'P040', 'P042', 'P043', 'P045', 'PD053', 'PD055', 'PD060', 'PD061', 'PD062',
        #                       'PD063','PD064', 'PD065', 'PD069', 'PD074', 'PD078', 'PD079', 'PD080', 'SLEEP106',
        #                       'SLEEP134', 'SLEEP135', 'SLEEP136', 'SLEEP137', 'SLEEP138', 'SLEEP139', 'SLEEP140',
        #                       'SLEEP141', 'SLEEP142', 'SLEEP143', 'SLEEP144', 'SLEEP145']]
        #test_patient_lst = [['ET019', 'ET020', 'PD081_3T']]
        #test_patient_lst = [['PD081_3T']]
        #test_patient_lst = [['PD061_071319']]
        #test_patient_lst = [['PD061_071519', 'PD074_071519', 'PD086_071519']]
        #test_patient_lst = [['PD074_071519', 'PD086_071519']]
        #test_patient_lst = [['PD086_071919']]
        #test_patient_lst = [['P044_071919', 'PD057_071919', 'PD070_071919']] # 5th
        #test_patient_lst = [['ET030_080219']]  # 5th
        #test_patient_lst = [['ET023_080519', 'ET024_080519', 'ET025_080519']]   # 5th
        #test_patient_lst = [['ET024_080519']]  # 5th (manual roi setting)
        test_patient_lst = [['ET025_080519']]   # 5th
        train_test_lst = zip(train_patient_lst, test_patient_lst)
    else:
        raise NotImplementedError('mode ' + mode + 'does not exist')

    k = 0
    for train_idx, test_idx in train_test_lst:

        if mode is '0': # k-fold cross validation

            if is_new_trn_label == 3:
                new_trn_label = ['C052', 'C053', 'C056', 'C057', 'C060', 'P042', 'P043', 'PD055', 'PD060', 'PD061',
                                 'PD062', 'PD063', 'PD064', 'PD069', 'PD074', 'PD078', 'PD079', 'PD080', 'SLEEP106',
                                 'SLEEP134', 'SLEEP135', 'SLEEP136', 'SLEEP137', 'SLEEP138', 'SLEEP139', 'SLEEP140',
                                 'SLEEP141', 'SLEEP142', 'SLEEP143', 'SLEEP144', 'SLEEP145']
                train_patient_lst = [patient_id[i] for i in train_idx] + new_trn_label
            else:
                train_patient_lst = [patient_id[i] for i in train_idx]
            test_patient_lst = [patient_id[i] for i in test_idx]

            if k in []:
                k += 1
                continue

            # set training set and test set to skip
            skip_train_lst = [''] #['C052', 'C053', 'C056', 'PD062', 'PD063', 'PD078', 'PD079', 'PD080', 'PD081']
            for skip_patient_id in skip_train_lst:
                if skip_patient_id in train_patient_lst:
                    train_patient_lst.remove(skip_patient_id)

            skip_test_lst = ['']
            for skip_patient_id in skip_test_lst:
                if skip_patient_id in test_patient_lst:
                    test_patient_lst.remove(skip_patient_id)

            if len(test_patient_lst) == 0:
                k += 1
                continue
        else:

            if k in []:
                k += 1
                continue

            if mode is '2':
                k = trn_set_num - 1

            train_patient_lst = train_idx
            test_patient_lst = test_idx

        # prepare for the files to write
        f = prepare_files_to_write_dcn(train_patient_lst, test_patient_lst, file_output_dir, approach, loss, k, mode,
                                       args, target, multi_output)

        # load training images, labels, and test images on the roi
        if is_unseen_case == 0:
            train_img_lst, label_lst, test_img_lst, train_fname_lst, label_fname_lst, test_fname_lst  = \
                read_dentate_interposed_dataset(gen_conf, train_conf, train_patient_lst, test_patient_lst, preprocess_trn,
                                     preprocess_tst, file_output_dir, target)
        else:
            is_scaling = False
            is_reg = True # false: manual roi setting
            roi_pos = [[55, 30, 3], [85, 50, 20]] # for ET024
            #[[120, 100, 50], [200, 145, 80]] for PD086
            # [[50, 25, 3], [85, 45, 16]] for C058, [[70, 50, 3], [120, 80, 10]] for PD081_3T # [[50, 30, 1], [85, 50, 20]] for ET019, [[50, 30, 1], [85, 50, 10]] for ET020 if is_reg is False
            # [[70, 50, 3], [120, 80, 10]] for PD081_3T (in a original res. space)
            train_img_lst, label_lst, test_img_lst, train_fname_lst, label_fname_lst, test_fname_lst, is_res_diff  = \
                read_dentate_interposed_dataset_unseen(gen_conf, train_conf, train_patient_lst, test_patient_lst,
                                                       preprocess_trn, preprocess_tst, file_output_dir, target,
                                                       is_scaling, is_reg, roi_pos)

        # train the model
        if mode is not '2':
            cur_epochs = -1
            metric_best = 0
            metric_thres = 0.997  # 0.992 for 16 patch size # 0.997 for 32 patch size; retrain when it was early stopped # 0.024 for loss
            num_retrain = 0
            lim_epochs = 7

            while cur_epochs <= lim_epochs and num_epochs > cur_epochs and metric_best <= metric_thres:
                # (num_epochs > cur_epochs) and current epoch is less than 8 (originally 6) (by val_acc or val_acc_dc)
                if cur_epochs != -1:
                    num_retrain += 1
                    train_conf['num_retrain'] = num_retrain
                    train_conf['continue_tr'] = 1
                    lim_epochs = 5
                else:
                    train_conf['num_retrain'] = num_retrain_init
                    if train_conf['continue_tr'] == 1:
                        print('continuing training with the trained model...')
                model, cur_epochs, metric_best, mean, std = train_model(gen_conf, train_conf, train_img_lst, label_lst,
                                                                   train_fname_lst, label_fname_lst, k+1)
                print('current epoch (total # of epochs): ' + str(cur_epochs) + '(' + str(num_epochs) + ')')

            if mode is '1' and test_patient_lst[0] is None: # training only
                k += 1
                continue
        else: # test only
            # read the trained model
            case_name = k + 1 # or a designated case: train_cases = [0,1], case_name = train_cases[k]
            model = read_model(gen_conf, train_conf, case_name) # model filename should have 'mode_2_'
            mean = []
            std = []

        # predict the test set
        for test_img, test_patient_id, test_fname in zip(test_img_lst, test_patient_lst, test_fname_lst):
            print('#' + str(k + 1) + ': processing test_patient_id - ' + test_patient_id)

            # preprocess test image
            test_vol = preprocess_test_image(test_img, train_img_lst, num_modality, mean, std, preprocess_tst)

            # inference from the learned model
            rec_vol_crop, prob_vol_crop, test_patches = inference(gen_conf, train_conf, test_conf, test_vol, model)

            # uncrop and save the segmentation result
            if is_unseen_case == 0:
                save_volume_dentate_interposed(gen_conf, train_conf, test_conf, rec_vol_crop, prob_vol_crop, test_fname,
                                    test_patient_id, file_output_dir, target)
                # compute DC
                if is_measure == 1:
                    _ = measure_dentate_interposed(gen_conf, train_conf, test_conf, test_patient_id, k, mode, target)

            else:
                save_volume_dentate_interposed_unseen(gen_conf, train_conf, test_conf, rec_vol_crop, prob_vol_crop, test_fname,
                                    test_patient_id, file_output_dir, target, is_res_diff)

            del test_patches

        k += 1
        f.close()

    return True


def evaluate_interposed_fastigial_seg(gen_conf, train_conf, test_conf):
    args = gen_conf['args']
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    root_path = gen_conf['root_path']
    results_path = gen_conf['results_path']
    mode = gen_conf['validation_mode']
    approach = train_conf['approach']
    loss = train_conf['loss']
    preprocess_trn = train_conf['preprocess']
    num_k_fold = train_conf['num_k_fold']
    num_epochs = train_conf['num_epochs']
    num_retrain_init = train_conf['num_retrain']
    preprocess_tst = test_conf['preprocess']

    folder_names = dataset_info['folder_names']
    modality = dataset_info['image_modality']
    patient_id = dataset_info['patient_id']
    target = dataset_info['target']

    num_data = len(patient_id)
    num_modality = len(modality)

    file_output_dir = root_path + results_path + dataset + '/' + folder_names[0]
    if not os.path.exists(file_output_dir):
        os.makedirs(os.path.join(root_path, results_path, dataset, folder_names[0]))

    if mode is '0': # k-fold cross validation
        kfold = KFold(num_k_fold, True, 1)  # StratifiedKFold, RepeatedKFold
        train_test_lst = kfold.split(range(num_data))
    elif mode is '1': # designated training and test set (if test set is None, training only)
        # patient id
        train_patient_lst = [patient_id]
        test_patient_lst = [patient_id]
        train_test_lst = zip(train_patient_lst, test_patient_lst)  # set test_patient_lst as None for only training
    elif mode is '2': # test only # combine to mode 1
        # patient id
        train_patient_lst = [[None], [None]]
        test_patient_lst = [patient_id, patient_id]
        train_test_lst = zip(train_patient_lst, test_patient_lst)
    else:
        raise NotImplementedError('mode ' + mode + 'does not exist')

    k = 0
    for train_idx, test_idx in train_test_lst:

        if mode is '0': # k-fold cross validation
            train_patient_lst = [patient_id[i] for i in train_idx]
            test_patient_lst = [patient_id[i] for i in test_idx]

            if k in []:
                k += 1
                continue

            # set training set and test set to skip
            skip_train_lst = [''] #['C052', 'C053', 'C056', 'PD062', 'PD063', 'PD078', 'PD079', 'PD080', 'PD081']
            for skip_patient_id in skip_train_lst:
                if skip_patient_id in train_patient_lst:
                    train_patient_lst.remove(skip_patient_id)

            skip_test_lst = ['']
            for skip_patient_id in skip_test_lst:
                if skip_patient_id in test_patient_lst:
                    test_patient_lst.remove(skip_patient_id)

            if len(test_patient_lst) == 0:
                k += 1
                continue
        else:

            if k in []:
                k += 1
                continue

            train_patient_lst = train_idx
            test_patient_lst = test_idx

        # prepare for the files to write
        f = prepare_files_to_write(train_patient_lst, test_patient_lst, file_output_dir, approach, loss, k, mode, args)

        # load training images, labels, and test images on the roi
        train_img_suit_dentate_lst, dentate_label_crop_lst, test_img_suit_interposed_fastigial_lst, train_fname_lst, \
        dentate_label_fname_lst, test_fname_lst = read_interposed_fastigial_dataset_rev(gen_conf, train_conf,
                                                                                    train_patient_lst, test_patient_lst,
                                                                                    preprocess_trn, preprocess_tst,
                                                                                    file_output_dir)
        # train_img_suit_dentate_lst, suit_interposed_fasigial_lst, test_img_dentate_label_lst, train_fname_lst, \
        # dentate_label_fname_lst, test_fname_lst = read_interposed_fastigial_dataset(gen_conf, train_conf,
        #                                                                             train_patient_lst, test_patient_lst,
        #                                                                             preprocess_trn, preprocess_tst,
        #                                                                             file_output_dir)

        # train the model
        if mode is not '2':
            cur_epochs = -1
            metric_best = 0
            metric_thres = 0.997 # retrain when it was early stopped
            num_retrain = 0
            lim_epochs = 7

            while cur_epochs <= lim_epochs and num_epochs > cur_epochs and metric_best <= metric_thres:
                # (num_epochs > cur_epochs) and current epoch is less than 8 (originally 6) (by val_acc or val_acc_dc)
                if cur_epochs != -1:
                    num_retrain += 1
                    train_conf['num_retrain'] = num_retrain
                    train_conf['continue_tr'] = 1
                    lim_epochs = 5
                else:
                    train_conf['num_retrain'] = num_retrain_init
                    if train_conf['continue_tr'] == 1:
                        print('continuing training with the trained model...')
                model, cur_epochs, metric_best, mean, std = train_model(gen_conf, train_conf, train_img_suit_dentate_lst,
                                                                   dentate_label_crop_lst, train_fname_lst,
                                                                   dentate_label_fname_lst, k+1)
                print ('current epoch (total # of epochs): ' + str(cur_epochs) + '(' + str(num_epochs) + ')')

            if mode is '1' and test_patient_lst[0] is None: # training only
                k += 1
                continue
        else: # test only
            # read the trained model
            case_name = k + 1 # or a designated case: train_cases = [0,1], case_name = train_cases[k]
            model = read_model(gen_conf, train_conf, case_name) # model filename should have 'mode_2_'
            mean = []
            std = []

        # predict the test set
        #label_mapper = {0: 0, 1: 10, 2: 150}
        for test_img_suit_interposed_fastigial_mask, test_patient_id, test_fname in \
                zip(test_img_suit_interposed_fastigial_lst, test_patient_lst, test_fname_lst):
            print('#' + str(k + 1) + ': processing test_patient_id - ' + test_patient_id)

            # preprocess test image

            test_suit_interposed_fastigial_mask_vol = preprocess_test_image(test_img_suit_interposed_fastigial_mask,
                                                           train_img_suit_dentate_lst, num_modality, mean,
                                                           std, preprocess_tst)
            # inference from the learned model
            interposed_fastigial_rec_vol_crop, interposed_fastigial_prob_vol_crop, test_interposed_fastigial_patches = \
                inference(gen_conf, train_conf, test_conf, test_suit_interposed_fastigial_mask_vol, model)

            # for key in label_mapper.keys():
            #     interposed_fastigial_rec_vol_crop[interposed_fastigial_rec_vol_crop == key] = label_mapper[key]
            #
            # interposed_rec_vol_crop = np.zeros(interposed_fastigial_rec_vol_crop.shape)
            # fastigial_rec_vol_crop = np.zeros(interposed_fastigial_rec_vol_crop.shape)
            #
            # interposed_rec_vol_crop[interposed_fastigial_rec_vol_crop == 10] = 1
            # fastigial_rec_vol_crop[interposed_fastigial_rec_vol_crop == 150] = 1

            # uncrop and save the segmentation result
            #if 'interposed' in target:
            save_volume_interposed_fastigial(gen_conf, train_conf, test_conf, interposed_fastigial_rec_vol_crop,
                                interposed_fastigial_prob_vol_crop, test_fname, test_patient_id, 'interposed+fastigial',
                                file_output_dir)

            # if 'fastigial' in target:
            #     save_volume_dcn(gen_conf, train_conf, test_conf, interposed_fastigial_rec_vol_crop,
            #                     interposed_fastigial_prob_vol_crop, test_fname, test_patient_id, 'fastigial',
            #                     file_output_dir)

            del test_interposed_fastigial_patches

            # compute DC
            #_ = measure_dentate(gen_conf, train_conf, test_conf, test_patient_id, k, mode)

        k += 1
        f.close()

    return True


def evaluate_sar_prediction(gen_conf, train_conf, test_conf):
    args = gen_conf['args']
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    root_path = gen_conf['root_path']
    results_path = gen_conf['results_path']
    mode = gen_conf['validation_mode']
    approach = train_conf['approach']
    loss = train_conf['loss']
    num_k_fold = train_conf['num_k_fold']
    preprocess_tst = test_conf['preprocess']
    is_measure = test_conf['is_measure']
    folder_names = dataset_info['folder_names']
    modality = dataset_info['image_modality']
    case_id = dataset_info['case_id']
    augment_sar_data = dataset_info['augment_sar_data']
    data_augment = augment_sar_data[0]
    num_gen = augment_sar_data[1]
    num_epochs = train_conf['num_epochs']
    trn_dim = dataset_info['trn_dim']

    num_data = len(case_id)
    num_modality = len(modality)

    file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])
    if not os.path.exists(file_output_dir):
        os.makedirs(file_output_dir)

    if mode is '0': # k-fold cross validation
        kfold = KFold(num_k_fold, True, 1)  # StratifiedKFold, RepeatedKFold
        train_test_id_lst = kfold.split(range(num_data))
    elif mode is '1': # designated training and test set (if test set is None, training only)
        # patient id
        train_id_lst = [[]]
        test_id_lst = [[None]]
        train_test_id_lst = zip(train_id_lst, test_id_lst)  # set test_patient_lst as None for only training
    elif mode is '2': # test only # combine to mode 1
        # patient id
        train_id_lst = [[]]
        test_id_lst = [[]]
        trn_set_num = 5
        train_test_id_lst = zip(train_id_lst, test_id_lst)
    else:
        raise NotImplementedError('mode ' + mode + 'does not exist')

    k = 0
    for train_idx, test_idx in train_test_id_lst:

        if mode is '0': # k-fold cross validation

            train_id_lst = [case_id[i] for i in train_idx]
            test_id_lst = [case_id[i] for i in test_idx]

            if k in []:
                k += 1
                continue

            # set training set and test set to skip
            skip_train_lst = [''] #['C052', 'C053', 'C056', 'PD062', 'PD063', 'PD078', 'PD079', 'PD080', 'PD081']
            for skip_patient_id in skip_train_lst:
                if skip_patient_id in train_id_lst:
                    train_id_lst.remove(skip_patient_id)

            skip_test_lst = ['']
            for skip_patient_id in skip_test_lst:
                if skip_patient_id in test_id_lst:
                    test_id_lst.remove(skip_patient_id)

            if len(test_id_lst) == 0:
                k += 1
                continue
        else:

            if k in []:
                k += 1
                continue

            if mode is '2':
                k = trn_set_num - 1

            train_id_lst = train_idx
            test_id_lst = test_idx

        # prepare for the files to write
        f = prepare_files_to_write_sar(train_id_lst, test_id_lst, file_output_dir, approach, loss, k, mode, args)

        # load training images, labels, and test images on the roi
        train_src_data, train_sar_data, test_src_data, test_sar_lst = read_sar_dataset(gen_conf, train_conf,
                                                                                       train_id_lst, test_id_lst)

        # train the model
        if mode is not '2':
            model, cur_epochs, metric_best, mean, std = train_sar_model(gen_conf, train_conf, train_src_data, train_sar_data, k+1)

            if mode is '1' and test_id_lst[0] is None: # training only
                k += 1
                continue

        else: # test only
            # read the trained model
            case_name = k + 1 # or a designated case: train_cases = [0,1], case_name = train_cases[k]
            model, _, _ = load_gan_model(gen_conf, train_conf, case_name)

        # predict the training set
        if data_augment == 1 and num_epochs > 0:
            num_augmented_data = 4 + num_gen * 16
            train_src_data_temp, train_sar_data_temp = train_src_data, train_sar_data
            train_src_data, train_sar_data = [], []
            for i in range(len(train_id_lst)):
                train_src_data.append(train_src_data_temp[num_augmented_data * i])
                train_sar_data.append(train_sar_data_temp[num_augmented_data * i])

        for train_src, train_sar, train_id in zip(train_src_data, train_sar_data, train_id_lst):
            print('#' + str(k + 1) + ': processing train_patient_id - ' + train_id)

            # preprocess test image
            train_vol = preprocess_test_image(train_src, train_src_data, num_modality, mean, std, preprocess_tst)

            # inference from the learned model
            rec_vol, _, _, _ = inference_sar(gen_conf, train_conf, test_conf, train_vol, model)

            # uncrop and save the segmentation result
            #save_sar_volume(gen_conf, train_conf, test_vol, test_sar, rec_vol, prob_vol_crop, ovr_vol_crop, test_id, file_output_dir)
            # compute DC
            if is_measure == 1:
                measure_sar_prediction(gen_conf, train_conf, train_sar, str(train_id) + ' (trn)', rec_vol, k+1)

        # predict the test set
        for test_src, test_sar, test_id in zip(test_src_data, test_sar_lst, test_id_lst):
            print('#' + str(k + 1) + ': processing test_patient_id - ' + test_id)

            # preprocess test image
            test_vol = preprocess_test_image(test_src, train_src_data, num_modality, mean, std, preprocess_tst)

            # inference from the learned model
            rec_vol, prob_vol_crop, ovr_vol_crop, _ = inference_sar(gen_conf, train_conf, test_conf, test_vol, model)

            # uncrop and save the segmentation result
            save_sar_volume(gen_conf, train_conf, test_vol, test_sar, rec_vol, prob_vol_crop, ovr_vol_crop, test_id, file_output_dir)
            # compute DC
            if is_measure == 1:
                measure_sar_prediction(gen_conf, train_conf, test_sar, str(test_id) + ' (tst)', rec_vol, k+1)

        k += 1
        f.close()

    return True


def evaluate_sar_prediction_2_5D(gen_conf, train_conf, test_conf):
    args = gen_conf['args']
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    root_path = gen_conf['root_path']
    results_path = gen_conf['results_path']
    mode = gen_conf['validation_mode']
    approach = train_conf['approach']
    loss = train_conf['loss']
    num_k_fold = train_conf['num_k_fold']
    preprocess_tst = test_conf['preprocess']
    is_measure = test_conf['is_measure']
    folder_names = dataset_info['folder_names']
    modality = dataset_info['image_modality']
    case_id = dataset_info['case_id']
    num_epochs = train_conf['num_epochs']
    ensemble_weight = dataset_info['ensemble_weight']
    num_classes = train_conf['GAN']['generator']['num_classes']

    trn_dim = dataset_info['trn_dim']
    dim_num = []
    axis_num = []
    for d in trn_dim:
        if d == 'axial':
            dim_num.append(0)
            axis_num.append(2)
        elif d == 'sagittal':
            dim_num.append(1)
            axis_num.append(1)
        elif d == 'coronal':
            dim_num.append(2)
            axis_num.append(0)

    augment_sar_data = dataset_info['augment_sar_data']
    data_augment = augment_sar_data[0]
    num_gen = augment_sar_data[1]

    g_patch_shape = train_conf['GAN']['generator']['patch_shape']
    g_output_shape = train_conf['GAN']['generator']['output_shape']
    d_patch_shape = train_conf['GAN']['discriminator']['patch_shape']
    d_output_shape = train_conf['GAN']['discriminator']['output_shape']

    num_data = len(case_id)
    num_modality = len(modality)

    file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])
    if not os.path.exists(file_output_dir):
        os.makedirs(file_output_dir)

    if mode is '0': # k-fold cross validation
        kfold = KFold(num_k_fold, True, 1)  # StratifiedKFold, RepeatedKFold
        train_test_id_lst = kfold.split(range(num_data))
    elif mode is '1': # designated training and test set (if test set is None, training only)
        # patient id
        train_id_lst = [[]]
        test_id_lst = [[None]]
        train_test_id_lst = zip(train_id_lst, test_id_lst)  # set test_patient_lst as None for only training
    elif mode is '2': # test only # combine to mode 1
        # patient id
        train_id_lst = [[]]
        test_id_lst = [[]]
        trn_set_num = 5
        train_test_id_lst = zip(train_id_lst, test_id_lst)
    else:
        raise NotImplementedError('mode ' + mode + 'does not exist')

    k = 0
    for train_idx, test_idx in train_test_id_lst:

        if mode is '0': # k-fold cross validation

            train_id_lst = [case_id[i] for i in train_idx]
            test_id_lst = [case_id[i] for i in test_idx]

            if k in []:
                k += 1
                continue

            # set training set and test set to skip
            skip_train_lst = [''] #['C052', 'C053', 'C056', 'PD062', 'PD063', 'PD078', 'PD079', 'PD080', 'PD081']
            for skip_patient_id in skip_train_lst:
                if skip_patient_id in train_id_lst:
                    train_id_lst.remove(skip_patient_id)

            skip_test_lst = ['']
            for skip_patient_id in skip_test_lst:
                if skip_patient_id in test_id_lst:
                    test_id_lst.remove(skip_patient_id)

            if len(test_id_lst) == 0:
                k += 1
                continue
        else:

            if k in []:
                k += 1
                continue

            if mode is '2':
                k = trn_set_num - 1

            train_id_lst = train_idx
            test_id_lst = test_idx

        # prepare for the files to write
        f = prepare_files_to_write_sar(train_id_lst, test_id_lst, file_output_dir, approach, loss, k, mode, args)

        # load training images, labels, and test images on the roi
        train_src_data, train_sar_data, test_src_data, test_sar_data = read_sar_dataset(gen_conf, train_conf,
                                                                                       train_id_lst, test_id_lst)

        g_patch_shape = train_sar_data[0][0, 0].shape
        g_output_shape = train_sar_data[0][0, 0].shape
        d_patch_shape = train_sar_data[0][0, 0].shape
        d_output_shape = np.divide(train_sar_data[0][0, 0].shape, 8).astype(int)

        # train the model
        if mode is not '2':
            model, cur_epochs, metric_best, mean, std = train_sar_model_2_5D(gen_conf, train_conf, train_src_data,
                                                                             train_sar_data, g_patch_shape,
                                                                             g_output_shape, d_patch_shape,
                                                                             d_output_shape, dim_num, trn_dim, k+1)

            if mode is '1' and test_id_lst[0] is None: # training only
                k += 1
                continue

        else: # test only
            # read the trained model
            case_name = k + 1 # or a designated case: train_cases = [0,1], case_name = train_cases[k]
            model = []
            for dim_label in trn_dim:
                if dim_label == 'axial':
                    g_patch_shape_2d = (g_patch_shape[0], g_patch_shape[1])
                    g_output_shape_2d = (g_output_shape[0], g_output_shape[1])
                    d_patch_shape_2d = (d_patch_shape[0], d_patch_shape[1])
                    d_output_shape_2d = (d_output_shape[0], d_output_shape[1])
                elif dim_label == 'sagittal':
                    g_patch_shape_2d = (g_patch_shape[0], g_patch_shape[2])
                    g_output_shape_2d = (g_output_shape[0], g_output_shape[2])
                    d_patch_shape_2d = (d_patch_shape[0], d_patch_shape[2])
                    d_output_shape_2d = (d_output_shape[0], d_output_shape[2])
                elif dim_label == 'coronal':
                    g_patch_shape_2d = (g_patch_shape[1], g_patch_shape[2])
                    g_output_shape_2d = (g_output_shape[1], g_output_shape[2])
                    d_patch_shape_2d = (d_patch_shape[1], d_patch_shape[2])
                    d_output_shape_2d = (d_output_shape[1], d_output_shape[2])

                train_conf['GAN']['generator']['patch_shape'] = g_patch_shape_2d
                train_conf['GAN']['generator']['output_shape'] = g_output_shape_2d
                train_conf['GAN']['discriminator']['patch_shape'] = d_patch_shape_2d
                train_conf['GAN']['discriminator']['output_shape'] = d_output_shape_2d

                generator, _, _ = load_gan_model(gen_conf, train_conf, str(case_name) + '_' + dim_label)
                model.append(generator)

        # print(len(model))
        # print(model)

        test_conf['GAN']['generator']['output_shape'] = g_output_shape
        test_conf['GAN']['generator']['num_classes'] = num_classes

        # predict the training set
        if data_augment == 1 and num_epochs > 0:
            num_augmented_data = 4 + num_gen * 16
            train_src_data_temp, train_sar_data_temp = train_src_data, train_sar_data
            train_src_data, train_sar_data = [], []
            for i in range(len(train_id_lst)):
                train_src_data.append(train_src_data_temp[num_augmented_data*i])
                train_sar_data.append(train_sar_data_temp[num_augmented_data*i])

        if num_epochs > 0:
            for train_src, train_sar, train_id in zip(train_src_data, train_sar_data, train_id_lst):
                print('#' + str(k + 1) + ': processing train_patient_id - ' + train_id)

                # preprocess test image
                train_vol = preprocess_test_image(train_src, train_src_data, num_modality, mean, std, preprocess_tst)

                # inference from the learned model
                ensemble_vol, vol_out_total = inference_sar_2_5D(gen_conf, test_conf, train_vol, model,
                                                                 ensemble_weight, dim_num, axis_num, trn_dim)

                # uncrop and save the segmentation result
                save_sar_volume_2_5D(gen_conf, train_conf, train_vol, train_sar, ensemble_vol, vol_out_total, train_id,
                                     trn_dim, file_output_dir)
                # compute DC
                if is_measure == 1:
                    measure_sar_prediction(gen_conf, train_conf, train_sar, str(train_id) + ' (trn)', ensemble_vol, k+1)

        # predict the test set
        for test_src, test_sar, test_id in zip(test_src_data, test_sar_data, test_id_lst):
            print('#' + str(k + 1) + ': processing test_patient_id - ' + test_id)

            # preprocess test image
            test_vol = preprocess_test_image(test_src, train_src_data, num_modality, mean, std, preprocess_tst)

            # inference from the learned model
            ensemble_vol, vol_out_total = inference_sar_2_5D(gen_conf, test_conf, test_vol, model,
                                                             ensemble_weight, dim_num, axis_num, trn_dim)

            # uncrop and save the segmentation result
            save_sar_volume_2_5D(gen_conf, train_conf, test_vol, test_sar, ensemble_vol, vol_out_total, test_id, trn_dim,
                                 file_output_dir)
            # compute DC
            if is_measure == 1:
                measure_sar_prediction(gen_conf, train_conf, test_sar, str(test_id) + ' (tst)', ensemble_vol, k+1)

        k += 1
        f.close()

    return True


def prepare_files_to_write(train_patient_lst, test_patient_lst, file_output_dir, approach, loss, k, mode, args):

    measure_pkl_filepath = file_output_dir + '/' + 'mode_' + mode + '_#' + str(k + 1) + '_' + 'measurement_' + \
                               approach + '_' + loss + '.pkl'

    if os.path.exists(measure_pkl_filepath):
        os.remove(measure_pkl_filepath)

    measure_pkl_filepath_staple = file_output_dir + '/' + 'mode_' + mode + '_#' + str(k + 1) + '_' + 'measurement_' + \
                                  'staple' + '_' + loss + '.pkl'
    if os.path.exists(measure_pkl_filepath_staple):
        os.remove(measure_pkl_filepath_staple)

    k_fold_mode_filepath = file_output_dir + '/' + 'mode_' + mode + '_#'+ str(k + 1) + '_' + \
                           'training_test_patient_id_' + approach + '_' + loss + '.txt'
    if os.path.exists(k_fold_mode_filepath):
        os.remove(k_fold_mode_filepath)

    k_fold_patient_list = '#' + str(k + 1) + '\ntrain sets: %s \ntest sets: %s \nparameter sets: %s' \
                          % (train_patient_lst, test_patient_lst, args)

    f = ioutils.create_log(k_fold_mode_filepath, k_fold_patient_list, is_debug=1)

    failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
    if os.path.exists(failed_cases_filepath):
        os.remove(failed_cases_filepath)

    return f



def prepare_files_to_write_dcn(train_patient_lst, test_patient_lst, file_output_dir, approach, loss, k, mode, args,
                               target, multi_output):

    if multi_output == 1:
        loss=loss[0]

    for seg_label in target:
        measure_pkl_filepath = file_output_dir + '/' + 'mode_' + mode + '_#' + str(k + 1) + '_' + 'measurement_' + \
                               approach + '_' + loss + '_' + seg_label + '_seg.pkl'

        if os.path.exists(measure_pkl_filepath):
            os.remove(measure_pkl_filepath)

    k_fold_mode_filepath = file_output_dir + '/' + 'mode_' + mode + '_#'+ str(k + 1) + '_' + \
                           'training_test_patient_id_' + approach + '_' + loss + '.txt'
    if os.path.exists(k_fold_mode_filepath):
        os.remove(k_fold_mode_filepath)

    k_fold_patient_list = '#' + str(k + 1) + '\ntrain sets: %s \ntest sets: %s \nparameter sets: %s' \
                          % (train_patient_lst, test_patient_lst, args)

    f = ioutils.create_log(k_fold_mode_filepath, k_fold_patient_list, is_debug=1)

    failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
    if os.path.exists(failed_cases_filepath):
        os.remove(failed_cases_filepath)

    return f


def prepare_files_to_write_sar(train_patient_lst, test_patient_lst, file_output_dir, approach, loss, k, mode, args):

    measure_pkl_filename = 'mode_' + mode + '_#' + str(k + 1) + '_measurement.pkl'
    measure_pkl_filepath = os.path.join(file_output_dir, measure_pkl_filename)

    if os.path.exists(measure_pkl_filepath):
        os.remove(measure_pkl_filepath)

    k_fold_mode_filename = 'mode_' + mode + '_#' + str(k + 1) + '_training_test_patient_id_' + approach + '.txt'
    k_fold_mode_filepath = os.path.join(file_output_dir, k_fold_mode_filename)
    if os.path.exists(k_fold_mode_filepath):
        os.remove(k_fold_mode_filepath)

    k_fold_patient_list = '#' + str(k + 1) + '\ntrain sets: %s \ntest sets: %s \nparameter sets: %s' \
                          % (train_patient_lst, test_patient_lst, args)

    f = ioutils.create_log(k_fold_mode_filepath, k_fold_patient_list, is_debug=1)

    return f


def evaluate_using_loo(gen_conf, train_conf, test_conf):
    root_path = gen_conf['root_path']
    dataset_path = root_path + gen_conf['dataset_path']
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    num_volumes = dataset_info['num_volumes']
    dimension = train_conf['dimension']
    patch_shape = train_conf['patch_shape']
    output_shape = train_conf['output_shape']
    preprocess_trn = train_conf['preprocess']
    preprocess_tst = test_conf['preprocess']
    bg_value = train_conf['bg_value']

    modality = dataset_info['image_modality']
    num_modality = len(modality)

    input_data, labels, data_filename_ext_list, label_filename_ext_list = read_dataset(gen_conf, train_conf)
    # input_data, labels, data_filename_ext_list, label_filename_ext_list = read_3T7T_dataset(dataset_path, dataset_info,
    #                                                                                         preprocess_trn)


    loo = LeaveOneOut()
    for train_index, test_index in loo.split(range(num_volumes)):
        #if test_index in [0,1,2,3,4,5,6,7,9,10,11,12,13,14]:
         #if test_index in [0]:
        if test_index in [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
            continue

        print(train_index, test_index)

        train_img_list = [input_data[i] for i in train_index]
        train_label_list = [labels[i] for i in train_index]
        test_img = input_data[test_index[0]]

        model, _, _, mean, std = train_model(
              gen_conf, train_conf, train_img_list, train_label_list, data_filename_ext_list[train_index],
            label_filename_ext_list[train_index], test_index[0] + 1)

        if preprocess_tst == 1 or preprocess_tst == 3:
            if mean == [] or std == []:
                mean, std = compute_statistics(train_img_list, num_modality)
            test_vol = standardize_volume(test_img, num_modality, mean, std)
        elif preprocess_tst == 4:
            ref_num = 0
            if mean == [] or std == []:
                mean, std = compute_statistics(train_img_list, num_modality)
            ref_training_vol= input_data[train_index[ref_num]]
            test_vol = normalize_image(test_img, [np.min(ref_training_vol.flatten()), np.max(ref_training_vol.flatten())])
            #test_vol2 = test_vol
            #test_vol2 = hist_match(input_data[test_index[0]], ref_training_vol)
            #test_vol = standardize_volume(test_vol, num_modality, mean, std)
            #test_vol = standardize_volume(input_data[test_index[0]], num_modality, mean, std)
            save_intermediate_volume(gen_conf, train_conf, test_vol, test_index[0] + 1, [],
                                     'test_data_hist_matched_standardized')
        elif preprocess_tst == 5:
            ref_num = 0
            ref_training_vol= input_data[train_index[ref_num]]
            print(np.shape(test_img))
            print(np.shape(ref_training_vol))
            test_img = normalize_image(test_img, [np.min(ref_training_vol.flatten()),
                                                  np.max(ref_training_vol.flatten())])
            test_vol = hist_match(test_img, ref_training_vol)
            print(np.shape(test_vol))
            save_intermediate_volume(gen_conf, train_conf, test_vol, test_index[0] + 1, [],
                                     'test_data_normalized_hist_matched')

        else:
            test_vol = test_img

        if patch_shape != output_shape :
            pad_size = ()
            for dim in range(dimension) :
                pad_size += (output_shape[dim], )
            test_vol[0] = pad_both_sides(dimension, test_vol[0], pad_size, bg_value)

        x_test = build_testing_set(gen_conf, train_conf, test_vol)

        test_vol_size = test_vol[0].shape
        gen_conf['dataset_info'][dataset]['size'] = test_vol_size[1:4] # output image size

        rec_vol = test_model(gen_conf, train_conf, x_test, model)

        save_dataset(gen_conf, train_conf, test_conf, rec_vol, data_filename_ext_list[test_index[0]], test_index[0] + 1)
        #save_volume(gen_conf, train_conf, rec_vol, test_index[0]+1)
        #save_volume_3T7T(gen_conf, train_conf, rec_vol, test_index[0]+1)

        # compute_measures
        if num_modality < 2:
            DC = measure(gen_conf, train_conf, test_conf, test_index[0]+1)
        else:
            DC = measure_4d(gen_conf, train_conf, test_conf, test_index[0] + 1)
        print(DC)

        del x_test

    return True


def evaluate_training_only(gen_conf, train_conf):
    root_path = gen_conf['root_path']
    dataset_path = root_path + gen_conf['dataset_path']
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    num_volumes = dataset_info['num_volumes']
    preprocess = train_conf['preprocess']
    modality = dataset_info['image_modality']
    num_modality = len(modality)

    input_data, labels, data_filename_ext_list, label_filename_ext_list  = read_dataset(gen_conf, train_conf)
    # input_data, labels, data_filename_ext_list, label_filename_ext_list = read_3T7T_dataset(dataset_path, dataset_info,
    #                                                                                         preprocess)

    train_index = range(num_volumes)
    print(train_index)

    train_img_list = [input_data[i] for i in train_index]
    train_label_list = [labels[i] for i in train_index]

    train_model(gen_conf, train_conf, train_img_list, train_label_list, data_filename_ext_list[train_index],
                label_filename_ext_list[train_index], 0)

    return True


def evaluate_testing_only(gen_conf, train_conf, test_conf):
    root_path = gen_conf['root_path']
    dataset_path = root_path + gen_conf['dataset_path']

    trainset = train_conf['dataset']
    trainset_info = gen_conf['dataset_info'][trainset]

    testset = test_conf['dataset']
    testset_info = gen_conf['dataset_info'][testset]
    dimension = test_conf['dimension']
    patch_shape = test_conf['patch_shape']
    output_shape = test_conf['output_shape']
    preprocess = test_conf['preprocess']
    modality = trainset_info['image_modality']
    num_modality = len(modality)
    bg_value = train_conf['bg_value']

    # train_input_set, _ = read_dataset(gen_conf, train_conf)
    # test_input_data, filename_ext_list = read_dataset(gen_conf, test_conf)

    train_input_set, _ = read_ADNI_dataset(dataset_path, trainset_info, preprocess)
    test_input_data, filename_ext_list = read_ADNI_dataset(dataset_path, testset_info, preprocess)
    num_volumes = np.size(filename_ext_list)

    model = read_model(gen_conf, train_conf, 0)

    for test_index in range(num_volumes):

        # input_test_data_list = []
        # for modal_idx in range(num_modality):
        #     input_test_data_list += [test_input_data[modal_idx][test_index]]

        if preprocess == 1 or preprocess == 3:
            mean, std = compute_statistics(train_input_set, num_modality)
            test_vol = standardize_volume(test_input_data[test_index], num_modality, mean, std)
        elif preprocess == 4:
            ref_num = 0
            mean, std = compute_statistics(train_input_set, num_modality)
            ref_training_vol= train_input_set[0][ref_num]
            print(np.min(ref_training_vol.flatten()))
            print(np.max(ref_training_vol.flatten()))

            test_vol = normalize_image(test_input_data[test_index],
                                       [np.min(ref_training_vol.flatten()), np.max(ref_training_vol.flatten())])

            #test_vol = hist_match(test_input_data[test_index], ref_training_vol)
            #test_vol = standardize_volume(test_vol, num_modality, mean, std)
            save_intermediate_volume(gen_conf, test_conf, test_vol, test_index + 1, filename_ext_list[test_index],
                                     'test_data_hist_matched_standardized')
        elif preprocess == 5:
            ref_num = 0
            ref_training_vol= train_input_set[0][ref_num]
            print(np.min(ref_training_vol.flatten()))
            print(np.max(ref_training_vol.flatten()))

            test_input_data[test_index] = normalize_image(test_input_data[test_index], [np.min(ref_training_vol.flatten()),
                                                                                    np.max(ref_training_vol.flatten())])

            test_vol = hist_match(test_input_data[test_index], ref_training_vol)
            save_intermediate_volume(gen_conf, test_conf, test_vol, test_index + 1, filename_ext_list[test_index],
                                     'test_data_normalized_hist_matched')

        else:
            test_vol = test_input_data[test_index]

        if patch_shape != output_shape :
            pad_size = ()
            for dim in range(dimension) :
                pad_size += (output_shape[dim], )
            test_vol[0] = pad_both_sides(dimension, test_vol[0], pad_size, bg_value)

        x_test = build_testing_set(gen_conf, test_conf, test_vol)

        dataset = test_conf['dataset']
        test_vol_size = test_vol[0].shape
        gen_conf['dataset_info'][dataset]['size'] = test_vol_size[1:4] # output image size

        rec_vol = test_model(gen_conf, test_conf, x_test, model)
        save_volume_ADNI(gen_conf, train_conf, test_conf, rec_vol, filename_ext_list[test_index])

        # compute_measures
        # measure(gen_conf, test_conf, test_index+1)

        del x_test

    return True


def train_model(
    gen_conf, train_conf, input_data, labels, data_filename_ext_list, label_filename_ext_list, case_name):
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    time_series = dataset_info['time_series']
    folder_names = dataset_info['folder_names']
    preprocess = train_conf['preprocess']
    use_saved_patches = train_conf['use_saved_patches']
    num_epochs = train_conf['num_epochs']
    modality = dataset_info['image_modality']
    approach = train_conf['approach']
    num_modality = len(modality)

    mean = []
    std = []
    cur_epochs = 0
    metric_best = 0
    model = None
    if num_epochs > 0:

        train_index, val_index = split_train_val(range(len(input_data)), train_conf['validation_split'])

        if preprocess == 1 or preprocess == 3:
            mean, std = compute_statistics(input_data, num_modality)
            input_data = standardize_set(input_data, num_modality, mean, std)
        elif preprocess == 4:
            ref_num = 0
            mean, std = compute_statistics(input_data, num_modality)
            ref_training_vol = input_data[ref_num]
            input_data = hist_match_set(input_data, ref_training_vol, num_modality)
            input_data = standardize_set(input_data, num_modality, mean, std)
            for idx in range(len(input_data)):
                save_intermediate_volume(gen_conf, train_conf, input_data[idx], idx + 1, [],
                                         'train_data_hist_matched_standardized')
        elif preprocess == 5:
            ref_num = 0
            ref_training_vol = input_data[ref_num]
            input_data[0] = hist_match_set(input_data, ref_training_vol, num_modality)
            for idx in range(len(input_data)):
                save_intermediate_volume(gen_conf, train_conf, input_data[idx], idx + 1, [],
                                         'train_data_hist_matched')

        train_img_list = [input_data[i] for i in train_index]
        train_label_list = [labels[i] for i in train_index]
        val_img_list = [input_data[i] for i in val_index]
        val_label_list = [labels[i] for i in val_index]

        if use_saved_patches is True:
            train_data = read_patches(gen_conf, train_conf, case_name)
        else:
            train_data = []

        if train_data == []:
            print('Building training samples (patches)...')
            if time_series is False:
                x_train, y_train = build_training_set(
                    gen_conf, train_conf, train_img_list, train_label_list)
                x_val, y_val = build_training_set(
                    gen_conf, train_conf, val_img_list, val_label_list)
            else: # if data is time series
                x_train, y_train = build_training_set_4d(
                    gen_conf, train_conf, train_img_list, train_label_list)
                x_val, y_val = build_training_set_4d(
                    gen_conf, train_conf, val_img_list, val_label_list)

            train_data = [x_train, y_train, x_val, y_val]
            if use_saved_patches is True:
                save_patches(gen_conf, train_conf, train_data, case_name)
                print('Saved training samples (patches)')
        else:
            x_train = train_data[0]
            y_train = train_data[1]
            x_val = train_data[2]
            y_val = train_data[3]
            print('Loaded training samples (patches')

        if approach == 'cgan':
            cur_epochs, metric_best = __train_cgan_model(gen_conf, train_conf, x_train, y_train, x_val, y_val, case_name)
        else:
            callbacks = generate_callbacks(gen_conf, train_conf, case_name)
            cur_epochs, metric_best = __train_model(gen_conf, train_conf, x_train, y_train, x_val, y_val, callbacks)

    if approach == 'cgan':
        generator, _, _ = load_gan_model(gen_conf, train_conf, case_name)
        model = generator
    else:
        model = read_model(gen_conf, train_conf, case_name)

    return model, cur_epochs, metric_best, mean, std


def train_sar_model(gen_conf, train_conf, train_src_data, train_sar_data, case_name):
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    use_saved_patches = train_conf['use_saved_patches']
    preprocess = train_conf['preprocess']
    num_epochs = train_conf['num_epochs']
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    g_patch_shape = train_conf['GAN']['generator']['patch_shape']
    d_patch_shape = train_conf['GAN']['discriminator']['patch_shape']

    mean = []
    std = []
    cur_epochs = 0
    metric_best = 0
    if num_epochs > 0:

        train_idx, val_idx = split_train_val(range(len(train_src_data)), train_conf['validation_split'])

        if preprocess == 1 or preprocess == 3:
            mean, std = compute_statistics(train_src_data, num_modality)
            train_src_data = standardize_set(train_src_data, num_modality, mean, std)
        elif preprocess == 4:
            ref_num = 0
            mean, std = compute_statistics(train_src_data, num_modality)
            ref_training_vol = train_src_data[ref_num]
            train_src_data = hist_match_set(train_src_data, ref_training_vol, num_modality)
            train_src_data = standardize_set(train_src_data, num_modality, mean, std)
            for idx in range(len(train_src_data)):
                save_intermediate_volume(gen_conf, train_conf, train_src_data[idx], idx + 1, [],
                                         'train_data_hist_matched_standardized')
        elif preprocess == 5:
            ref_num = 0
            ref_training_vol = train_src_data[ref_num]
            train_src_data[0] = hist_match_set(train_src_data, ref_training_vol, num_modality)
            for idx in range(len(train_src_data)):
                save_intermediate_volume(gen_conf, train_conf, train_src_data[idx], idx + 1, [],
                                         'train_data_hist_matched')

        train_src_lst = [train_src_data[i] for i in train_idx]
        train_sar_lst = [train_sar_data[i] for i in train_idx]
        val_src_lst = [train_src_data[i] for i in val_idx]
        val_sar_lst = [train_sar_data[i] for i in val_idx]

        if use_saved_patches is True:
            train_data = read_patches(gen_conf, train_conf, case_name)
            print('Loaded training samples (patches)')
        else:
            train_data = []

        if train_data == []:
            print('Building training samples (patches)...')
            x_train_g, y_train_g = build_training_set_sar(gen_conf, train_conf, train_src_lst, train_sar_lst, 'g')
            x_val_g, y_val_g = build_training_set_sar(gen_conf, train_conf, val_src_lst, val_sar_lst, 'g')

            #x_train_d, y_train_d, x_val_d, y_val_d = None, None, None, None
            if g_patch_shape != d_patch_shape:
                x_train_d, y_train_d = build_training_set_sar(gen_conf, train_conf, train_src_lst, train_sar_lst, 'd')
                x_val_d, y_val_d = build_training_set_sar(gen_conf, train_conf, val_src_lst, val_sar_lst, 'd')
                train_data = [x_train_g, y_train_g, x_val_g, y_val_g, x_train_d, y_train_d, x_val_d, y_val_d]
            else:
                train_data = [x_train_g, y_train_g, x_val_g, y_val_g]

            if use_saved_patches is True:
                save_patches(gen_conf, train_conf, train_data, case_name)
                print('Saved training samples (patches)')

        cur_epochs, metric_best = __train_sar_pred_model(gen_conf, train_conf, train_data, case_name)

    generator, _, _ = load_gan_model(gen_conf, train_conf, case_name)
    model = generator

    return model, cur_epochs, metric_best, mean, std



def train_sar_model_2_5D(gen_conf, train_conf, train_src_data, train_sar_data, g_patch_shape, g_output_shape,
                         d_patch_shape, d_output_shape, dim_num, trn_dim, case_name):
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    use_saved_patches = train_conf['use_saved_patches']
    preprocess = train_conf['preprocess']
    num_epochs = train_conf['num_epochs']
    modality = dataset_info['image_modality']
    num_modality = len(modality)

    mean = []
    std = []
    cur_epochs = 0
    metric_best = 0
    if num_epochs > 0:

        train_idx, val_idx = split_train_val(range(len(train_src_data)), train_conf['validation_split'])

        if preprocess == 1 or preprocess == 3:
            mean, std = compute_statistics(train_src_data, num_modality)
            train_src_data = standardize_set(train_src_data, num_modality, mean, std)
        elif preprocess == 4:
            ref_num = 0
            mean, std = compute_statistics(train_src_data, num_modality)
            ref_training_vol = train_src_data[ref_num]
            train_src_data = hist_match_set(train_src_data, ref_training_vol, num_modality)
            train_src_data = standardize_set(train_src_data, num_modality, mean, std)
            for idx in range(len(train_src_data)):
                save_intermediate_volume(gen_conf, train_conf, train_src_data[idx], idx + 1, [],
                                         'train_data_hist_matched_standardized')
        elif preprocess == 5:
            ref_num = 0
            ref_training_vol = train_src_data[ref_num]
            train_src_data[0] = hist_match_set(train_src_data, ref_training_vol, num_modality)
            for idx in range(len(train_src_data)):
                save_intermediate_volume(gen_conf, train_conf, train_src_data[idx], idx + 1, [],
                                         'train_data_hist_matched')

        train_src_lst = [train_src_data[i] for i in train_idx]
        train_sar_lst = [train_sar_data[i] for i in train_idx]
        val_src_lst = [train_src_data[i] for i in val_idx]
        val_sar_lst = [train_sar_data[i] for i in val_idx]

        if use_saved_patches is True:
            train_data = read_patches(gen_conf, train_conf, case_name)
            print('Loaded training samples (patches)')
        else:
            train_data = []

        if train_data == []:
            print('Building training samples (patches)...')

            train_data_total_g = build_training_set_sar_2_5D_MIMO(gen_conf, train_conf, train_src_lst, train_sar_lst,
                                                             g_patch_shape, g_output_shape)
            val_data_total_g = build_training_set_sar_2_5D_MIMO(gen_conf, train_conf, val_src_lst, val_sar_lst,
                                                           g_patch_shape, g_output_shape)

            if g_patch_shape != d_patch_shape:
                train_data_total_d = build_training_set_sar_2_5D_MIMO(gen_conf, train_conf, train_src_lst, train_sar_lst,
                                                                 d_patch_shape, d_output_shape)
                val_data_total_d = build_training_set_sar_2_5D_MIMO(gen_conf, train_conf, val_src_lst, val_sar_lst,
                                                               d_patch_shape, d_output_shape)
                for i in range(3):
                    train_data.append([train_data_total_g[i][0], train_data_total_g[i][1],
                                       val_data_total_g[i][0], val_data_total_g[i][1],
                                       train_data_total_d[i][0], train_data_total_d[i][1],
                                       val_data_total_d[i][0], val_data_total_d[i][1]])
            else:
                for i in range(3):
                    train_data.append([train_data_total_g[i][0], train_data_total_g[i][1],
                                       val_data_total_g[i][0], val_data_total_g[i][1]])

            if use_saved_patches is True:
                save_patches(gen_conf, train_conf, train_data, case_name)
                print('Saved training samples (patches)')

        for i, dim_label in zip(dim_num, trn_dim):
            cur_epochs, metric_best = __train_sar_pred_model_2_5D(gen_conf, train_conf, g_patch_shape, g_output_shape,
                                                                  d_patch_shape, d_output_shape, train_data[i],
                                                                  case_name, dim_label)

    model = []
    for dim_label in trn_dim:
        print(dim_label)
        if dim_label == 'axial':
            g_patch_shape_2d = (g_patch_shape[0], g_patch_shape[1])
            g_output_shape_2d = (g_output_shape[0], g_output_shape[1])
            d_patch_shape_2d = (d_patch_shape[0], d_patch_shape[1])
            d_output_shape_2d = (d_output_shape[0], d_output_shape[1])
        elif dim_label == 'sagittal':
            g_patch_shape_2d = (g_patch_shape[0], g_patch_shape[2])
            g_output_shape_2d = (g_output_shape[0], g_output_shape[2])
            d_patch_shape_2d = (d_patch_shape[0], d_patch_shape[2])
            d_output_shape_2d = (d_output_shape[0], d_output_shape[2])
        elif dim_label == 'coronal':
            g_patch_shape_2d = (g_patch_shape[1], g_patch_shape[2])
            g_output_shape_2d = (g_output_shape[1], g_output_shape[2])
            d_patch_shape_2d = (d_patch_shape[1], d_patch_shape[2])
            d_output_shape_2d = (d_output_shape[1], d_output_shape[2])

        train_conf['GAN']['generator']['patch_shape'] = g_patch_shape_2d
        train_conf['GAN']['generator']['output_shape'] = g_output_shape_2d
        train_conf['GAN']['discriminator']['patch_shape'] = d_patch_shape_2d
        train_conf['GAN']['discriminator']['output_shape'] = d_output_shape_2d

        generator, _, _ = load_gan_model(gen_conf, train_conf, str(case_name) + '_' + dim_label)
        model.append(generator)

    return model, cur_epochs, metric_best, mean, std



def __train_model(gen_conf, train_conf, x_train, y_train, x_val, y_val, callbacks):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    num_classes = gen_conf['num_classes']
    multi_output = gen_conf['multi_output']
    output_name = gen_conf['output_name']
    data_augment = train_conf['data_augment']
    is_continue = train_conf['continue_tr']
    is_shuffle = train_conf['shuffle']
    metric_opt = train_conf['metric']
    batch_size = train_conf['batch_size'] # default
    importance_spl = train_conf['importance_sampling']
    is_oversampling = train_conf['oversampling']
    attention_loss = train_conf['attention_loss']
    overlap_penalty_loss = train_conf['overlap_penalty_loss']
    exclusive_train = train_conf['exclusive_train']
    exclude_label_num = dataset_info['exclude_label_num']
    target = dataset_info['target']

    if exclusive_train == 1 and multi_output == 0:
        y_train = np.delete(y_train, exclude_label_num[0], 2)
        y_val = np.delete(y_val, exclude_label_num[0], 2)
    elif exclusive_train == 0 and multi_output == 1:
        # fill interposed labels and remove interposed labels for output 1
        y_train_1 = np.delete(y_train, exclude_label_num[0], 2)
        y_val_1 = np.delete(y_val, exclude_label_num[0], 2)

        if exclude_label_num[0] == 2:
            y_train_bg_interposed_one_hot = y_train[:, :, 0] + y_train[:, :, 2]
            y_train_bg_interposed_one_hot[y_train_bg_interposed_one_hot == np.max(y_train_bg_interposed_one_hot)] = 1
            y_train_1[:, :, 0] = y_train_bg_interposed_one_hot

            y_val_bg_interposed_one_hot = y_val[:, :, 0] + y_val[:, :, 2]
            y_val_bg_interposed_one_hot[y_val_bg_interposed_one_hot == np.max(y_val_bg_interposed_one_hot)] = 1
            y_val_1[:, :, 0] = y_val_bg_interposed_one_hot

        # fill interposed labels and remove interposed labels for output 2
        y_train_2 = np.delete(y_train, exclude_label_num[1], 2)
        y_val_2 = np.delete(y_val, exclude_label_num[1], 2)

        if exclude_label_num[1] == 1:
            y_train_bg_dentate_one_hot = y_train[:, :, 0] + y_train[:, :, 1]
            y_train_bg_dentate_one_hot[y_train_bg_dentate_one_hot == np.max(y_train_bg_dentate_one_hot)] = 1
            y_train_2[:, :, 0] = y_train_bg_dentate_one_hot

            y_val_bg_dentate_one_hot = y_val[:, :, 0] + y_val[:, :, 1]
            y_val_bg_dentate_one_hot[y_val_bg_dentate_one_hot == np.max(y_val_bg_dentate_one_hot)] = 1
            y_val_2[:, :, 0] = y_val_bg_dentate_one_hot

        if attention_loss == 1:
            # only labels: dentate + interposed
            y_train_3 = np.delete(y_train, 2, 2)
            y_val_3 = np.delete(y_val, 2, 2)

            y_train_dentate_interposed_one_hot = y_train[:, :, 1] + y_train[:, :, 2]
            y_train_dentate_interposed_one_hot[y_train_dentate_interposed_one_hot ==
                                               np.max(y_train_dentate_interposed_one_hot)] = 1
            y_train_3[:, :, 1] = y_train_dentate_interposed_one_hot

            y_val_dentate_interposed_one_hot = y_val[:, :, 1] + y_val[:, :, 2]
            y_val_dentate_interposed_one_hot[y_val_dentate_interposed_one_hot ==
                                               np.max(y_val_dentate_interposed_one_hot)] = 1
            y_val_3[:, :, 1] = y_val_dentate_interposed_one_hot

            if overlap_penalty_loss == 1:
                y_train_bg_one = y_train[:, :, 0]
                y_train_bg_one[y_train_bg_one == 0] = 1

                y_val_bg_one = y_val[:, :, 0]
                y_val_bg_one[y_val_bg_one == 0] = 1

                print (y_train_bg_one.shape)
                print(y_val_bg_one.shape)

                y_train = {
                    output_name[0]: y_train_1,
                    output_name[1]: y_train_2,
                    'attention_maps': y_train_3,
                    'overlap_dentate_interposed': y_train_bg_one
                }
                y_val = {
                    output_name[0]: y_val_1,
                    output_name[1]: y_val_2,
                    'attention_maps': y_val_3,
                    'overlap_dentate_interposed': y_val_bg_one
                }
            else:
                y_train = {
                    output_name[0]: y_train_1,
                    output_name[1]: y_train_2,
                    'attention_maps': y_train_3
                }
                y_val = {
                    output_name[0]: y_val_1,
                    output_name[1]: y_val_2,
                    'attention_maps': y_val_3
                }
        else:
            if overlap_penalty_loss == 1:
                y_train_bg_one = y_train[:, :, 0]
                y_train_bg_one[y_train_bg_one == 0] = 1

                y_val_bg_one = y_val[:, :, 0]
                y_val_bg_one[y_val_bg_one == 0] = 1

                print (y_train_bg_one.shape)
                print(y_val_bg_one.shape)

                y_train = {
                    output_name[0]: y_train_1,
                    output_name[1]: y_train_2,
                    'overlap_dentate_interposed': y_train_bg_one
                }
                y_val = {
                    output_name[0]: y_val_1,
                    output_name[1]: y_val_2,
                    'overlap_dentate_interposed': y_val_bg_one
                }
            else:
                y_train = {
                    output_name[0]: y_train_1,
                    output_name[1]: y_train_2
                }
                y_val = {
                    output_name[0]: y_val_1,
                    output_name[1]: y_val_2
                }
    elif exclusive_train == 1 and multi_output == 1:
        print('In multi_output option, exclusive_train should be off')
        exit()
    else:
        if attention_loss == 1:
            if target == 'both':
                # only labels: dentate + interposed
                y_train_3 = np.delete(y_train, 2, 2)
                y_val_3 = np.delete(y_val, 2, 2)

                y_train_dentate_interposed_one_hot = y_train[:, :, 1] + y_train[:, :, 2]
                y_train_dentate_interposed_one_hot[y_train_dentate_interposed_one_hot ==
                                                   np.max(y_train_dentate_interposed_one_hot)] = 1
                y_train_3[:, :, 1] = y_train_dentate_interposed_one_hot

                y_val_dentate_interposed_one_hot = y_val[:, :, 1] + y_val[:, :, 2]
                y_val_dentate_interposed_one_hot[y_val_dentate_interposed_one_hot ==
                                                 np.max(y_val_dentate_interposed_one_hot)] = 1
                y_val_3[:, :, 1] = y_val_dentate_interposed_one_hot

                y_train = {
                    output_name: y_train,
                    'attention_maps': y_train_3
                }
                y_val = {
                    output_name: y_val,
                    'attention_maps': y_val_3
                }
            else:
                y_train = {
                    output_name: y_train,
                    'attention_maps': y_train
                }
                y_val = {
                    output_name: y_val,
                    'attention_maps': y_val
                }

    print('Generating a model to be trained...')
    model = generate_model(gen_conf, train_conf)

    if is_continue == 1:
        model_filename = load_saved_model_filename(gen_conf, train_conf, 'pre_trained')
        if os.path.isfile(model_filename):
            #model = load_model(model_filename)
            print('Loading saved weights from a trained model (%s)' % model_filename)
            # If `by_name` is True, weights are loaded into layers only if they share the same name.
            # This is useful for fine-tuning or transfer-learning models where some of the layers have changed.
            #model.load_weights(model_filename, by_name=True)
            model.load_weights(model_filename)
        else:
            print('No Found a trained model in the path (%s). Newly starting training...' % model_filename)

    # computation to informative/important samples (by sampling mini-batches from a distribution other than uniform)
    # thus accelerating the convergence
    if importance_spl == 1:
        model = ImportanceTraining(model)

    if data_augment == 1: # mixup
        training_generator = MixupGenerator(x_train, y_train, batch_size, alpha=0.2)()
        model_fit = model.fit_generator(generator=training_generator,
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=train_conf['num_epochs'],
                            validation_data=(x_val, y_val),
                            verbose=train_conf['verbose'],
                            callbacks=callbacks, shuffle=is_shuffle)

    elif data_augment == 2: #datagen in keras
        datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            channel_shift_range=0.1,
            horizontal_flip=True)

        model_fit = model.fit_generator(datagen.flow(x_train, y_train, batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=train_conf['num_epochs'],
                            validation_data=(x_val, y_val),
                            verbose=train_conf['verbose'],
                            callbacks=callbacks, shuffle=is_shuffle)

    elif data_augment == 3: #mixup + datagen
        datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            channel_shift_range=0.1,
            horizontal_flip=True)

        training_generator = MixupGenerator(x_train, y_train, batch_size, alpha=0.2, datagen=datagen)()
        model_fit = model.fit_generator(generator=training_generator,
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=train_conf['num_epochs'],
                            validation_data=(x_val, y_val),
                            verbose=train_conf['verbose'],
                            callbacks=callbacks, shuffle=is_shuffle)

    else: # no data augmentation
        if is_oversampling == 1:
            from imblearn.keras import BalancedBatchGenerator
            from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE
            patch_shape = train_conf['patch_shape']
            #num_classes = gen_conf['num_classes']
            num_modality = len(gen_conf['dataset_info'][train_conf['dataset']]['image_modality'])

            print(x_train.shape)
            print(y_train.shape)

            sm = SMOTE()
            x_train_sampled, y_train_sampled = [], []
            x_train = x_train.reshape(x_train.shape[0], np.prod(patch_shape), num_modality)
            for i in range(x_train.shape[0]):
                x_train_tr, y_train_tr = sm.fit_resample(x_train[i], y_train[i])
                x_train_sampled.append(x_train_tr)
                y_train_sampled.append(y_train_tr)
                print(x_train_tr.shape)
                print(y_train_tr.shape)

            model_fit = model.fit(
                x_train_sampled, y_train_sampled, batch_size=batch_size,
                epochs=train_conf['num_epochs'],
                validation_data=(x_val, y_val),
                verbose=train_conf['verbose'],
                callbacks=callbacks, shuffle=is_shuffle)

            # x_train = x_train.reshape(x_train.shape[0]*np.prod(patch_shape), num_modality)
            # y_train = y_train.reshape(y_train.shape[0]*np.prod(patch_shape), num_classes)
            # print(x_train.shape)
            # print(y_train.shape)
            #
            # training_generator = BalancedBatchGenerator(x_train, y_train, sampler=SVMSMOTE(),
            #                                             batch_size=batch_size*np.prod(patch_shape))
            # model_fit = model.fit_generator(generator=training_generator,
            #                                 steps_per_epoch=x_train.shape[0] // batch_size,
            #                                 epochs=train_conf['num_epochs'],
            #                                 validation_data=(x_val, y_val),
            #                                 verbose=train_conf['verbose'],
            #                                 callbacks=callbacks, shuffle=is_shuffle)
        else:
            model_fit = model.fit(
                x_train, y_train, batch_size=batch_size,
                epochs=train_conf['num_epochs'],
                validation_data=(x_val, y_val),
                verbose=train_conf['verbose'],
                callbacks=callbacks, shuffle=is_shuffle)

    cur_epochs = len(model_fit.history['loss'])
    metric_best = None
    if multi_output == 1:
        if metric_opt in ['acc', 'acc_dc', 'loss']:
            metric_monitor = 'val_' + output_name[0] + '_' + metric_opt
        elif metric_opt == 'loss_total':
            metric_monitor = 'val_loss'
        else:
            print('unknown metric for early stopping')
            metric_monitor = None
        metric = model_fit.history[metric_monitor]
        if metric_opt in ['loss', 'loss_total']:
            metric_best = np.min(metric)
        else:
            metric_best = np.max(metric)
    else:
        if attention_loss == 1:
            if metric_opt in ['acc', 'acc_dc', 'loss']:
                metric_monitor = 'val_' + output_name + '_' + metric_opt
            elif metric_opt == 'loss_total':
                metric_monitor = 'val_loss'
            else:
                print('unknown metric for early stopping')
                metric_monitor = None
            metric = model_fit.history[metric_monitor]
            if metric_opt == ['loss', 'loss_total']:
                metric_best = np.min(metric)
            else:
                metric_best = np.max(metric)
        else:
            metric = model_fit.history['val_' + metric_opt]
            if metric_opt == 'loss':
                metric_best = np.min(metric)
            else:
                metric_best = np.max(metric)

    return cur_epochs, metric_best


def __train_cgan_model(gen_conf, train_conf, x_train, y_train, x_val, y_val, case_name):
    from time import time

    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    #multi_output = gen_conf['multi_output']
    #output_name = gen_conf['output_name']
    is_continue = train_conf['continue_tr']
    #metric_opt = train_conf['metric']
    num_epochs = train_conf['num_epochs']
    batch_size = train_conf['batch_size'] # default
    #attention_loss = train_conf['attention_loss']
    #overlap_penalty_loss = train_conf['overlap_penalty_loss']
    exclusive_train = train_conf['exclusive_train']
    exclude_label_num = dataset_info['exclude_label_num']
    target = dataset_info['target']

    patience = train_conf['patience']

    root_path = gen_conf['root_path']
    model_path = gen_conf['model_path']
    results_path = gen_conf['results_path']
    folder_names = dataset_info['folder_names']

    g_metric_opt = train_conf['GAN']['generator']['metric']
    g_multi_output = train_conf['GAN']['generator']['multi_output']
    g_output_name = train_conf['GAN']['generator']['output_name']
    g_attention_loss = train_conf['GAN']['generator']['attention_loss']
    g_overlap_penalty_loss = train_conf['GAN']['generator']['overlap_penalty_loss']

    d_metric_opt = train_conf['GAN']['discriminator']['metric']
    d_output_shape = train_conf['GAN']['discriminator']['output_shape']

    file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])

    if exclusive_train == 1 and g_multi_output == 0:
        y_train = np.delete(y_train, exclude_label_num[0], 2)
        y_val = np.delete(y_val, exclude_label_num[0], 2)
    elif exclusive_train == 0 and g_multi_output == 1:
        # fill interposed labels and remove interposed labels for output 1
        y_train_1 = np.delete(y_train, exclude_label_num[0], 2)
        y_val_1 = np.delete(y_val, exclude_label_num[0], 2)

        if exclude_label_num[0] == 2:
            y_train_bg_interposed_one_hot = y_train[:, :, 0] + y_train[:, :, 2]
            y_train_bg_interposed_one_hot[y_train_bg_interposed_one_hot == np.max(y_train_bg_interposed_one_hot)] = 1
            y_train_1[:, :, 0] = y_train_bg_interposed_one_hot

            y_val_bg_interposed_one_hot = y_val[:, :, 0] + y_val[:, :, 2]
            y_val_bg_interposed_one_hot[y_val_bg_interposed_one_hot == np.max(y_val_bg_interposed_one_hot)] = 1
            y_val_1[:, :, 0] = y_val_bg_interposed_one_hot

        # fill interposed labels and remove interposed labels for output 2
        y_train_2 = np.delete(y_train, exclude_label_num[1], 2)
        y_val_2 = np.delete(y_val, exclude_label_num[1], 2)

        if exclude_label_num[1] == 1:
            y_train_bg_dentate_one_hot = y_train[:, :, 0] + y_train[:, :, 1]
            y_train_bg_dentate_one_hot[y_train_bg_dentate_one_hot == np.max(y_train_bg_dentate_one_hot)] = 1
            y_train_2[:, :, 0] = y_train_bg_dentate_one_hot

            y_val_bg_dentate_one_hot = y_val[:, :, 0] + y_val[:, :, 1]
            y_val_bg_dentate_one_hot[y_val_bg_dentate_one_hot == np.max(y_val_bg_dentate_one_hot)] = 1
            y_val_2[:, :, 0] = y_val_bg_dentate_one_hot

        if g_attention_loss == 1:
            # only labels: dentate + interposed
            y_train_3 = np.delete(y_train, 2, 2)
            y_val_3 = np.delete(y_val, 2, 2)

            y_train_dentate_interposed_one_hot = y_train[:, :, 1] + y_train[:, :, 2]
            y_train_dentate_interposed_one_hot[y_train_dentate_interposed_one_hot ==
                                               np.max(y_train_dentate_interposed_one_hot)] = 1
            y_train_3[:, :, 1] = y_train_dentate_interposed_one_hot

            y_val_dentate_interposed_one_hot = y_val[:, :, 1] + y_val[:, :, 2]
            y_val_dentate_interposed_one_hot[y_val_dentate_interposed_one_hot ==
                                               np.max(y_val_dentate_interposed_one_hot)] = 1
            y_val_3[:, :, 1] = y_val_dentate_interposed_one_hot

            if g_overlap_penalty_loss == 1:
                y_train_bg_one = y_train[:, :, 0]
                y_train_bg_one[y_train_bg_one == 0] = 1

                y_val_bg_one = y_val[:, :, 0]
                y_val_bg_one[y_val_bg_one == 0] = 1

                print (y_train_bg_one.shape)
                print(y_val_bg_one.shape)

                y_train = {
                    g_output_name[0]: y_train_1,
                    g_output_name[1]: y_train_2,
                    'attention_maps': y_train_3,
                    'overlap_dentate_interposed': y_train_bg_one # not meaningful (garbage values)
                }
                y_val = {
                    g_output_name[0]: y_val_1,
                    g_output_name[1]: y_val_2,
                    'attention_maps': y_val_3,
                    'overlap_dentate_interposed': y_val_bg_one
                }
            else:
                y_train = {
                    g_output_name[0]: y_train_1,
                    g_output_name[1]: y_train_2,
                    'attention_maps': y_train_3
                }
                y_val = {
                    g_output_name[0]: y_val_1,
                    g_output_name[1]: y_val_2,
                    'attention_maps': y_val_3
                }
        else:
            if g_overlap_penalty_loss == 1:
                y_train_bg_one = y_train[:, :, 0]
                y_train_bg_one[y_train_bg_one == 0] = 1

                y_val_bg_one = y_val[:, :, 0]
                y_val_bg_one[y_val_bg_one == 0] = 1

                print (y_train_bg_one.shape)
                print(y_val_bg_one.shape)

                y_train = {
                    g_output_name[0]: y_train_1,
                    g_output_name[1]: y_train_2,
                    'overlap_dentate_interposed': y_train_bg_one
                }
                y_val = {
                    g_output_name[0]: y_val_1,
                    g_output_name[1]: y_val_2,
                    'overlap_dentate_interposed': y_val_bg_one
                }
            else:
                y_train = {
                    g_output_name[0]: y_train_1,
                    g_output_name[1]: y_train_2
                }
                y_val = {
                    g_output_name[0]: y_val_1,
                    g_output_name[1]: y_val_2
                }
    elif exclusive_train == 1 and g_multi_output == 1:
        print('In multi_output option, exclusive_train should be off')
        exit()
    else:
        if g_attention_loss == 1:
            if target == 'both':
                # only labels: dentate + interposed
                y_train_3 = np.delete(y_train, 2, 2)
                y_val_3 = np.delete(y_val, 2, 2)

                y_train_dentate_interposed_one_hot = y_train[:, :, 1] + y_train[:, :, 2]
                y_train_dentate_interposed_one_hot[y_train_dentate_interposed_one_hot ==
                                                   np.max(y_train_dentate_interposed_one_hot)] = 1
                y_train_3[:, :, 1] = y_train_dentate_interposed_one_hot

                y_val_dentate_interposed_one_hot = y_val[:, :, 1] + y_val[:, :, 2]
                y_val_dentate_interposed_one_hot[y_val_dentate_interposed_one_hot ==
                                                 np.max(y_val_dentate_interposed_one_hot)] = 1
                y_val_3[:, :, 1] = y_val_dentate_interposed_one_hot

                y_train = {
                    g_output_name: y_train,
                    'attention_maps': y_train_3
                }
                y_val = {
                    g_output_name: y_val,
                    'attention_maps': y_val_3
                }
            else:
                y_train = {
                    g_output_name: y_train,
                    'attention_maps': y_train
                }
                y_val = {
                    g_output_name: y_val,
                    'attention_maps': y_val
                }

    if is_continue == 1:
        gan_model_path = os.path.join(root_path, model_path, dataset, folder_names[0])
        if os.path.isfile(gan_model_path):
            [g_model, d_model, cgan_model] = load_gan_model(gen_conf, train_conf, 'pre_trained')
            #model = load_model(model_filename)
            print('Loading saved weights from trained modeld (%s)' % gan_model_path)
        else:
            print('No Found trained models in the path (%s). Newly starting training...' % gan_model_path)
            print('Generating a model to be trained...')
            [g_model, d_model, cgan_model] = generate_model(gen_conf, train_conf)
    else:
        print('Generating a model to be trained...')
        [g_model, d_model, cgan_model] = generate_model(gen_conf, train_conf)

    # x_train: source real image
    # y_train: target real image
    # x_val: source real image (validation set)
    # y_val: target real image (validation set)

    _y_train = []
    _y_val = []
    for output_name in [g_output_name[0], g_output_name[1], 'attention_maps', 'overlap_dentate_interposed']:
        _y_train.append(y_train[output_name])
        _y_val.append(y_val[output_name])

    trn_dataset = [x_train, _y_train]
    val_dataset = [x_val, _y_val]
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(x_train) / batch_size)
    # calculate the number of training iterations
    n_steps = bat_per_epo * num_epochs
    #d_patch_size = d_output_shape[0]

    trn_d_accuracies = []
    trn_losses = []
    val_d_accuracies = []
    val_losses = []
    min_g_val_loss = 1000
    patience_cnt = 0
    #val_ratio = 0.2 # a ratio of validation set and training set

    traing_log_file = os.path.join(file_output_dir, 'training_log_#%d.txt' % case_name)
    is_debug = 1
    if os.path.isfile(traing_log_file):
        os.remove(traing_log_file)

    # manually enumerate epochs
    for e in range(num_epochs):
        t1 = time()
        for i in range(bat_per_epo):
            # select a batch of real samples
            [x_real, y_real], c_real = gen_real_samples(trn_dataset, batch_size, d_output_shape)
            # generate a batch of fake samples
            y_pred, c_fake = gen_pred_samples(g_model, x_real, d_output_shape)

            # update discriminator for real samples
            d_loss1 = d_model.train_on_batch([x_real, y_real[0], y_real[1]], c_real) # discriminator loss (real)
            # update discriminator for generated samples
            d_loss2 = d_model.train_on_batch([x_real, y_pred[0], y_pred[1]], c_fake) # discriminator loss (fake)
            d_loss = 0.5 * np.add(d_loss1, d_loss2)
            #d_loss = [d_loss, d_acc, d_acc_dc]

            # update the generator
            #g_sim_loss = g_model.train_on_batch(x_real, y_real) # similarity loss of a generator
            g_loss = cgan_model.train_on_batch(x_real, [c_real, y_real[0], y_real[1], y_real[2], y_real[3]]) # output: weighted sum of adversarial loss and L1 loss, adversarial loss, L1 loss of a generator
            # g_loss = [g_loss_weighted_sum, g_adv_loss, g_den_loss, g_int_loss, g_attn_loss, g_ovr_loss]

            if i < bat_per_epo-1: # summarize performance
                status = 'epoch#: %d/%d, step#: %d/%d, training - d_loss[%.3f], d_acc[%.2f], g_loss[%.3f]' \
                         % (e + 1, num_epochs, i + 1, bat_per_epo, d_loss[0], 100 * d_loss[1], g_loss[0])
                _ = ioutils.create_log(traing_log_file, status, is_debug=is_debug)

            else: # do validation at the end of the step of each epoch
                # validation set
                [x_val_real, y_val_real], c_real = gen_real_samples(val_dataset, batch_size, d_output_shape)
                y_val_pred, c_fake = gen_pred_samples(g_model, x_val_real, d_output_shape)
                d_val_loss1 = d_model.test_on_batch([x_val_real, y_val_real[0], y_val_real[1]],
                                                    c_real)  # discriminator loss (real)
                d_val_loss2 = d_model.test_on_batch([x_val_real, y_val_pred[0], y_val_pred[1]],
                                                    c_fake)  # discriminator loss (fake)
                d_val_loss = 0.5 * np.add(d_val_loss1, d_val_loss2)

                g_val_loss = cgan_model.test_on_batch(x_val_real,
                                                      [c_real, y_val_real[0], y_val_real[1], y_val_real[2],
                                                       y_val_real[3]])

                status = 'epoch#: %d/%d, step#: %d/%d, training - d_loss[%.3f], d_acc[%.2f], g_loss[%.3f], validation ' \
                         '- d_loss[%.3f], d_acc[%.2f], g_val_loss[%.3f]' \
                         % (e + 1, num_epochs, i + 1, bat_per_epo, d_loss[0], 100 * d_loss[1], g_loss[0], d_val_loss[0],
                            100 * d_val_loss[1], g_val_loss[0])
                _ = ioutils.create_log(traing_log_file, status, is_debug=is_debug)


        # trn_losses.append((d_loss[0], g_loss[0]))
        # trn_d_accuracies.append(d_loss[1])
        #
        # val_losses.append((d_val_loss[0], g_val_loss[0]))
        # val_d_accuracies.append(d_val_loss[1])

        cur_g_val_loss = g_val_loss[0]

        if cur_g_val_loss < min_g_val_loss:
            # save model

            save_gan_model(gen_conf, train_conf, case_name, [g_model, d_model, cgan_model])
            min_g_val_loss = cur_g_val_loss
            patience_cnt = 0
        else:
            patience_cnt += 1

        t2 = time()
        elapsed = t2 - t1
        f = ioutils.create_log(traing_log_file, 'Elapsed time is %f seconds (g_val_loss: %f, min_g_val_loss: %f, '
                                                'patience_cnt: %d)' % (elapsed, g_val_loss[0], min_g_val_loss,
                                                                       patience_cnt), is_debug=is_debug)

        if patience_cnt == patience:
            break

    cur_epochs = e + 1
    metric_best = min_g_val_loss
    if is_debug == 1:
        f.close()

    # model_fit = model.fit(
    #     x_train, y_train, batch_size=batch_size,
    #     epochs=train_conf['num_epochs'],
    #     validation_data=(x_val, y_val),
    #     verbose=train_conf['verbose'],
    #     callbacks=callbacks, shuffle=is_shuffle)

    # cur_epochs = len(model_fit.history['loss'])
    # metric_best = None
    #
    # if multi_output == 1:
    #     if metric_opt in ['acc', 'acc_dc', 'loss']:
    #         metric_monitor = 'val_' + output_name[0] + '_' + metric_opt
    #     elif metric_opt == 'loss_total':
    #         metric_monitor = 'val_loss'
    #     else:
    #         print('unknown metric for early stopping')
    #         metric_monitor = None
    #     #metric = model_fit.history[metric_monitor]
    #     if metric_opt in ['loss', 'loss_total']:
    #         metric_best = np.min(metric)
    #     else:
    #         metric_best = np.max(metric)
    # else:
    #     if attention_loss == 1:
    #         if metric_opt in ['acc', 'acc_dc', 'loss']:
    #             metric_monitor = 'val_' + output_name + '_' + metric_opt
    #         elif metric_opt == 'loss_total':
    #             metric_monitor = 'val_loss'
    #         else:
    #             print('unknown metric for early stopping')
    #             metric_monitor = None
    #         #metric = model_fit.history[metric_monitor]
    #         if metric_opt == ['loss', 'loss_total']:
    #             metric_best = np.min(metric)
    #         else:
    #             metric_best = np.max(metric)
    #     else:
    #         #metric = model_fit.history['val_' + metric_opt]
    #         if metric_opt == 'loss':
    #             metric_best = np.min(metric)
    #         else:
    #             metric_best = np.max(metric)

    return cur_epochs, metric_best


def __train_sar_pred_model(gen_conf, train_conf, train_data, case_name):
    from time import time

    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    is_continue = train_conf['continue_tr']
    num_epochs = train_conf['num_epochs']
    batch_size = train_conf['batch_size']
    patience = train_conf['patience']

    root_path = gen_conf['root_path']
    model_path = gen_conf['model_path']
    results_path = gen_conf['results_path']
    folder_names = dataset_info['folder_names']

    g_patch_shape = train_conf['GAN']['generator']['patch_shape']
    d_patch_shape = train_conf['GAN']['discriminator']['patch_shape']
    d_output_shape = train_conf['GAN']['discriminator']['output_shape']

    file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])

    d_patch_shape_2_arr = np.divide(d_patch_shape, 2)
    d_patch_shape_2 = tuple(int(s) for s in d_patch_shape_2_arr)
    d_patch_shape_4_arr = np.divide(d_patch_shape, 4)
    d_patch_shape_4 = tuple(int(s) for s in d_patch_shape_4_arr)

    interm_f1_real_one_sample = np.ones(((batch_size, 1, ) + d_patch_shape))
    interm_f2_real_one_sample = np.ones(((batch_size, 1, ) + d_patch_shape_2))
    interm_f3_real_one_sample = np.ones(((batch_size, 1, ) + d_patch_shape_4))
    interm_f_real_one_sample = [interm_f1_real_one_sample, interm_f2_real_one_sample, interm_f3_real_one_sample]

    interm_f1_real_zero_sample = np.zeros(((batch_size, 1, ) + d_patch_shape))
    interm_f2_real_zero_sample = np.zeros(((batch_size, 1, ) + d_patch_shape_2))
    interm_f3_real_zero_sample = np.zeros(((batch_size, 1, ) + d_patch_shape_4))
    interm_f_real_zero_sample = [interm_f1_real_zero_sample, interm_f2_real_zero_sample, interm_f3_real_zero_sample]

    if g_patch_shape != d_patch_shape:
        x_train_g = train_data[0]
        y_train_g = train_data[1]
        x_val_g = train_data[2]
        y_val_g = train_data[3]
        x_train_d = train_data[4]
        y_train_d = train_data[5]
        x_val_d = train_data[6]
        y_val_d = train_data[7]

        trn_dataset_g = [x_train_g, y_train_g]
        val_dataset_g = [x_val_g, y_val_g]

        trn_dataset_d = [x_train_d, y_train_d]
        val_dataset_d = [x_val_d, y_val_d]

        num_train = min(len(x_train_g), len(x_train_d))
    else:
        x_train_g = train_data[0]
        y_train_g = train_data[1]
        x_val_g = train_data[2]
        y_val_g = train_data[3]

        trn_dataset_g = [x_train_g, y_train_g]
        val_dataset_g = [x_val_g, y_val_g]

        trn_dataset_d = []
        val_dataset_d = []

        num_train = len(x_train_g)

    if is_continue == 1:
        gan_model_path = os.path.join(root_path, model_path, dataset, folder_names[0])
        if os.path.isfile(gan_model_path):
            model = load_gan_model(gen_conf, train_conf, 'pre_trained')
            #model = load_model(model_filename)
            print('Loading saved weights from trained modeld (%s)' % gan_model_path)
        else:
            print('No Found trained models in the path (%s). Newly starting training...' % gan_model_path)
            print('Generating a model to be trained...')
            model = generate_model(gen_conf, train_conf)
    else:
        print('Generating a model to be trained...')
        model = generate_model(gen_conf, train_conf)

    # x_train: source real image
    # y_train: target real image
    # x_val: source real image (validation set)
    # y_val: target real image (validation set)

    # calculate the number of batches per training epoch
    bat_per_epo = int(num_train / batch_size) # total_steps = bat_per_epo * num_epochs
    # calculate the number of training iterations
    # d_patch_size = d_output_shape[0]

    trn_d_accuracies = []
    trn_losses = []
    val_d_accuracies = []
    val_losses = []
    min_g_val_loss = 1000
    patience_cnt = 0

    traing_log_file = os.path.join(file_output_dir, 'training_log_#%d.txt' % case_name)
    is_debug = 1
    if os.path.isfile(traing_log_file):
        os.remove(traing_log_file)

    # manually enumerate epochs
    d_loss, g_loss, d_val_loss, g_val_loss = [], [], [], []
    for e in range(num_epochs):
        t1 = time()
        for i in range(bat_per_epo):
            d_loss, g_loss, model = fit_cgan_sar_model(g_patch_shape, d_patch_shape, d_output_shape, batch_size, model,
                                                       trn_dataset_g, trn_dataset_d, interm_f_real_one_sample,
                                                       interm_f_real_zero_sample, is_train=1)

            if i < bat_per_epo-1: # summarize performance
                status = 'epoch#: %d/%d, step#: %d/%d, training - d_loss[%.3f], d_acc[%.2f], g_loss[%.3f], g_adv_loss[%.3f], ' \
                         'g_l1_loss[%.3f], g_percep_loss1[%.4f], g_percep_loss2[%.4f], g_percep_loss3[%.4f], g_peak_loss[%.3f], ' \
                         'g_sar_real_peak[%.3f], g_sar_pred_peak[%.3f], g_neg_loss[%.3f]' \
                         % (e + 1, num_epochs, i + 1, bat_per_epo, d_loss[0], 100 * d_loss[5], g_loss[0], g_loss[1],
                            g_loss[2], g_loss[3], g_loss[4], g_loss[5], g_loss[6], g_loss[7], g_loss[8], g_loss[9])
                _ = ioutils.create_log(traing_log_file, status, is_debug=is_debug)

            else: # do validation at the end of the step of each epoch
                # validation set
                d_val_loss, g_val_loss, _ = fit_cgan_sar_model(g_patch_shape, d_patch_shape, d_output_shape,
                                                                           batch_size, model, val_dataset_g,
                                                                           val_dataset_d, interm_f_real_one_sample,
                                                                           interm_f_real_zero_sample, is_train=0)

                status = 'epoch#: %d/%d, step#: %d/%d, training - d_loss[%.3f], d_acc[%.2f], g_loss[%.3f], g_adv_loss[%.3f], ' \
                         'g_l1_loss[%.3f], g_percep_loss1[%.4f], g_percep_loss2[%.4f], g_percep_loss3[%.4f], g_peak_loss[%.3f], ' \
                         'g_sar_real_peak[%.3f], g_sar_pred_peak[%.3f], g_neg_loss[%.3f];' \
                         'validation - d_loss[%.3f], d_acc[%.2f], g_val_loss[%.3f], g_val_adv_loss[%.3f], g_val_l1_loss[%.3f], ' \
                         'g_val_percep_loss1[%.4f], g_val_percep_loss2[%.4f], g_val_percep_loss3[%.4f], g_val_peak_loss[%.3f], ' \
                         'g_val_sar_real_peak[%.3f], g_val_sar_pred_peak[%.3f], g_val_neg_loss[%.3f]' \
                         % (e + 1, num_epochs, i + 1, bat_per_epo, d_loss[0], 100 * d_loss[5], g_loss[0], g_loss[1],
                            g_loss[2], g_loss[3], g_loss[4], g_loss[5], g_loss[6], g_loss[7], g_loss[8], g_loss[9], d_val_loss[0],
                            100 * d_val_loss[1], g_val_loss[0], g_val_loss[1], g_val_loss[2], g_val_loss[3], g_val_loss[4],
                            g_val_loss[5], g_val_loss[6], g_val_loss[7], g_val_loss[8], g_val_loss[9])
                _ = ioutils.create_log(traing_log_file, status, is_debug=is_debug)

        # trn_losses.append((d_loss[0], g_loss[0]))
        # trn_d_accuracies.append(d_loss[1])
        #
        # val_losses.append((d_val_loss[0], g_val_loss[0]))
        # val_d_accuracies.append(d_val_loss[1])


        t2 = time()
        elapsed = t2 - t1

        if patience > 0:
            cur_g_val_loss = g_val_loss[0]
            if cur_g_val_loss < min_g_val_loss:
                # save model
                save_gan_model(gen_conf, train_conf, case_name, model)
                min_g_val_loss = cur_g_val_loss
                patience_cnt = 0
            else:
                patience_cnt += 1

            f = ioutils.create_log(traing_log_file, 'Elapsed time is %f seconds (g_val_loss: %f, min_g_val_loss: %f, '
                                                    'patience_cnt: %d)' % (elapsed, g_val_loss[0], min_g_val_loss,
                                                                           patience_cnt), is_debug=is_debug)

            if patience_cnt == patience:
                break
        else: # save the model every epoch
            save_gan_model(gen_conf, train_conf, case_name, model)

            f = ioutils.create_log(traing_log_file, 'Elapsed time is %f seconds (g_val_loss: %f)'
                                   % (elapsed, g_val_loss[0]), is_debug=is_debug)

    cur_epochs = e + 1
    metric_best = min_g_val_loss
    if is_debug == 1:
        f.close()

    return cur_epochs, metric_best


def __train_sar_pred_model_2_5D(gen_conf, train_conf, g_patch_shape, g_output_shape, d_patch_shape, d_output_shape,
                                train_data, case_name, dim_label):
    from time import time

    approach = train_conf['approach']
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dimension = train_conf['dimension']
    is_continue = train_conf['continue_tr']
    num_epochs = train_conf['num_epochs']
    batch_size = train_conf['batch_size']
    patience = train_conf['patience']

    root_path = gen_conf['root_path']
    model_path = gen_conf['model_path']
    results_path = gen_conf['results_path']
    folder_names = dataset_info['folder_names']

    if dim_label == 'axial':
        g_patch_shape_2d = (g_patch_shape[0], g_patch_shape[1])
        g_output_shape_2d = (g_output_shape[0], g_output_shape[1])
        d_patch_shape_2d = (d_patch_shape[0], d_patch_shape[1])
        d_output_shape_2d = (d_output_shape[0], d_output_shape[1])
    elif dim_label == 'sagittal':
        g_patch_shape_2d = (g_patch_shape[0], g_patch_shape[2])
        g_output_shape_2d = (g_output_shape[0], g_output_shape[2])
        d_patch_shape_2d = (d_patch_shape[0], d_patch_shape[2])
        d_output_shape_2d = (d_output_shape[0], d_output_shape[2])
    elif dim_label == 'coronal':
        g_patch_shape_2d = (g_patch_shape[1], g_patch_shape[2])
        g_output_shape_2d = (g_output_shape[1], g_output_shape[2])
        d_patch_shape_2d = (d_patch_shape[1], d_patch_shape[2])
        d_output_shape_2d = (d_output_shape[1], d_output_shape[2])

    train_conf['GAN']['generator']['patch_shape'] = g_patch_shape_2d
    train_conf['GAN']['generator']['output_shape'] = g_output_shape_2d
    train_conf['GAN']['discriminator']['patch_shape'] = d_patch_shape_2d
    train_conf['GAN']['discriminator']['output_shape'] = d_output_shape_2d

    file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])

    d_patch_shape_2d_2_arr = np.divide(d_patch_shape_2d, 2)
    d_patch_shape_2d_2 = tuple(int(s) for s in d_patch_shape_2d_2_arr)
    d_patch_shape_2d_4_arr = np.divide(d_patch_shape_2d, 4)
    d_patch_shape_2d_4 = tuple(int(s) for s in d_patch_shape_2d_4_arr)
    d_patch_shape_2d_8_arr = np.divide(d_patch_shape_2d, 8)
    d_patch_shape_2d_8 = tuple(int(s) for s in d_patch_shape_2d_8_arr)

    print(dim_label)
    print(d_patch_shape_2d)
    print(d_patch_shape_2d_2)
    print(d_patch_shape_2d_4)
    print(d_patch_shape_2d_8)

    interm_f1_real_one_sample = np.ones((batch_size, d_patch_shape_2d_2[0], d_patch_shape_2d_2[1], 1))
    interm_f2_real_one_sample = np.ones((batch_size, d_patch_shape_2d_4[0], d_patch_shape_2d_4[1], 1))
    interm_f3_real_one_sample = np.ones((batch_size, d_patch_shape_2d_8[0], d_patch_shape_2d_8[1], 1))
    interm_f_real_one_sample = [interm_f1_real_one_sample, interm_f2_real_one_sample, interm_f3_real_one_sample]

    interm_f1_real_zero_sample = np.zeros((batch_size, d_patch_shape_2d_2[0], d_patch_shape_2d_2[1], 1))
    interm_f2_real_zero_sample = np.zeros((batch_size, d_patch_shape_2d_4[0], d_patch_shape_2d_4[1], 1))
    interm_f3_real_zero_sample = np.zeros((batch_size, d_patch_shape_2d_8[0], d_patch_shape_2d_8[1], 1))
    interm_f_real_zero_sample = [interm_f1_real_zero_sample, interm_f2_real_zero_sample, interm_f3_real_zero_sample]

    print(interm_f1_real_one_sample.shape)
    print(interm_f2_real_one_sample.shape)
    print(interm_f3_real_one_sample.shape)
    print(interm_f1_real_zero_sample.shape)
    print(interm_f2_real_zero_sample.shape)
    print(interm_f3_real_zero_sample.shape)

    if g_patch_shape != d_patch_shape:
        x_train_g = train_data[0]
        y_train_g = train_data[1]
        x_val_g = train_data[2]
        y_val_g = train_data[3]
        x_train_d = train_data[4]
        y_train_d = train_data[5]
        x_val_d = train_data[6]
        y_val_d = train_data[7]

        x_train_g = np.transpose(x_train_g, [0, 2, 3, 1])
        x_val_g = np.transpose(x_val_g, [0, 2, 3, 1])
        x_train_d = np.transpose(x_train_d, [0, 2, 3, 1])
        x_val_d = np.transpose(x_val_d, [0, 2, 3, 1])

        # x_train_g = np.moveaxis(x_train_g, 1, 3)
        # x_val_g = np.moveaxis(x_val_g, 1, 3)
        # x_train_d = np.moveaxis(x_train_d, 1, 3)
        # x_val_d = np.moveaxis(x_val_d, 1, 3)

        trn_dataset_g = [x_train_g, y_train_g]
        val_dataset_g = [x_val_g, y_val_g]

        trn_dataset_d = [x_train_d, y_train_d]
        val_dataset_d = [x_val_d, y_val_d]

        num_train = min(x_train_g.shape[0], x_train_d.shape[0])
    else:
        x_train_g = train_data[0]
        y_train_g = train_data[1]
        x_val_g = train_data[2]
        y_val_g = train_data[3]

        print(x_train_g.shape)
        print(y_train_g.shape)

        x_train_g = np.transpose(x_train_g, [0, 2, 3, 1])
        x_val_g = np.transpose(x_val_g, [0, 2, 3, 1])

        # x_train_g = np.moveaxis(x_train_g, 1, 3)
        # x_val_g = np.moveaxis(x_val_g, 1, 3)

        print(x_train_g.shape)
        print(y_train_g.shape)

        trn_dataset_g = [x_train_g, y_train_g]
        val_dataset_g = [x_val_g, y_val_g]

        trn_dataset_d = []
        val_dataset_d = []

        num_train = x_train_g.shape[0]
        print(num_train)


    # ### debug: save training samples
    #
    # dataset = train_conf['dataset']
    # root_path = gen_conf['root_path']
    # results_path = gen_conf['results_path']
    # folder_names = dataset_info['folder_names']
    # modality = dataset_info['image_modality']
    # file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])
    # if not os.path.exists(file_output_dir):
    #     os.makedirs(file_output_dir)
    #
    # for i in range(50):
    #     for m, src_label in zip(range(len(modality)), modality):
    #         src_filename = os.path.join(file_output_dir, 'src_%s_%s_%s.%s' % (src_label, dim_label, i, 'nii.gz'))
    #         __save_sar_volume(x_train_g[i, :, :, m], src_filename, 'nii.gz', 'float32', is_compressed=False)
    #
    #     sar_filename = os.path.join(file_output_dir, 'sar_%s_%s.%s' % (dim_label, i, 'nii.gz'))
    #     __save_sar_volume(y_train_g[i, :, :], sar_filename, 'nii.gz', 'float32', is_compressed=False)
    #
    # ###


    if is_continue == 1:
        gan_model_path = os.path.join(root_path, model_path, dataset, folder_names[0])
        if os.path.isfile(gan_model_path):
            model = load_gan_model(gen_conf, train_conf, 'pre_trained')
            print('Loading saved weights from trained modeld (%s)' % gan_model_path)
        else:
            print('No Found trained models in the path (%s). Newly starting training...' % gan_model_path)
            print('Generating a model to be trained...')
            model = generate_model(gen_conf, train_conf)
    else:
        print('Generating a model to be trained...')
        model = generate_model(gen_conf, train_conf)

    # x_train: source real image
    # y_train: target real image
    # x_val: source real image (validation set)
    # y_val: target real image (validation set)

    # calculate the number of batches per training epoch
    bat_per_epo = int(num_train / batch_size) # total_steps = bat_per_epo * num_epochs
    # calculate the number of training iterations
    #d_patch_size = d_output_shape[0]

    trn_d_accuracies = []
    trn_losses = []
    val_d_accuracies = []
    val_losses = []
    min_g_val_loss = 1000
    patience_cnt = 0

    traing_log_file = os.path.join(file_output_dir, 'training_log_%s_#%d.txt' % (dim_label, case_name))
    is_debug = 1
    if os.path.isfile(traing_log_file):
        os.remove(traing_log_file)

    # manually enumerate epochs
    d_loss, g_loss, d_val_loss, g_val_loss = [0], [0], [0], [0]
    if approach in ['single_cyclegan_sar', 'multi_cyclegan_sar', 'composite_cyclegan_sar']:
        x_pool, y_pool = list(), list()

    for e in range(num_epochs):
        t1 = time()
        for i in range(bat_per_epo):
            if approach == 'cgan_sar':
                d_loss, g_loss, model = fit_cgan_sar_model(g_patch_shape, d_patch_shape, d_output_shape_2d, batch_size, model,
                                                trn_dataset_g, trn_dataset_d, interm_f_real_one_sample,
                                                interm_f_real_zero_sample, is_train=1)

                if i < bat_per_epo-1: # summarize performance
                    status = 'epoch#: %d/%d, step#: %d/%d, training - d_loss[%.3f], d_acc[%.2f], g_loss[%.3f], g_adv_loss[%.3f], ' \
                             'g_l1_loss[%.3f], g_percep_loss1[%.6f], g_percep_loss2[%.6f], g_percep_loss3[%.6f], g_peak_loss[%.3f], ' \
                             'g_sar_real_peak[%.3f], g_sar_pred_peak[%.3f], g_neg_loss[%.3f]' \
                             % (e + 1, num_epochs, i + 1, bat_per_epo, d_loss[0], 100 * d_loss[5], g_loss[0], g_loss[1],
                                g_loss[2], g_loss[3], g_loss[4], g_loss[5], g_loss[6], g_loss[7], g_loss[8], g_loss[9])
                    _ = ioutils.create_log(traing_log_file, status, is_debug=is_debug)

                else: # do validation at the end of the step of each epoch
                    d_val_loss, g_val_loss, _ = fit_cgan_sar_model(g_patch_shape, d_patch_shape, d_output_shape_2d, batch_size,
                                                                model, val_dataset_g, val_dataset_d, interm_f_real_one_sample,
                                                                interm_f_real_zero_sample, is_train=0)

                    status = 'epoch#: %d/%d, step#: %d/%d, training - d_loss[%.3f], d_acc[%.2f], g_loss[%.3f], g_adv_loss[%.3f], ' \
                             'g_l1_loss[%.3f], g_percep_loss1[%.4f], g_percep_loss2[%.4f], g_percep_loss3[%.4f], g_peak_loss[%.3f], ' \
                             'g_sar_real_peak[%.3f], g_sar_pred_peak[%.3f], g_neg_loss[%.3f];' \
                             ' validation - d_loss[%.3f], d_acc[%.2f], g_val_loss[%.3f], g_val_adv_loss[%.3f], g_val_l1_loss[%.3f], ' \
                             'g_val_percep_loss1[%.6f], g_val_percep_loss2[%.6f], g_val_percep_loss3[%.6f], g_val_peak_loss[%.3f], ' \
                             'g_val_sar_real_peak[%.3f], g_val_sar_pred_peak[%.3f], g_val_neg_loss[%.3f]' \
                             % (e + 1, num_epochs, i + 1, bat_per_epo, d_loss[0], 100 * d_loss[5], g_loss[0], g_loss[1],
                                g_loss[2], g_loss[3], g_loss[4], g_loss[5], g_loss[6], g_loss[7], g_loss[8], g_loss[9], d_val_loss[0],
                                100 * d_val_loss[5], g_val_loss[0], g_val_loss[1], g_val_loss[2], g_val_loss[3], g_val_loss[4],
                                g_val_loss[5], g_val_loss[6], g_val_loss[7], g_val_loss[8], g_val_loss[9])
                    _ = ioutils.create_log(traing_log_file, status, is_debug=is_debug)

            elif approach == 'single_cyclegan_sar':
                d_loss_x, d_loss_y, g_loss2, g_loss1, model = \
                    fit_single_cyclegan_sar_model(g_output_shape_2d, d_output_shape_2d, d_output_shape_2d, batch_size, model,
                                                trn_dataset_g, trn_dataset_d, x_pool, y_pool, interm_f_real_one_sample,
                                                interm_f_real_zero_sample, is_train=1)

                if i < bat_per_epo-1: # summarize performance
                    status = 'epoch#: %d/%d, step#: %d/%d, training - d_loss_x[%.3f], d_acc_x[%.2f], d_loss_y[%.3f], ' \
                             'd_acc_y[%.2f], g_loss_XtoY[%.3f], g_adv_loss_XtoY[%.3f], g_id_loss_XtoY[%.3f], ' \
                             'g_fwd_loss_XtoY[%.3f], g_bwd_loss_XtoY[%.3f], g_percep_loss1_XtoY[%.6f], ' \
                             'g_percep_loss2_XtoY[%.6f], g_percep_loss3_XtoY[%.6f], g_peak_loss_XtoY[%.3f], ' \
                             'g_sar_real_peak_XtoY[%.3f], g_sar_pred_peak_XtoY[%.3f], g_neg_loss_XtoY[%.3f], ' \
                             'g_loss_YtoX[%.3f], g_adv_loss_YtoX[%.3f], g_id_loss_YtoX[%.3f], g_fwd_loss_YtoX[%.3f], ' \
                             'g_bwd_loss_YtoX[%.3f], g_percep_loss1_YtoX[%.6f], g_percep_loss2_YtoX[%.6f], ' \
                             'g_percep_loss3_YtoX[%.6f], g_peak_loss_YtoX[%.3f], g_sar_real_peak_YtoX[%.3f], ' \
                             'g_sar_pred_peak_YtoX[%.3f], g_neg_loss_YtoX[%.3f], ' \
                             % (e + 1, num_epochs, i + 1, bat_per_epo, d_loss_x[0], 100 * d_loss_x[5], d_loss_y[0],
                                100 * d_loss_y[5], g_loss1[0], g_loss1[1], g_loss1[2], g_loss1[3], g_loss1[4],
                                g_loss1[5], g_loss1[6], g_loss1[7], g_loss1[8], g_loss1[9], g_loss1[10], g_loss1[11],
                                g_loss2[0], g_loss2[1], g_loss2[2], g_loss2[3], g_loss2[4], g_loss2[5], g_loss2[6],
                                g_loss2[7], g_loss2[8], g_loss2[9], g_loss2[10], g_loss2[11])
                    _ = ioutils.create_log(traing_log_file, status, is_debug=is_debug)

                else: # do validation at the end of the step of each epoch
                    d_val_loss_x, d_val_loss_y, g_val_loss2, g_val_loss1, _ = \
                        fit_single_cyclegan_sar_model(g_output_shape_2d, d_output_shape_2d, d_output_shape_2d, batch_size, model,
                                               val_dataset_g, val_dataset_d, x_pool, y_pool, interm_f_real_one_sample,
                                               interm_f_real_zero_sample, is_train=0)

                    status = 'epoch#: %d/%d, step#: %d/%d, training - d_loss_x[%.3f], d_acc_x[%.2f], d_loss_y[%.3f], ' \
                             'd_acc_y[%.2f], g_loss_XtoY[%.3f], g_adv_loss_XtoY[%.3f], g_id_loss_XtoY[%.3f], ' \
                             'g_fwd_loss_XtoY[%.3f], g_bwd_loss_XtoY[%.3f], g_percep_loss1_XtoY[%.6f], ' \
                             'g_percep_loss2_XtoY[%.6f], g_percep_loss3_XtoY[%.6f], g_peak_loss_XtoY[%.3f], ' \
                             'g_sar_real_peak_XtoY[%.3f], g_sar_pred_peak_XtoY[%.3f], g_neg_loss_XtoY[%.3f], ' \
                             'g_loss_YtoX[%.3f], g_adv_loss_YtoX[%.3f], g_id_loss_YtoX[%.3f], g_fwd_loss_YtoX[%.3f], ' \
                             'g_bwd_loss_YtoX[%.3f], g_percep_loss1_YtoX[%.6f], g_percep_loss2_YtoX[%.6f], ' \
                             'g_percep_loss3_YtoX[%.6f], g_peak_loss_YtoX[%.3f], g_sar_real_peak_YtoX[%.3f], ' \
                             'g_sar_pred_peak_YtoX[%.3f], g_neg_loss_YtoX[%.3f]; ' \
                             'validation - d_val_loss_x[%.3f], d_val_acc_x[%.2f], d_val_loss_y[%.3f], ' \
                             'd_val_acc_y[%.2f], g_val_loss_XtoY[%.3f], g_val_adv_loss_XtoY[%.3f], g_id_val_loss_XtoY[%.3f], ' \
                             'g_val_fwd_loss_XtoY[%.3f], g_val_bwd_loss_XtoY[%.3f], g_val_percep_loss1_XtoY[%.6f], ' \
                             'g_val_percep_loss2_XtoY[%.6f], g_val_percep_loss3_XtoY[%.6f], g_val_peak_loss_XtoY[%.3f], ' \
                             'g_val_sar_real_peak_XtoY[%.3f], g_val_sar_pred_peak_XtoY[%.3f], g_val_neg_loss_XtoY[%.3f], ' \
                             'g_val_loss_YtoX[%.3f], g_val_adv_loss_YtoX[%.3f], g_id_val_loss_YtoX[%.3f], ' \
                             'g_val_fwd_loss_YtoX[%.3f], g_val_bwd_loss_YtoX[%.3f], g_val_percep_loss1_YtoX[%.6f], ' \
                             'g_val_percep_loss2_YtoX[%.6f], g_val_percep_loss3_YtoX[%.6f], g_val_peak_loss_YtoX[%.3f], ' \
                             'g_val_sar_real_peak_YtoX[%.3f], g_val_sar_pred_peak_YtoX[%.3f], g_val_neg_loss_YtoX[%.3f], ' \
                             % (e + 1, num_epochs, i + 1, bat_per_epo, d_loss_x[0], 100 * d_loss_x[5], d_loss_y[0],
                               100 * d_loss_y[5], g_loss1[0], g_loss1[1], g_loss1[2], g_loss1[3], g_loss1[4],
                               g_loss1[5], g_loss1[6], g_loss1[7], g_loss1[8], g_loss1[9], g_loss1[10], g_loss1[11],
                               g_loss2[0], g_loss2[1], g_loss2[2], g_loss2[3], g_loss2[4], g_loss2[5], g_loss2[6],
                               g_loss2[7], g_loss2[8], g_loss2[9], g_loss2[10], g_loss2[11],
                               d_val_loss_x[0], 100 * d_val_loss_x[5], d_val_loss_y[0], 100 * d_val_loss_y[5],
                               g_val_loss1[0], g_val_loss1[1], g_val_loss1[2], g_val_loss1[3], g_val_loss1[4],
                               g_val_loss1[5], g_val_loss1[6], g_val_loss1[7], g_val_loss1[8], g_val_loss1[9],
                               g_val_loss1[10], g_val_loss1[11], g_val_loss2[0], g_val_loss2[1], g_val_loss2[2],
                               g_val_loss2[3], g_val_loss2[4], g_val_loss2[5], g_val_loss2[6], g_val_loss2[7],
                               g_val_loss2[8], g_val_loss2[9], g_val_loss2[10], g_val_loss2[11])
                    _ = ioutils.create_log(traing_log_file, status, is_debug=is_debug)

                    g_val_loss[0] = 0.5 * g_val_loss1[0] + 0.5 * g_val_loss2[0]

            elif approach == 'multi_cyclegan_sar':
                d_loss_x, d_loss_y, g_loss2, g_loss1, model = \
                    fit_multi_cyclegan_sar_model(g_output_shape_2d, d_output_shape_2d, d_output_shape_2d, batch_size, model,
                                                trn_dataset_g, trn_dataset_d, x_pool, y_pool, interm_f_real_one_sample,
                                                interm_f_real_zero_sample, is_train=1)

                if i < bat_per_epo-1: # summarize performance
                    status = 'epoch#: %d/%d, step#: %d/%d, training - d_loss_x[%.3f], d_acc_x[%.2f], d_loss_y[%.3f], ' \
                             'd_acc_y[%.2f], g_loss_XtoY[%.3f], g_adv_loss_XtoY[%.3f], g_id_loss_XtoY[%.3f], ' \
                             'g_fwd_loss_XtoY[%.3f], g_bwd_loss_XtoY[%.3f], g_percep_loss1_XtoY[%.6f], ' \
                             'g_percep_loss2_XtoY[%.6f], g_percep_loss3_XtoY[%.6f], g_peak_loss_XtoY[%.3f], ' \
                             'g_sar_real_peak_XtoY[%.3f], g_sar_pred_peak_XtoY[%.3f], g_mean_loss_XtoY[%.3f], g_neg_loss_XtoY[%.3f], ' \
                             'g_loss_YtoX[%.3f], g_adv_loss_YtoX[%.3f], g_id_loss_YtoX[%.3f], g_fwd_loss_YtoX[%.3f], ' \
                             'g_bwd_loss_YtoX[%.3f], g_percep_loss1_YtoX[%.6f], g_percep_loss2_YtoX[%.6f], ' \
                             'g_percep_loss3_YtoX[%.6f], g_peak_loss_YtoX[%.3f], g_sar_real_peak_YtoX[%.3f], ' \
                             'g_sar_pred_peak_YtoX[%.3f], g_mean_loss_YtoX[%.3f], g_neg_loss_YtoX[%.3f], ' \
                             % (e + 1, num_epochs, i + 1, bat_per_epo, d_loss_x[0], 100 * d_loss_x[5], d_loss_y[0],
                                100 * d_loss_y[5], g_loss1[0], g_loss1[1], g_loss1[2], g_loss1[3], g_loss1[4],
                                g_loss1[5], g_loss1[6], g_loss1[7], g_loss1[8], g_loss1[9], g_loss1[10], g_loss1[11], g_loss1[12],
                                g_loss2[0], g_loss2[1], g_loss2[2], g_loss2[3], g_loss2[4], g_loss2[5], g_loss2[6],
                                g_loss2[7], g_loss2[8], g_loss2[9], g_loss2[10], g_loss2[11], g_loss2[12])
                    _ = ioutils.create_log(traing_log_file, status, is_debug=is_debug)

                else: # do validation at the end of the step of each epoch
                    d_val_loss_x, d_val_loss_y, g_val_loss2, g_val_loss1, _ = \
                        fit_multi_cyclegan_sar_model(g_output_shape_2d, d_output_shape_2d, d_output_shape_2d, batch_size, model,
                                               val_dataset_g, val_dataset_d, x_pool, y_pool, interm_f_real_one_sample,
                                               interm_f_real_zero_sample, is_train=0)

                    status = 'epoch#: %d/%d, step#: %d/%d, training - d_loss_x[%.3f], d_acc_x[%.2f], d_loss_y[%.3f], ' \
                             'd_acc_y[%.2f], g_loss_XtoY[%.3f], g_adv_loss_XtoY[%.3f], g_id_loss_XtoY[%.3f], ' \
                             'g_fwd_loss_XtoY[%.3f], g_bwd_loss_XtoY[%.3f], g_percep_loss1_XtoY[%.6f], ' \
                             'g_percep_loss2_XtoY[%.6f], g_percep_loss3_XtoY[%.6f], g_peak_loss_XtoY[%.3f], ' \
                             'g_sar_real_peak_XtoY[%.3f], g_sar_pred_peak_XtoY[%.3f], g_mean_loss_XtoY[%.3f], g_neg_loss_XtoY[%.3f], ' \
                             'g_loss_YtoX[%.3f], g_adv_loss_YtoX[%.3f], g_id_loss_YtoX[%.3f], g_fwd_loss_YtoX[%.3f], ' \
                             'g_bwd_loss_YtoX[%.3f], g_percep_loss1_YtoX[%.6f], g_percep_loss2_YtoX[%.6f], ' \
                             'g_percep_loss3_YtoX[%.6f], g_peak_loss_YtoX[%.3f], g_sar_real_peak_YtoX[%.3f], ' \
                             'g_sar_pred_peak_YtoX[%.3f], g_mean_loss_YtoX[%.3f], g_neg_loss_YtoX[%.3f]; ' \
                             'validation - d_val_loss_x[%.3f], d_val_acc_x[%.2f], d_val_loss_y[%.3f], ' \
                             'd_val_acc_y[%.2f], g_val_loss_XtoY[%.3f], g_val_adv_loss_XtoY[%.3f], g_id_val_loss_XtoY[%.3f], ' \
                             'g_val_fwd_loss_XtoY[%.3f], g_val_bwd_loss_XtoY[%.3f], g_val_percep_loss1_XtoY[%.6f], ' \
                             'g_val_percep_loss2_XtoY[%.6f], g_val_percep_loss3_XtoY[%.6f], g_val_peak_loss_XtoY[%.3f], ' \
                             'g_val_sar_real_peak_XtoY[%.3f], g_val_sar_pred_peak_XtoY[%.3f], g_val_mean_loss_XtoY[%.3f], ' \
                             'g_val_neg_loss_XtoY[%.3f], g_val_loss_YtoX[%.3f], g_val_adv_loss_YtoX[%.3f], g_id_val_loss_YtoX[%.3f], ' \
                             'g_val_fwd_loss_YtoX[%.3f], g_val_bwd_loss_YtoX[%.3f], g_val_percep_loss1_YtoX[%.6f], ' \
                             'g_val_percep_loss2_YtoX[%.6f], g_val_percep_loss3_YtoX[%.6f], g_val_peak_loss_YtoX[%.3f], ' \
                             'g_val_sar_real_peak_YtoX[%.3f], g_val_sar_pred_peak_YtoX[%.3f], g_val_mean_loss_YtoX[%.3f], ' \
                             'g_val_neg_loss_YtoX[%.3f], ' \
                             % (e + 1, num_epochs, i + 1, bat_per_epo, d_loss_x[0], 100 * d_loss_x[5], d_loss_y[0],
                               100 * d_loss_y[5], g_loss1[0], g_loss1[1], g_loss1[2], g_loss1[3], g_loss1[4],
                               g_loss1[5], g_loss1[6], g_loss1[7], g_loss1[8], g_loss1[9], g_loss1[10], g_loss1[11], g_loss1[12],
                               g_loss2[0], g_loss2[1], g_loss2[2], g_loss2[3], g_loss2[4], g_loss2[5], g_loss2[6],
                               g_loss2[7], g_loss2[8], g_loss2[9], g_loss2[10], g_loss2[11], g_loss2[12],
                               d_val_loss_x[0], 100 * d_val_loss_x[5], d_val_loss_y[0], 100 * d_val_loss_y[5],
                               g_val_loss1[0], g_val_loss1[1], g_val_loss1[2], g_val_loss1[3], g_val_loss1[4],
                               g_val_loss1[5], g_val_loss1[6], g_val_loss1[7], g_val_loss1[8], g_val_loss1[9],
                               g_val_loss1[10], g_val_loss1[11], g_val_loss1[12], g_val_loss2[0], g_val_loss2[1], g_val_loss2[2],
                               g_val_loss2[3], g_val_loss2[4], g_val_loss2[5], g_val_loss2[6], g_val_loss2[7],
                               g_val_loss2[8], g_val_loss2[9], g_val_loss2[10], g_val_loss2[11], g_val_loss2[12])
                    _ = ioutils.create_log(traing_log_file, status, is_debug=is_debug)

                    g_val_loss[0] = 0.5 * g_val_loss1[0] + 0.5 * g_val_loss2[0]

            elif approach == 'composite_cyclegan_sar':
                d_loss_x, d_loss_y, g_loss, model = \
                    fit_cyclegan_sar_model2(g_output_shape_2d, d_output_shape_2d, d_output_shape_2d, batch_size, model,
                                           trn_dataset_g, trn_dataset_d, x_pool, y_pool, interm_f_real_one_sample,
                                           interm_f_real_zero_sample, is_train=1)

                if i < bat_per_epo - 1:  # summarize performance
                    status = 'epoch#: %d/%d, step#: %d/%d, training - d_loss_x[%.3f], d_acc_x[%.2f], d_loss_y[%.3f], ' \
                             'd_acc_y[%.2f], g_loss[%.3f], g_adv_loss_fwd[%.3f], g_adv_loss_bwd[%.3f], g1_loss[%.3f], ' \
                             'g_cycle_loss_fwd[%.3f], g2_loss[%.3f], g_cycle_loss_bwd[%.3f], g_percep_loss1_fwd[%.6f], ' \
                             'g_percep_loss2_fwd[%.6f], g_percep_loss3_fwd[%.6f], g_percep_loss1_bwd[%.6f], ' \
                             'g_percep_loss2_bwd[%.6f], g_percep_loss3_bwd[%.6f], g_sar_peak_loss[%.3f], ' \
                             'g_sar_real_peak[%.3f], g_sar_pred_peak[%.3f], g_sar_neg_loss[%.3f], ' \
                             % (e + 1, num_epochs, i + 1, bat_per_epo, d_loss_x[0], 100 * d_loss_x[5], d_loss_y[0],
                                100 * d_loss_y[5], g_loss[0], g_loss[1], g_loss[2], g_loss[3], g_loss[4],
                                g_loss[5], g_loss[6], g_loss[7], g_loss[8], g_loss[9], g_loss[10], g_loss[11],
                                g_loss[12], g_loss[13], g_loss[14], g_loss[15], g_loss[16])
                    _ = ioutils.create_log(traing_log_file, status, is_debug=is_debug)

                else:  # do validation at the end of the step of each epoch
                    d_val_loss_x, d_val_loss_y, g_val_loss, _ = \
                        fit_cyclegan_sar_model2(g_output_shape_2d, d_output_shape_2d, d_output_shape_2d, batch_size,
                                               model, val_dataset_g, val_dataset_d, x_pool, y_pool, interm_f_real_one_sample,
                                               interm_f_real_zero_sample, is_train=0)

                    status = 'epoch#: %d/%d, step#: %d/%d, training - d_loss_x[%.3f], d_acc_x[%.2f], d_loss_y[%.3f], ' \
                             'd_acc_y[%.2f], g_loss[%.3f], g_adv_loss_fwd[%.3f], g_adv_loss_bwd[%.3f], g1_loss[%.3f], ' \
                             'g_cycle_loss_fwd[%.3f], g2_loss[%.3f], g_cycle_loss_bwd[%.3f], g_percep_loss1_fwd[%.6f], ' \
                             'g_percep_loss2_fwd[%.6f], g_percep_loss3_fwd[%.6f], g_percep_loss1_bwd[%.6f], ' \
                             'g_percep_loss2_bwd[%.6f], g_percep_loss3_bwd[%.6f], g_sar_peak_loss[%.3f], ' \
                             'g_sar_real_peak[%.3f], g_sar_pred_peak[%.3f], g_sar_neg_loss[%.3f], ' \
                             'validation -  d_val_loss_x[%.3f], d_val_acc_x[%.2f], d_val_loss_y[%.3f], ' \
                             'd_val_acc_y[%.2f], g_val_loss[%.3f], g_val_adv_loss_fwd[%.3f], g_val_adv_loss_bwd[%.3f], g1_val_loss[%.3f], ' \
                             'g_val_cycle_loss_fwd[%.3f], g2_val_loss[%.3f], g_val_cycle_loss_bwd[%.3f], g_val_percep_loss1_fwd[%.6f], ' \
                             'g_val_percep_loss2_fwd[%.6f], g_val_percep_loss3_fwd[%.6f], g_val_percep_loss1_bwd[%.6f], ' \
                             'g_val_percep_loss2_bwd[%.6f], g_val_percep_loss3_bwd[%.6f], g_val_sar_peak_loss[%.3f], ' \
                             'g_val_sar_real_peak[%.3f], g_val_sar_pred_peak[%.3f], g_val_sar_neg_loss[%.3f], ' \
                             % (e + 1, num_epochs, i + 1, bat_per_epo, d_loss_x[0], 100 * d_loss_x[5], d_loss_y[0],
                                100 * d_loss_y[5], g_loss[0], g_loss[1], g_loss[2], g_loss[3], g_loss[4],
                                g_loss[5], g_loss[6], g_loss[7], g_loss[8], g_loss[9], g_loss[10], g_loss[11],
                                g_loss[12], g_loss[13], g_loss[14], g_loss[15], g_loss[16],
                                d_val_loss_x[0], 100 * d_val_loss_x[5], d_val_loss_y[0], 100 * d_val_loss_y[5],
                                g_val_loss[0], g_val_loss[1], g_val_loss[2], g_val_loss[3],
                                g_val_loss[4], g_val_loss[5], g_val_loss[6], g_val_loss[7], g_val_loss[8],
                                g_val_loss[9], g_val_loss[10], g_val_loss[11], g_val_loss[12], g_val_loss[13],
                                g_val_loss[14], g_val_loss[15], g_val_loss[16])
                    _ = ioutils.create_log(traing_log_file, status, is_debug=is_debug)

        t2 = time()
        elapsed = t2 - t1

        if patience > 0:
            cur_g_val_loss = g_val_loss[0]
            if cur_g_val_loss < min_g_val_loss:
                # save model
                if approach == 'cgan_sar':
                    trn_model = model
                elif approach in ['single_cyclegan_sar', 'multi_cyclegan_sar', 'composite_cyclegan_sar']:
                    trn_model = [model[0], model[2], model[4]]
                save_gan_model(gen_conf, train_conf, str(case_name) + '_' + dim_label, trn_model)
                min_g_val_loss = cur_g_val_loss
                patience_cnt = 0
            else:
                patience_cnt += 1

            f = ioutils.create_log(traing_log_file, 'Elapsed time is %f seconds (g_val_loss: %f, min_g_val_loss: %f, '
                                                    'patience_cnt (early stop): %d)' % (elapsed, g_val_loss[0], min_g_val_loss,
                                                                           patience_cnt), is_debug=is_debug)

            if patience_cnt == patience: # early stop
                break

        else: # save the model every epoch (without early stop)
            if approach == 'cgan_sar':
                trn_model = model
            elif approach in ['single_cyclegan_sar', 'multi_cyclegan_sar', 'composite_cyclegan_sar']:
                trn_model = [model[0], model[2], model[4]]
            save_gan_model(gen_conf, train_conf, str(case_name) + '_' + dim_label, trn_model)

            cur_g_val_loss = g_val_loss[0]
            if cur_g_val_loss < min_g_val_loss:
                min_g_val_loss = cur_g_val_loss
                patience_cnt = 0
            else:
                patience_cnt += 1

            f = ioutils.create_log(traing_log_file, 'Elapsed time is %f seconds (g_val_loss: %f, min_g_val_loss: %f, '
                                                    'patience_cnt (no early stop): %d)' % (elapsed, g_val_loss[0], min_g_val_loss,
                                                                           patience_cnt), is_debug=is_debug)

    cur_epochs = e + 1
    metric_best = min_g_val_loss
    if is_debug == 1:
        f.close()

    return cur_epochs, metric_best


def fit_cgan_sar_model(g_patch_shape, d_patch_shape, d_output_shape, batch_size, model, trn_dataset_g, trn_dataset_d,
                       interm_f_real_one_sample, interm_f_real_zero_sample, is_train):

    [g_model, d_model, cgan_model] = model

    if g_patch_shape != d_patch_shape:
        [x_real_g, y_real_g, x_real_d, y_real_d], c_real = gen_sar_real_samples(trn_dataset_g, trn_dataset_d,
                                                                                batch_size, d_output_shape)
        # generate a batch of fake samples
        y_pred_g, c_fake = gen_sar_pred_samples(g_model, x_real_g, d_output_shape)

        # update discriminator for real samples
        if is_train == 1:
            d_loss1 = d_model.train_on_batch([x_real_d, y_real_g], [c_real, interm_f_real_one_sample[0],
                                                                    interm_f_real_one_sample[1],
                                                                    interm_f_real_one_sample[2]])  # discriminator loss (real)

            # update discriminator for generated samples
            d_loss2 = d_model.train_on_batch([x_real_d, y_pred_g], [c_fake, interm_f_real_zero_sample[0],
                                                                    interm_f_real_zero_sample[1],
                                                                    interm_f_real_zero_sample[2]])  # discriminator loss (fake)
            d_loss = 0.5 * np.add(d_loss1, d_loss2)

            # add perceptual loss (E[|(discriminator output (y_real) - discriminator ouput (y_pred))|]
            _, interm_f1_real, interm_f2_real, interm_f3_real = d_model.predict_on_batch([x_real_d, y_real_g])
            # update the generator
            g_loss = cgan_model.train_on_batch([x_real_g, x_real_d, y_real_g],
                                               [c_real, y_real_g, interm_f1_real, interm_f2_real, interm_f3_real,
                                                y_real_g, y_real_g, y_real_g,
                                                y_real_g])  # output: weighted sum of adversarial loss and L1 loss, adversarial loss, L1 loss of a generator
        else:
            d_loss1 = d_model.test_on_batch([x_real_d, y_real_g], [c_real, interm_f_real_one_sample[0],
                                                                    interm_f_real_one_sample[1],
                                                                    interm_f_real_one_sample[2]])  # discriminator loss (real)

            # update discriminator for generated samples
            d_loss2 = d_model.test_on_batch([x_real_d, y_pred_g], [c_fake, interm_f_real_zero_sample[0],
                                                                    interm_f_real_zero_sample[1],
                                                                    interm_f_real_zero_sample[2]])  # discriminator loss (fake)
            d_loss = 0.5 * np.add(d_loss1, d_loss2)

            # add perceptual loss (E[|(discriminator output (y_real) - discriminator ouput (y_pred))|]
            _, interm_f1_real, interm_f2_real, interm_f3_real = d_model.predict_on_batch([x_real_d, y_real_g])
            # update the generator
            g_loss = cgan_model.test_on_batch([x_real_g, x_real_d, y_real_g],
                                               [c_real, y_real_g, interm_f1_real, interm_f2_real, interm_f3_real,
                                                y_real_g, y_real_g, y_real_g,
                                                y_real_g])  # output: weighted sum of adversarial loss and L1 loss, adversarial loss, L1 loss of a generator
    else:
        [x_real_g, y_real_g], c_real = gen_sar_real_samples(trn_dataset_g, None, batch_size, d_output_shape)
        # generate a batch of fake samples
        y_pred_g, c_fake = gen_sar_pred_samples(g_model, x_real_g, d_output_shape)

        if is_train == 1:
            # update discriminator for real samples
            d_loss1 = d_model.train_on_batch([x_real_g, y_real_g], [c_real, interm_f_real_one_sample[0],
                                                                    interm_f_real_one_sample[1],
                                                                    interm_f_real_one_sample[2]])  # discriminator loss (real)

            # update discriminator for generated samples
            d_loss2 = d_model.train_on_batch([x_real_g, y_pred_g], [c_fake, interm_f_real_zero_sample[0],
                                                                    interm_f_real_zero_sample[1],
                                                                    interm_f_real_zero_sample[2]])  # discriminator loss (fake)
            d_loss = 0.5 * np.add(d_loss1, d_loss2)

            # add perceptual loss (E[|(discriminator output (y_real) - discriminator ouput (y_pred))|]
            _, interm_f1_real, interm_f2_real, interm_f3_real = d_model.predict_on_batch([x_real_g, y_real_g])
            # update the generator
            g_loss = cgan_model.train_on_batch([x_real_g, y_real_g],
                                               [c_real, y_real_g, interm_f1_real, interm_f2_real, interm_f3_real,
                                                y_real_g, y_real_g, y_real_g,
                                                y_real_g])  # output: weighted sum of adversarial loss and L1 loss, adversarial loss, L1 loss of a generator
        else:
            # update discriminator for real samples
            d_loss1 = d_model.test_on_batch([x_real_g, y_real_g], [c_real, interm_f_real_one_sample[0],
                                                                    interm_f_real_one_sample[1],
                                                                    interm_f_real_one_sample[2]]) # discriminator loss (real)

            # update discriminator for generated samples
            d_loss2 = d_model.test_on_batch([x_real_g, y_pred_g], [c_fake, interm_f_real_zero_sample[0],
                                                                    interm_f_real_zero_sample[1],
                                                                    interm_f_real_zero_sample[2]]) # discriminator loss (fake)
            d_loss = 0.5 * np.add(d_loss1, d_loss2)

            # add perceptual loss (E[|(discriminator output (y_real) - discriminator ouput (y_pred))|]
            _, interm_f1_real, interm_f2_real, interm_f3_real = d_model.predict_on_batch([x_real_g, y_real_g])
            # update the generator
            g_loss = cgan_model.test_on_batch([x_real_g, y_real_g],
                                               [c_real, y_real_g, interm_f1_real, interm_f2_real, interm_f3_real,
                                                y_real_g, y_real_g, y_real_g,
                                                y_real_g])  # output: weighted sum of adversarial loss and L1 loss, adversarial loss, L1 loss of a generator

    trained_model = [g_model, d_model, cgan_model]

    return d_loss, g_loss, trained_model


def fit_single_cyclegan_sar_model(g_patch_shape, d_patch_shape, d_output_shape, batch_size, model, trn_dataset_g,
                           trn_dataset_d, x_pool, y_pool, interm_f_real_one_sample, interm_f_real_zero_sample, is_train):

    [g_model_XtoY, g_model_YtoX, d_model_Y, d_model_X, c_model_XtoY, c_model_YtoX] = model

    [x_in_real_g, y_out_real_g], c_real = gen_sar_real_samples(trn_dataset_g, None, batch_size, d_output_shape)

    x_out_patch_shape = (np.prod(g_patch_shape), 1)
    y_in_patch_shape = (g_patch_shape[0], g_patch_shape[1], 1)

    x_out_real_g = np.reshape(x_in_real_g, (batch_size, ) + x_out_patch_shape)
    y_in_real_g = np.reshape(y_out_real_g, (batch_size, ) + y_in_patch_shape)

    # generate a batch of fake samples
    x_pred_g, c_real_x, c_fake_x = gen_sar_pred_samples_YtoX(g_model_YtoX, y_in_real_g, d_output_shape)
    y_pred_g, c_fake_y = gen_sar_pred_samples(g_model_XtoY, x_in_real_g, d_output_shape)

    # update fakes from pool
    x_pred_g = update_image_pool(x_pool, x_pred_g)
    y_pred_g = update_image_pool(y_pool, y_pred_g)

    if is_train == 1:

        # update generator Y->X via adversarial and cycle loss
        _, interm_f1_x_real, interm_f2_x_real, interm_f3_x_real = d_model_X.predict_on_batch([y_in_real_g, x_out_real_g])
        g_loss2 = c_model_YtoX.train_on_batch([y_in_real_g, x_in_real_g, y_out_real_g],
                                              [c_real_x, x_out_real_g, y_out_real_g, x_out_real_g, interm_f1_x_real,
                                               interm_f2_x_real, interm_f3_x_real, y_out_real_g, y_out_real_g,
                                               y_out_real_g, y_out_real_g])
        # update discriminator for X -> [real/fake]
        d_loss_x1 = d_model_X.train_on_batch([y_in_real_g, x_out_real_g], [c_real_x, interm_f_real_one_sample[0],
                                                                    interm_f_real_one_sample[1],
                                                                    interm_f_real_one_sample[2]])
        d_loss_x2 = d_model_X.train_on_batch([y_in_real_g, x_pred_g], [c_fake_x, interm_f_real_zero_sample[0],
                                                                    interm_f_real_zero_sample[1],
                                                                    interm_f_real_zero_sample[2]])
        d_loss_x = 0.5 * np.add(d_loss_x1, d_loss_x2)

        # update generator X->Y via adversarial and cycle loss
        _, interm_f1_y_real, interm_f2_y_real, interm_f3_y_real = d_model_Y.predict_on_batch([x_in_real_g, y_out_real_g])
        g_loss1 = c_model_XtoY.train_on_batch([x_in_real_g, y_in_real_g, y_out_real_g],
                                              [c_real, y_out_real_g, x_out_real_g, y_out_real_g, interm_f1_y_real,
                                               interm_f2_y_real, interm_f3_y_real, y_out_real_g, y_out_real_g,
                                               y_out_real_g, y_out_real_g])
        # update discriminator for Y -> [real/fake]
        d_loss_y1 = d_model_Y.train_on_batch([x_in_real_g, y_out_real_g], [c_real, interm_f_real_one_sample[0],
                                                                    interm_f_real_one_sample[1],
                                                                    interm_f_real_one_sample[2]])
        d_loss_y2 = d_model_Y.train_on_batch([x_in_real_g, y_pred_g], [c_fake_y, interm_f_real_zero_sample[0],
                                                                    interm_f_real_zero_sample[1],
                                                                    interm_f_real_zero_sample[2]])
        d_loss_y = 0.5 * np.add(d_loss_y1, d_loss_y2)

    else:

        # update generator Y->X via adversarial and cycle loss
        _, interm_f1_x_real, interm_f2_x_real, interm_f3_x_real = d_model_X.predict_on_batch([y_in_real_g, x_out_real_g])
        g_loss2 = c_model_YtoX.test_on_batch([y_in_real_g, x_in_real_g, y_out_real_g],
                                              [c_real_x, x_out_real_g, y_out_real_g, x_out_real_g, interm_f1_x_real,
                                               interm_f2_x_real, interm_f3_x_real, y_out_real_g, y_out_real_g,
                                               y_out_real_g, y_out_real_g])
        # update discriminator for X -> [real/fake]
        d_loss_x1 = d_model_X.test_on_batch([y_in_real_g, x_out_real_g], [c_real_x, interm_f_real_one_sample[0],
                                                                    interm_f_real_one_sample[1],
                                                                    interm_f_real_one_sample[2]])
        d_loss_x2 = d_model_X.test_on_batch([y_in_real_g, x_pred_g], [c_fake_x, interm_f_real_zero_sample[0],
                                                                    interm_f_real_zero_sample[1],
                                                                    interm_f_real_zero_sample[2]])
        d_loss_x = 0.5 * np.add(d_loss_x1, d_loss_x2)

        # update generator X->Y via adversarial and cycle loss
        _, interm_f1_y_real, interm_f2_y_real, interm_f3_y_real = d_model_Y.predict_on_batch([x_in_real_g, y_out_real_g])
        g_loss1 = c_model_XtoY.test_on_batch([x_in_real_g, y_in_real_g, y_out_real_g],
                                              [c_real, y_out_real_g, x_out_real_g, y_out_real_g, interm_f1_y_real,
                                               interm_f2_y_real, interm_f3_y_real, y_out_real_g, y_out_real_g,
                                               y_out_real_g, y_out_real_g])
        # update discriminator for Y -> [real/fake]
        d_loss_y1 = d_model_Y.test_on_batch([x_in_real_g, y_out_real_g], [c_real, interm_f_real_one_sample[0],
                                                                    interm_f_real_one_sample[1],
                                                                    interm_f_real_one_sample[2]])
        d_loss_y2 = d_model_Y.test_on_batch([x_in_real_g, y_pred_g], [c_fake_y, interm_f_real_zero_sample[0],
                                                                    interm_f_real_zero_sample[1],
                                                                    interm_f_real_zero_sample[2]])
        d_loss_y = 0.5 * np.add(d_loss_y1, d_loss_y2)


    trained_model = [g_model_XtoY, g_model_YtoX, d_model_Y, d_model_X, c_model_XtoY, c_model_YtoX]

    return d_loss_x, d_loss_y, g_loss2, g_loss1, trained_model


def fit_multi_cyclegan_sar_model(g_patch_shape, d_patch_shape, d_output_shape, batch_size, model, trn_dataset_g,
                           trn_dataset_d, x_pool, y_pool, interm_f_real_one_sample, interm_f_real_zero_sample, is_train):

    [g_model_XtoY, g_model_YtoX, d_model_Y, d_model_X, c_model_XtoY, c_model_YtoX] = model

    [x_in_real_g, y_out_real_g], c_real = gen_sar_real_samples(trn_dataset_g, None, batch_size, d_output_shape)

    # x_out_patch_shape = (np.prod(g_patch_shape), 3)
    # y_in_patch_shape = (g_patch_shape[0], g_patch_shape[1], 1)

    x_out_patch_shape = (np.prod(g_patch_shape), 2)
    y_in_patch_shape = (g_patch_shape[0], g_patch_shape[1], 2)

    # x_out_patch_shape = (np.prod(g_patch_shape), 5)
    # y_in_patch_shape = (g_patch_shape[0], g_patch_shape[1], 5)

    x_out_real_g = np.reshape(x_in_real_g, (batch_size, ) + x_out_patch_shape)
    y_in_real_g = np.reshape(y_out_real_g, (batch_size, ) + y_in_patch_shape)

    # generate a batch of fake samples
    x_pred_g, c_real_x, c_fake_x = gen_sar_pred_samples_YtoX(g_model_YtoX, y_in_real_g, d_output_shape)
    y_pred_g, c_fake_y = gen_sar_pred_samples(g_model_XtoY, x_in_real_g, d_output_shape)

    # update fakes from pool
    x_pred_g = update_image_pool(x_pool, x_pred_g)
    y_pred_g = update_image_pool(y_pool, y_pred_g)

    if is_train == 1:

        # update generator Y->X via adversarial and cycle loss
        #_, interm_f1_x_real, interm_f2_x_real, interm_f3_x_real = d_model_X.predict_on_batch([y_in_real_g, x_out_real_g])
        _, interm_f1_x_real, interm_f2_x_real, interm_f3_x_real = d_model_X.predict_on_batch(x_out_real_g)
        g_loss2 = c_model_YtoX.train_on_batch([y_in_real_g, x_in_real_g, y_out_real_g],
                                              [c_real_x, x_out_real_g, y_out_real_g, x_out_real_g, interm_f1_x_real,
                                               interm_f2_x_real, interm_f3_x_real, y_out_real_g, y_out_real_g,
                                               y_out_real_g, y_out_real_g, y_out_real_g])
        # update discriminator for X -> [real/fake]
        # d_loss_x1 = d_model_X.train_on_batch([y_in_real_g, x_out_real_g], [c_real_x, interm_f_real_one_sample[0],
        #                                                             interm_f_real_one_sample[1],
        #                                                             interm_f_real_one_sample[2]])
        # d_loss_x2 = d_model_X.train_on_batch([y_in_real_g, x_pred_g], [c_fake_x, interm_f_real_zero_sample[0],
        #                                                             interm_f_real_zero_sample[1],
        #                                                             interm_f_real_zero_sample[2]])
        d_loss_x1 = d_model_X.train_on_batch(x_out_real_g, [c_real_x, interm_f_real_one_sample[0],
                                                                           interm_f_real_one_sample[1],
                                                                           interm_f_real_one_sample[2]])
        d_loss_x2 = d_model_X.train_on_batch(x_pred_g, [c_fake_x, interm_f_real_zero_sample[0],
                                                                       interm_f_real_zero_sample[1],
                                                                       interm_f_real_zero_sample[2]])
        d_loss_x = 0.5 * np.add(d_loss_x1, d_loss_x2)

        # update generator X->Y via adversarial and cycle loss
        #_, interm_f1_y_real, interm_f2_y_real, interm_f3_y_real = d_model_Y.predict_on_batch([x_in_real_g, y_out_real_g])
        _, interm_f1_y_real, interm_f2_y_real, interm_f3_y_real = d_model_Y.predict_on_batch(y_out_real_g)
        g_loss1 = c_model_XtoY.train_on_batch([x_in_real_g, y_in_real_g, y_out_real_g],
                                              [c_real, y_out_real_g, x_out_real_g, y_out_real_g, interm_f1_y_real,
                                               interm_f2_y_real, interm_f3_y_real, y_out_real_g, y_out_real_g,
                                               y_out_real_g, y_out_real_g, y_out_real_g])
        # update discriminator for Y -> [real/fake]
        # d_loss_y1 = d_model_Y.train_on_batch([x_in_real_g, y_out_real_g], [c_real, interm_f_real_one_sample[0],
        #                                                             interm_f_real_one_sample[1],
        #                                                             interm_f_real_one_sample[2]])
        # d_loss_y2 = d_model_Y.train_on_batch([x_in_real_g, y_pred_g], [c_fake_y, interm_f_real_zero_sample[0],
        #                                                             interm_f_real_zero_sample[1],
        #                                                             interm_f_real_zero_sample[2]])
        d_loss_y1 = d_model_Y.train_on_batch(y_out_real_g, [c_real, interm_f_real_one_sample[0],
                                                                    interm_f_real_one_sample[1],
                                                                    interm_f_real_one_sample[2]])
        d_loss_y2 = d_model_Y.train_on_batch(y_pred_g, [c_fake_y, interm_f_real_zero_sample[0],
                                                                    interm_f_real_zero_sample[1],
                                                                    interm_f_real_zero_sample[2]])
        d_loss_y = 0.5 * np.add(d_loss_y1, d_loss_y2)

    else:

        # update generator Y->X via adversarial and cycle loss
        #_, interm_f1_x_real, interm_f2_x_real, interm_f3_x_real = d_model_X.predict_on_batch([y_in_real_g, x_out_real_g])
        _, interm_f1_x_real, interm_f2_x_real, interm_f3_x_real = d_model_X.predict_on_batch(x_out_real_g)
        g_loss2 = c_model_YtoX.test_on_batch([y_in_real_g, x_in_real_g, y_out_real_g],
                                              [c_real_x, x_out_real_g, y_out_real_g, x_out_real_g, interm_f1_x_real,
                                               interm_f2_x_real, interm_f3_x_real, y_out_real_g, y_out_real_g,
                                               y_out_real_g, y_out_real_g, y_out_real_g])
        # update discriminator for X -> [real/fake]
        # d_loss_x1 = d_model_X.test_on_batch([y_in_real_g, x_out_real_g], [c_real_x, interm_f_real_one_sample[0],
        #                                                             interm_f_real_one_sample[1],
        #                                                             interm_f_real_one_sample[2]])
        # d_loss_x2 = d_model_X.test_on_batch([y_in_real_g, x_pred_g], [c_fake_x, interm_f_real_zero_sample[0],
        #                                                             interm_f_real_zero_sample[1],
        #                                                             interm_f_real_zero_sample[2]])
        d_loss_x1 = d_model_X.test_on_batch(x_out_real_g, [c_real_x, interm_f_real_one_sample[0],
                                                                    interm_f_real_one_sample[1],
                                                                    interm_f_real_one_sample[2]])
        d_loss_x2 = d_model_X.test_on_batch(x_pred_g, [c_fake_x, interm_f_real_zero_sample[0],
                                                                    interm_f_real_zero_sample[1],
                                                                    interm_f_real_zero_sample[2]])
        d_loss_x = 0.5 * np.add(d_loss_x1, d_loss_x2)

        # update generator X->Y via adversarial and cycle loss
        #_, interm_f1_y_real, interm_f2_y_real, interm_f3_y_real = d_model_Y.predict_on_batch([x_in_real_g, y_out_real_g])
        _, interm_f1_y_real, interm_f2_y_real, interm_f3_y_real = d_model_Y.predict_on_batch(y_out_real_g)
        g_loss1 = c_model_XtoY.test_on_batch([x_in_real_g, y_in_real_g, y_out_real_g],
                                              [c_real, y_out_real_g, x_out_real_g, y_out_real_g, interm_f1_y_real,
                                               interm_f2_y_real, interm_f3_y_real, y_out_real_g, y_out_real_g,
                                               y_out_real_g, y_out_real_g, y_out_real_g])
        # update discriminator for Y -> [real/fake]
        # d_loss_y1 = d_model_Y.test_on_batch([x_in_real_g, y_out_real_g], [c_real, interm_f_real_one_sample[0],
        #                                                             interm_f_real_one_sample[1],
        #                                                             interm_f_real_one_sample[2]])
        # d_loss_y2 = d_model_Y.test_on_batch([x_in_real_g, y_pred_g], [c_fake_y, interm_f_real_zero_sample[0],
        #                                                             interm_f_real_zero_sample[1],
        #                                                             interm_f_real_zero_sample[2]])
        d_loss_y1 = d_model_Y.test_on_batch(y_out_real_g, [c_real, interm_f_real_one_sample[0],
                                                                    interm_f_real_one_sample[1],
                                                                    interm_f_real_one_sample[2]])
        d_loss_y2 = d_model_Y.test_on_batch(y_pred_g, [c_fake_y, interm_f_real_zero_sample[0],
                                                                    interm_f_real_zero_sample[1],
                                                                    interm_f_real_zero_sample[2]])
        d_loss_y = 0.5 * np.add(d_loss_y1, d_loss_y2)


    trained_model = [g_model_XtoY, g_model_YtoX, d_model_Y, d_model_X, c_model_XtoY, c_model_YtoX]

    return d_loss_x, d_loss_y, g_loss2, g_loss1, trained_model



def fit_cyclegan_sar_model2(g_patch_shape, d_patch_shape, d_output_shape, batch_size, model, trn_dataset_g,
                           trn_dataset_d, x_pool, y_pool, interm_f_real_one_sample, interm_f_real_zero_sample, is_train):

    [g_model_XtoY, g_model_YtoX, d_model_Y, d_model_X, c_model] = model

    [x_in_real_g, y_out_real_g], c_real = gen_sar_real_samples(trn_dataset_g, None, batch_size, d_output_shape)

    #x_out_patch_shape = (np.prod(g_patch_shape), 3)
    x_out_patch_shape = (np.prod(g_patch_shape), 1)
    y_in_patch_shape = (g_patch_shape[0], g_patch_shape[1], 1)

    x_out_real_g = np.reshape(x_in_real_g, (batch_size, ) + x_out_patch_shape)
    y_in_real_g = np.reshape(y_out_real_g, (batch_size, ) + y_in_patch_shape)

    # generate a batch of fake samples
    x_pred_g, c_real_x, c_fake_x = gen_sar_pred_samples_YtoX(g_model_YtoX, y_in_real_g, d_output_shape)
    y_pred_g, c_fake_y = gen_sar_pred_samples(g_model_XtoY, x_in_real_g, d_output_shape)

    # update fakes from pool
    x_pred_g = update_image_pool(x_pool, x_pred_g)
    y_pred_g = update_image_pool(y_pool, y_pred_g)

    if is_train == 1:

        # update discriminator for Y -> [real/fake]
        d_loss_y1 = d_model_Y.train_on_batch([x_in_real_g, y_out_real_g], [c_real, interm_f_real_one_sample[0],
                                                                    interm_f_real_one_sample[1],
                                                                    interm_f_real_one_sample[2]])
        d_loss_y2 = d_model_Y.train_on_batch([x_in_real_g, y_pred_g], [c_fake_y, interm_f_real_zero_sample[0],
                                                                    interm_f_real_zero_sample[1],
                                                                    interm_f_real_zero_sample[2]])
        d_loss_y = 0.5 * np.add(d_loss_y1, d_loss_y2)

        # update discriminator for X -> [real/fake]
        d_loss_x1 = d_model_X.train_on_batch([y_in_real_g, x_out_real_g], [c_real_x, interm_f_real_one_sample[0],
                                                                    interm_f_real_one_sample[1],
                                                                    interm_f_real_one_sample[2]])
        d_loss_x2 = d_model_X.train_on_batch([y_in_real_g, x_pred_g], [c_fake_x, interm_f_real_zero_sample[0],
                                                                    interm_f_real_zero_sample[1],
                                                                    interm_f_real_zero_sample[2]])
        d_loss_x = 0.5 * np.add(d_loss_x1, d_loss_x2)

        # update generator X->Y via adversarial and cycle loss
        _, interm_f1_y_real, interm_f2_y_real, interm_f3_y_real = d_model_Y.predict_on_batch([x_in_real_g, y_out_real_g])

        # update generator Y->X via adversarial and cycle loss
        _, interm_f1_x_real, interm_f2_x_real, interm_f3_x_real = d_model_X.predict_on_batch([y_in_real_g, x_out_real_g])

        g_loss = c_model.train_on_batch([x_in_real_g, y_in_real_g, y_out_real_g],
                                              [c_real, c_real, y_out_real_g, x_out_real_g, x_out_real_g, y_out_real_g,
                                               interm_f1_y_real, interm_f2_y_real, interm_f3_y_real, interm_f1_x_real,
                                               interm_f2_x_real, interm_f3_x_real, y_out_real_g, y_out_real_g,
                                               y_out_real_g, y_out_real_g])

    else:

        # update discriminator for Y -> [real/fake]
        d_loss_y1 = d_model_Y.test_on_batch([x_in_real_g, y_out_real_g], [c_real, interm_f_real_one_sample[0],
                                                                    interm_f_real_one_sample[1],
                                                                    interm_f_real_one_sample[2]])
        d_loss_y2 = d_model_Y.test_on_batch([x_in_real_g, y_pred_g], [c_fake_y, interm_f_real_zero_sample[0],
                                                                    interm_f_real_zero_sample[1],
                                                                    interm_f_real_zero_sample[2]])
        d_loss_y = 0.5 * np.add(d_loss_y1, d_loss_y2)


        # update discriminator for X -> [real/fake]
        d_loss_x1 = d_model_X.test_on_batch([y_in_real_g, x_out_real_g], [c_real_x, interm_f_real_one_sample[0],
                                                                    interm_f_real_one_sample[1],
                                                                    interm_f_real_one_sample[2]])
        d_loss_x2 = d_model_X.test_on_batch([y_in_real_g, x_pred_g], [c_fake_x, interm_f_real_zero_sample[0],
                                                                    interm_f_real_zero_sample[1],
                                                                    interm_f_real_zero_sample[2]])
        d_loss_x = 0.5 * np.add(d_loss_x1, d_loss_x2)

        # update generator X->Y via adversarial and cycle loss
        _, interm_f1_y_real, interm_f2_y_real, interm_f3_y_real = d_model_Y.predict_on_batch([x_in_real_g, y_out_real_g])

        # update generator Y->X via adversarial and cycle loss
        _, interm_f1_x_real, interm_f2_x_real, interm_f3_x_real = d_model_X.predict_on_batch([y_in_real_g, x_out_real_g])

        g_loss = c_model.test_on_batch([x_in_real_g, y_in_real_g, y_out_real_g],
                                              [c_real, c_real, y_out_real_g, x_out_real_g, x_out_real_g, y_out_real_g,
                                               interm_f1_y_real, interm_f2_y_real, interm_f3_y_real, interm_f1_x_real,
                                               interm_f2_x_real, interm_f3_x_real, y_out_real_g, y_out_real_g,
                                               y_out_real_g, y_out_real_g])


    trained_model = [g_model_XtoY, g_model_YtoX, d_model_Y, d_model_X, c_model]

    return d_loss_x, d_loss_y, g_loss, trained_model


# select a batch of random samples, returns images and target
def gen_real_samples(dataset, batch_size, class_patch_shape):

    # unpack dataset
    [x_train, y_train] = dataset
    # choose random instances
    ix = np.random.randint(0, x_train.shape[0], batch_size)
    # retrieve selected images
    X1 = x_train[ix, :, :, :, :]
    X2 = [y_train[0][ix, :, :], y_train[1][ix, :, :], y_train[2][ix, :, :], y_train[3][ix, :]]
    # generate 'real' class labels (1)
    y = np.ones((batch_size, np.prod(class_patch_shape), 1))
    return [X1, X2], y


def gen_sar_real_samples(dataset_g, dataset_d, batch_size, class_patch_shape):

    # unpack dataset
    [x_train_g, y_train_g] = dataset_g
    y = np.ones((batch_size, np.prod(class_patch_shape), 1))
    if dataset_d is not None:
        [x_train_d, y_train_d] = dataset_d
        ix = np.random.randint(0, min(x_train_g.shape[0], x_train_d.shape[0]), batch_size)
        X1 = x_train_g[ix]
        X2 = y_train_g[ix]
        X3 = x_train_d[ix]
        X4 = y_train_d[ix]

        return [X1, X2, X3, X4], y
    else:
        ix = np.random.randint(0, x_train_g.shape[0], batch_size)
        X1 = x_train_g[ix]
        X2 = y_train_g[ix]

        return [X1, X2], y


# generate a batch of images, returns images and targets
def gen_pred_samples(g_model, x_real, class_patch_shape):
    # generate fake instance
    y_pred = g_model.predict_on_batch(x_real)
    # create 'fake' class labels (0)
    c_fake = np.zeros((len(y_pred[0]), np.prod(class_patch_shape), 1))
    return [y_pred[0], y_pred[1]], c_fake


def gen_sar_pred_samples(g_model, x_real, class_patch_shape):
    # generate fake instance
    y_pred = g_model.predict_on_batch(x_real)
    # create 'fake' class labels (0)
    c_fake = np.zeros((len(y_pred), np.prod(class_patch_shape), 1))
    return y_pred, c_fake


def gen_sar_pred_samples_YtoX(g_model, y_real, class_patch_shape):
    # generate fake instance
    x_pred = g_model.predict_on_batch(y_real)
    # create 'fake' class labels (0)
    c_real = np.ones((len(x_pred), np.prod(class_patch_shape), 1))
    c_fake = np.zeros((len(x_pred), np.prod(class_patch_shape), 1))
    return x_pred, c_real, c_fake


def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif np.random.random() < 0.5:
            # use image, but don't add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            ix = np.random.randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return np.asarray(selected)


def load_saved_model_filename(gen_conf, train_conf, case_name):
    root_path = gen_conf['root_path']
    model_path = gen_conf['model_path']
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    num_classes = gen_conf['num_classes']
    folder_names = dataset_info['folder_names']
    mode = gen_conf['validation_mode']
    multi_output = gen_conf['multi_output']
    approach = train_conf['approach']
    loss = train_conf['loss']
    dimension = train_conf['dimension']
    patch_shape = train_conf['patch_shape']
    extraction_step = train_conf['extraction_step']
    data_augment = train_conf['data_augment']
    preprocess_trn = train_conf['preprocess']

    if data_augment == 1:
        data_augment_label = 'mixup'
    elif data_augment == 2:
        data_augment_label = 'datagen'
    elif data_augment == 3:
        data_augment_label = 'mixup+datagen'
    else:
        data_augment_label = ''

    if multi_output == 1:
        loss = loss[0] + '_' + loss[1]
    model_filename = generate_output_filename(root_path + model_path, dataset + '/' + folder_names[0], 'mode_'+ mode,
                                              case_name, approach, loss, 'dim_' + str(dimension), 'n_classes_' +
                                              str(num_classes), str(patch_shape), str(extraction_step),
                                              data_augment_label, 'preproc_trn_opt_' + str(preprocess_trn), 'h5')

    return model_filename


def read_model(gen_conf, train_conf, case_name):

    model = generate_model(gen_conf, train_conf)
    model_filename = load_saved_model_filename(gen_conf, train_conf, case_name)
    model.load_weights(model_filename)

    return model


def save_gan_model(gen_conf, train_conf, case_name, model):

    approach = train_conf['approach']
    root_path = gen_conf['root_path']
    model_path = gen_conf['model_path']
    mode = gen_conf['validation_mode']
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    folder_names = dataset_info['folder_names']

    gan_model_path = os.path.join(root_path, model_path, dataset, folder_names[0])
    if not os.path.exists(gan_model_path):
        os.makedirs(gan_model_path)
    prefix = 'mode_'+ str(mode) + '-' + str(case_name) + '_'

    if approach in ['cgan', 'cgan_sar', 'single_cyclegan_sar', 'multi_cyclegan_sar', 'composite_cyclegan_sar']:
        [generator, discriminator, cgan] = model
        discriminator.trainable = False
        cgan.save(os.path.join(gan_model_path, prefix + 'gan.h5'))
        discriminator.trainable = True
        generator.save(os.path.join(gan_model_path, prefix + 'generator.h5'))
        discriminator.save(os.path.join(gan_model_path, prefix + 'discriminator.h5'))
    else:
        raise NotImplementedError('choose dilated_densenet or patch_gan')


def load_gan_model(gen_conf, train_conf, case_name):

    approach = train_conf['approach']
    root_path = gen_conf['root_path']
    model_path = gen_conf['model_path']
    mode = gen_conf['validation_mode']
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    folder_names = dataset_info['folder_names']

    gan_model_path = os.path.join(root_path, model_path, dataset, folder_names[0])
    prefix = 'mode_'+ str(mode) + '-' + str(case_name) + '_'

    # discriminator = load_model(os.path.join(file_path, prefix + 'discriminator.h5'))
    # generator = load_model(os.path.join(file_path, prefix + 'generator.h5'))
    # gan = load_model(os.path.join(file_path, prefix + 'gan.h5'))
    # gan.summary()
    # discriminator.summary()
    # generator.summary()

    g_model, d_model, gan_model = [], [], []
    trn_model = generate_model(gen_conf, train_conf)
    if approach in ['cgan', 'cgan_sar']:
        g_model, d_model, gan_model = trn_model[0], trn_model[1], trn_model[2]
    elif approach in ['single_cyclegan_sar', 'multi_cyclegan_sar', 'composite_cyclegan_sar']:
        g_model, d_model, gan_model = trn_model[0], trn_model[2], trn_model[4]

    g_model.load_weights(os.path.join(gan_model_path, prefix + 'generator.h5'))
    d_model.load_weights(os.path.join(gan_model_path, prefix + 'discriminator.h5'))
    gan_model.load_weights(os.path.join(gan_model_path, prefix + 'gan.h5'))

    return g_model, d_model, gan_model


def test_model(gen_conf, test_conf, x_test, model):
    num_classes = gen_conf['num_classes']
    output_shape = test_conf['output_shape']

    pred = model.predict(x_test, verbose=1)
    pred = pred.reshape((len(pred),) + output_shape + (num_classes,))

    return reconstruct_volume(gen_conf, test_conf, pred)


def test_model_modified(gen_conf, train_conf, test_conf, x_test, model):
    exclusive_train = train_conf['exclusive_train']
    attention_loss = train_conf['attention_loss']
    num_classes = gen_conf['num_classes']
    multi_output = gen_conf['multi_output']
    output_shape = test_conf['output_shape']

    if exclusive_train == 1:
        num_classes -= 1

    if multi_output == 1:
        pred = model.predict(x_test, verbose=1)
        pred_multi = []
        for pred_i, n_class in zip(pred[:len(num_classes)], num_classes):
            pred_multi.append(pred_i.reshape((len(pred_i),) + output_shape + (n_class,)))
        pred_recon = reconstruct_volume_modified(gen_conf, train_conf, test_conf, pred_multi, num_classes)
    else:
        if attention_loss == 1:
            pred = model.predict(x_test, verbose=1)
            pred = pred[0].reshape((len(pred[0]),) + output_shape + (num_classes,))
            pred_recon = reconstruct_volume_modified(gen_conf, train_conf, test_conf, pred, num_classes)
        else:
            pred = model.predict(x_test, verbose=1)
            pred = pred.reshape((len(pred),) + output_shape + (num_classes,))
            pred_recon = reconstruct_volume_modified(gen_conf, train_conf, test_conf, pred, num_classes)
    return pred_recon


def test_model_sar(gen_conf, train_conf, test_conf, x_test, model):

    num_classes = gen_conf['num_classes']
    output_shape = test_conf['output_shape']

    pred = model.predict(x_test, verbose=1)
    pred = pred.reshape((len(pred),) + output_shape + (num_classes,))
    pred_recon = reconstruct_volume_sar(gen_conf, train_conf, test_conf, pred, num_classes)

    return pred_recon


def test_model_sar_2_5D(gen_conf, test_conf, x_test, model, dim_label):

    num_classes = gen_conf['num_classes']
    output_shape = test_conf['GAN']['generator']['output_shape']

    if dim_label == 'axial':
        output_shape = (output_shape[0], output_shape[1])
    elif dim_label == 'sagittal':
        output_shape = (output_shape[0], output_shape[2])
    elif dim_label == 'coronal':
        output_shape = (output_shape[1], output_shape[2])

    x_test = np.transpose(x_test, [0, 2, 3, 1])
    pred = model.predict(x_test, verbose=1)
    pred = pred.reshape((len(pred), output_shape[0], output_shape[1], num_classes))

    return pred


def inference(gen_conf, train_conf, test_conf, test_vol, trained_model):
    dataset = test_conf['dataset']
    dimension = test_conf['dimension']
    patch_shape = test_conf['patch_shape']
    output_shape = test_conf['output_shape']
    extraction_step = test_conf['extraction_step']
    dataset_info = gen_conf['dataset_info'][dataset]
    approach = train_conf['approach']
    multi_output = gen_conf['multi_output']
    bg_value = train_conf['bg_value']

    # if test_vol_size[1] < output_shape[0] or test_vol_size[2] < output_shape[1] or test_vol_size[3] \
    #         < output_shape[2]:
    #     output_shape_real = test_vol_size[1:4]
    # else:
    #     output_shape_real = output_shape

    test_vol_org_size = test_vol[0].shape
    print(test_vol_org_size)

    pad_size_total = np.zeros(np.shape(output_shape)).astype(int)
    for dim in range(dimension):
        if test_vol_org_size[dim+1] < output_shape[dim]:
            pad_size_total[dim] = np.ceil((output_shape[dim] - test_vol_org_size[dim+1]) / 2)

    # add zeros with same size to both side
    #if patch_shape != output_shape_real:
    if np.sum(pad_size_total) != 0:
        pad_size = ()
        for dim in range(dimension):
            pad_size += (pad_size_total[dim],)
        test_vol_pad_org = pad_both_sides(dimension, test_vol[0], pad_size, bg_value)
    else:
        test_vol_pad_org = test_vol[0]
    print(test_vol_pad_org.shape)

    # To avoid empty regions (which is not processed) around the edge of input (prob) image
    test_vol_pad_org_size = test_vol_pad_org.shape
    extra_pad_size = ()
    extra_pad_value = np.zeros(np.shape(output_shape)).astype(int)
    for dim in range(dimension):
        extra_pad_value[dim] = np.ceil((output_shape[dim] + extraction_step[dim] *
                          np.ceil((test_vol_pad_org_size[dim + 1] - output_shape[dim]) / extraction_step[dim]) -
                          test_vol_pad_org_size[dim + 1]) / 2)
        extra_pad_size += (extra_pad_value[dim], )
        pad_size_total[dim] += extra_pad_value[dim]
    test_vol_pad_extra = pad_both_sides(dimension, test_vol_pad_org, extra_pad_size, bg_value)
    print(test_vol_pad_extra.shape)

    # only for building test patches
    tst_data_pad_size = ()
    for dim in range(dimension):
        tst_data_pad_size += ((patch_shape[dim] - output_shape[dim]) // 2, )
    test_vol_pad = pad_both_sides(dimension, test_vol_pad_extra, tst_data_pad_size, bg_value)
    test_vol_size = test_vol_pad.shape
    print(test_vol_size)

    #test_vol_crop_array = np.zeros((1, num_modality) + test_vol_pad.shape[1:4])
    test_vol_crop_array = np.zeros((1, test_vol_pad.shape[0]) + test_vol_pad.shape[1:4])
    test_vol_crop_array[0] = test_vol_pad
    print(test_vol_crop_array[0].shape)

    x_test = build_testing_set(gen_conf, test_conf, test_vol_crop_array)

    #gen_conf['dataset_info'][dataset]['size'] = test_vol_crop_array[0].shape[1:4]  # output image size
    gen_conf['dataset_info'][dataset]['size'] = test_vol_pad_extra.shape[1:4]  # output image size
    # rec_vol = test_model(gen_conf, train_conf, x_test, model)
    rec_vol, prob_vol = test_model_modified(gen_conf, train_conf, test_conf, x_test, trained_model)
    if multi_output == 1:
        for r in rec_vol:
            print(r.shape)
        for p in prob_vol:
            print(p.shape)
    else:
        print(rec_vol.shape)
        print(prob_vol.shape)

    # re-crop zero-padded vol
    if np.sum(pad_size_total) != 0:
    #if patch_shape != output_shape_real:
    # org_size = [test_vol_size[1], test_vol_size[2], test_vol_size[3]]
        start_ind = np.zeros(dimension).astype(int)
        end_ind = np.zeros(dimension).astype(int)
        for dim in range(dimension):
            if pad_size_total[dim] != 0:
                start_ind[dim] = pad_size_total[dim]
                end_ind[dim] = pad_size_total[dim] + test_vol_org_size[dim + 1]
            else:
                start_ind[dim] = 0
                end_ind[dim] = test_vol_org_size[dim + 1]

        if multi_output == 1:
            rec_vol_crop, prob_vol_crop = [], []
            for rec, prob in zip(rec_vol, prob_vol):
                rec_vol_crop.append(rec[start_ind[0]:end_ind[0], start_ind[1]:end_ind[1], start_ind[2]:end_ind[2]])
                prob_vol_crop.append(prob[start_ind[0]:end_ind[0], start_ind[1]:end_ind[1], start_ind[2]:end_ind[2]])
        else:
            rec_vol_crop = rec_vol[start_ind[0]:end_ind[0], start_ind[1]:end_ind[1], start_ind[2]:end_ind[2]]
            prob_vol_crop = prob_vol[start_ind[0]:end_ind[0], start_ind[1]:end_ind[1], start_ind[2]:end_ind[2]]

        # rec_vol_crop = rec_vol[org_size[0]:org_size[0] * 2, org_size[1]:org_size[1] * 2, org_size[2]:org_size[2] * 2]
        # prob_vol_crop = prob_vol[org_size[0]:org_size[0] * 2, org_size[1]:org_size[1] * 2,
        #                 org_size[2]:org_size[2] * 2]
    else:
        rec_vol_crop = rec_vol
        prob_vol_crop = prob_vol
    if multi_output == 1:
        for rc in rec_vol_crop:
            print(rc.shape)
        for pc in prob_vol_crop:
            print(pc.shape)
    else:
        print(rec_vol_crop.shape)
        print(prob_vol_crop.shape)

    return rec_vol_crop, prob_vol_crop, x_test


def inference_sar(gen_conf, train_conf, test_conf, test_vol, trained_model):
    dataset = test_conf['dataset']
    dimension = test_conf['dimension']
    patch_shape = test_conf['patch_shape']
    output_shape = test_conf['output_shape']
    extraction_step = test_conf['extraction_step']
    bg_value = train_conf['bg_value']

    test_vol_org_size = test_vol[0].shape
    print(test_vol_org_size)

    pad_size_total = np.zeros(np.shape(output_shape)).astype(int)
    for dim in range(dimension):
        if test_vol_org_size[dim+1] < output_shape[dim]:
            pad_size_total[dim] = np.ceil((output_shape[dim] - test_vol_org_size[dim+1]) / 2)

    # add zeros with same size to both side
    if np.sum(pad_size_total) != 0:
        pad_size = ()
        for dim in range(dimension):
            pad_size += (pad_size_total[dim],)
        test_vol_pad_org = pad_both_sides(dimension, test_vol[0], pad_size, bg_value)
    else:
        test_vol_pad_org = test_vol[0]
    print(test_vol_pad_org.shape)

    # To avoid empty regions (which is not processed) around the edge of input (prob) image
    test_vol_pad_org_size = test_vol_pad_org.shape
    extra_pad_size = ()
    extra_pad_value = np.zeros(np.shape(output_shape)).astype(int)
    for dim in range(dimension):
        extra_pad_value[dim] = np.ceil((output_shape[dim] + extraction_step[dim] *
                          np.ceil((test_vol_pad_org_size[dim + 1] - output_shape[dim]) / extraction_step[dim]) -
                          test_vol_pad_org_size[dim + 1]) / 2)
        extra_pad_size += (extra_pad_value[dim], )
        pad_size_total[dim] += extra_pad_value[dim]
    test_vol_pad_extra = pad_both_sides(dimension, test_vol_pad_org, extra_pad_size, bg_value)
    print(test_vol_pad_extra.shape)

    # only for building test patches
    tst_data_pad_size = ()
    for dim in range(dimension):
        tst_data_pad_size += ((patch_shape[dim] - output_shape[dim]) // 2, )
    test_vol_pad = pad_both_sides(dimension, test_vol_pad_extra, tst_data_pad_size, bg_value)
    test_vol_size = test_vol_pad.shape
    print(test_vol_size)

    test_vol_crop_array = np.zeros((1, test_vol_pad.shape[0]) + test_vol_pad.shape[1:4])
    test_vol_crop_array[0] = test_vol_pad
    print(test_vol_crop_array[0].shape)

    x_test = build_testing_set(gen_conf, test_conf, test_vol_crop_array)

    gen_conf['dataset_info'][dataset]['size'] = test_vol_pad_extra.shape[1:4]  # output image size
    rec_vol, [prob_vol, ovr_vol] = test_model_sar(gen_conf, train_conf, test_conf, x_test, trained_model)

    print(rec_vol.shape)
    print(prob_vol.shape)

    # re-crop zero-padded vol
    if np.sum(pad_size_total) != 0:
        start_ind = np.zeros(dimension).astype(int)
        end_ind = np.zeros(dimension).astype(int)
        for dim in range(dimension):
            if pad_size_total[dim] != 0:
                start_ind[dim] = pad_size_total[dim]
                end_ind[dim] = pad_size_total[dim] + test_vol_org_size[dim + 1]
            else:
                start_ind[dim] = 0
                end_ind[dim] = test_vol_org_size[dim + 1]

        rec_vol_crop = rec_vol[start_ind[0]:end_ind[0], start_ind[1]:end_ind[1], start_ind[2]:end_ind[2]]
        prob_vol_crop = prob_vol[start_ind[0]:end_ind[0], start_ind[1]:end_ind[1], start_ind[2]:end_ind[2]]
        ovr_vol_crop = ovr_vol[start_ind[0]:end_ind[0], start_ind[1]:end_ind[1], start_ind[2]:end_ind[2]]

    else:
        rec_vol_crop = rec_vol
        prob_vol_crop = prob_vol
        ovr_vol_crop = ovr_vol

    # sar_pred = normalize_image(rec_vol_crop[:, :, :, 0], [0, 1])
    # sar_prob = normalize_image(prob_vol_crop[:, :, :, 0], [0, 1])

    sar_pred = rec_vol_crop[:, :, :, 0]
    sar_prob = prob_vol_crop[:, :, :, 0]
    ovr_pat = ovr_vol_crop[:, :, :, 0]

    print(sar_pred.shape)
    print(sar_prob.shape)
    print(ovr_pat.shape)

    return sar_pred, sar_prob, ovr_pat, x_test


def inference_sar_2_5D(gen_conf, test_conf, test_vol, trained_model, weights, dim_num, axis_num, trn_dim):
    dataset = test_conf['dataset']
    output_shape = test_conf['GAN']['generator']['output_shape']

    print(test_vol.shape)
    gen_conf['dataset_info'][dataset]['size'] = test_vol[0].shape[1:4]  # output image size

    test_data_total = build_testing_set_2_5D(gen_conf, test_conf, test_vol)
    ensemble = np.zeros(output_shape)
    vol_out_total = []

    test_data_total = [test_data_total[idx] for idx in dim_num]

    for test, model, k, w, dim_label in zip(test_data_total, trained_model, axis_num, weights, trn_dim):
    #for test, model, k, dim_label in zip([test_data_total[0]], [trained_model[0]], [1], ['coronal']):
        pred_vol = test_model_sar_2_5D(gen_conf, test_conf, test, model, dim_label)
        print('processing %s: %s' %(dim_label,  pred_vol.shape))
        vol_out = np.zeros(output_shape)
        for n in range(vol_out.shape[k]):
            if k == 2:
                vol_out[:, :, n] = pred_vol[n, :, :, 0]
            elif k == 1:
                vol_out[:, n, :] = pred_vol[n, :, :, 0]
            elif k == 0:
                vol_out[n, :, :] = pred_vol[n, :, :, 0]
        vol_out_total.append(vol_out)
        ensemble += np.multiply(vol_out, w)

    return ensemble, vol_out_total


def measure(gen_conf, train_conf, test_conf, idx):
    dataset = train_conf['dataset']
    approach = train_conf['approach']
    patch_shape = train_conf['patch_shape']
    extraction_step = train_conf['extraction_step']
    root_path = gen_conf['root_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    results_path = gen_conf['results_path']
    path = dataset_info['path']
    folder_names= dataset_info['folder_names']
    data_augment = train_conf['data_augment']
    preprocess_trn = train_conf['preprocess']
    preprocess_tst = test_conf['preprocess']

    if data_augment == 1:
        data_augment_label = '_mixup'
    elif data_augment == 2:
        data_augment_label = '_datagen'
    elif data_augment == 3:
        data_augment_label = '_mixup+datagen'
    else:
        data_augment_label = ''

    DC = []
    if dataset.find("3T7T") != -1:
        pattern = dataset_info['general_pattern']

        seg_filename = root_path + results_path + path + pattern[1].format(folder_names[0], str(idx) + '_' +
                                                                           approach + '_' + str(patch_shape) + '_' +
                                                                           str(extraction_step) + data_augment_label +
                                                                           '_preproc_trn_opt_' + str(preprocess_trn) +
                                                                           '_preproc_tst_opt_' + str(preprocess_tst))

        gt_filename = root_path + dataset_path + path + pattern[1].format(folder_names[2], str(idx))

        print(seg_filename)
        print(gt_filename)

        if os.path.exists(seg_filename) and os.path.exists(gt_filename):
            seg_label_vol = read_volume(seg_filename)
            gt_label_vol = read_volume(gt_filename)

            label_mapper = {0: 10, 1: 150, 2: 250} # 0 : CSF, 1: GM, 2: WM
            DC = computeDice(seg_label_vol, gt_label_vol, label_mapper)
        else:
            DC = []

    elif dataset.find("CBCT") != -1:
        pattern = dataset_info['general_pattern']

        file_path = root_path + dataset_path + path + folder_names[0]
        lb1_file_lst = glob.glob(file_path + '/' + '*mandible.nii.gz')  # mandible label
        lb2_file_lst = glob.glob(file_path + '/' + '*midface.nii.gz')  # midface label
        lb1_file_lst.sort()
        lb2_file_lst.sort()

        seg_filename = root_path + results_path + path + pattern[3].format(folder_names[0], str(idx) + '_' + approach +
                                                                           '_' + str(patch_shape) + '_' +
                                                                           str(extraction_step) + data_augment_label +
                                                                           '_preproc_trn_opt_' + str(preprocess_trn) +
                                                                           '_preproc_tst_opt_' + str(preprocess_tst))

        print(seg_filename)
        print(lb1_file_lst[idx-1])
        print(lb2_file_lst[idx-1])

        if os.path.exists(seg_filename) and os.path.exists(lb1_file_lst[idx-1]) and os.path.exists(lb2_file_lst[idx-1]):
            seg_label_vol = read_volume(seg_filename)
            label1_vol = read_volume(lb1_file_lst[idx-1])
            label2_vol = read_volume(lb2_file_lst[idx-1])

            if np.size(np.shape(label1_vol)) == 4:
                label1_data = label1_vol[:, :, :, 0]
            else:
                label1_data = label1_vol

            if np.size(np.shape(label2_vol)) == 4:
                label2_data = label2_vol[:, :, :, 0]
            else:
                label2_data = label2_vol

            label1_data[label1_data == np.max(label1_data)] = 150  # mandible label
            label2_data[label2_data == np.max(label2_data)] = 250  # midface label
            label_vol = label1_data + label2_data

            label_mapper = {0: 150, 1: 250} # 0 : mandible, 1: midface
            DC = computeDice(seg_label_vol, label_vol, label_mapper)
        else:
            DC = []
    else:
        raise NotImplementedError('Unknown approach for measuring')

    return DC


def measure_thalamus(gen_conf, train_conf, test_conf, idx, seg_label, k, mode):

    import pandas as pd
    import pickle

    dataset = train_conf['dataset']
    approach = train_conf['approach']
    loss = train_conf['loss']
    root_path = gen_conf['root_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    results_path = gen_conf['results_path']
    path = dataset_info['path']
    folder_names= dataset_info['folder_names']
    data_augment = train_conf['data_augment']
    file_format = dataset_info['format']

    test_patient_id = idx
    file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])
    test_patient_dir = os.path.join(file_output_dir, test_patient_id)
    seg_output_dir = os.path.join(test_patient_dir, path)

    crop_tst_label_pattern = dataset_info['crop_tst_label_pattern']
    staple_pattern = dataset_info['staple_pattern']
    crop_staple_pattern = dataset_info['crop_staple_pattern']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']
    test_roi_mask_file = os.path.join(test_patient_dir, test_roi_mask_pattern)
    test_crop_mask = find_crop_mask(test_roi_mask_file)

    cmd, msd, dc, vol = [], [], [], []
    cmd_staple, msd_staple, dc_staple, vol_staple = [], [], [], []

    # measure on the ROI
    for side in ['left', 'right']:
        label_file_lst = []
        for ver in ['2', '3']:
            label_file = os.path.join(test_patient_dir, crop_tst_label_pattern.format(side, ver))
            label_file_lst.append(label_file)
            if os.path.exists(label_file):
                break
        if os.path.exists(label_file):
            label_data = read_volume_data(label_file)
            label_image = label_data.get_data()
            if np.size(np.shape(label_image)) == 4:
                label_image_f = label_image[:, :, :, 0]
            else:
                label_image_f = label_image
            label_image_vox_size = label_data.header.get_zooms()
            label_vol = len(np.where(label_image_f == 1)[0]) * label_image_vox_size[0] * label_image_vox_size[1] * \
                        label_image_vox_size[2]
        else:
            print('No found: %s or %s' % (label_file[0], label_file[1]))
            label_image_vox_size = None
            label_vol = None
            label_image_f = None

        # read cropped smoothen threhold image for measure
        seg_filename = side + '_' + seg_label + '_seg_crop_' + approach + '_' + loss + '.' + \
                       file_format # measure original one (not smoothed/normalized/threshold)
        non_smoothed_crop_output_dir = os.path.join(seg_output_dir, 'non_smoothed', 'crop')
        seg_out_file = os.path.join(non_smoothed_crop_output_dir, seg_filename)

        if os.path.exists(seg_out_file):
            seg_data = read_volume_data(seg_out_file)
            seg_image = seg_data.get_data()
            if np.size(np.shape(seg_image)) == 4:
                seg_image_f = seg_image[:, :, :, 0]
            else:
                seg_image_f = seg_image
            seg_image_vox_size = seg_data.header.get_zooms()
            seg_vol = len(np.where(seg_image_f == 1)[0]) * seg_image_vox_size[0] * seg_image_vox_size[1] * \
                      seg_image_vox_size[2]
        else:
            print('No found: ' + seg_out_file)
            seg_image_vox_size = None
            seg_vol = None
            seg_image_f = None

        if label_image_vox_size is None or seg_image_vox_size is None:
            cmd.append(None)
            msd.append(None)
            dc.append(None)
        else:
            if label_image_f.shape != seg_image_f.shape:
                print('image size mismatches: manual - ' + str(label_image_f.shape), ', seg - ' +
                      str(seg_image_f.shape))
                cmd.append(None)
                msd.append(None)
                dc.append(None)
            else:
                cm_dist, _ = measure_cmd(label_image_f, seg_image_f, label_image_vox_size)
                cmd.append(cm_dist)
                msd.append(measure_msd(label_image_f, seg_image_f, label_image_vox_size))
                dc.append(dice(label_image_f, seg_image_f))
        vol.append([seg_vol, label_vol])

        staple_file = os.path.join(dataset_path, test_patient_id, 'fusion', staple_pattern.format(side))
        crop_staple_pattern_file = os.path.join(test_patient_dir, crop_staple_pattern.format(side))
        if (not os.path.exists(crop_staple_pattern_file)):
            crop_image(staple_file, test_crop_mask, crop_staple_pattern_file)

        if os.path.exists(crop_staple_pattern_file):
            staple_data = read_volume_data(crop_staple_pattern_file)
            staple_image = staple_data.get_data()
            if np.size(np.shape(staple_image)) == 4:
                staple_image_f = staple_image[:, :, :, 0]
            else:
                staple_image_f = staple_image
            staple_image_vox_size = staple_data.header.get_zooms()
            staple_vol = len(np.where(staple_image_f == 1)[0]) * staple_image_vox_size[0] * staple_image_vox_size[1] * \
                      staple_image_vox_size[2]
        else:
            print('No found: ' + crop_staple_pattern_file)
            staple_image_vox_size = None
            staple_vol = None
            staple_image_f = None

        if label_image_vox_size is None or staple_image_vox_size is None:
            cmd_staple.append(None)
            msd_staple.append(None)
            dc_staple.append(None)
        else:
            if label_image_f.shape != staple_image_f.shape:
                print('image size mismatches: manual - ' + str(label_image_f.shape), ', seg - ' +
                      str(staple_image_f.shape))
                cmd_staple.append(None)
                msd_staple.append(None)
                dc_staple.append(None)
            else:
                cm_dist_staple, _ = measure_cmd(label_image_f, staple_image_f, label_image_vox_size)
                cmd_staple.append(cm_dist_staple)
                msd_staple.append(measure_msd(label_image_f, staple_image_f, label_image_vox_size))
                dc_staple.append(dice(label_image_f, staple_image_f))
        vol_staple.append([staple_vol, label_vol])


    # Save segmentation results in dataframe as pkl
    file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])
    measure_pkl_filename = 'mode_' + mode + '_#' + str(k + 1) + '_' + 'measurement_' + \
                           approach + '_' + loss + '.pkl'
    measure_pkl_filepath = os.path.join(file_output_dir, measure_pkl_filename)
    patient_results = {'CMD_L': [cmd[0]], 'CMD_R': [cmd[1]],
                       'MSD_L': [msd[0]], 'MSD_R': [msd[1]],
                       'DC_L': [dc[0]], 'DC_R': [dc[1]],
                       'VOL_L': [vol[0][0]], 'VOL_R': [vol[1][0]],
                       'VOL_manual_L': [vol[0][1]], 'VOL_manual_R': [vol[1][1]]}
    columns_name_lst = ['CMD_L', 'CMD_R', 'MSD_L', 'MSD_R', 'DC_L', 'DC_R', 'VOL_L', 'VOL_R',
                        'VOL_manual_L', 'VOL_manual_R']
    patient_results_df = pd.DataFrame(patient_results, columns=columns_name_lst)
    patient_results_df.insert(0, 'patient_id', [test_patient_id])

    #staple results
    file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])
    measure_pkl_filename_staple = 'mode_' + mode + '_#' + str(k + 1) + '_' + 'measurement_' + 'staple' + '_' + \
                                  loss + '.pkl'
    measure_pkl_filepath_staple = os.path.join(file_output_dir, measure_pkl_filename_staple)
    patient_staple_results = {'CMD_L': [cmd_staple[0]], 'CMD_R': [cmd_staple[1]],
                       'MSD_L': [msd_staple[0]], 'MSD_R': [msd_staple[1]],
                       'DC_L': [dc_staple[0]], 'DC_R': [dc_staple[1]],
                       'VOL_L': [vol_staple[0][0]], 'VOL_R': [vol_staple[1][0]],
                       'VOL_manual_L': [vol_staple[0][1]], 'VOL_manual_R': [vol_staple[1][1]]}
    columns_name_lst = ['CMD_L', 'CMD_R', 'MSD_L', 'MSD_R', 'DC_L', 'DC_R', 'VOL_L', 'VOL_R',
                        'VOL_manual_L', 'VOL_manual_R']
    patient_staple_results_df = pd.DataFrame(patient_staple_results, columns=columns_name_lst)
    patient_staple_results_df.insert(0, 'patient_id', [test_patient_id])

    if os.path.exists(measure_pkl_filepath):
        with open(measure_pkl_filepath, 'rb') as handle:
            patient_results_df_total = pickle.load(handle)
            patient_results_df_total = pd.concat([patient_results_df_total, patient_results_df])
    else:
        patient_results_df_total = patient_results_df

    patient_results_df_total.to_pickle(measure_pkl_filepath)
    print(patient_results_df_total)
    print(patient_results_df_total.describe(percentiles=[0.25, 0.5, 0.75, 0.9]))

    if os.path.exists(measure_pkl_filepath_staple):
        with open(measure_pkl_filepath_staple, 'rb') as handle:
            patient_staple_results_df_total = pickle.load(handle)
            patient_staple_results_df_total = pd.concat([patient_staple_results_df_total, patient_staple_results_df])
    else:
        patient_staple_results_df_total = patient_staple_results_df

    patient_staple_results_df_total.to_pickle(measure_pkl_filepath_staple)
    print(patient_staple_results_df_total)
    print(patient_staple_results_df_total.describe(percentiles=[0.25, 0.5, 0.75, 0.9]))

    return patient_results_df_total, patient_staple_results_df_total


def measure_dentate(gen_conf, train_conf, test_conf, idx, k, mode):

    import pandas as pd
    import pickle

    dataset = train_conf['dataset']
    approach = train_conf['approach']
    loss = train_conf['loss']
    patch_shape = train_conf['patch_shape']
    extraction_step = train_conf['extraction_step']
    root_path = gen_conf['root_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    results_path = gen_conf['results_path']
    path = dataset_info['path']
    folder_names= dataset_info['folder_names']
    data_augment = train_conf['data_augment']
    preprocess_trn = train_conf['preprocess']
    preprocess_tst = test_conf['preprocess']
    file_format = dataset_info['format']

    if data_augment == 1:
        data_augment_label = '_mixup'
    elif data_augment == 2:
        data_augment_label = '_datagen'
    elif data_augment == 3:
        data_augment_label = '_mixup+datagen'
    else:
        data_augment_label = ''

    test_patient_id = idx
    label_pattern = dataset_info['manual_corrected_pattern']
    label_v2_pattern = dataset_info['manual_corrected_dentate_v2_pattern']

    file_output_dir = root_path + results_path + dataset + '/' + folder_names[0] + '/' + test_patient_id + '/' + \
                      path + '/'

    cmd, msd, dc, vol = [], [], [], []
    for side in ['left', 'right']:

        updated_label_filepath = dataset_path + test_patient_id + label_v2_pattern.format(test_patient_id, side)
        if os.path.exists(updated_label_filepath):
            label_file = updated_label_filepath
        else:
            label_file = dataset_path + test_patient_id + label_pattern.format(test_patient_id, side)

        seg_file = file_output_dir + side + '_dentate_seg_' + approach + '_' + loss + '_' + str(patch_shape) + \
                   '_' + str(extraction_step) + data_augment_label + '_preproc_trn_opt_' + str(preprocess_trn) + \
                   '_preproc_tst_opt_' + str(preprocess_tst) + '.' + file_format

        if os.path.exists(label_file):
            label_data = read_volume_data(label_file)
            label_image = label_data.get_data()
            if np.size(np.shape(label_image)) == 4:
                label_image_f = label_image[:, :, :, 0]
            else:
                label_image_f = label_image
            label_image_vox_size = label_data.header.get_zooms()
            label_vol = len(np.where(label_image_f == 1)[0]) * label_image_vox_size[0] * label_image_vox_size[1] * \
                        label_image_vox_size[2]
        else:
            print('No found: ' + label_file)
            label_image_vox_size = None
            label_vol = None
            label_image_f = None

        if os.path.exists(seg_file):
            seg_data = read_volume_data(seg_file)
            seg_image = seg_data.get_data()
            if np.size(np.shape(seg_image)) == 4:
                seg_image_f = seg_image[:, :, :, 0]
            else:
                seg_image_f = seg_image
            seg_image_vox_size = seg_data.header.get_zooms()
            seg_vol = len(np.where(seg_image_f == 1)[0]) * seg_image_vox_size[0] * seg_image_vox_size[1] * \
                      seg_image_vox_size[2]
        else:
            print('No found: ' + seg_file)
            seg_image_vox_size = None
            seg_vol = None
            seg_image_f = None

        if label_image_vox_size is None or seg_image_vox_size is None:
            cmd.append(None)
            msd.append(None)
            dc.append(None)
        else:
            if label_image_vox_size[0:3] != seg_image_vox_size[0:3]:
                print('voxel size mismatches: manual - ' + str(label_image_vox_size), ', seg - ' +
                      str(seg_image_vox_size))
                cmd.append(None)
                msd.append(None)
                dc.append(None)
            else:
                cm_dist, _ = measure_cmd(label_image_f, seg_image_f, label_image_vox_size)
                cmd.append(cm_dist)
                msd.append(measure_msd(label_image_f, seg_image_f, label_image_vox_size))
                dc.append(dice(label_image_f, seg_image_f))
        vol.append([seg_vol, label_vol])

    # Save results in dataframe as pkl
    file_output_dir = root_path + results_path + dataset + '/' + folder_names[0]
    measure_pkl_filepath = file_output_dir + '/' + 'mode_' + mode + '_#' + str(k + 1) + '_' + 'measurement_' + \
                           approach + '_' + loss + '.pkl'
    patient_results = {'CMD_L': [cmd[0]], 'CMD_R': [cmd[1]],
                       'MSD_L': [msd[0]], 'MSD_R': [msd[1]],
                       'DC_L': [dc[0]], 'DC_R': [dc[1]],
                       'VOL_L': [vol[0][0]], 'VOL_R': [vol[1][0]],
                       'VOL_manual_L': [vol[0][1]], 'VOL_manual_R': [vol[1][1]]}
    columns_name_lst = ['CMD_L', 'CMD_R', 'MSD_L', 'MSD_R', 'DC_L', 'DC_R', 'VOL_L', 'VOL_R',
                        'VOL_manual_L', 'VOL_manual_R']
    patient_results_df = pd.DataFrame(patient_results, columns=columns_name_lst)
    patient_results_df.insert(0, 'patient_id', [test_patient_id])

    if os.path.exists(measure_pkl_filepath):
        with open(measure_pkl_filepath, 'rb') as handle:
            patient_results_df_total = pickle.load(handle)
            patient_results_df_total = pd.concat([patient_results_df_total, patient_results_df])
    else:
        patient_results_df_total = patient_results_df

    patient_results_df_total.to_pickle(measure_pkl_filepath)
    print(patient_results_df_total.sort_values(['patient_id'], ascending=[1]))
    print(patient_results_df_total.describe(percentiles=[0.25, 0.5, 0.75, 0.9]))

    return patient_results_df_total


def measure_dentate_interposed(gen_conf, train_conf, test_conf, idx, k, mode, target):

    import pandas as pd
    import pickle

    dataset = train_conf['dataset']
    approach = train_conf['approach']
    loss = train_conf['loss']
    root_path = gen_conf['root_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    results_path = gen_conf['results_path']
    num_classes = gen_conf['num_classes']
    multi_output = gen_conf['multi_output']
    path = dataset_info['path']
    folder_names= dataset_info['folder_names']
    data_augment = train_conf['data_augment']
    file_format = dataset_info['format']

    test_patient_id = idx
    dentate_label_pattern = dataset_info['crop_tst_manual_dentate_v2_corrected_pattern']
    interposed_label_pattern = dataset_info['crop_tst_manual_interposed_v2_pattern']

    file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])
    test_patient_dir = os.path.join(file_output_dir, test_patient_id)
    seg_output_dir = os.path.join(test_patient_dir, path)

    if len(target) == 2:
        target = 'both'

    if target == 'dentate':
        label_pattern_lst = [dentate_label_pattern]
        seg_label_lst = ['dentate_seg']
        if multi_output == 1:
            loss_ch = [loss[0]]
            num_classes_ch = [num_classes[0]]
        else:
            loss_ch = [loss]
            num_classes_ch = [num_classes]
    elif target == 'interposed':
        label_pattern_lst = [interposed_label_pattern]
        seg_label_lst = ['interposed_seg']
        if multi_output == 1:
            loss_ch = [loss[1]]
            num_classes_ch = [num_classes[1]]
        else:
            loss_ch = [loss]
            num_classes_ch = [num_classes]
    else:
        label_pattern_lst = [dentate_label_pattern, interposed_label_pattern]
        seg_label_lst = ['dentate_seg', 'interposed_seg']
        if multi_output == 1:
            loss_ch = [loss[0], loss[1]]
            num_classes_ch = [num_classes[0], num_classes[1]]
        else:
            loss_ch = [loss, loss]
            num_classes_ch = [num_classes, num_classes]

    for label_pattern, seg_label, l_ch, n_ch in zip(label_pattern_lst, seg_label_lst, loss_ch, num_classes_ch):
        cmd, msd, dc, vol = [], [], [], []
        for side in ['left', 'right']:
            label_file = os.path.join(test_patient_dir, label_pattern.format(test_patient_id, side))
            seg_filename = side + '_' + seg_label + '_crop_' + approach + '_' + l_ch + '.' + \
                           file_format # measure original one (not smoothed/normalized/threshold)
            non_smoothed_crop_output_dir = os.path.join(seg_output_dir, 'non_smoothed', 'crop')
            seg_out_file = os.path.join(non_smoothed_crop_output_dir, seg_filename)
            if os.path.exists(label_file):
                label_data = read_volume_data(label_file)
                label_image = label_data.get_data()
                if np.size(np.shape(label_image)) == 4:
                    label_image_f = label_image[:, :, :, 0]
                else:
                    label_image_f = label_image
                label_image_vox_size = label_data.header.get_zooms()
                label_vol = len(np.where(label_image_f == 1)[0]) * label_image_vox_size[0] * label_image_vox_size[1] * \
                            label_image_vox_size[2]
            else:
                print('No found: ' + label_file)
                label_image_vox_size = None
                label_vol = None
                label_image_f = None

            if os.path.exists(seg_out_file):
                seg_data = read_volume_data(seg_out_file)
                seg_image = seg_data.get_data()
                if np.size(np.shape(seg_image)) == 4:
                    seg_image_f = seg_image[:, :, :, 0]
                else:
                    seg_image_f = seg_image
                seg_image_vox_size = seg_data.header.get_zooms()
                seg_vol = len(np.where(seg_image_f == 1)[0]) * seg_image_vox_size[0] * seg_image_vox_size[1] * \
                          seg_image_vox_size[2]
            else:
                print('No found: ' + seg_out_file)
                seg_image_vox_size = None
                seg_vol = None
                seg_image_f = None

            if label_image_vox_size is None or seg_image_vox_size is None or seg_vol == 0:
                cmd.append(None)
                msd.append(None)
                dc.append(None)
            else:
                if label_image_vox_size[0:3] != seg_image_vox_size[0:3]:
                    print('voxel size mismatches: manual - ' + str(label_image_vox_size), ', seg - ' +
                          str(seg_image_vox_size))
                    cmd.append(None)
                    msd.append(None)
                    dc.append(None)
                else:
                    cm_dist, _ = measure_cmd(label_image_f, seg_image_f, label_image_vox_size)
                    cmd.append(cm_dist)
                    msd.append(measure_msd(label_image_f, seg_image_f, label_image_vox_size))
                    dc.append(dice(label_image_f, seg_image_f))
            vol.append([seg_vol, label_vol])

        # Save results in dataframe as pkl
        measure_pkl_filename = 'mode_' + mode + '_#' + str(k + 1) + '_' + 'measurement_' + \
                               approach + '_' + l_ch + '_n_classes_' + str(n_ch) + '_' + seg_label + '.pkl'
        measure_pkl_filepath = os.path.join(file_output_dir, measure_pkl_filename)
        patient_results = {'CMD_L': [cmd[0]], 'CMD_R': [cmd[1]],
                           'MSD_L': [msd[0]], 'MSD_R': [msd[1]],
                           'DC_L': [dc[0]], 'DC_R': [dc[1]],
                           'VOL_L': [vol[0][0]], 'VOL_R': [vol[1][0]],
                           'VOL_manual_L': [vol[0][1]], 'VOL_manual_R': [vol[1][1]]}
        columns_name_lst = ['CMD_L', 'CMD_R', 'MSD_L', 'MSD_R', 'DC_L', 'DC_R', 'VOL_L', 'VOL_R',
                            'VOL_manual_L', 'VOL_manual_R']
        patient_results_df = pd.DataFrame(patient_results, columns=columns_name_lst)
        patient_results_df.insert(0, 'patient_id', [test_patient_id])
        patient_results_df.insert(0, 'structure', [seg_label])

        if os.path.exists(measure_pkl_filepath):
            with open(measure_pkl_filepath, 'rb') as handle:
                patient_results_df_total = pickle.load(handle)
                patient_results_df_total = pd.concat([patient_results_df_total, patient_results_df])
        else:
            patient_results_df_total = patient_results_df

        patient_results_df_total.to_pickle(measure_pkl_filepath)
        print(patient_results_df_total)
        print(patient_results_df_total.describe(percentiles=[0.25,0.5,0.75,0.9]))

        #todo: add measures of suits and level set results


    return patient_results_df_total


def measure_sar_prediction(gen_conf, train_conf, gt, test_id, sar_pred, case_name):

    import pandas as pd
    import pickle
    from sklearn.metrics import mean_squared_error

    dataset = train_conf['dataset']
    root_path = gen_conf['root_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    results_path = gen_conf['results_path']
    mode = gen_conf['validation_mode']
    path = dataset_info['path']
    folder_names= dataset_info['folder_names']
    file_format = dataset_info['format']

    file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])
    traing_log_file = os.path.join(file_output_dir, 'training_log_#%d.txt' % case_name)
    #test_id_dir = os.path.join(file_output_dir, test_id)
    #sar_output_dir = os.path.join(test_id_dir, path)

    sar_gt = gt[0, 0, :, :, :]
    sar_gt_array = sar_gt.flatten()
    sar_pred_array = sar_pred.flatten()
    non_zero_gt_indices = np.where(np.array(sar_gt_array) > 0)[0]
    sar_gt_non_zero_array = sar_gt_array[non_zero_gt_indices]
    sar_pred_non_zero_array = sar_pred_array[non_zero_gt_indices]

    # mse = mean_squared_error(sar_gt_array, sar_pred_array)
    # rmse = np.sqrt(mse)
    # nrmse_max_min = rmse/(np.max(sar_gt_array)-np.min(sar_gt_array))
    # nrmse_avg = rmse / np.mean(sar_gt_array)

    mse = mean_squared_error(sar_gt_non_zero_array, sar_pred_non_zero_array)
    rmse = np.sqrt(mse)
    nrmse_max_min = rmse/(np.max(sar_gt_non_zero_array)-np.min(sar_gt_non_zero_array))
    nrmse_avg = rmse / np.mean(sar_gt_non_zero_array)
    f = ioutils.create_log(traing_log_file, 'RMSE of %s : %s' % (test_id, rmse), is_debug=1)
    f = ioutils.create_log(traing_log_file, 'normalized RMSE (by max-min) of %s : %s' % (test_id, nrmse_max_min),
                           is_debug=1)
    f = ioutils.create_log(traing_log_file, 'normalized RMSE (by avg) of %s : %s' % (test_id, nrmse_avg),
                           is_debug=1)
    f.close()

    # Save results in dataframe as pkl
    measure_pkl_filename = 'mode_' + mode + '_#' + str(case_name) + '_' + 'measurement.pkl'
    measure_pkl_filepath = os.path.join(file_output_dir, measure_pkl_filename)
    measure_results = {'RMSE': [rmse], 'NRMSE (max-min)': [nrmse_max_min], 'NRMSE (avg)': [nrmse_avg]}
    columns_name_lst = ['RMSE', 'NRMSE (max-min)', 'NRMSE (avg)']
    measure_results_df = pd.DataFrame(measure_results, columns=columns_name_lst)
    measure_results_df.insert(0, 'test_id', [test_id])

    if os.path.exists(measure_pkl_filepath):
        with open(measure_pkl_filepath, 'rb') as handle:
            measure_results_df_total = pickle.load(handle)
            measure_results_df_total = pd.concat([measure_results_df_total, measure_results_df])
    else:
        measure_results_df_total = measure_results_df

    measure_results_df_total.to_pickle(measure_pkl_filepath)
    print(measure_results_df_total)
    print(measure_results_df_total.describe(percentiles=[0.25, 0.5, 0.75, 0.9]))

    return measure_results_df_total


def measure_4d(gen_conf, train_conf, test_conf, idx):
    dataset = train_conf['dataset']
    approach = train_conf['approach']
    patch_shape = train_conf['patch_shape']
    extraction_step = train_conf['extraction_step']
    root_path = gen_conf['root_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    results_path = gen_conf['results_path']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    folder_names= dataset_info['folder_names']
    data_augment = train_conf['data_augment']
    preprocess_trn = train_conf['preprocess']
    preprocess_tst = test_conf['preprocess']
    modality = dataset_info['image_modality']
    num_modality = len(modality)

    if data_augment == 1:
        data_augment_label = '_mixup'
    elif data_augment == 2:
        data_augment_label = '_datagen'
    elif data_augment == 3:
        data_augment_label = '_mixup+datagen'
    else:
        data_augment_label = ''

    DC_total = []
    for t in range(num_modality):
        if dataset.find("3T7T") != -1:
            seg_filename = results_path + path + pattern[1].format(folder_names[0], str(idx) + '_' + approach + '_' +
                                                                   str(patch_shape) + '_' + str(extraction_step) +
                                                                   data_augment_label + '_preproc_trn_opt_' +
                                                                   str(preprocess_trn) + '_preproc_tst_opt_' +
                                                                   str(preprocess_tst) + 'time_point_' + str(t+1))


            gt_filename = root_path + dataset_path + path + pattern[1].format(folder_names[2], str(idx))

            print(seg_filename)
            print(gt_filename)

            if os.path.exists(seg_filename) and os.path.exists(gt_filename):
                seg_label_vol = read_volume(seg_filename)
                gt_label_vol = read_volume(gt_filename)

                label_mapper = {0: 10, 1: 150, 2: 250} # 0 : CSF, 1: GM, 2: WM
                DC = computeDice(seg_label_vol, gt_label_vol, label_mapper)
            else:
                DC = []

        elif dataset.find("CBCT") != -1:
            file_path = root_path + dataset_path + path + folder_names
            lb1_file_lst = glob.glob(file_path + '/' + '*mandible.nii.gz')  # mandible label
            lb2_file_lst = glob.glob(file_path + '/' + '*midface.nii.gz')  # midface label
            lb1_file_lst.sort()
            lb2_file_lst.sort()

            seg_filename = root_path + results_path + path + pattern[3].format(folder_names[0], str(idx) + '_' + approach + '_' +
                                                                       str(patch_shape) + '_' + str(extraction_step) +
                                                                       data_augment_label + '_preproc_trn_opt_' +
                                                                       str(preprocess_trn) + '_preproc_tst_opt_' +
                                                                       str(preprocess_tst) + 'time_point_' + str(t+1))

            print(seg_filename)
            print(lb1_file_lst[idx-1])
            print(lb2_file_lst[idx-1])

            if os.path.exists(seg_filename) and os.path.exists(lb1_file_lst[idx-1]) and os.path.exists(lb2_file_lst[idx-1]):
                seg_label_vol = read_volume(seg_filename)
                label1_vol = read_volume(lb1_file_lst[idx-1])
                label2_vol = read_volume(lb2_file_lst[idx-1])

                if np.size(np.shape(label1_vol)) == 4:
                    label1_data = label1_vol[:, :, :, 0]
                else:
                    label1_data = label1_vol

                if np.size(np.shape(label2_vol)) == 4:
                    label2_data = label2_vol[:, :, :, 0]
                else:
                    label2_data = label2_vol

                label1_data[label1_data == np.max(label1_data)] = 150  # mandible label
                label2_data[label2_data == np.max(label2_data)] = 250  # midface label
                label_vol = label1_data + label2_data

                label_mapper = {0: 150, 1: 250} # 0 : mandible, 1: midface
                DC = computeDice(seg_label_vol, label_vol, label_mapper)
            else:
                DC = []

        else:
            raise NotImplementedError('Unknown approach for measuring')

        DC_total.append(DC)

    return DC_total

