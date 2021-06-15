
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Lambda, ReLU, Subtract, Reshape, Permute, Activation
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from utils import optimizers_builtin, loss_functions, metrics
import numpy as np
import keras.backend as K


# define the combined generator and discriminator model, for updating the generator
def build_cgan(gen_conf, train_conf, g_model, d_model):

    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    patch_shape = train_conf['patch_shape']
    optimizer = train_conf['optimizer']
    initial_lr = train_conf['initial_lr']
    loss_opt = train_conf['GAN']['generator']['loss']
    d_loss = train_conf['GAN']['discriminator']['loss']
    num_classes = train_conf['GAN']['generator']['num_classes']
    metric_opt = train_conf['GAN']['generator']['metric']
    multi_output = train_conf['GAN']['generator']['multi_output']
    output_name = train_conf['GAN']['generator']['output_name']
    attention_loss = train_conf['GAN']['generator']['attention_loss']
    overlap_penalty_loss = train_conf['GAN']['generator']['overlap_penalty_loss']
    lamda = train_conf['GAN']['generator']['lamda']
    adv_loss_weight = train_conf['GAN']['generator']['adv_loss_weight']

    input_shape = (num_modality,) + patch_shape

    # make weights in the discriminator not trainable
    d_model.trainable = False

    # define the source image
    in_src = Input(shape=input_shape)

    # connect the source image to the generator input
    gen_out = g_model(in_src)

    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out[0], gen_out[1]])

    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out[0], gen_out[1], gen_out[2], gen_out[3]])
    model.summary()

    dentate_prob_output = Lambda(lambda x: x[:, :, num_classes[0] - 1])(gen_out[0])
    interposed_prob_output = Lambda(lambda x: x[:, :, num_classes[1] - 1])(gen_out[1])

    # loss_multi = {
    #     g_output_name[0]: loss_functions.select(g_num_classes[0], loss_opt[0]),
    #     g_output_name[1]: loss_functions.select(g_num_classes[1], loss_opt[1]),
    #     'attention_maps': 'categorical_crossentropy',
    #     'overlap_dentate_interposed': loss_functions.dc_btw_dentate_interposed(dentate_prob_output,
    #                                                                            interposed_prob_output)
    # }
    # g_loss_weights = {g_output_name[0]: g_lamda[0], g_output_name[1]: g_lamda[1], 'attention_maps': g_lamda[2],
    #                 'overlap_dentate_interposed': g_lamda[3]}
    #metric_f = ['loss', metrics.select(g_metric_opt)]

    # compile model
    loss_f = [d_loss, loss_functions.select(num_classes[0], loss_opt[0]),
              loss_functions.select(num_classes[1], loss_opt[1]), 'categorical_crossentropy',
              loss_functions.dc_btw_dentate_interposed(dentate_prob_output, interposed_prob_output)]  # adversarial loss via discriminator output + L1 loss via the direct image output
    loss_weights_f = [adv_loss_weight, lamda[0], lamda[1], lamda[2], lamda[3]]
    #loss_f = ['binary_crossentropy', loss_functions.select(2, loss_opt[0]), loss_functions.select(2, loss_opt[1])]  # adversarial loss via discriminator output + L1 loss via the direct image output
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_f, optimizer=optimizers_builtin.select(optimizer, initial_lr),
                  loss_weights=loss_weights_f) #metrics=metrics.select(metric_opt)

    d_model.trainable = True

    return model


def build_cgan_sar(gen_conf, train_conf, g_model, d_model):

    num_of_gpu = train_conf['num_of_gpu']
    dataset = train_conf['dataset']
    dimension = train_conf['dimension']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    patch_shape = train_conf['patch_shape']
    optimizer = train_conf['optimizer']
    initial_lr = train_conf['initial_lr']
    g_loss = train_conf['GAN']['generator']['loss']
    d_loss = train_conf['GAN']['discriminator']['loss']
    lamda = train_conf['GAN']['generator']['lamda']
    adv_loss_weight = train_conf['GAN']['generator']['adv_loss_weight']
    g_patch_shape = train_conf['GAN']['generator']['patch_shape']
    g_output_shape = train_conf['GAN']['generator']['output_shape']
    d_patch_shape = train_conf['GAN']['discriminator']['patch_shape']

    if dimension == 2:
        g_input_shape = (g_patch_shape[0], g_patch_shape[1], num_modality)
        d_input_shape = (d_patch_shape[0], d_patch_shape[1], num_modality)
    else:
        g_input_shape = (num_modality,) + g_patch_shape
        d_input_shape = (num_modality,) + d_patch_shape

    # make weights in the discriminator not trainable
    d_model.trainable = False

    # define the source image
    g_in_src = Input(shape=g_input_shape)
    if g_patch_shape != d_patch_shape:
        d_in_src = Input(shape=d_input_shape)

    output_shape_prod = (np.prod(g_output_shape), 1)
    g_sar_real = Input(shape=output_shape_prod)

    gen_out = g_model(g_in_src)

    print(g_sar_real.shape)
    print(gen_out.shape)

    tf_g_sar_real = Lambda(lambda x: x[:, :, 0])(g_sar_real)
    tf_gen_out = Lambda(lambda x: x[:, :, 0])(gen_out)

    print(tf_g_sar_real.shape)
    print(tf_gen_out.shape)

    # connect the source input and generator output to the discriminator input
    if g_patch_shape != d_patch_shape:
        [dis_out, interm_f1_out, interm_f2_out, interm_f3_out] = d_model([d_in_src, gen_out])
        model = Model([g_in_src, d_in_src, g_sar_real], [dis_out, gen_out, interm_f1_out, interm_f2_out, interm_f3_out,
                                                         g_sar_real, g_sar_real, g_sar_real, g_sar_real]) # output g_sar_real is garbage for peak loss
    else:
        [dis_out, interm_f1_out, interm_f2_out, interm_f3_out] = d_model([g_in_src, gen_out])
        model = Model([g_in_src, g_sar_real], [dis_out, gen_out, interm_f1_out, interm_f2_out, interm_f3_out,
                                               g_sar_real, g_sar_real, g_sar_real, g_sar_real])
    model.summary()

    # compile model
    loss_f = [d_loss, g_loss, 'mae', 'mae', 'mae',
              loss_functions.local_sar_peak_loss(tf_g_sar_real, tf_gen_out),
              loss_functions.local_sar_real_peak(tf_g_sar_real),
              loss_functions.local_sar_pred_peak(tf_gen_out),
              loss_functions.local_sar_pred_neg_loss(tf_gen_out)]  # adversarial loss via discriminator output + L1 loss + L2 perceptual loss via the direct image output
    #loss_weights_f = [adv_loss_weight, lamda, 0.5/3, 0.5/3, 0.5/3, 0.001, 0, 0, 0.001]
    loss_weights_f = [adv_loss_weight, lamda[0], lamda[1], lamda[2], lamda[3], lamda[4], 0, 0, lamda[5]]
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_f, optimizer=optimizers_builtin.select(optimizer, initial_lr), loss_weights=loss_weights_f)

    d_model.trainable = True

    return model