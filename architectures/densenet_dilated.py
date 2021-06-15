import numpy as np
from keras import backend as K

from keras.models import Model
from keras.layers import Input, Dropout, Reshape, Conv2D, Conv3D, concatenate, BatchNormalization, Activation
from keras.regularizers import l2
from keras.layers.core import Permute
from keras.initializers import he_normal
from keras.engine.topology import get_source_inputs
from keras.utils import multi_gpu_model

from utils import loss_functions, metrics, optimizers_builtin
from .squeeze_excitation import spatial_and_channel_squeeze_excite_block2D, spatial_and_channel_squeeze_excite_block3D

K.set_image_dim_ordering('th')

# Ref.
# Jegou et al., CVPRW 17, "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation"
# Kamnitsas et al., MedIA17, "Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation"
# Chen et al., ECCV18, "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"

def generate_densenet_dilated(gen_conf, train_conf) :

    '''Instantiate the DenseNet FCN architecture.
        Note that when using TensorFlow,
        for best performance you should set
        `image_data_format='channels_last'` in your Keras config
        at ~/.keras/keras.json.
        # Arguments
            nb_dense_block: number of dense blocks to add to end (generally = 3)
            growth_rate: number of filters to add per dense block
            nb_layers_per_block: number of layers in each dense block.
                Can be a positive integer or a list.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
            reduction: reduction factor of transition blocks.
                Note : reduction value is inverted to compute compression.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            init_conv_filters: number of layers in the initial convolution layer
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: one of `None` (random initialization) or
                'cifar10' (pre-training on CIFAR-10)..
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(32, 32, 3)` (with `channels_last` dim ordering)
                or `(3, 32, 32)` (with `channels_first` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
            activation: Type of activation at the top layer. Can be one of 'softmax'
                or 'sigmoid'. Note that if sigmoid is used, classes must be 1.
            upsampling_type: Can be one of 'deconv', 'upsampling' and
                'subpixel'. Defines type of upsampling algorithm used.
            batchsize: Fixed batch size. This is a temporary requirement for
                computation of output shape in the case of Deconvolution2D layers.
                Parameter will be removed in next iteration of Keras, which infers
                output shape of deconvolution layers automatically.
            early_transition: Start with an extra initial transition down and end with
                an extra transition up to reduce the network size.
            initial_kernel_size: The first Conv2D kernel might vary in size based on the
                application, this parameter makes it configurable.
        # Returns
            A Keras model instance.
    '''

    dataset = train_conf['dataset']
    activation = train_conf['activation']
    dimension = train_conf['dimension']
    num_classes = gen_conf['num_classes']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    expected_output_shape = train_conf['output_shape']
    patch_shape = train_conf['patch_shape']

    loss_opt = train_conf['loss']
    metric_opt = train_conf['metric']
    optimizer = train_conf['optimizer']
    initial_lr = train_conf['initial_lr']
    is_set_random_seed = train_conf['is_set_random_seed']
    random_seed_num = train_conf['random_seed_num']
    exclusive_train = train_conf['exclusive_train']

    if is_set_random_seed == 1:
        kernel_init = he_normal(seed=random_seed_num)
    else:
        kernel_init = he_normal()

    #nb_dense_block = 3 # for dentate 3 #for thalamus
    #nb_layers_per_block = (3, 4, 5) #[3,4,5,6] # for dentate[3,4,5,6] #for thalamus #[3,4,5,6] # 4 #
    growth_rate = 8
    dropout_rate = 0.2

    is_densenet = True # densenet or conventional convnet
    is_dilated_conv = True # dilated densenet db or densenet db
    is_aggregation = False

    if is_densenet:
        if is_dilated_conv:
            dilation_rate_per_block = [(1, 1, 2), (1, 1, 2, 4), (1, 1, 2, 4, 8)]
            #dilation_rate_per_block = [(1, 1, 2, 4, 8)]
        else:
            dilation_rate_per_block = [(1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1, 1)]
    else:
        if is_dilated_conv:
            dilation_rate_per_block = [(1, 1, 2, 4, 8, 1)]
        else:
            dilation_rate_per_block = [(1, 1, 1, 1, 1, 1)]

    nb_dense_block = len(dilation_rate_per_block)
    nb_layers_per_block = []
    for i in dilation_rate_per_block:
        nb_layers_per_block.append(len(i))
    nb_layers_per_block = tuple(nb_layers_per_block)

    # Global-local attention module
    glam = False #True
    glam_arrange_type = 'two_way_sequential' #'two_way_sequential' #concurrent_scSE
    glam_input_short_cut = False #True #False
    glam_final_conv = False # False: addition # True: applying fully connected conv layer (1x1x1) after concatenation instead of adding up
    glam_position = 'before_shortcut' #before_shortcut #after_shortcut

    reduction = 0.0
    weight_decay = 1E-4
    init_conv_filters = 16 #16 #32
    weights = None
    input_tensor = None

    if dimension == 2:
        initial_kernel_size = (3, 3)
    else:
        initial_kernel_size = (3, 3, 3)

    if exclusive_train == 1:
        num_classes -= 1

    input_shape = (num_modality, ) + patch_shape
    output_shape = (num_classes, np.prod(expected_output_shape))

    if weights not in {None}:
        raise ValueError('The `weights` argument should be '
                         '`None` (random initialization) as no '
                         'model weights are provided.')

    if input_shape is None:
        raise ValueError('For fully convolutional models, input shape must be supplied.')

    if type(nb_layers_per_block) is not list and nb_dense_block < 1:
        raise ValueError('Number of dense layers per block must be greater than 1. '
                         'Argument value was %s.' % nb_layers_per_block)

    if activation not in ['softmax', 'sigmoid']:
        raise ValueError('activation must be one of "softmax" or "sigmoid"')

    if activation == 'sigmoid' and num_classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')

    assert dimension in [2, 3]

    model = __generate_dilated_densenet(dimension, num_classes, kernel_init, random_seed_num, input_shape, output_shape,
                                        input_tensor, nb_dense_block, nb_layers_per_block, growth_rate, reduction,
                                        dropout_rate, weight_decay, init_conv_filters, activation, initial_kernel_size,
                                        is_densenet, is_dilated_conv, is_aggregation, dilation_rate_per_block, glam,
                                        glam_arrange_type, glam_input_short_cut, glam_final_conv, glam_position)

    model.summary()
    if num_of_gpu > 1:
        model = multi_gpu_model(model, gpus=num_of_gpu)
    model.compile(loss=loss_functions.select(num_classes, loss_opt),
                  optimizer=optimizers_builtin.select(optimizer, initial_lr),
                  metrics=metrics.select(metric_opt))
    return model


def __generate_dilated_densenet(dimension, num_classes, kernel_init, random_seed_num, input_shape=None,
                                output_shape=None, input_tensor=None, nb_dense_block=3,
                                nb_layers_per_block=(3, 4, 5, 6), growth_rate=12, reduction=0.0,
                                dropout_rate=None, weight_decay=1e-4, init_conv_filters=48, activation='softmax',
                                initial_kernel_size=(3, 3, 3), is_densenet=True, is_dilated_conv=True,
                                is_aggregation=False, dilation_rate_per_block=(2, 4, 8), glam=True,
                                glam_arrange_type='two_way_sequential', glam_input_short_cut=True,
                                glam_final_conv=False, glam_position='before_shortcut'):

    ''' Build the DenseNet-FCN model
    # Arguments
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        reduction: reduction factor of transition blocks. Note : reduction value
            is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay
        nb_layers_per_block: number of layers in each dense block.
            Can be a positive integer or a list.
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided. Note that list size must
            be (nb_dense_block + 1)
        upsampling_type: Can be one of 'upsampling', 'deconv' and 'subpixel'. Defines
            type of upsampling algorithm used.
        input_shape: Only used for shape inference in fully convolutional networks.
        activation: Type of activation at the top layer. Can be one of 'softmax' or
            'sigmoid'. Note that if sigmoid is used, classes must be 1.
        early_transition: Start with an extra initial transition down and end with an
            extra transition up to reduce the network size.
        transition_pooling: 'max' for max pooling (default), 'avg' for average pooling,
            None for no pooling. Please note that this default differs from the DenseNet
            paper in accordance with the DenseNetFCN paper.
        initial_kernel_size: The first Conv2D kernel might vary in size based on the
            application, this parameter makes it configurable.
    # Returns
        a keras tensor
    # Raises
        ValueError: in case of invalid argument for `reduction`,
            `nb_dense_block`.
    '''
    with K.name_scope('DilatedDenseNet'):

        # Determine proper input shape
        min_size = 2 ** nb_dense_block

        if K.image_data_format() == 'channels_first':

            if dimension == 2:
                if input_shape is not None:
                    if ((input_shape[1] is not None and input_shape[1] < min_size) or
                            (input_shape[2] is not None and input_shape[2] < min_size)):
                        raise ValueError('Input size must be at least ' +
                                         str(min_size) + 'x' + str(min_size) +
                                         ', got `input_shape=' + str(input_shape) + '`')
            else:
                if input_shape is not None:
                    if ((input_shape[1] is not None and input_shape[1] < min_size) or
                            (input_shape[2] is not None and input_shape[2] < min_size) or
                            (input_shape[3] is not None and input_shape[3] < min_size)):
                        raise ValueError('Input size must be at least ' +
                                         str(min_size) + 'x' + str(min_size) +
                                         ', got `input_shape=' + str(input_shape) + '`')
        else:
            if dimension == 2:
                if input_shape is not None:
                    if ((input_shape[0] is not None and input_shape[0] < min_size) or
                            (input_shape[1] is not None and input_shape[1] < min_size)):
                        raise ValueError('Input size must be at least ' +
                                         str(min_size) + 'x' + str(min_size) +
                                         ', got `input_shape=' + str(input_shape) + '`')
            else:
                if input_shape is not None:
                    if ((input_shape[0] is not None and input_shape[0] < min_size) or
                            (input_shape[1] is not None and input_shape[1] < min_size) or
                            (input_shape[2] is not None and input_shape[2] < min_size)):
                        raise ValueError('Input size must be at least ' +
                                         str(min_size) + 'x' + str(min_size) +
                                         ', got `input_shape=' + str(input_shape) + '`')

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        if concat_axis == 1:  # channels_first dim ordering
            if dimension == 2:
                _, rows, cols = input_shape
            else:
                _, rows, cols, axes = input_shape
        else:
            if dimension == 2:
                rows, cols, _ = input_shape
            else:
                rows, cols, axes, _ = input_shape

        if reduction != 0.0:
            if not (reduction <= 1.0 and reduction > 0.0):
                raise ValueError('`reduction` value must lie between 0.0 and 1.0')

        # layers in each dense block
        # if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        #     nb_layers = list(nb_layers_per_block)  # Convert tuple to list
        #
        #     if len(nb_layers) != (nb_dense_block + 1):
        #         raise ValueError('If `nb_dense_block` is a list, its length must be '
        #                          '(`nb_dense_block` + 1)')
        #
        #     bottleneck_nb_layers = nb_layers[-1]
        #     rev_layers = nb_layers[::-1]
        #     nb_layers.extend(rev_layers[1:])
        # else:
        #     bottleneck_nb_layers = nb_layers_per_block
        #     nb_layers = [nb_layers_per_block] * (2 * nb_dense_block + 1)

        # compute compression factor
        #compression = 1.0 - reduction

        # dilated densenet
        output_temp = dilated_dense_conv(dimension,
                                         num_classes,
                                         img_input,
                                         init_conv_filters,
                                         initial_kernel_size,
                                         kernel_init,
                                         random_seed_num,
                                         weight_decay,
                                         concat_axis,
                                         nb_dense_block,
                                         nb_layers_per_block,
                                         growth_rate,
                                         dropout_rate,
                                         is_densenet,
                                         is_dilated_conv,
                                         is_aggregation,
                                         dilation_rate_per_block,
                                         glam,
                                         glam_arrange_type,
                                         glam_input_short_cut,
                                         glam_final_conv,
                                         glam_position,
                                         block_prefix_num=1)

        output = organise_output(output_temp, output_shape, activation)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            input = get_source_inputs(input_tensor)
        else:
            input = img_input

        print(input.shape)
        print(output.shape)

        return Model(inputs=[input], outputs=[output], name='ms_fc_densenet')


def organise_output(input, output_shape, activation):
    pred = Reshape(output_shape)(input)
    pred = Permute((2, 1))(pred)
    return Activation(activation)(pred)


def dilated_dense_conv(dimension, num_classes, img_input, init_conv_filters, initial_kernel_size, kernel_init,
                       random_seed_num, weight_decay, concat_axis, nb_dense_block, nb_layers, growth_rate, dropout_rate,
                       is_densenet, is_dilated_conv, is_aggregation, dilation_rate_per_block, glam, glam_arrange_type,
                       glam_input_short_cut, glam_final_conv, glam_position, block_prefix_num):

    # Initial convolution
    if dimension == 2:
        x = Conv2D(init_conv_filters, initial_kernel_size,
                   kernel_initializer=kernel_init, padding='same',
                   name='initial_conv2D_%i' % block_prefix_num, use_bias=False,
                   kernel_regularizer=l2(weight_decay))(img_input)
    else:
        x = Conv3D(init_conv_filters, initial_kernel_size,
                   kernel_initializer=kernel_init, padding='same',
                   name='initial_conv3D_%i' % block_prefix_num, use_bias=False,
                   kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5, name='initial_bn_%i' % block_prefix_num)(x)
    x = Activation('elu')(x)

    if is_densenet:
        skip_list = []
        nb_filter = init_conv_filters
        # dense blocks and transition down blocks
        for block_idx in range(nb_dense_block):
            if glam:
                org_input = x
            else:
                org_input = None

            kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
            if is_dilated_conv:
                dilation_rate_lst = []
                for d in dilation_rate_per_block[block_idx]:
                    dilation_rate = (d, d) if dimension == 2 else (d, d, d)
                    dilation_rate_lst.append(dilation_rate)
                x, nb_filter, concat_list = __dilated_dense_block(dimension, x, kernel_size, kernel_init,
                                                                  random_seed_num, nb_layers[block_idx],
                                                                  dilation_rate_lst, nb_filter, growth_rate,
                                                                  dropout_rate=dropout_rate, weight_decay=weight_decay,
                                                                  return_concat_list=True,
                                                                  block_prefix='dilated_dense_%i_%i' % (block_idx,
                                                                                                        block_prefix_num))
            else:
                dilation_rate = (1, 1) if dimension == 2 else (1, 1, 1)
                x, nb_filter, concat_list = __dense_block(dimension, x, kernel_size, kernel_init, random_seed_num,
                                                          nb_layers[block_idx], dilation_rate, nb_filter, growth_rate,
                                                          dropout_rate=dropout_rate, weight_decay=weight_decay,
                                                          return_concat_list=True,
                                                          block_prefix='dense_%i_%i' % (block_idx, block_prefix_num))

            if glam:
                if glam_position == 'before_shortcut':
                    l = concatenate(concat_list[1:], axis=concat_axis)
                    if dimension == 2:
                        l = spatial_and_channel_squeeze_excite_block2D(l, kernel_init, arrange_type=glam_arrange_type,
                                                                       input_short_cut=glam_input_short_cut,
                                                                       final_conv=glam_final_conv)
                    else:
                        l = spatial_and_channel_squeeze_excite_block3D(l, kernel_init, arrange_type=glam_arrange_type,
                                                                       input_short_cut=glam_input_short_cut,
                                                                       final_conv=glam_final_conv)
                    x = concatenate([org_input, l], axis=concat_axis)
                else:
                    if dimension == 2:
                        x = spatial_and_channel_squeeze_excite_block2D(x, kernel_init, arrange_type=glam_arrange_type,
                                                                       input_short_cut=glam_input_short_cut,
                                                                       final_conv=glam_final_conv)
                    else:
                        x = spatial_and_channel_squeeze_excite_block3D(x, kernel_init, arrange_type=glam_arrange_type,
                                                                       input_short_cut=glam_input_short_cut,
                                                                       final_conv=glam_final_conv)
            skip_list.append(x)

        if is_aggregation:
            x = concatenate(skip_list, axis=concat_axis)

        # The last dense_block does not have a transition_down_block
        # return the concatenated feature maps without the concatenation of the input

    else:
        kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
        dilation_rate_lst = dilation_rate_per_block[0]
        filter_num_lst = [16, 32, 32, 64, 64, 128]
        i = 0
        for d, f in zip(dilation_rate_lst, filter_num_lst):
            i += 1
            dilation_rate = (d, d) if dimension == 2 else (d, d, d)
            x = __conv_block(dimension, x, kernel_size, kernel_init, random_seed_num, f, dropout_rate, False,
                             weight_decay, dilation_rate, block_prefix=name_or_none('dilated_conv', '_%i' % i))

            if glam:
                if dimension == 2:
                    x = spatial_and_channel_squeeze_excite_block2D(x, kernel_init, arrange_type=glam_arrange_type,
                                                                   input_short_cut=glam_input_short_cut,
                                                                   final_conv=glam_final_conv)
                else:
                    x = spatial_and_channel_squeeze_excite_block3D(x, kernel_init, arrange_type=glam_arrange_type,
                                                                   input_short_cut=glam_input_short_cut,
                                                                   final_conv=glam_final_conv)

    if dimension == 2:
        output = Conv2D(num_classes, (1, 1), activation='linear', padding='same', kernel_initializer=kernel_init,
                        use_bias=False)(x)
    else:
        output = Conv3D(num_classes, (1, 1, 1), activation='linear', padding='same', kernel_initializer=kernel_init,
                        use_bias=False)(x)

    return output


def __conv_block(dimension, ip, kernel_size, kernel_init, random_seed_num, nb_filter, bottleneck=False,
                 dropout_rate=None, weight_decay=1e-4, dilation_rate=(1, 1, 1), block_prefix=None):
    '''
    Adds a convolution layer (with batch normalization and elu),
    and optionally a bottleneck layer.
    # Arguments
        ip: Input tensor
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        bottleneck: if True, adds a bottleneck convolution block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        block_prefix: str, for unique layer naming
     # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        output tensor of block
    '''
    with K.name_scope('ConvBlock'):

        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                               name=name_or_none(block_prefix, '_bn'))(ip)
        x = Activation('elu')(x)

        if bottleneck:
            inter_channel = nb_filter * 4

            if dimension == 2:
                x = Conv2D(inter_channel, (1, 1), kernel_initializer=kernel_init,
                           padding='same', use_bias=False, dilation_rate=dilation_rate,
                           kernel_regularizer=l2(weight_decay),
                           name=name_or_none(block_prefix, '_bottleneck_conv2D'))(x)
            else:
                x = Conv3D(inter_channel, (1, 1, 1), kernel_initializer=kernel_init,
                           padding='same', use_bias=False, dilation_rate=dilation_rate,
                           kernel_regularizer=l2(weight_decay),
                           name=name_or_none(block_prefix, '_bottleneck_conv3D'))(x)
            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                                   name=name_or_none(block_prefix, '_bottleneck_bn'))(x)
            x = Activation('elu')(x)

        if dimension == 2:
            x = Conv2D(nb_filter, kernel_size, kernel_initializer=kernel_init, padding='same',
                       use_bias=False, dilation_rate=dilation_rate, name=name_or_none(block_prefix, '_conv2D'))(x)
        else:
            x = Conv3D(nb_filter, kernel_size, kernel_initializer=kernel_init, padding='same',
                       use_bias=False, dilation_rate=dilation_rate, name=name_or_none(block_prefix, '_conv3D'))(x)
        if dropout_rate:
            x = Dropout(dropout_rate, seed=random_seed_num)(x)

    return x


def __dense_block(dimension, x, kernel_size, kernel_init, random_seed_num, nb_layers, dilation_rate, nb_filter,
                  growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True,
                  return_concat_list=False, block_prefix=None):
    '''
    Build a dense_block where the output of each conv_block is fed
    to subsequent ones
    # Arguments
        x: input keras tensor
        nb_layers: the number of conv_blocks to append to the model
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        growth_rate: growth rate of the dense block
        bottleneck: if True, adds a bottleneck convolution block to
            each conv_block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: if True, allows number of filters to grow
        return_concat_list: set to True to return the list of
            feature maps along with the actual output
        block_prefix: str, for block unique naming
    # Return
        If return_concat_list is True, returns a list of the output
        keras tensor, the number of filters and a list of all the
        dense blocks added to the keras tensor
        If return_concat_list is False, returns a list of the output
        keras tensor and the number of filters
    '''
    with K.name_scope('DenseBlock'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x_list = [x]

        for i in range(nb_layers):
            cb = __conv_block(dimension, x, kernel_size, kernel_init, random_seed_num, growth_rate, bottleneck,
                              dropout_rate, weight_decay, dilation_rate, block_prefix=name_or_none(block_prefix, '_%i' % i))
            x_list.append(cb)

            x = concatenate([x, cb], axis=concat_axis)

            if grow_nb_filters:
                nb_filter += growth_rate

        if return_concat_list:
            return x, nb_filter, x_list
        else:
            return x, nb_filter


def __dilated_dense_block(dimension, x, kernel_size, kernel_init, random_seed_num, nb_layers, dilation_rate_lst,
                          nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                          grow_nb_filters=True, return_concat_list=False, block_prefix=None):
    '''
    Build a dense_block where the output of each conv_block is fed
    to subsequent ones
    # Arguments
        x: input keras tensor
        nb_layers: the number of conv_blocks to append to the model
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        growth_rate: growth rate of the dense block
        bottleneck: if True, adds a bottleneck convolution block to
            each conv_block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: if True, allows number of filters to grow
        return_concat_list: set to True to return the list of
            feature maps along with the actual output
        block_prefix: str, for block unique naming
    # Return
        If return_concat_list is True, returns a list of the output
        keras tensor, the number of filters and a list of all the
        dense blocks added to the keras tensor
        If return_concat_list is False, returns a list of the output
        keras tensor and the number of filters
    '''
    with K.name_scope('DilatedDenseBlock'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x_list = [x]

        for i in range(nb_layers):
            dilation_rate = dilation_rate_lst[i]
            cb = __conv_block(dimension, x, kernel_size, kernel_init, random_seed_num, growth_rate, bottleneck,
                              dropout_rate, weight_decay, dilation_rate,
                              block_prefix=name_or_none(block_prefix, '_%i' % i))
            x_list.append(cb)

            x = concatenate([x, cb], axis=concat_axis)

            if grow_nb_filters:
                nb_filter += growth_rate

        if return_concat_list:
            return x, nb_filter, x_list
        else:
            return x, nb_filter


def name_or_none(prefix, name):
    return prefix + name if (prefix is not None and name is not None) else None

