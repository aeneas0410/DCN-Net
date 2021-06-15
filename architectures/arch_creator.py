
def generate_model(gen_conf, train_conf):
    mode = gen_conf['validation_mode']
    approach = train_conf['approach']

    model = None
    if approach == 'deepmedic' :
        from .deepmedic import generate_deepmedic_model
        model =  generate_deepmedic_model(gen_conf, train_conf)
    if approach == 'livianet' :
        from .livianet import generate_livianet_model
        model = generate_livianet_model(gen_conf, train_conf)
    if approach == 'unet' :
        from .unet import generate_unet_model
        model = generate_unet_model(gen_conf, train_conf)
    if approach == 'uresnet' :
        from .uresnet import generate_uresnet_model
        model = generate_uresnet_model(gen_conf, train_conf)
    if approach == 'cenet' :
        from .cenet import generate_cenet_model
        model = generate_cenet_model(gen_conf, train_conf)
    if approach == 'cc_3d_fcn' :
        from .cc_3d_fcn import generate_cc_3d_fcn_model
        model = generate_cc_3d_fcn_model(gen_conf, train_conf)
    if approach == 'wnet' :
        from .wnet import generate_wunet_model
        model = generate_wunet_model(gen_conf, train_conf)
    if approach == 'fc_densenet':
        from .fc_densenet import generate_fc_densenet_model
        model = generate_fc_densenet_model(gen_conf, train_conf)
    if approach == 'fc_densenet_ms':
        from .fc_densenet_ms import generate_fc_densenet_ms
        model = generate_fc_densenet_ms(gen_conf, train_conf)
    if approach == 'fc_densenet_dilated':
        from .fc_densenet_dilated import generate_fc_densenet_dilated
        model = generate_fc_densenet_dilated(gen_conf, train_conf)
    if approach == 'densenet_dilated':
        from .densenet_dilated import generate_densenet_dilated
        model = generate_densenet_dilated(gen_conf, train_conf)
    if approach == 'fc_capsnet':
        from .fc_capsnet import generate_fc_capsnet_model
        model = generate_fc_capsnet_model(gen_conf, train_conf, mode)
    if approach == 'attention_unet':
        from .attention_unet import generate_attention_unet_model
        model = generate_attention_unet_model(gen_conf, train_conf)
    if approach == 'attention_se_fcn':
        from .attention_se_fcn import generate_attention_se_fcn_model
        model = generate_attention_se_fcn_model(gen_conf, train_conf)
    if approach == 'multires_net':
        from .multires_net import generate_multires_net
        model = generate_multires_net(gen_conf, train_conf)
    if approach == 'attention_onet':
        from .attention_onet import generate_attention_onet_model
        model = generate_attention_onet_model(gen_conf, train_conf)
    if approach == 'fc_rna':
        from .fc_rna import generate_fc_rna_model
        model = generate_fc_rna_model(gen_conf, train_conf)

    if approach == 'cgan':
        from .generator import build_fc_dense_contextnet, build_unet
        from .discriminator import build_dilated_densenet, build_patch_gan
        from .cgan import build_cgan
        d_model = build_dilated_densenet(gen_conf, train_conf) # optional
        g_model = build_fc_dense_contextnet(gen_conf, train_conf) # optional
        gan_model = build_cgan(gen_conf, train_conf, g_model, d_model)
        model = [g_model, d_model, gan_model]

    if approach == 'cgan_sar':
        from .generator import build_fc_dense_contextnet, build_g_dilated_densenet, build_unet
        from .discriminator import build_d_dilated_densenet, build_patch_gan
        from .cgan import build_cgan_sar

        g_network = train_conf['GAN']['generator']['g_network']
        d_network = train_conf['GAN']['discriminator']['d_network']

        if g_network == 'fc_dense_contextnet':
            g_model = build_fc_dense_contextnet(gen_conf, train_conf)  # optional
        elif g_network == 'dilated_densenet':
            g_model = build_g_dilated_densenet(gen_conf, train_conf)
        elif g_network == 'u_net':
            g_model = build_unet(gen_conf, train_conf)  # optional
        else:
            raise NotImplementedError('choose fc_dense_contextnet or unet')

        if d_network == 'dilated_densenet':
            d_model = build_d_dilated_densenet(gen_conf, train_conf) # optional
        elif d_network == 'patch_gan':
            d_model = build_patch_gan(gen_conf, train_conf) # optional
        else:
            raise NotImplementedError('choose dilated_densenet or patch_gan')

        gan_model = build_cgan_sar(gen_conf, train_conf, g_model, d_model)
        model = [g_model, d_model, gan_model]

    if approach in ['single_cyclegan_sar', 'multi_cyclegan_sar', 'composite_cyclegan_sar']:
        from .generator import build_fc_dense_contextnet, build_g_dilated_densenet, build_unet
        from .discriminator import build_d_dilated_densenet, build_patch_gan
        from .cyclegan import build_multi_cyclegan_sar_fwd, build_multi_cyclegan_sar_bwd, build_single_cyclegan_sar_fwd, \
            build_single_cyclegan_sar_bwd, build_cyclegan_sar

        g_network = train_conf['GAN']['generator']['g_network']
        d_network = train_conf['GAN']['discriminator']['d_network']
        dataset = train_conf['dataset']

        if g_network == 'fc_dense_contextnet':
            gen_conf['dataset_info'][dataset]['image_modality'] = ['B1_real', 'B1_imag']
            #gen_conf['dataset_info'][dataset]['image_modality'] = ['B1_real','B1_imag','Epsilon','Rho','Sigma']  #['B1_mag'] #['B1_real','B1_imag','Sigma']
            train_conf['GAN']['generator']['num_classes'] = 2 #1
            train_conf['GAN']['generator']['model_name'] = 'g_model_fc_dense_contextnet_XtoY'
            g_model_XtoY = build_fc_dense_contextnet(gen_conf, train_conf)  # generator: A -> B

            gen_conf['dataset_info'][dataset]['image_modality'] = ['sar', 'sar']
            #gen_conf['dataset_info'][dataset]['image_modality'] = ['sar','sar','sar','sar','sar'] #['sar']
            train_conf['GAN']['generator']['num_classes'] = 2 #1 #3
            train_conf['GAN']['generator']['model_name'] = 'g_model_fc_dense_contextnet_YtoX'
            g_model_YtoX = build_fc_dense_contextnet(gen_conf, train_conf)  # generator: B -> A
        elif g_network == 'dilated_densenet':
            gen_conf['dataset_info'][dataset]['image_modality'] = ['B1_real','B1_imag','Epsilon','Rho','Sigma']  #['B1_real','B1_imag','Sigma']
            train_conf['GAN']['generator']['num_classes'] = 2 #1
            train_conf['GAN']['generator']['model_name'] = 'g_model_dilated_densenet_XtoY'
            g_model_XtoY = build_g_dilated_densenet(gen_conf, train_conf)  # generator: A -> B

            gen_conf['dataset_info'][dataset]['image_modality'] = ['sar','sar','sar','sar','sar']
            train_conf['GAN']['generator']['num_classes'] = 2 #1 #3
            train_conf['GAN']['generator']['model_name'] = 'g_model_dilated_densenet_YtoX'
            g_model_YtoX = build_g_dilated_densenet(gen_conf, train_conf)  # generator: B -> A
        elif g_network == 'u_net':
            gen_conf['dataset_info'][dataset]['image_modality'] = ['B1_real','B1_imag','Epsilon','Rho','Sigma']  #['B1_real','B1_imag','Sigma']
            train_conf['GAN']['generator']['num_classes'] = 2 #1
            train_conf['GAN']['generator']['model_name'] = 'g_model_u_net_XtoY'
            g_model_XtoY = build_unet(gen_conf, train_conf)  # generator: A -> B

            gen_conf['dataset_info'][dataset]['image_modality'] = ['sar','sar','sar','sar','sar']
            train_conf['GAN']['generator']['num_classes'] = 2 #1 #3
            train_conf['GAN']['generator']['model_name'] = 'g_model_u_net_YtoX'
            g_model_YtoX = build_unet(gen_conf, train_conf)  # generator: B -> A
        else:
            raise NotImplementedError('choose fc_dense_contextnet or unet')

        if d_network == 'dilated_densenet':
            gen_conf['dataset_info'][dataset]['image_modality'] = ['B1_real', 'B1_imag']
            #gen_conf['dataset_info'][dataset]['image_modality'] = ['B1_real','B1_imag','Epsilon','Rho','Sigma']  #['B1_mag'] #['B1_real','B1_imag','Sigma']
            train_conf['GAN']['generator']['num_classes'] = 2 #1
            train_conf['GAN']['discriminator']['model_name'] = 'd_model_dilated_densenet_Y'
            d_model_Y = build_d_dilated_densenet(gen_conf, train_conf)  # discriminator: B -> [real/fake]

            gen_conf['dataset_info'][dataset]['image_modality'] = ['sar', 'sar']
            #gen_conf['dataset_info'][dataset]['image_modality'] = ['sar','sar','sar','sar','sar'] #['sar']
            train_conf['GAN']['generator']['num_classes'] = 2 #1 #3
            train_conf['GAN']['discriminator']['model_name'] = 'd_model_dilated_densenet_X'
            d_model_X = build_d_dilated_densenet(gen_conf, train_conf)  # discriminator: A -> [real/fake]
        elif d_network == 'patch_gan':
            gen_conf['dataset_info'][dataset]['image_modality'] = ['B1_real','B1_imag','Epsilon','Rho','Sigma'] #['B1_mag'] #['B1_real','B1_imag','Sigma']
            train_conf['GAN']['generator']['num_classes'] = 2 #1
            train_conf['GAN']['discriminator']['model_name'] = 'd_model_patch_gan_Y'
            d_model_Y= build_patch_gan(gen_conf, train_conf)  # discriminator: B -> [real/fake]

            gen_conf['dataset_info'][dataset]['image_modality'] = ['sar','sar','sar','sar','sar'] #['sar']
            train_conf['GAN']['generator']['num_classes'] = 2 #1 #3
            train_conf['GAN']['discriminator']['model_name'] = 'd_model_patch_gan_X'
            d_model_X = build_patch_gan(gen_conf, train_conf)  # discriminator: A -> [real/fake]
        else:
            raise NotImplementedError('choose dilated_densenet or patch_gan')

        # composite: A -> B -> [real/fake, A]
        gen_conf['dataset_info'][dataset]['image_modality'] = ['B1_real', 'B1_imag']
        #gen_conf['dataset_info'][dataset]['image_modality'] = ['B1_real','B1_imag','Epsilon','Rho','Sigma']  #['B1_mag'] #['B1_real','B1_imag','Sigma']
        train_conf['GAN']['generator']['num_classes'] = 2

        if approach == 'single_cyclegan_sar':
            c_model_XtoY = build_single_cyclegan_sar_fwd(gen_conf, train_conf, g_model_XtoY, d_model_Y, g_model_YtoX)
            # composite: B -> A -> [real/fake, B]
            c_model_YtoX = build_single_cyclegan_sar_bwd(gen_conf, train_conf, g_model_YtoX, d_model_X, g_model_XtoY)

            model = [g_model_XtoY, g_model_YtoX, d_model_Y, d_model_X, c_model_XtoY, c_model_YtoX]
        elif approach == 'multi_cyclegan_sar':
            c_model_XtoY = build_multi_cyclegan_sar_fwd(gen_conf, train_conf, g_model_XtoY, d_model_Y, g_model_YtoX)
            # composite: B -> A -> [real/fake, B]
            c_model_YtoX = build_multi_cyclegan_sar_bwd(gen_conf, train_conf, g_model_YtoX, d_model_X, g_model_XtoY)

            model = [g_model_XtoY, g_model_YtoX, d_model_Y, d_model_X, c_model_XtoY, c_model_YtoX]
        elif approach == 'composite_cyclegan_sar':
            c_model = build_cyclegan_sar(gen_conf, train_conf, g_model_XtoY, d_model_Y, g_model_YtoX, d_model_X)
            model = [g_model_XtoY, g_model_YtoX, d_model_Y, d_model_X, c_model]

    print(gen_conf['args'])
    if train_conf['num_retrain'] > 0:
        print('retraining...#' + str(train_conf['num_retrain']))

    return model


