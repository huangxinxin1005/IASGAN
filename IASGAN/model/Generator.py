import segmentation_models as sm

def define_generator():
    backbone = 'resnet50'
    backbone_weights = None
    class_num = 3
    pyramid_block_filters = 256
    model = sm.FPN(backbone_name=backbone,
                   encoder_weights=backbone_weights,
                   classes=class_num,
                   pyramid_block_filters=pyramid_block_filters, activation='sigmoid')
    model.summary()
    #model.compile(optimizer=adam_v2.Adam(learning_rate=0.0001), loss=sm.losses.binary_focal_loss)
    return model