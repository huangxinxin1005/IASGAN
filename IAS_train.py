from keras.optimizers import rmsprop_v2, adam_v2
from keras.models import Model
from keras.layers import Input
from keras.layers import BatchNormalization
import segmentation_models as sm

def define_gan(g_model, d_model, image_shape):
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False

    in_src = Input(shape=image_shape)
    gen_out = g_model(in_src)
    dis_out = d_model(gen_out)
    model = Model(in_src, [dis_out, gen_out])
    opt = adam_v2.Adam(learning_rate=0.0001)
    model.compile(loss=['mae', sm.losses.binary_focal_loss], optimizer=opt)
    model.summary()
    return model

def train(d_model, g_model, gan_model, epochs):

    for i in range(196001, epochs):
        x_train, y_train =
        # update the discriminator
        X_fakeB = g_model.predict(x_train)
        y_real = d_model(y_train)
        y_fake = d_model(X_fakeB)
        d_loss_real = d_model.train_on_batch(y_train, y_real)
        d_loss_fake = d_model.train_on_batch(X_fakeB, y_fake)
        # update the generator
        g_loss = gan_model.train_on_batch(x_train, [y_real, y_train])

        if i % 200 == 0:
            plt.figure(figsize=(24, 8))
            plt.subplot(1, 3, 1)
            plt.imshow(x_train[0, :, :, :])
            plt.xlim(0, 250)
            plt.ylim(0, 250)

            plt.subplot(1, 3, 2)
            plt.imshow(X_fakeB[0, :, :, :])
            plt.xlim(0, 250)
            plt.ylim(0, 250)

            plt.subplot(1, 3, 3)
            plt.imshow(y_train[0, :, :, :])
            plt.xlim(0, 250)
            plt.ylim(0, 250)

            filename1 =
            plt.savefig(filename1)
            plt.close()
        if i % 2000 == 0:
            filename2 =
            g_model.save(filename2)
        if i % epochs == 0:
            filename3 =
            d_model.save(filename3)
