import keras.models as models
import keras.layers as layers

class DoubleConv(layers.Layer):
    def __init__(self, size, batch_norm = False):
        super(DoubleConv, self).__init__()
        self.size = size
        self.isBatchNorm = batch_norm

        self.conv1 = layers.Conv2D(size, (3,3), padding='same')
        self.conv2 = layers.Conv2D(size, (3,3), padding='same')

    def call(self, input):
        x = self.conv1(input)
        if self.isBatchNorm:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = self.conv2(x)
        if self.isBatchNorm:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

class DeconvBlock(layers.Layer):
    def __init__(self, size, conv_layer):
        super(DeconvBlock, self).__init__()
        self.size = size
        self.conv_layer = conv_layer
        self.deconv = layers.Conv2DTranspose(size, (3,3), strides=(2,2), padding='same')

    def call(self, input):
        x = self.deconv(input)
        x = layers.concatenate([x, self.conv_layer])
        return x

class UNet(models.Model):
    def __init__(self, input_layer, n_size, out_filters_nb):
        super(UNet, self).__init__()
        self._input_layer = input_layer
        self.size = n_size
        self.out_size = out_filters_nb


        
