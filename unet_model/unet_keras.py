import keras.models as models
import keras.layers as layers

def double_conv(output_size, layer, max_pool=True):
    conv_1 = layers.Conv2D(output_size, (3,3), padding='same', activation='relu')(layer)
    conv_2 = layers.Conv2D(output_size, (3,3), padding='same', activation='relu')(conv_1)
    return conv_2
    
def deconv_layer(output_size, conv_layer, layer_last):
    deconv = layers.Conv2DTranspose(output_size, (3,3), strides=(2,2), padding='same')(layer_last)
    uconv = layers.concatenate([deconv, conv_layer])
    return uconv
    
def UNet(input_layer, size, out):
    convDown1 = double_conv(size, input_layer)
    poolDown1 = layers.MaxPooling2D((2,2))(convDown1)
    dropoutD1 = layers.Dropout(0.25)(poolDown1)
    
    convDown2 = double_conv(size*2, dropoutD1)
    poolDown2 = layers.MaxPooling2D((2,2))(convDown2)
    
    convDown3 = double_conv(size*4, poolDown2)
    poolDown3 = layers.MaxPooling2D((2,2))(convDown3)
    
    convDown4 = double_conv(size*8, poolDown3)
    poolDown4 = layers.MaxPooling2D((2,2))(convDown4)
    
    convMiddle = double_conv(size*16, poolDown4, max_pool=False)

    deconv4 = deconv_layer(size*8, convDown4, convMiddle)
    convUp4 = double_conv(size*8, deconv4, max_pool=False)
     
    deconv3 = deconv_layer(size*4, convDown3, convUp4)
    convUp3 = double_conv(size*4, deconv3, max_pool=False)
        
    deconv2 = deconv_layer(size*2, convDown2, convUp3)
    convUp2 = double_conv(size*2, deconv2, max_pool=False)
        
    deconv1 = deconv_layer(size, convDown1, convUp2)
    convUp1 = double_conv(size, deconv1, max_pool=False)

    convOut = layers.Conv2D(out, (1,1), padding='same', activation='linear')(convUp1)
    
    return convOut