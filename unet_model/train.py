import keras
import keras.backend as K
from keras.models import Model
from unet_keras import UNet

def train():
    # Load dataset 
    # ...
    img_size = (128,128,1)
    input_layer = keras.layers.Input(img_size)
    output_layer = UNet(input_layer, size=16, out = 1)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss=custom_loss, optimizer='adam', metrics=["accuracy"])
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks = [checkpoint]

    batch_size = 20
    epochs = 400
    # Training
    hist = model.fit(x_train, y_train, batch_size = batch_size,\
            epochs = epochs, validation_data = (x_val, y_val),  \
            callbacks = callbacks)
    model.summary()

if __name__ == '__main__':
    train()