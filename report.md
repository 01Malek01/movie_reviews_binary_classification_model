first try settings :
optimizer="rmsprop",
loss="binary_crossentropy",
metrics=["accuracy"],

## epochs = 20
batch_size=512,

result:
Test loss: 0.5823825597763062
Test accuracy: 0.8596799969673157

## I noticed that val loss is increasing starting from epoch 5, and the val accuracy was at its peak at epoch 5 and slightly dropped till epoch 20.

second try settings :
same exact settings but:
## epochs = 5

result:
Test loss: 0.30768507719039917
Test accuracy: 0.8751599788665771
