from torchinfo import summary
from unet import *

model = UNet(3, 2)
# Print a summary using torchinfo (uncomment for actual output)
summary(model=model, 
        input_size=(1, 3, 720, 720), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)