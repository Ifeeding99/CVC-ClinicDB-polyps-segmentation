import torch
import torch.nn as nn

# to calculate the output of the conv2d and convtransposed2d layers I used the standard formulas also present in the torch docs
# I will put them below for convenience
# for conv2d: H_out = floor((H_in + 2 * padding - dilation * (kernel_size - 1) - 1)/ stride + 1)
# for convtransposed2d: H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

img_size = 256

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride, padding, n_layers):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_size = k_size
        self.stride = stride
        self.padding = padding
        self.n_layers = n_layers # the number of convolution layers
        self.first_conv = nn.Conv2d(self.in_channels, self.out_channels, self.k_size, self.stride, self.padding)

        self.block = nn.ModuleList([
            nn.Conv2d(self.out_channels, self.out_channels, self.k_size, self.stride, self.padding),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        ])

        self.layers = nn.ModuleList([
            self.block for i in range(self.n_layers - 1)
        ])

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.layers:
            for layer in block:
                x = layer(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.input_channels = input_channels
        self.pool = nn.MaxPool2d(2)

        # using kernel size of 3, stride of 1 and padding of 1 keep the input image dimensions unchanged, as you can verify
        self.descending_block_1 = ConvBlock(self.input_channels, out_channels=32, k_size=3, stride=1, padding=1, n_layers=3)
        self.descending_block_2 = ConvBlock(in_channels=32, out_channels=64, k_size=3, stride=1, padding=1, n_layers=3)
        self.descending_block_3 = ConvBlock(in_channels=64, out_channels=128, k_size=3, stride=1, padding=1, n_layers=3)
        self.descending_block_4 = ConvBlock(in_channels=128, out_channels=256, k_size=3, stride=1, padding=1, n_layers=3)
        self.descending_block_5 = ConvBlock(in_channels=256, out_channels=512, k_size=3, stride=1, padding=1, n_layers=3)

        # I used transposed conv instead of upsampling
        # the chosen kernel size, stride and padding allow an input image to be doubled in height and width, as you can verify
        self.upconv_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)
        self.upconv_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.upconv_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.upconv_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0)

        self.ascending_block_1 = ConvBlock(in_channels=256*2, out_channels=256, k_size=3, stride=1, padding=1, n_layers=3)
        self.ascending_block_2 = ConvBlock(in_channels=128*2, out_channels=128, k_size=3, stride=1, padding=1, n_layers=3)
        self.ascending_block_3 = ConvBlock(in_channels=64*2, out_channels=64, k_size=3, stride=1, padding=1, n_layers=3)
        self.ascending_block_4 = ConvBlock(in_channels=32*2, out_channels=32, k_size=3, stride=1, padding=1, n_layers=3)

        self.final_conv = nn.Conv2d(in_channels=32, out_channels=1,kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.descending_block_1(x)
        block_1_output = x

        x = self.pool(x)
        x = self.descending_block_2(x)
        block_2_output = x

        x = self.pool(x)
        x = self.descending_block_3(x)
        block_3_output = x

        x = self.pool(x)
        x = self.descending_block_4(x)
        block_4_output = x

        x = self.pool(x)
        x = self.descending_block_5(x)

        x = self.upconv_1(x)
        x = torch.cat([block_4_output, x], dim=1) # concatenated along the channel dimension
        x = self.ascending_block_1(x)

        x = self.upconv_2(x)
        x = torch.cat([block_3_output, x], dim=1)
        x = self.ascending_block_2(x)

        x = self.upconv_3(x)
        x = torch.cat([block_2_output,x], dim=1)
        x = self.ascending_block_3(x)

        x = self.upconv_4(x)
        x = torch.cat([block_1_output, x], dim=1)
        x = self.ascending_block_4(x)

        x = self.final_conv(x)
        return x
