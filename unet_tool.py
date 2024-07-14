'''
Date: 2024 - WBF Group - Tongji University
Authors: Lin Shicong, Mo Tongtong
Description: U-Net model & SNR function
'''
import torch
import torch.nn as nn

# ===================-==================== U-Net ==========================================
class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Downsample, self).__init__()
        self.Conv_BN_ReLU2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0),
        )

        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        out_x = self.Conv_BN_ReLU2(x)
        out = self.downsample(out_x)
        return out_x, out

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Upsample, self).__init__()
        self.Conv_BN_ReLU2 = nn.Sequential(
            nn.Conv2d(in_ch, 2*out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2*out_ch),
            nn.ReLU(),
            nn.Conv2d(2*out_ch, 2*out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2*out_ch),
            nn.ReLU()
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(2*out_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.Conv2d(2 * out_ch, out_ch * 2 * 2, kernel_size=3, stride=1, padding=1),
            # nn.PixelShuffle(2),  # Sub-Pixel conv
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, out_x):
        out = self.Conv_BN_ReLU2(x)
        out = self.upsample(out)
        assert(out.shape[2:3] == out_x.shape[2:3])
        out = torch.cat((out, out_x), dim=1)
        return out

class Unet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Unet, self).__init__()
        out_ch = [2**(5+i) for i in range(3)] # [32 64 128]

        ## define Downsampling
        self.d1 = Downsample(in_channel, out_ch[0])
        self.d2 = Downsample(out_ch[0], out_ch[1])

        ## define Upsampling
        self.u1 = Upsample(out_ch[1], out_ch[1])
        self.u2 = Upsample(out_ch[2], out_ch[0])

        # define output
        self.o = nn.Sequential(
            nn.Conv2d(out_ch[1], out_ch[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch[0]),
            nn.ReLU(),
            nn.Conv2d(out_ch[0], out_channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out_1, out1 = self.d1(x)
        out_2, out2 = self.d2(out1)
        out3 = self.u1(out2, out_2)
        out4 = self.u2(out3, out_1)
        out = self.o(out4)
        return out

# ====================================== modified U-Net =========================================
class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        # x = self.bn(x)
        x = self.relu(x)

        return x

class Unet2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Unet2, self).__init__()
        out_ch = [2 ** (5 + i) for i in range(3)]  # [32 64 128]

        ## define Downsampling
        self.d1 = Conv2d(in_channel, out_ch[0])
        self.d2 = Conv2d(out_ch[0], out_ch[0])
        self.d3 = Conv2d(out_ch[0], out_ch[1])
        self.d4 = Conv2d(out_ch[1], out_ch[1])

        ## define Upsampling
        self.u1 = Conv2d(out_ch[1], out_ch[2])
        self.u2 = Conv2d(out_ch[2], out_ch[2])
        self.u3 = Conv2d(out_ch[2], out_ch[1])
        self.u4 = Conv2d(out_ch[1], out_ch[1])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.upsample1 = nn.ConvTranspose2d(out_ch[2], out_ch[1], kernel_size=3, stride=2, padding=1,
                                               output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(out_ch[1], out_ch[0], kernel_size=3, stride=2, padding=1,
                                               output_padding=1)

        # define output
        self.o = nn.Sequential(
            Conv2d(out_ch[1], out_ch[0]),
            nn.Conv2d(out_ch[0], out_channel, kernel_size=3, stride=1, padding=1)
        )

        # define dropout
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        # downsample
        out_1 = self.d2(self.d1(x))
        out1 = self.maxpool(out_1)
        out1 = self.drop(out1)
        out_2 = self.d4(self.d3(out1))
        out2 = self.maxpool(out_2)
        out2 = self.drop(out2)
        # upsample
        out3 = self.upsample1(self.u2(self.u1(out2)))
        out3 = self.drop(out3)
        out4 = torch.cat([out3, out_2], dim=1)
        out5 = self.upsample2(self.u4(self.u3(out4)))
        out5 = self.drop(out5)
        out6 = torch.cat([out5, out_1], dim=1)
        out = self.o(out6)

        return out

# ===================================== 2D SNR function =====================================
def snr_fn2d(y_hat, y):
    tmp1 = torch.sum(torch.mul(y,y) , dim = (2,3))
    tmp2 = torch.sum(torch.mul(y-y_hat, y-y_hat), dim = (2,3))
    out = tmp1/tmp2
    out = torch.squeeze(out)
    out = 10*torch.log10(out)
    out = torch.mean(out)
    return out.cpu().item()
