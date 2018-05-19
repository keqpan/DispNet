import torch
import torch.nn as nn


def downsample_conv(in_planes, out_planes, kernel_size=3, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
    )


def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes, kernel_size=4):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )


def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]


class DispNet(nn.Module):

    def __init__(self, alpha=10, beta=0.01):
        super(DispNet, self).__init__()

        self.alpha = alpha
        self.beta = beta

        conv_planes = [64, 128, 256, 512, 512, 1024]
        self.conv1 = downsample_conv(6,              conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3a = downsample_conv(conv_planes[1], conv_planes[2], kernel_size=5)
        self.conv3b = downsample_conv(conv_planes[2], conv_planes[2], stride=1)
        self.conv4a = downsample_conv(conv_planes[2], conv_planes[3])
        self.conv4b = downsample_conv(conv_planes[3], conv_planes[3], stride=1)
        self.conv5a = downsample_conv(conv_planes[3], conv_planes[4])
        self.conv5b = downsample_conv(conv_planes[4], conv_planes[4], stride=1)
        self.conv6a = downsample_conv(conv_planes[4], conv_planes[5])
        self.conv6b = downsample_conv(conv_planes[5], conv_planes[5], stride=1)

        upconv_planes = [512, 256, 128, 64, 32]
        self.upconv5 = upconv(conv_planes[5],   upconv_planes[0])
        self.upconv4 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv3 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv2 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv1 = upconv(upconv_planes[3], upconv_planes[4])


        self.iconv5 = conv(1 + upconv_planes[0] + conv_planes[4], upconv_planes[0])
        self.iconv4 = conv(1 + upconv_planes[1] + conv_planes[3], upconv_planes[1])
        self.iconv3 = conv(1 + upconv_planes[2] + conv_planes[2], upconv_planes[2])
        self.iconv2 = conv(1 + upconv_planes[3] + conv_planes[1], upconv_planes[3])
        self.iconv1 = conv(1 + upconv_planes[4] + conv_planes[0], upconv_planes[4])

        self.pr6 = predict_disp(conv_planes[5])
        self.upsample6 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pr5 = predict_disp(upconv_planes[0])
        self.upsample5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pr4 = predict_disp(upconv_planes[1])
        self.upsample4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pr3 = predict_disp(upconv_planes[2])
        self.upsample3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pr2 = predict_disp(upconv_planes[3])
        self.upsample2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pr1 = predict_disp(upconv_planes[4])
        self.upsample1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out_conv1 = self.conv1(x)             # 64, 270, 480
        print("out_conv1", out_conv1.size())
        out_conv2 = self.conv2(out_conv1)     # 128, 135, 240
        print("out_conv2", out_conv2.size())
        out_conv3a = self.conv3a(out_conv2)
        out_conv3b = self.conv3b(out_conv3a)  # 256, 68, 120
        print("out_conv3b", out_conv3b.size())
        out_conv4a = self.conv4a(out_conv3b)
        out_conv4b = self.conv4b(out_conv4a)  # 512, 34, 60
        print("out_conv4b", out_conv4b.size())
        out_conv5a = self.conv5a(out_conv4b)
        out_conv5b = self.conv5b(out_conv5a)  # 512, 17, 30
        print("out_conv5b", out_conv5b.size())
        
        disp5 = self.pr5(out_conv5b)
        print("disp5", disp5.size())
#         disp5 = crop_like(disp5, out_conv5b)
#         print("disp5", disp5.size())

        out_upconv4 = self.upconv4(out_conv4b)
        print("out_upconv4", out_upconv4.size())        
        concat4 = torch.cat((crop_like(out_upconv4, out_conv4b), disp5, out_conv4b), 1)
        out_iconv4 = self.iconv4(concat4)
        disp4 = self.pr4(out_iconv4)
        print("disp4", disp4.size())
        disp4 = crop_like(disp4, out_conv3b)
        print("disp4", disp4.size())
        
        out_upconv3 = self.upconv3(out_iconv4)
        print("out_upconv3", out_upconv3.size())    
        concat3 = torch.cat((crop_like(out_upconv3, out_conv3b), disp4, out_conv3b), 1)
        out_iconv3 = self.iconv3(concat3)
        disp3 = self.pr3(out_iconv3)
        print("disp3", disp3.size())
        disp3 = crop_like(disp3, out_conv2b)
        print("disp3", disp3.size())

        out_upconv2 = self.upconv3(out_iconv3)
        print("out_upconv2", out_upconv2.size())
        concat2 = torch.cat((crop_like(out_upconv2, out_conv2), disp3, out_conv2), 1)
        out_iconv2 = self.iconv2(concat2)
        disp2 = self.pr2(out_iconv2)
        print("disp2", disp2.size())
        disp2 = crop_like(disp2, out_conv2)
        print("disp2", disp2.size())
              
        out_upconv1 = self.upconv1(out_iconv2)
        print("out_upconv1", out_upconv1.size())
        concat1 = torch.cat((crop_like(out_upconv1, out_conv1), disp2, out_conv1), 1)
        out_iconv1 = self.iconv1(concat1)
        disp1 = self.pr1(out_iconv1)
        print("disp1", disp1.size())
        disp1 = crop_like(disp1, x)
        print("disp1", disp1.size())

        out_upconv1 = self.upconv1(out_iconv3)
        concat2 = torch.cat((crop_like(out_upconv2, out_conv2), disp3, out_conv2), 1)
        out_iconv2 = self.iconv2(concat2)
        disp2 = self.pr2(out_iconv2)
        disp2 = crop_like(self.upsample2(disp2), out_conv1)
#         print("disp2", disp2.size())


        if self.training:
            return disp1, disp2, disp3, disp4, disp5
        else:
            return disp1