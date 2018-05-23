import torch
import torch.nn as nn
import torch.nn.functional as F


def downsample_conv(in_planes, out_planes, kernel_size=3, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes, kernel_size=4):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )

def predict_disp_segment(in_planes, n_segments = 10):
    return nn.Sequential(
        nn.Conv2d(in_planes, n_segments, kernel_size=3, padding=1),
        nn.ReLU()
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
        self.pr5 = predict_disp(upconv_planes[0])
        self.pr4 = predict_disp(upconv_planes[1])
        self.pr3 = predict_disp(upconv_planes[2])
        self.pr2 = predict_disp(upconv_planes[3])
#         self.pr1 = predict_disp(upconv_planes[4])
        
#         self.pr1_0 = predict_disp(upconv_planes[4])
#         self.pr1_1 = predict_disp(upconv_planes[4])
#         self.pr1_2 = predict_disp(upconv_planes[4])
#         self.pr1_3 = predict_disp(upconv_planes[4])
#         self.pr1_4 = predict_disp(upconv_planes[4])
#         self.pr1_5 = predict_disp(upconv_planes[4])
#         self.pr1_6 = predict_disp(upconv_planes[4])
#         self.pr1_7 = predict_disp(upconv_planes[4])
#         self.pr1_8 = predict_disp(upconv_planes[4])
#         self.pr1_9 = predict_disp(upconv_planes[4])
#         [self.pr1_0, self.pr1_1, self.pr1_2, self.pr1_3, self.pr1_4, self.pr1_5, self.pr1_6, self.pr1_7, self.pr1_8, self.pr1_9]  
        
        self.pr1_segment = nn.Conv2d(upconv_planes[4], 10, kernel_size=3, padding=1)
        
        self.pr1_delta = predict_disp_segment(upconv_planes[4])
        
#         self.upsample6 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
#         self.upsample5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
#         self.upsample4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
#         self.upsample3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
#         self.upsample2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
#         self.upsample1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out_conv1 = self.conv1(x)
#         print("out_conv1", out_conv1.size())
        out_conv2 = self.conv2(out_conv1)
#         print("out_conv2", out_conv2.size())
        out_conv3a = self.conv3a(out_conv2)
        out_conv3b = self.conv3b(out_conv3a)
#         print("out_conv3b", out_conv3b.size())
        out_conv4a = self.conv4a(out_conv3b)
        out_conv4b = self.conv4b(out_conv4a)
#         print("out_conv4b", out_conv4b.size())
        out_conv5a = self.conv5a(out_conv4b)
        out_conv5b = self.conv5b(out_conv5a)
#         print("out_conv5b", out_conv5b.size())
        out_conv6a = self.conv6a(out_conv5b)
        out_conv6b = self.conv6b(out_conv6a)
#         print("out_conv6b", out_conv6b.size())

        disp6_out = self.pr6(out_conv6b)
        disp6 = F.upsample(disp6_out, list(out_conv5b.size()[-2:]), mode='bilinear', align_corners=True)
#         print("disp6", disp6.size())

        out_upconv5 = self.upconv5(out_conv6b)
#         print("out_upconv5", out_upconv5.size())        
        concat5 = torch.cat((crop_like(out_upconv5, out_conv5b), disp6, out_conv5b), 1)
        out_iconv5 = self.iconv5(concat5)
        disp5_out = self.pr5(out_iconv5)
        disp5 = F.upsample(disp5_out, list(out_conv4b.size()[-2:]), mode='bilinear', align_corners=True)
#         print("disp5", disp5.size())
        
        out_upconv4 = self.upconv4(out_iconv5)
#         print("out_upconv4", out_upconv4.size())    
        concat4 = torch.cat((crop_like(out_upconv4, out_conv4b), disp5, out_conv4b), 1)
        out_iconv4 = self.iconv4(concat4)
        disp4_out = self.pr4(out_iconv4)
        disp4 = F.upsample(disp4_out, list(out_conv3b.size()[-2:]), mode='bilinear', align_corners=True)
#         print("disp4", disp4.size())

        out_upconv3 = self.upconv3(out_iconv4)
        concat3 = torch.cat((crop_like(out_upconv3, out_conv3b), disp4, out_conv3b), 1)
        out_iconv3 = self.iconv3(concat3)
        disp3_out = self.pr3(out_iconv3)
        disp3 = F.upsample(disp3_out, list(out_conv2.size()[-2:]), mode='bilinear', align_corners=True)
#         print("disp3", disp3.size())
        
        out_upconv2 = self.upconv2(out_iconv3)
        concat2 = torch.cat((crop_like(out_upconv2, out_conv2), disp3, out_conv2), 1)
        out_iconv2 = self.iconv2(concat2)
        disp2_out = self.pr2(out_iconv2)
        disp2 = F.upsample(disp2_out, list(out_conv1.size()[-2:]), mode='bilinear', align_corners=True)
#         print("disp2", disp2.size())
        
        out_upconv1 = self.upconv1(out_iconv2)
        concat1 = torch.cat((crop_like(out_upconv1, out_conv1), disp2, out_conv1), 1)
        out_iconv1 = self.iconv1(concat1)
        
        disp1_probs = self.pr1_segment(out_iconv1)
        pr1_shifts = self.pr1_delta(out_iconv1)
#         print("pr1_shifts", pr1_shifts.size())
#         print("disp1", disp1.size())

        if self.training:
            return disp1_probs, pr1_shifts, disp2_out, disp3_out, disp4_out, disp5_out, disp6_out
        else:
            disp1_out = disp1_probs.argmax(dim=1, keepdim=True)
            disp1_out = pr1_shifts.gather(1, disp1_out) + 30*disp1_out.float()
            disp = F.upsample(disp1_out, list(x.size()[-2:]), mode='bilinear', align_corners=True)            
            return disp