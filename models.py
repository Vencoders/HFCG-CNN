import torch
import torch.nn.functional as F
import torch.nn as nn
import octconv as oc

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, act=nn.ReLU(True)):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            # m.append(CBAMBlock(channel=n_feat,reduction=16,kernel_size=kernel_size))
            # m.append(SEAttention(channel=n_feat,reduction=8))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class MS_RB(nn.Module):
    def __init__(self, num_feats, kernel_size):
        super(MS_RB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
                               kernel_size=kernel_size, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
                               kernel_size=kernel_size, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
                               kernel_size=1, padding=0)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x))
        x3 = x1 + x2
        x4 = self.conv4(x3)
        out = x4 + x

        return out



class HNet(nn.Module):
    def __init__(self, num_feats=32, kernel_size=3):
        super(HNet, self).__init__()
        self.conv_rgb1 = nn.Conv2d(in_channels=1, out_channels=num_feats,
                                   kernel_size=kernel_size, padding=1)
        #self.conv_rgb2 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)
        #self.conv_rgb3 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)
        #self.conv_rgb4 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)
        #self.conv_rgb5 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)
                                   
        self.rgb_cbl2 = oc.OctaveConv(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, alpha_in=0, alpha_out=0.25,
                                    stride=1, padding=1,dilation=1,groups=1, bias=False)
        self.rgb_cbl3 = oc.OctaveConv(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, alpha_in=0.25, alpha_out=0.25,
                                    stride=1, padding=1,dilation=1,groups=1, bias=False)
        self.rgb_cbl4 = oc.OctaveConv(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, alpha_in=0.25, alpha_out=0.25,
                                    stride=1, padding=1,dilation=1,groups=1, bias=False)
        self.rgb_cbl5 = oc.OctaveConv(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, alpha_in=0.25, alpha_out=0.25,
                                    stride=1, padding=1,dilation=1,groups=1, bias=False)
        #self.rgb_cbl5 = oc.Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, alpha_in=0.125, alpha_out=0.125,
        #                            stride=1, padding=1,dilation=1,groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU(negative_slope=0.2, inplace=True))                         
                                   




        self.conv_dp1 = nn.Conv2d(in_channels=1, out_channels=num_feats,
                                  kernel_size=kernel_size, padding=1)
        self.conv_dp2 = nn.Conv2d(in_channels=1, out_channels=num_feats,
                                  kernel_size=kernel_size, padding=1)
        self.conv_dp3 = nn.Conv2d(in_channels=1, out_channels=num_feats,
                                  kernel_size=kernel_size, padding=1)

        self.MSBY1 = MS_RB(num_feats, kernel_size)
        self.MSBU1 = MS_RB(num_feats, kernel_size)
        self.MSBV1 = MS_RB(num_feats, kernel_size)

        self.MSBY2 = MS_RB(56, kernel_size)
        self.MSBU2 = MS_RB(56, kernel_size)
        self.MSBV2 = MS_RB(56, kernel_size)

        self.MSBY3 = MS_RB(80, kernel_size)
        self.MSBU3 = MS_RB(80, kernel_size)
        self.MSBV3 = MS_RB(80, kernel_size)

        self.MSBY4 = MS_RB(104, kernel_size)
        self.MSBU4 = MS_RB(104, kernel_size)
        self.MSBV4 = MS_RB(104, kernel_size)

        # self.MSBY5 = MS_RB(170, kernel_size)
        # self.MSBU5 = MS_RB(170, kernel_size)
        # self.MSBV5 = MS_RB(170, kernel_size)



        self.Yrestore = nn.Conv2d(in_channels=104, out_channels=1, kernel_size=kernel_size, padding=1)
        self.Urestore = nn.Conv2d(in_channels=104, out_channels=1, kernel_size=kernel_size, padding=1)
        self.Vrestore = nn.Conv2d(in_channels=104, out_channels=1, kernel_size=kernel_size, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, Y, U, V):
        #高频层1
        rgb1 = self.act(self.conv_rgb1(Y))#(高频Y特征提取)
        #rgb2 = self.act(self.conv_rgb2(rgb1))
        rgb2 = self.rgb_cbl2(rgb1)#(高频Y1分离)
        #增强层1
        Y_in = self.act(self.conv_dp1(Y))#(Y特征提取)
        Ydp1 = self.MSBY1(Y_in)#(YUV,Y特征增强)
        U_in = self.act(self.conv_dp2(U))#(U特征提取)
        Udp1 = self.MSBU1(U_in)#(YUV,U特征增强)
        V_in = self.act(self.conv_dp3(V))#(V特征提取)
        Vdp1 = self.MSBV1(V_in)#(YUV,V特征增强)
        Udp1 = F.interpolate(Udp1,scale_factor=2)
        Vdp1 = F.interpolate(Vdp1, scale_factor=2)


        Yca1_in = torch.cat([Ydp1, rgb2[0]], dim=1)#  (高频Y指导Y恢复)
        Uca1_in = torch.cat([Udp1, rgb2[0]], dim=1) # (高频Y指导U恢复)
        Vca1_in = torch.cat([Vdp1, rgb2[0]], dim=1) # (高频Y指导V恢复)


        #高频层2
        #rgb3 = self.conv_rgb3(rgb2)
        rgb3 = self.rgb_cbl3(rgb2)
        #增强层2
        Ydp2 = self.MSBY2(Yca1_in)
        Udp2 = self.MSBU2(Uca1_in)
        Vdp2 = self.MSBV2(Vca1_in)
        #ca2_in = dp2 + rgb3

        Yca2_in = torch.cat([Ydp2, rgb3[0]], dim=1)
        Uca2_in = torch.cat([Udp2, rgb3[0]], dim=1)
        Vca2_in = torch.cat([Vdp2, rgb3[0]], dim=1)

        #高频层3
        # rgb4 = self.conv_rgb4(rgb3)
        rgb4 = self.rgb_cbl4(rgb3)
        #增强层3
        Ydp3 = self.MSBY3(Yca2_in)
        Udp3 = self.MSBU3(Uca2_in)
        Vdp3 = self.MSBV3(Vca2_in)
        #ca3_in = rgb4 + dp3

        Yca3_in = torch.cat([Ydp3, rgb4[0]], dim=1)
        Uca3_in = torch.cat([Udp3, rgb4[0]], dim=1)
        Vca3_in = torch.cat([Vdp3, rgb4[0]], dim=1)

        #高频层4
        # rgb5 = self.conv_rgb5(rgb4)
        rgb5 = self.rgb_cbl5(rgb4)
        #增强层4
        Ydp4 = self.MSBY4(Yca3_in)
        Udp4 = self.MSBU4(Uca3_in)
        Vdp4 = self.MSBV4(Vca3_in)
        #ca4_in = rgb5 + dp4
        Yca4_in = torch.cat([Ydp4, rgb5[0]], dim=1)
        Uca4_in = torch.cat([Udp4, rgb5[0]], dim=1)
        Vca4_in = torch.cat([Vdp4, rgb5[0]], dim=1)


        # Ydp5 = self.MSBY5(Yca4_in)
        # Udp5 = self.MSBU5(Uca4_in)
        # Vdp5 = self.MSBV5(Vca4_in)

        # Uup1 = self.ps1(self.conv_recon1(self.act(Udp5)))
        # Vup1 = self.ps1(self.conv_recon2(self.act(Vdp5)))
        Udown = self.downsample(Udp4)
        Vdown = self.downsample(Vdp4)
        Yout = self.Yrestore(Ydp4)
        Uout = self.Urestore(Udown)
        Vout = self.Vrestore(Vdown)
        Yout = Y + Yout
        Uout = U + Uout
        Vout = V + Vout

        return Yout , Uout , Vout

class EDAR(nn.Module):
    def __init__(self, conv=default_conv):
        super(EDAR, self).__init__()

        n_resblock = 30
        n_feats = 64
        kernel_size = 3

        #DIV 2K mean
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        # rgb_mean = (0.485, 0.456, 0.406)
        # rgb_std = (0.229, 0.224, 0.225)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)

        # define head module
        m_head = [conv(1, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size
            )
            for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            conv(n_feats, 1, kernel_size)
        ]

        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)
        # return x
        return x


class Net(nn.Module):
    def __init__(self, conv=default_conv):
        super(Net, self).__init__()
        self.layer1 = HNet(num_feats=32,kernel_size=3)
        self.layer2 = EDAR()

    def forward(self, Y,U,V):

        Y,U,V = self.layer1(Y,U,V)
        Y1 = self.layer2(Y)
        U1 = self.layer2(U)
        V1 = self.layer2(V)

        return Y1,U1,V1

if __name__ == '__main__':
    device='cpu'
    input1 = torch.randn(1, 1, 40, 40).to(device)
    input2 = torch.randn(1, 1, 20, 20).to(device)
    input3 = torch.randn(1, 1, 20, 20).to(device)

    net = Net().to(device)
    out1,out2,out3=net(input1,input2,input3)

    print(out1.shape,out2.shape,out3.shape)