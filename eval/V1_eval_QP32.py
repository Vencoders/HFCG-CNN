import time
from utils import calculate_psnr, import_yuv, write_ycbcr,get_logger
from models import EDAR
from models import Net
import torch
from numpy import *

num = 240
logger = get_logger('./eval/V1_QP32.log')
model_path1 = './YUV_model/OCT_QP32.pth'
model_path2 = './YUV_model/OCT_QP32.pth'
input1_path = './eval/Validation1/Validation1_QP32.yuv'#compressed
input2_path = './ISCAS_Grand_Challenge_Validation1.yuv'#
logger.info("Model loaded")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model1 = Net().to(device)
model2 = EDAR().to(device)

state_dict = model1.state_dict()
for n, p in torch.load(model_path1, map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
        state_dict[n].copy_(p)
    else:
        raise KeyError(n)

state_dict = model2.state_dict()
for n, p in torch.load(model_path2, map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
        state_dict[n].copy_(p)
    else:
        raise KeyError(n)


ypsnr = 0
upsnr = 0
vpsnr = 0
_psnr = 0
psnr = 0


YY = []
UU = []
VV = []
width = 1920
height = 1080
d00 = width // 2
d01 = height // 2
Yt = zeros((height, width), float64, 'C')
Ut = zeros((d00, d01), float64, 'C')
Vt = zeros((d00, d01), float64, 'C')

data1 = import_yuv(input1_path, height, width, num, yuv_type='420p', start_frm=0, only_y=False)

data2 = import_yuv(input2_path, height, width, num, yuv_type='420p', start_frm=0, only_y=False)
time_start = time.time()
for i in range(num):

    Y = torch.from_numpy(data1[0][i]).unsqueeze(0).unsqueeze(0).float().cuda().to(device)
    U = torch.from_numpy(data1[1][i]).unsqueeze(0).unsqueeze(0).float().cuda().to(device)
    V = torch.from_numpy(data1[2][i]).unsqueeze(0).unsqueeze(0).float().cuda().to(device)
    with torch.no_grad():
        out_Y = model2(Y)
        _,out_U,out_V = model1(Y,U,V)

    Y1 = out_Y.byte().squeeze(0).squeeze(0).cpu().numpy()
    U1 = out_U.byte().squeeze(0).squeeze(0).cpu().numpy()
    V1 = out_V.byte().squeeze(0).squeeze(0).cpu().numpy()
    YY = YY + [Y1]
    UU = UU + [U1]
    VV = VV + [V1]


    Y2 = data2[0][i]
    U2 = data2[1][i]
    V2 = data2[2][i]
    psnr1 = calculate_psnr(Y1, Y2)
    psnr2 = calculate_psnr(U1, U2)
    psnr3 = calculate_psnr(V1, V2)
    logger.info('poc:{} Y-PSNR: {} U-PSNR: {} V-PSNR: {} '.format(i, '%.4f' % psnr1, '%.4f' % psnr2, '%.4f' % psnr3))
    _psnr = (6 * psnr1 + psnr2 + psnr3) / 8
    ypsnr += psnr1
    upsnr += psnr2
    vpsnr += psnr3
    psnr += _psnr
ave_psnry = ypsnr / num
ave_psnru = upsnr / num
ave_psnrv = vpsnr / num
ave_psnr = psnr / num


logger.info('new: Y-PSNR: {} U-PSNR: {} V-PSNR: {} PSNR: {}'.format('%.4f' % ave_psnry, '%.4f' % ave_psnru,
                                                                    '%.4f' % ave_psnrv, '%.4f' % ave_psnr))
logger.info('Evaluation over.')
time_end = time.time()
time_c = time_end - time_start
logger.info('time cost:{},s'.format('%.3f' %time_c))