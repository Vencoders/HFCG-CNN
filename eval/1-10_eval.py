import gc
import time
import xlwt
from utils import calculate_psnr, import_yuv, write_ycbcr,get_logger
from models import EDAR
from models import Net
import torch
from numpy import *
workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('My Worksheet')
num = [0,300,300,300,240,250,300,300,300,300,300]
# logger = ['0','logger1','logger2','logger3','logger4','logger5','logger6','logger7','logger8','logger9','logger10']
# num = [0,1,1,1,1,1,1,1,1,1,1]


for m in range(1,11):
    for K in range(22,38,5):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model1 = Net().to(device)
        # model2 = EDAR().to(device)
        logger = get_logger('/opt/data/private/HFCG-CNN/test/OCT_Test{}_QP{}.log'.format(str(m),str(K)))
        logger.info("V{}Model{} loaded".format(str(m),str(K)))
        vid_path = '/opt/data/private/HFCG-CNN/test/Test_Enhance/OCT_En_Test{}_QP{}.yuv'.format(str(m),str(K))
        model_path1 = '/opt/data/private/HFCG-CNN/YUV_model/OCT_QP{}.pth'.format(str(K))
        # model_path2 = '/opt/data/private/HFCG-CNN/YUV_model/QP{}.pth'.format(str(K))
        input1_path = '/opt/data/private/HFCG-CNN/test/test{}/Test{}_QP{}.yuv'.format(str(m),str(m),str(K))
        input2_path = '/opt/data/private/HFCG-CNN/test/test{}/Test{}.yuv'.format(str(m),str(m))
        state_dict = model1.state_dict()
        for n, p in torch.load(model_path1, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

        # state_dict = model2.state_dict()
        # for n, p in torch.load(model_path2, map_location=lambda storage, loc: storage).items():
        #     if n in state_dict.keys():
        #         state_dict[n].copy_(p)
        #     else:
        #         raise KeyError(n)


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

        data1 = import_yuv(input1_path, height, width, num[m], yuv_type='420p', start_frm=0, only_y=False)

        data2 = import_yuv(input2_path, height, width, num[m], yuv_type='420p', start_frm=0, only_y=False)
        time_start = time.time()
        for i in range(num[m]):

            Y = torch.from_numpy(data1[0][i]).unsqueeze(0).unsqueeze(0).float().cuda().to(device)
            U = torch.from_numpy(data1[1][i]).unsqueeze(0).unsqueeze(0).float().cuda().to(device)
            V = torch.from_numpy(data1[2][i]).unsqueeze(0).unsqueeze(0).float().cuda().to(device)
            with torch.no_grad():
                # out_Y = model2(Y)
                # _,out_U,out_V = model1(Y,U,V)
                out_Y, out_U, out_V = model1(Y, U, V)


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
            logger.info('T{}QP{} poc:{} Y-PSNR: {} U-PSNR: {} V-PSNR: {} '.format(str(m),str(K), i, '%.4f' % psnr1, '%.4f' % psnr2, '%.4f' % psnr3))
            _psnr = (6 * psnr1 + psnr2 + psnr3) / 8
            ypsnr += psnr1
            upsnr += psnr2
            vpsnr += psnr3
            psnr += _psnr
        ave_psnry = ypsnr / num[m]
        ave_psnru = upsnr / num[m]
        ave_psnrv = vpsnr / num[m]
        ave_psnr = psnr / num[m]

        data3 = write_ycbcr(YY, UU, VV, vid_path)
        # logger.info('old: Y-PSNR: 39.9750 U-PSNR: 43.5548 V-PSNR: 46.0669 PSNR: 41.0210')
        logger.info('new: Y-PSNR: {} U-PSNR: {} V-PSNR: {} PSNR: {}'.format('%.4f' % ave_psnry, '%.4f' % ave_psnru,
                                                                            '%.4f' % ave_psnrv, '%.4f' % ave_psnr))
        logger.info('Evaluation over.')
        time_end = time.time()
        time_c = time_end - time_start
        logger.info('time cost:{},s'.format('%.3f' %time_c))
        if K==22:
            worksheet.write(int(4*m-3), 1, label ='%.4f' % ave_psnry)
            worksheet.write(int(4*m-3), 2, label ='%.4f' % ave_psnru)
            worksheet.write(int(4*m-3), 3, label ='%.4f' % ave_psnrv)
            worksheet.write(int(4*m-3), 4, label ='%.4f' % ave_psnr)
            worksheet.write(int(4*m-3), 5, label ='%.3f' %time_c)
        elif K==27:
            worksheet.write(int(4*m-2), 1, label ='%.4f' % ave_psnry)
            worksheet.write(int(4*m-2), 2, label ='%.4f' % ave_psnru)
            worksheet.write(int(4*m-2), 3, label ='%.4f' % ave_psnrv)
            worksheet.write(int(4*m-2), 4, label ='%.4f' % ave_psnr)
            worksheet.write(int(4*m-2), 5, label ='%.3f' %time_c)
        elif K==32:
            worksheet.write(int(4*m-1), 1, label ='%.4f' % ave_psnry)
            worksheet.write(int(4*m-1), 2, label ='%.4f' % ave_psnru)
            worksheet.write(int(4*m-1), 3, label ='%.4f' % ave_psnrv)
            worksheet.write(int(4*m-1), 4, label ='%.4f' % ave_psnr)
            worksheet.write(int(4*m-1), 5, label ='%.3f' %time_c)
        elif K==37:
            worksheet.write(int(4*m), 1, label ='%.4f' % ave_psnry)
            worksheet.write(int(4*m), 2, label ='%.4f' % ave_psnru)
            worksheet.write(int(4*m), 3, label ='%.4f' % ave_psnrv)
            worksheet.write(int(4*m), 4, label ='%.4f' % ave_psnr)
            worksheet.write(int(4*m), 5, label ='%.3f' %time_c)
        else:print('error')
        workbook.save('/opt/data/private/HFCG-CNN/test/OCT_eval1-10.xls')
        del data1,data2,data3,device,Y,U,V,YY,UU,VV,out_Y,out_U,out_V,Y2,U2,V2,Yt,Ut,Vt
        gc.collect()
