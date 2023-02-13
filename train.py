import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from QP22dataset import QPDataset, yuv_import, yuv_write, calculate_psnr
from codes.models1 import Net
from file_io import import_yuv
from log import get_logger
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import transforms
from torchvision.models.vgg import vgg16
from utils import AverageMeter
from tqdm import tqdm

if __name__ == '__main__':

    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--tensorboard_path', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--rec', type=float, default=1.0)
    parser.add_argument('--per', type=float, default=0)
    parser.add_argument('--adv', type=float, default=0)
    parser.add_argument('--log_path', type=str, default='', required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    log_path = opt.log_path
    logger = get_logger(log_path)

    torch.manual_seed(opt.seed)

    logger.info('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    # 实例化SummaryWriter对象
    tb_writer = SummaryWriter(log_dir=opt.tensorboard_path)

    logger.info("Data processing started")

    dataset = QPDataset(opt.dataroot)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True)
    logger.info("Data loading completed")
    # vgg = vgg16(pretrained=True).cuda()
    # loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
    #     for param in loss_network.parameters():
    #         param.requires_grad = False
    logger.info('Length of the dataset is {}'.format((len(dataset))))

    model = Net().to(device)
    # 将模型写入tensorboard
    init_imgY = torch.zeros((1, 1, 224, 224), device=device)
    init_imgU = torch.zeros((1, 1, 112, 112), device=device)
    init_imgV = torch.zeros((1, 1, 112, 112), device=device)
    tb_writer.add_graph(model, [init_imgY,init_imgU,init_imgV])
    logger.info("Model loaded")

    if opt.resume:
        if os.path.isfile(opt.resume):
            state_dict = model.state_dict()
            for n, p in torch.load(opt.resume, map_location=lambda storage, loc: storage).items():
                if n in state_dict.keys():
                    state_dict[n].copy_(p)
                else:
                    raise KeyError(n)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)


    for epoch in range(1,opt.num_epochs+1):
        epoch_losses_Y = AverageMeter()
        epoch_losses_U = AverageMeter()
        epoch_losses_V = AverageMeter()
        epoch_losses = AverageMeter()
        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch, opt.num_epochs))
            for data in dataloader:
                img_in_Y,img_in_U,img_in_V,img_QP_Y,img_QP_U,img_QP_V = data
                inputs_Y = img_QP_Y.to(device).float()
                labels_Y = img_in_Y.to(device).float()
                inputs_U = img_QP_U.to(device).float()
                labels_U = img_in_U.to(device).float()
                inputs_V = img_QP_V.to(device).float()
                labels_V = img_in_V.to(device).float()
                outs_Y,outs_U,outs_V = model(inputs_Y,inputs_U,inputs_V)
                loss_Y = criterion(outs_Y, labels_Y)
                loss_U = criterion(outs_U, labels_U)
                loss_V = criterion(outs_V, labels_V)
                loss = 0.6*loss_Y + 0.1*loss_U + 0.1*loss_V
                #perception_loss = criterion(loss_network(outs), loss_network(labels))
                #loss = loss + perception_loss*0.06
                # print(len(inputs))
                # logger.info('Epoch:[{}/{}]\t loss={:.5f}\t '.format(epoch + 1, opt.num_epochs, loss))
                epoch_losses_Y.update(loss_Y.item(), len(inputs_Y))
                epoch_losses_U.update(loss_U.item(), len(inputs_U))
                epoch_losses_V.update(loss_V.item(), len(inputs_V))
                epoch_losses.update(loss.item(), len(inputs_Y))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _tqdm.set_postfix(loss_Y='{:.6f}'.format(epoch_losses_Y.avg),loss_U='{:.6f}'.format(epoch_losses_U.avg),loss_V='{:.6f}'.format(epoch_losses_V.avg),loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(inputs_Y))
        tb_writer.add_scalar('Loss/LossY', epoch_losses_Y.avg, epoch)
        tb_writer.add_scalar('Loss/LossU', epoch_losses_U.avg, epoch)
        tb_writer.add_scalar('Loss/LossV', epoch_losses_V.avg, epoch)
        tb_writer.add_scalar('Loss/Loss', epoch_losses.avg, epoch)
        torch.save(model.state_dict(), os.path.join(opt.outputs_dir, '{}_epoch_{}.pth'.format("EDAR_", epoch)))

        #eval
        logger.info('Epoch ' + str(epoch) + ' evaluation process...')

        model.eval()
        with torch.no_grad():
            height = 1080
            width =1920
            num = 1
            ypsnr = 0
            upsnr = 0
            vpsnr = 0
            _psnr = 0
            psnr = 0
            for i in range(num):
                input1_path = '/data/yuli/EDAR-master/Eval/Validation1/Validation1_QP22.yuv'
                data1 = import_yuv(input1_path, height, width, num, yuv_type='420p', start_frm=0, only_y=False)
                Y = torch.from_numpy(data1[0][i]).unsqueeze(0).unsqueeze(0).float().cuda().to(device)
                U = torch.from_numpy(data1[1][i]).unsqueeze(0).unsqueeze(0).float().cuda().to(device)
                V = torch.from_numpy(data1[2][i]).unsqueeze(0).unsqueeze(0).float().cuda().to(device)
                with torch.no_grad():
                    out_Y,out_U,out_V = model(Y,U,V)

                Y1 = out_Y.byte().squeeze(0).squeeze(0).cpu().numpy()
                U1 = out_U.byte().squeeze(0).squeeze(0).cpu().numpy()
                V1 = out_V.byte().squeeze(0).squeeze(0).cpu().numpy()


                input2_path = '/data/yuli/EDAR-master/Eval/yuv/ISCAS_Grand_Challenge_Validation1.yuv'
                data2 = import_yuv(input2_path, height, width, num, yuv_type='420p', start_frm=0, only_y=False)
                Y2 = data2[0][i]
                U2 = data2[1][i]
                V2 = data2[2][i]
                psnr1 = calculate_psnr(Y1, Y2)
                psnr2 = calculate_psnr(U1, U2)
                psnr3 = calculate_psnr(V1, V2)
                _psnr = (6 * psnr1 + psnr2 + psnr3) / 8
                ypsnr += psnr1
                upsnr += psnr2
                vpsnr += psnr3
                psnr += _psnr
            ave_psnry = ypsnr / num
            ave_psnru = upsnr / num
            ave_psnrv = vpsnr / num
            ave_psnr = psnr / num
            logger.info('old: Y-PSNR: 48.5673 U-PSNR: 48.7616 V-PSNR: 52.7755')
            logger.info('new: Y-PSNR: {} U-PSNR: {} V-PSNR: {} PSNR: {}'.format('%.4f'%ave_psnry,'%.4f'%ave_psnru,'%.4f'%ave_psnrv,'%.4f'%ave_psnr))
            tb_writer.add_scalar('PSNR/PSNR_Y', ave_psnry, epoch)
            tb_writer.add_scalar('PSNR/PSNR_U', ave_psnru, epoch)
            tb_writer.add_scalar('PSNR/PSNR_V', ave_psnrv, epoch)
            tb_writer.add_scalar('PSNR/PSNR', ave_psnr, epoch)
        logger.info('Evaluation over.')
