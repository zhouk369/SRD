###### 测试
from __future__ import print_function, division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
 
import torch
import torch.nn as nn
from tools.new_roc import cal_metric
import argparse
import time

from network.srd import Convnext_H   # Proposed method

from tools.transforms import build_transforms

import numpy as np
import warnings

warnings.filterwarnings('ignore')
from network.loss import SupervisedContrastiveLoss
from tools.read_path import MyDataset_or
from PIL import Image
from torch.utils.data import Dataset
import os
import random
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True





def get_test_paths(test_txt):
    """
    Parse and validate test dataset paths. Supports multiple comma-separated paths.

    Parameters
    ----------
    test_txt : str
        Comma-separated test dataset paths.

    Returns
    -------
    list of str
        List of valid dataset paths.
    """
    if "," in test_txt:  # If multiple paths are provided, split them
        paths = test_txt.split(",")
    else:
        paths = [test_txt]

    # Validate paths
    valid_paths = [path.strip() for path in paths if os.path.isfile(path)]
    if not valid_paths:
        raise FileNotFoundError(f"No valid paths found in: {test_txt}")
    
    return valid_paths





def test_model(model_path, criterion, test_txt, batch_size):
    
    test_paths = get_test_paths(test_txt)

    for path in test_paths:
        print(f"Testing with dataset: {path}")
    
        transform_train, transform_test = build_transforms(args.resolution, args.resolution,
                                                        max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406],
                                                        norm_std=[0.229, 0.224, 0.225])

        test_dataset = MyDataset_or(txt_path=path, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                                                num_workers=8)    

        since = time.time()
        # model load
        model  = Convnext_H(num_classes=args.num_class).to(device)
        
        # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})  # 多加载
        model.load_state_dict(torch.load(model_path), strict=False)   # simple

        consistency_fn = SupervisedContrastiveLoss().cuda()  # supconloss

        model.eval()
        test_loss = 0
        correct = 0
        total_test = 0
        test_list = []
        output_list = []

        print('############################## test start ###########################################')
        for batch_idx, data in enumerate(test_loader):
            inputs, target = data

            inputs, target = inputs.cuda().float(), target.cuda()
            

            output, fre = model(inputs)


            L_con = consistency_fn(fre, target)

            total_test += target.size(0)

            loss_ce = criterion(output, target)
            _, pred = torch.max(output.data, 1)

            loss = 0.01 * L_con + loss_ce


            correct += torch.sum(pred == target.data)
            print_loss = loss.data.item()
            test_loss += print_loss

            test_list.extend(target.cpu().numpy().tolist())
            output_list.extend(output.data[:, 1].cpu().numpy().tolist())

        acc = correct / total_test
        avgloss = test_loss / total_test
        eer, TPRs, ave_auc, scaler, ap, best_f1 = cal_metric(test_list, output_list, False)  # calculate the auc

        print('Ave_Loss: {:.4f}, Ave_Acc: {:.4f}, Ave_Auc: {:.4f}, Ave_Ap: {:.4f}, Ave_F1: {:.4f}, Ave_Eer: {:.4f}'.format(avgloss, acc, ave_auc, ap, best_f1, eer))

        time_elapsed = time.time() - since
        print('Test complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()

    my_parser.add_argument("-b", type=int, default=6, help="Batch size")
    my_parser.add_argument("-e", type=int, default=6, help="Epoches")
    my_parser.add_argument('--num_class', type=int, default=2)
    my_parser.add_argument("--num_workers", type=int, default=4, help="Output for the model saving")


    ### --- 不同的篡改类型
    ### Test dataset paths

    

    my_parser.add_argument("-tl", type=str,
                           default="/mnt/home/Data/Celeb_V1.txt,"
                                   "/mnt/home/Data/DFDC/DFDC.txt,"
                                   '/mnt/home/Data/Celeb_DF_V2,'
                                   "/mnt/home/Data/FF_C23/Test.txt,",
                           help="Comma-separated test dataset paths")
    

    ###### C23的预训练权重   随机种子是42。
    my_parser.add_argument("-mp", type=str,
                           default="/mnt/home/my_weights/SRD/pro_C23_0.976471.pth",
                           help="model_path") 



    my_parser.add_argument('--resolution', type=int, default=224)
    my_parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')  # 设置随机种子


    args = my_parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

    model_path = args.mp
    batch_size = args.b
    test_list = args.tl

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = nn.CrossEntropyLoss().cuda()

    test_model(model_path, criterion, test_list, batch_size)






















