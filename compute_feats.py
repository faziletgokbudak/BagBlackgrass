import dsmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import re, sys, argparse, os, glob, copy
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.utils import shuffle

import cv2


class BagDataset():
    def __init__(self, csv_file, transform=None, input_c=5):
        self.files_list = csv_file
        self.transform = transform
        self.input_c = input_c
        # self.lists = []
        # self.root = "/mnt/yifan/data/blackgrass/single/"
        # for path in open("/mnt/yifan/data/blackgrass/blackgrass/data_table.txt"):
        #     if "tif" not in path:
        #         continue
        #     else:
        #         res = path.split()
        #         r_prefix = res[0].split("/")[-1]  # Red
        #         g_prefix = res[1].split("/")[-1]  # green
        #         b_prefix = res[2].split("/")[-1]  # blue
        #         e_prefix = res[3].split("/")[-1]  # red edge
        #         nir_prefix = res[4].split("/")[-1]  # NIR
        #         class_label = res[6]
        #         path_ids = os.listdir(self.root + class_label + '/' + r_prefix.replace(".tif", ""))
        #         for path_id in path_ids:
        #             self.lists.append(
        #                 r_prefix + "\t" + g_prefix + "\t" + b_prefix + "\t" + e_prefix + "\t" + nir_prefix + "\t" + path_id + "\t" + class_label)

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        # temp_path = self.lists[idx].split()
        # # path_id = "/" + choice(os.listdir(self.root + temp_path[-1] + '/' + temp_path[0].replace(".tif", "")))
        # input_channel = 5
        # img = cv2.imread(self.root + temp_path[-1] + '/' + temp_path[0].replace(".tif", "") + "/" + temp_path[-2])[:, :,
        #       0:1]
        # img = transforms.functional.to_tensor(img)
        # for ind in range(input_channel - 1):
        #     img_new = cv2.imread(
        #         self.root + temp_path[-1] + '/' + temp_path[ind + 1].replace(".tif", "") + "/" + temp_path[-2])[:,
        #               :, 0:1]
        #     img_new = transforms.functional.to_tensor(img_new)
        #     img = torch.cat((img, img_new), 0)

        temp_path = self.files_list[idx]
        #res = temp_path.split("_")

        input_channel = self.input_c
        img = cv2.imread(temp_path, -1)[:, :, 0:1]
        img = transforms.functional.to_tensor(img)
        for ind in range(input_channel - 1):
            pattern = re.compile("IMG_[0-9]+_")
            m = list(re.finditer(pattern, temp_path))[-1]
            dir_path = temp_path[:m.start(0)]
            img_path_list = temp_path[m.start(0):].split('_')
            new_path = dir_path + '_'.join(img_path_list[:2]) + '_' + str(ind + 2) + "_" + '_'.join(img_path_list[3:])
            print(dir_path)
            #new_path = "_".join(res[0:len(res) - 4]) + "_" + str(ind + 2) + "_" + "_".join(res[len(res) - 3:len(res)])
            #new_path = "_".join(res[0:len(res) - 5]) + "_" + str(ind + 2) + "_" + "_".join(res[len(res) - 4:len(res)])
            print('new path:', new_path)
            img_new = cv2.imread(new_path, -1)[:, :, 0:1]
            img_new = transforms.functional.to_tensor(img_new)
            img = torch.cat((img, img_new), 0)

        # if self.transform:
        #     sample = self.transform(img)
        return img


# class ToTensor(object):
#     def __call__(self, sample):
#         img = sample['input']
#         img = VF.to_tensor(img)
#         return {'input': img}
#
#
# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms
#
#     def __call__(self, img):
#         for t in self.transforms:
#             img = t(img)
#         return img
#

def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path, input_c=args.input_c)
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)


def compute_feats(args, bags_list, i_classifier, save_path=None, magnification='single'):
    i_classifier.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    for i in range(0, num_bags):
        feats_list = []
        if magnification == 'single' or magnification == 'low':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg')) + glob.glob(
                os.path.join(bags_list[i], '*.jpeg'))
	    
            print('csv file path:', csv_file_path)
        elif magnification == 'high':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*' + os.sep + '*.jpg')) + glob.glob(
                os.path.join(bags_list[i], '*' + os.sep + '*.jpeg'))
            print()
        dataloader, bag_size = bag_dataset(args, csv_file_path)

        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch.float().cuda()
                feats, classes = i_classifier(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i + 1, num_bags, iteration + 1, len(dataloader)))
        if len(feats_list) == 0:
            print('No valid patch extracted from: ' + bags_list[i])
        else:
            df = pd.DataFrame(feats_list)
            os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
            df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2],
                                   bags_list[i].split(os.path.sep)[-1] + '.csv'), index=False, float_format='%.4f')


def compute_tree_feats(args, bags_list, embedder_low, embedder_high, save_path=None, fusion='fusion'):
    embedder_low.eval()
    embedder_high.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    with torch.no_grad():
        for i in range(0, num_bags):
            low_patches = glob.glob(os.path.join(bags_list[i], '*.jpg')) + glob.glob(
                os.path.join(bags_list[i], '*.jpeg'))
            feats_list = []
            feats_tree_list = []
            dataloader, bag_size = bag_dataset(args, low_patches)
            for iteration, batch in enumerate(dataloader):
                patches = batch.float().cuda()
                print(patches.shape)
                feats, classes = embedder_low(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
            for idx, low_patch in enumerate(low_patches):
                high_patches = glob.glob(low_patch.replace('.jpeg', os.sep + '*.jpg')) + glob.glob(
                    low_patch.replace('.jpeg', os.sep + '*.jpeg'))
                high_patches = high_patches + glob.glob(low_patch.replace('.jpg', os.sep + '*.jpg')) + glob.glob(
                    low_patch.replace('.jpg', os.sep + '*.jpeg'))
                if len(high_patches) == 0:
                    pass
                else:
                    for high_patch in high_patches:
                        img = Image.open(high_patch)
                        img = VF.to_tensor(img).float().cuda()
                        feats, classes = embedder_high(img[None, :])
                        if fusion == 'fusion':
                            feats = feats.cpu().numpy() + 0.25 * feats_list[idx]
                        if fusion == 'cat':
                            feats = np.concatenate((feats.cpu().numpy(), 0.25 * feats_list[idx]), axis=-1)
                        feats_tree_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i + 1, num_bags, idx + 1, len(low_patches)))
            if len(feats_tree_list) == 0:
                print('No valid patch extracted from: ' + bags_list[i])
            else:
                df = pd.DataFrame(feats_tree_list)
                os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
                df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2],
                                       bags_list[i].split(os.path.sep)[-1] + '.csv'), index=False, float_format='%.4f')
            print('\n')


def main():
    parser = argparse.ArgumentParser(description='Compute TCGA features from SimCLR embedder')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--input_c', default=5, type=int, help='Number of input channels [5]')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size of dataloader [128]')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--backbone', default='resnet18', type=str, help='Embedder backbone [resnet18]')
    parser.add_argument('--norm_layer', default='instance', type=str, help='Normalization layer [instance]')
    parser.add_argument('--magnification', default='single', type=str,
                        help='Magnification to compute features. Use `tree` for multiple magnifications. Use `high` if patches are cropped for multiple resolution and only process higher level, `low` for only processing lower level.')
    parser.add_argument('--weights', default=None, type=str, help='Folder of the pretrained weights, simclr/runs/*')
    parser.add_argument('--weights_high', default=None, type=str,
                        help='Folder of the pretrained weights of high magnification, FOLDER < `simclr/runs/[FOLDER]`')
    parser.add_argument('--weights_low', default=None, type=str,
                        help='Folder of the pretrained weights of low magnification, FOLDER <`simclr/runs/[FOLDER]`')
    parser.add_argument('--dataset', default='/home/fg405/rds/hpc-work/IMAGE_ARCHIVE/single/', type=str,help='Dataset folder name [TCGA-lung-single]')
    #parser.add_argument('--dataset', default='/mnt/yifan/data/blackgrass/', type=str,help='Dataset folder name [TCGA-lung-single]')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    if args.norm_layer == 'instance':
        norm = nn.InstanceNorm2d
        pretrain = False
    elif args.norm_layer == 'batch':
        norm = nn.BatchNorm2d
        if args.weights == 'ImageNet':
            pretrain = True
        else:
            pretrain = False

    if args.backbone == 'resnet18':
        resnet = models.resnet18(pretrained=pretrain, norm_layer=norm)
        num_feats = 512
    if args.backbone == 'resnet34':
        resnet = models.resnet34(pretrained=pretrain, norm_layer=norm)
        num_feats = 512
    if args.backbone == 'resnet50':
        resnet = models.resnet50(pretrained=pretrain, norm_layer=norm)
        num_feats = 2048
    if args.backbone == 'resnet101':
        resnet = models.resnet101(pretrained=pretrain, norm_layer=norm)
        num_feats = 2048
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()

    if args.magnification == 'tree' and args.weights_high != None and args.weights_low != None:
        i_classifier_h = mil.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()
        i_classifier_l = mil.IClassifier(copy.deepcopy(resnet), num_feats, output_class=args.num_classes).cuda()

        if args.weights_high == 'ImageNet' or args.weights_low == 'ImageNet' or args.weights == 'ImageNet':
            if args.norm_layer == 'batch':
                print('Use ImageNet features.')
            else:
                raise ValueError('Please use batch normalization for ImageNet feature')
        else:
            weight_path = os.path.join('simclr', 'runs', args.weights_high, 'checkpoints', 'model.pth')
            state_dict_weights = torch.load(weight_path)
            for i in range(4):
                state_dict_weights.popitem()
            state_dict_init = i_classifier_h.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier_h.load_state_dict(new_state_dict, strict=False)
            os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
            torch.save(new_state_dict, os.path.join('embedder', args.dataset, 'embedder-high.pth'))

            weight_path = os.path.join('simclr', 'runs', args.weights_low, 'checkpoints', 'model.pth')
            state_dict_weights = torch.load(weight_path)
            for i in range(4):
                state_dict_weights.popitem()
            state_dict_init = i_classifier_l.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier_l.load_state_dict(new_state_dict, strict=False)
            os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
            torch.save(new_state_dict, os.path.join('embedder', args.dataset, 'embedder-low.pth'))
            print('Use pretrained features.')


    elif args.magnification == 'single' or args.magnification == 'high' or args.magnification == 'low':
        i_classifier = mil.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()

        if args.weights == 'ImageNet':
            if args.norm_layer == 'batch':
                print('Use ImageNet features.')
            else:
                print('Please use batch normalization for ImageNet feature')
        else:
            if args.weights is not None:
                weight_path = os.path.join('simclr', 'runs', args.weights, 'checkpoints', 'model.pth')
            else:
                weight_path = glob.glob('simclr/runs/*/checkpoints/*.pth')[-1]
            state_dict_weights = torch.load(weight_path)
            for i in range(4):
                state_dict_weights.popitem()
            state_dict_init = i_classifier.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier.load_state_dict(new_state_dict, strict=False)
            os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
            torch.save(new_state_dict, os.path.join('embedder', args.dataset, 'embedder.pth'))
            print('Use pretrained features.')

    if args.magnification == 'tree' or args.magnification == 'low' or args.magnification == 'high':
        bags_path = os.path.join('WSI', args.dataset, 'pyramid', '*', '*')
    else:
        bags_path = os.path.join('WSI', args.dataset, 'single', '*', '*')
    feats_path = os.path.join(args.dataset, "features")

    os.makedirs(feats_path, exist_ok=True)
    bags_list = []
    print('bag path', bags_path)
    bags_list_temp = glob.glob(bags_path)  ##get the bags list according to Blue Channel
    for i in bags_list_temp:
        print(i)

        r = re.compile("IMG_[0-9]+_1_")
        pattern = re.compile("IMG_[0-9]+_")
        print(re.search(pattern,i))
        if re.search(pattern,i):
        # if int(i.split("_")[2]) == 1:
            bags_list.append(i)
    print('bag:', bags_list)
    if args.magnification == 'tree':
        compute_tree_feats(args, bags_list, i_classifier_l, i_classifier_h, feats_path, 'fusion')
    else:
        compute_feats(args, bags_list, i_classifier, feats_path, args.magnification)
    print(os.path.join('datasets', os.path.join(args.dataset,'single'), '*' + os.path.sep))
    n_classes = glob.glob(os.path.join('datasets', os.path.join(args.dataset,'features'), '*' + os.path.sep))
    
    n_classes = sorted(n_classes)
    print('n_classes', n_classes)
    print('feats_path', feats_path)
    all_df = []
    for i, item in enumerate(n_classes):
        bag_csvs = glob.glob(os.path.join(item, '*.csv'))
        print(bag_csvs)
        bag_df = pd.DataFrame(bag_csvs)
        print(bag_df)
        bag_df['label'] = i
        print(os.path.join(args.dataset, "features", item.split(os.path.sep)[-2] + '.csv'))
        bag_df.to_csv(os.path.join(args.dataset, "features", item.split(os.path.sep)[-2] + '.csv'), index=False)
        all_df.append(bag_df)
        print(all_df)
    bags_path = pd.concat(all_df, axis=0, ignore_index=True)
    bags_path = shuffle(bags_path)
    bags_path.to_csv(os.path.join(args.dataset, "features", "bags_all" + '.csv'), index=False)


if __name__ == '__main__':
    main()
