from glob import glob
import os

def get_sub_list(train_img_dir):
    image_list = []
    image_files = glob(os.path.join(train_img_dir,"*.nii.gz"))
    image_files.sort()
    for name in image_files:
        image_list.append(os.path.basename(name)[:-7])
    return image_list, image_files


def get_sub_from_txt(train_txt):
    fp = open(train_txt, 'r')
    sublines = fp.readlines()
    train_img = []
    train_seg = []

    for subline in sublines:
        sub_info = subline.replace('\n', '').split(',')
        train_img.append(sub_info[0])
        train_seg.append(sub_info[1])
        # train_seg_subs.append(sub_info[2])
        # train_seg_files.append(sub_info[3])
    fp.close()
    return train_img, train_seg