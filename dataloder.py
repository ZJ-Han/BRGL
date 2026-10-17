
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
import cv2 as cv


transform_BZ= transforms.Normalize(
    mean=[0.5],
    std=[0.5]
)
class LoadData(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag

        self.train_tf = transforms.Compose([
                transforms.Resize(128),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transform_BZ
            ])
        self.val_tf = transforms.Compose([
                transforms.Resize(128),
                transforms.ToTensor(),
                transform_BZ
            ])

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))
        return imgs_info

    def padding_black(self, img):
        w, h  = img.size
        scale = 128. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = 128
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))
        img = img_bg
        return img

    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.padding_black(img)
        # if img.mode != 'L':
        #     img = img.convert('L')
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)


        return img, label ,index, img_path

    def __len__(self):
        return len(self.imgs_info)

if __name__ == "__main__":
    train_dataset = LoadData("train1.txt", True)
    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=10,
                                               shuffle=True)
    for i, (images, labels, indexes, img_path) in enumerate(train_loader):
        if i==0:
            img=images[0]
            img=img[0, :, :]
            cv.imshow('rgb image',img.numpy())
            cv.waitKey(0)
            cv.destroyAllWindows()
          #  cv.show()
           # print(images.shape)
            print(labels[0])
