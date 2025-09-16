from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
from datasets import Dataset as HFDataset

            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='',imb_factor=50): 
        
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode  
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
     
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']      
            elif dataset=='cifar10lt':
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100lt':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                        
        else:    
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
                num_data = 50000
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
                num_data = 50000
            elif dataset=='cifar10lt':
                # 读取 arrow 文件
                ds = HFDataset.from_file(f"/home/algroup/cyq/cp_pll/data/cifar-10-lt-{imb_factor}/cifar10-train.arrow")
                # 提取数据和标签
                train_data = np.stack([np.array(img) for img in ds['img']])
                train_label = np.array(ds['label'])
                 # 如果 img 还是 PIL.Image 或 list，可以进一步处理
                if not isinstance(train_data[0], np.ndarray):
                    train_data = np.stack([np.array(x) for x in train_data])
                num_data = len(ds)
            elif dataset=='cifar100lt':
                # 直接加载 arrow 文件
                train_dataset = HFDataset.from_file(f"/home/algroup/cyq/cp_pll/data/cifar-100-lt-{imb_factor}/cifar100-train.arrow")
                train_data = [x for x in train_dataset["img"]]
                if not isinstance(train_data[0], np.ndarray):
                    train_data = np.stack([np.array(x) for x in train_data])
                train_label = np.array(train_dataset["fine_label"])
                num_data = train_dataset.num_rows    # 标签
            train_data = train_data.reshape((num_data, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file,"r"))
            else:    #inject noise   
                noise_label = []
                idx = list(range(num_data))
                ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                random.shuffle(idx)
                num_noise = int(self.r*num_data)            
                noise_idx = idx[:num_noise]
                for i in range(num_data):
                    if i in noise_idx:
                        if dataset=='cifar10': 
                            noiselabel = random.randint(0,9)
                        elif dataset=='cifar100':    
                            noiselabel = random.randint(0,99)
                        elif dataset=='cifar10lt': 
                            noiselabel = random.randint(0,9)
                        elif dataset=='cifar100lt':    
                            noiselabel = random.randint(0,99)
                        noise_label.append(noiselabel)             
                    else:    
                        noise_label.append(train_label[i])   
                print("save noisy labels to %s ..."%noise_file)        
                noise_label_to_save = [int(x) for x in noise_label]  # 转成 Python int
                json.dump(noise_label_to_save, open(noise_file, "w"))     
            

            self.train_data = train_data
            self.noise_label = noise_label                       
            print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            
                
    def __getitem__(self, index):
        if self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
        else:
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index        

           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file='',imb_factor=50):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        self.imb_factor = imb_factor
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])   
        elif self.dataset=='cifar10lt':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])  
        elif self.dataset=='cifar100lt':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])  
    def run(self,mode,pred=[],prob=[]):       
        if mode == 'train':
            full_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file, imb_factor=self.imb_factor)   
            train_loader = DataLoader(
                dataset=full_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )
            return train_loader        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
     