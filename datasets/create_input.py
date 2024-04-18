import os
from utils.model_utils import *
from utils.segValidation import *
import numpy as np
from utils.model_utils import *
from utils.make_dir import *

from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler 
from torch.utils.data import SequentialSampler

class CreateDataInput(object):
    def __init__(self):
        pass

class DataTrain(CreateDataInput):
    labeled_set = []
    unlabeled_set = []
    test_dataset = []
    train_dataset = []
    true_labels = []
    files = []
    bk_dataset=[]
    noise_idx=[]
    mislabeled_targets = []
    WSI_Data = {
        'img_id': [],
        'patch_num': [],
        'grades0': [],
        'grades1': [],
        'grades2': [],
        'bmm': [],
        'pr': []
    }
    epoch_Data = {
        'epoch': [],
        'acc': [],
        'auc': [],
        'labeled': [],
        'unlabeled': [],
        'noise_in_labeled': [],
        'infor_in_unlabled': []
    }
    sample_data = {
        'patch_id': [],
        'img_id': [],
        'is_labeled': [],
        'scatter_x': [],
        'scatter_y': [],
       'grade': [],
        'class': [],
        'bmm':[],
        'bmm_num': [],
        'heat_score': [],
        'pr':[],
        'pr_num':[],
       'grades_num': [],
        'file_name': [],
        'CAM_file_name': [],
        'noise': [],
        'kmeans_label':[]
    }
    bk_data={
        'patch_id': [],
        'img_id': [],
        'class': [], # predict
        'file_name': [],
        'kmeans_label':[] # predict
        
    }
    def __init__(self, config, logger):
        super(DataTrain, self).__init__()
        self.path = config["dataset"]["path"]
        self.config = config
        self.logger = logger

    def init_data(self):
        self.logger.info("init datasets...")
        ls = np.load(os.path.join(
            self.path, "init_labeled_set.npy"), allow_pickle=True)
        uls = np.load(os.path.join(
            self.path, "init_unlabeled_set.npy"), allow_pickle=True)
        tes = np.load(os.path.join(
            self.path, "test_dataset.npy"), allow_pickle=True)
        trs = np.load(os.path.join(
            self.path, "train_dataset.npy"), allow_pickle=True)
        bkd=np.load(os.path.join(
            self.path, "bk_dataset.npy"), allow_pickle=True)
        bkdf=np.concatenate((
                np.load(os.path.join(self.path, "bk_patch_file.npy"), allow_pickle=True),
                np.load(os.path.join(self.path, "ts_patch_file.npy"), allow_pickle=True)
                ))
        DataTrain.files = np.load(os.path.join(
            self.path, "patch_file.npy"), allow_pickle=True)
        DataTrain.labeled_set = ls.tolist()
        DataTrain.unlabeled_set = uls.tolist()
        DataTrain.test_dataset = tes.tolist()
        DataTrain.train_dataset = trs.tolist()
        # DataTrain.bk_dataset=np.concatenate((bkd,tes)).tolist()
        DataTrain.bk_dataset=bkd.tolist()
        print(len(DataTrain.bk_dataset),len(bkdf))
     
        for i in range(len(DataTrain.train_dataset)):
            #DataTrain.true_labels.append(DataTrain.bk_dataset[i][1])
            DataTrain.true_labels.append(DataTrain.train_dataset[i][1])
        train_labels = np.asarray([[DataTrain.true_labels[i]]
                                   for i in range(len(DataTrain.true_labels))], dtype=int)
        # tl=np.copy(train_labels)
        # self.logger.info("We get %d cancer samples and %d no cancer samples "%(np.sum(tl==1),tl.sum(tl==0)))
        self.logger.info("Insert noise labels")
        train_noisy_labels, actual_noise_rate = noisify(nb_classes=self.config['dataset']['num_class'],
                                                        train_labels=train_labels,
                                                        noise_type=self.config['noise_method'],
                                                        noise_rate=self.config["noise_rate"],
                                                        random_state=self.config['seed'])
        #
        if self.config['noise_range']=='all': # all : add noise in unlabeled samples
            for i, (x, label, index, img_idx, global_idx) in enumerate(DataTrain.unlabeled_set):
                tnl=train_noisy_labels[global_idx][0]
                if tnl!=DataTrain.unlabeled_set[i][1]:
                    DataTrain.unlabeled_set[i][1] = tnl
                    DataTrain.noise_idx.append(global_idx) # get noise index 
        for i, (x, label, index, img_idx, global_idx) in enumerate(DataTrain.labeled_set):
            tnl=train_noisy_labels[global_idx][0]
            if tnl!=DataTrain.labeled_set[i][1]:
                DataTrain.labeled_set[i][1] = tnl
                DataTrain.noise_idx.append(global_idx) # get noise index 

        DataTrain.mislabeled_targets = [DataTrain.train_dataset[i][1] for i in range(len(DataTrain.train_dataset))]
        self.logger.info("We got %d noise samples which accounts for %.2f percent of train dataset."%(len(DataTrain.noise_idx),len(DataTrain.noise_idx)/train_labels.size))

        # init
        img_idxs = []
        for x, label, index, img_idx, global_idx in DataTrain.train_dataset:
            img_idxs.append(img_idx)

        img_idxs = list(set(img_idxs))
        DataTrain.WSI_Data['img_id'] = img_idxs
        DataTrain.WSI_Data['patch_num'] = np.zeros(len(img_idxs))
        DataTrain.WSI_Data['grades0'] = np.zeros(len(img_idxs))
        DataTrain.WSI_Data['grades1'] = np.zeros(len(img_idxs))
        DataTrain.WSI_Data['grades2'] = np.zeros(len(img_idxs))
        DataTrain.WSI_Data['bmm'] = np.zeros(len(img_idxs))
        DataTrain.WSI_Data['pr'] = np.zeros(len(img_idxs))
        

        for i,(x, label, pid, img_idx, global_idx)  in enumerate(DataTrain.bk_dataset):
            DataTrain.bk_data['patch_id'].append(int(pid))
            DataTrain.bk_data['img_id'].append(int(img_idx))
            DataTrain.bk_data['class'].append(label)
            DataTrain.bk_data['file_name'].append(bkdf[i][2])
            DataTrain.bk_data['kmeans_label'].append(0) #init 0


        for x, label, pid, img_idx, global_idx in DataTrain.train_dataset:
            DataTrain.WSI_Data['patch_num'][img_idx] += 1
            DataTrain.sample_data['patch_id'].append(int(pid))
            DataTrain.sample_data['img_id'].append(img_idx)
            DataTrain.sample_data['class'].append(label)
            DataTrain.sample_data['file_name'].append(DataTrain.files[global_idx][2])
            DataTrain.sample_data['CAM_file_name'].append(str(global_idx)+".png")
            DataTrain.sample_data['is_labeled'].append(0)
            # epoch data score
            DataTrain.sample_data['grades_num'].append(np.zeros(3))
            DataTrain.sample_data['bmm_num'].append([])
            DataTrain.sample_data['pr_num'].append([])

            

        for _, label, pid, imgid, global_idx in DataTrain.labeled_set:
            DataTrain.sample_data['is_labeled'][global_idx] = 1
        # 
        DataTrain.sample_data['scatter_x']=np.zeros(len(DataTrain.train_dataset))
        DataTrain.sample_data['scatter_y']=np.zeros(len(DataTrain.train_dataset))
        DataTrain.sample_data['grade']=np.zeros(len(DataTrain.train_dataset))
        DataTrain.sample_data['heat_score']=np.zeros(len(DataTrain.train_dataset))
        DataTrain.sample_data['noise']=np.zeros(len(DataTrain.train_dataset))
        DataTrain.sample_data['kmeans_label']=np.zeros(len(DataTrain.train_dataset))
        DataTrain.sample_data['bmm']=np.zeros(len(DataTrain.train_dataset))
        DataTrain.sample_data['pr']=np.zeros(len(DataTrain.train_dataset))

        self.logger.info("We have load train dataset:%d, back dataset:%d, labeled dataset:%d, unlabeled dataset:%d, test dataset:%d, files length:%d "
        %(len(DataTrain.train_dataset),len(DataTrain.bk_dataset),len(DataTrain.labeled_set),len(DataTrain.unlabeled_set),len(DataTrain.test_dataset),len(DataTrain.files)))
       
        return DataTrain.train_dataset, DataTrain.labeled_set, DataTrain.unlabeled_set, DataTrain.test_dataset

    def reset(self, noise_index, add_index_confident):
        self.logger.info(
            "We get %d noise labeled samples and %d informative unlabeled samples"%(len(noise_index),len(add_index_confident)))
        self.logger.info("Before reset,we have %d labeled samples and %d unlabeled samples" % (
            len(DataTrain.labeled_set), len(DataTrain.unlabeled_set)))
        
        # noise_index = noise_index.tolist()
        new_labeled_set = []
        new_unlabeled_set = []
        for i, (x, label, patch_idx, img_idx, global_idx) in enumerate(DataTrain.labeled_set):
            if global_idx in noise_index:
                new_labeled_set.append(
                    (x, DataTrain.true_labels[global_idx], patch_idx, img_idx, global_idx))
                continue
            else:
                new_labeled_set.append(
                    (x, label, patch_idx, img_idx, global_idx))

        for i, (x, label, patch_idx, img_idx, global_idx) in enumerate(DataTrain.unlabeled_set):
            if global_idx in add_index_confident:
                new_labeled_set.append(
                    (x, label, patch_idx, img_idx, global_idx))
                if global_idx not in DataTrain.labeled_set:
                    DataTrain.labeled_set.append(
                        (x, label, patch_idx, img_idx, global_idx))
            else:
                new_unlabeled_set.append(
                    (x, label, patch_idx, img_idx, global_idx))
        DataTrain.labeled_set = new_labeled_set
        DataTrain.unlabeled_set = new_unlabeled_set
        
        # caculate accuracy of noise fliter
        acc_noise=0
        for n in noise_index:
            if n in DataTrain.noise_idx:
                acc_noise+=1
        if len(noise_index) > 0:
            self.logger.info("The accuracy of noise sample filtering is [%.3f]" % (acc_noise / len(noise_index)))
        else:
            self.logger.info("No noise samples found, cannot calculate accuracy.")

        # self.logger.info("The accuracy of noise sample flitering is [%.3f] "%(acc_noise/len(noise_index))) 
        # caculate number of true labels 
        true_number=0
        for i, (x, label, patch_idx, img_idx, global_idx) in enumerate(DataTrain.labeled_set):
            if self.true_labels[global_idx]==label:
                true_number+=1

        self.logger.info("After reset, we got true [ %d / %d ] label samples "% (true_number, len(DataTrain.labeled_set)))
        self.logger.info("After reset, we have %d labeled samples and %d unlabeled samples" % (
            len(DataTrain.labeled_set), len(DataTrain.unlabeled_set)))

        return
