
import numpy as np
from torch.autograd import Variable
import logging

from PR.pr import pr

def pr_main(network, dataset ,logger ,config, train_loader):
    """
    pr 1
    """
    clean_set = None
    ep_stats = {} #创建一个字典
    # ep_stats['label'] = np.array(labeled_loader.dataset.mislabeled_targets).astype(int) # 读取标签
    ep_stats['label'] = np.array(dataset.mislabeled_targets).astype(int) # 读取标签
    num_train = len(ep_stats['label']) # 训练集的长度
    ep_stats['idx'] = np.empty(num_train) # 创建一个空的数组
    ep_stats['feature'] = np.zeros((num_train, config['num_features'])) # 创建一个全0的数组
    ep_stats['pred'] = np.zeros_like(ep_stats['label']) - 1 # 创建一个全-1的数组
    clean_set_all = [] # 保存所有的clean set
    noise_set_all = [] # 保存所有的noise set
    noise_prob_all = [] # 保存所有的noise prob

    # 预测模式
    network.eval()
    for e in range(1):
        # train models
        network.eval()

        visited = 0 # 记录访问的数量
        for i, (images, labels, _, _, global_idx,idx) in enumerate(train_loader):
            images = Variable(images).cuda()
            labels = labels.long().cuda()
            
            feature, logits = network(images)
            

            logits = logits.argmax(-1).detach().cpu().numpy()
            feature = feature.detach().cpu().numpy()
            for i in range(len(logits)):
                ep_stats['idx'][visited] = idx[i]
                visited += 1
                ep_stats['pred'][idx[i]] = logits[i]
                ep_stats['feature'][idx[i]] = feature[i]
            ##
            # scaler.scale(loss_1).backward()
            # scaler.step(optimizer)
            # scaler.update()
            #

        """
        pr 3
        """
        ep_stats['idx'] = ep_stats['idx'][:visited]
        clean_set, noise_set, noise_prob = pr(config, ep_stats, clean_set) 
        logger.info(f"clean_set num: {len(clean_set)}")

        """
        pr 4
        """
     
        clean_set_all = clean_set
        noise_set_all = noise_set
        noise_prob_all = noise_prob
    return clean_set_all, noise_set_all, noise_prob_all