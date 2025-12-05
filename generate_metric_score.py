import numpy as np
import os
from metric_data import test_dataset
from saliency_metric import cal_mae, cal_fm, cal_sm, cal_em, cal_wfm

dataset_path = '../RGBT_dataset/test/VT1000' ##gt_path

# dataset_path_pre = './output/vt1000/'  ##pre_salmap_path
sal_Path = './output/'
# test_datasets = ['']     ##test_datasets_name
lines = os.listdir(sal_Path)
# lines = ['vt1000-MAE-newbaseline','vt1000-MAE-4fusebaseline']
lines = ['vt1000-MAE-1fuse_30p']
for line in lines:
    sal_root = os.path.join(sal_Path, line)
    gt_root = dataset_path +'/GT/'
    test_loader = test_dataset(sal_root, gt_root)
    mae,fm,sm,em,wfm= cal_mae(),cal_fm(test_loader.size),cal_sm(),cal_em(),cal_wfm()
    for i in range(test_loader.size):
        #print ('predicting for %d / %d' % ( i + 1, test_loader.size))
        sal, gt = test_loader.load_data()
        if sal.size != gt.size:
            x, y = gt.size
            sal = sal.resize((x, y))
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        gt[gt > 0.5] = 1
        gt[gt != 1] = 0
        res = sal
        res = np.array(res)
        if res.max() == res.min():
            res = res/255
        else:
            res = (res - res.min()) / (res.max() - res.min())
        mae.update(res, gt)
        sm.update(res,gt)
        fm.update(res, gt)
        em.update(res,gt)
        wfm.update(res,gt)

    MAE = mae.show()
    maxf,meanf,_,_ = fm.show()
    sm = sm.show()
    em = em.show()
    wfm = wfm.show()
    print('epoch: {} MAE: {:.3f} maxF: {:.3f} avgF: {:.3f} wfm: {:.3f} Sm: {:.3f} Em: {:.3f}'.format(line, MAE, maxf,meanf,wfm,sm,em))