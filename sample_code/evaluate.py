import numpy_indexed as npi
import numpy as np
import pylas
import os


# gt_path = '../Assignment01-Data/Assignment01-Data-everybuilding/gt/BM1/'
# pr_path = '../Assignment01-Data/Assignment01-Data-result/BM1/'
# filename = 'BM10.las'
#
# gt_i = os.path.join(gt_path, filename)
# pr_i = os.path.join(pr_path, filename)

# read data
pcd_gt = pylas.read(r"C:\Users\24494\Downloads\code-assignment01\code-assignment01\Assignment01-Data\Assignment01-Data-everybuilding\gt\BM1/BM10.las") # Data01-cleanheight-1-roofblock.las: pcd.header.point_count =72590
pcd_pr = pylas.read(r"C:\Users\24494\Downloads\code-assignment01\code-assignment01\Assignment01-Data\Assignment01-Data-result\BM1/BM10.las")
print('statistic of pcd_gt: max z={}, min z={}, diff={}'.format(max(pcd_gt.z), min(pcd_gt.z), (max(pcd_gt.z)-min(pcd_gt.z))))


# arrange data
pcd_gt_p = np.array([pcd_gt.x, pcd_gt.y, pcd_gt.z]) # 3* n_points, gt
pcd_gt_p = pcd_gt_p.T   # n_points* 3
pcd_pr_p = np.array([pcd_pr.x, pcd_pr.y, pcd_pr.z]) # 3* n_points, predict result
pcd_pr_p = pcd_pr_p.T   # n_points* 3

# find intersection
intersection = npi.intersection(pcd_gt_p, pcd_pr_p)

# calculate accuracy
acc = intersection.shape[0] / pcd_gt_p.shape[0]

# print('acca of {} is:   {}'.format(gt_i, acc))

