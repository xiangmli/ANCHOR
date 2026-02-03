"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""

import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world

from world import cprint
from time import time

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, path="../data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData-= 1
        testData -= 1
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")

        # (users,users)
        self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items))

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return 1892

    @property
    def m_items(self):
        return 4489

    @property
    def trainDataSize(self):
        return len(self.trainUser)

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)

            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems



    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user

    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']

    def __len__(self):
        return len(self.trainUniqueUsers)

# class Loader(BasicDataset):
#     """
#     Dataset type for pytorch \n
#     Incldue graph information
#     gowalla dataset
#     """
#
#     def __init__(self,config = world.config,path="../data/gowalla"):
#         # train or test
#         cprint(f'loading [{path}]')
#         self.split = config['A_split']
#         self.folds = config['A_n_fold']
#         self.mode_dict = {'train': 0, "test": 1}
#         self.mode = self.mode_dict['train']
#         self.n_user = 0
#         self.m_item = 0
#         train_file = path + '/train.txt'
#         test_file = path + '/test.txt'
#         self.path = path
#         trainUniqueUsers, trainItem, trainUser = [], [], []
#         testUniqueUsers, testItem, testUser = [], [], []
#         self.traindataSize = 0
#         self.testDataSize = 0
#
#         with open(train_file) as f:
#             for l in f.readlines():
#                 if len(l) > 0:
#                     l = l.strip('\n').split(' ')
#                     items = [int(i) for i in l[1:]]
#                     uid = int(l[0])
#                     trainUniqueUsers.append(uid)
#                     trainUser.extend([uid] * len(items))
#                     trainItem.extend(items)
#                     self.m_item = max(self.m_item, max(items))
#                     self.n_user = max(self.n_user, uid)
#                     self.traindataSize += len(items)
#         self.trainUniqueUsers = np.array(trainUniqueUsers)
#         self.trainUser = np.array(trainUser)
#         self.trainItem = np.array(trainItem)
#
#         with open(test_file) as f:
#             for l in f.readlines():
#                 if len(l) > 0:
#                     l = l.strip('\n').split(' ')
#                     items = [int(i) for i in l[1:]]
#                     uid = int(l[0])
#                     testUniqueUsers.append(uid)
#                     testUser.extend([uid] * len(items))
#                     testItem.extend(items)
#                     self.m_item = max(self.m_item, max(items))
#                     self.n_user = max(self.n_user, uid)
#                     self.testDataSize += len(items)
#         self.m_item += 1
#         self.n_user += 1
#         self.testUniqueUsers = np.array(testUniqueUsers)
#         self.testUser = np.array(testUser)
#         self.testItem = np.array(testItem)
#
#         self.Graph = None
#         print(f"{self.trainDataSize} interactions for training")
#         print(f"{self.testDataSize} interactions for testing")
#         print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")
#
#         # (users,items), bipartite graph
#         self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
#                                       shape=(self.n_user, self.m_item))
#         self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
#         self.users_D[self.users_D == 0.] = 1
#         self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
#         self.items_D[self.items_D == 0.] = 1.
#         # pre-calculate
#         self._allPos = self.getUserPosItems(list(range(self.n_user)))
#         self.__testDict = self.__build_test()
#         print(f"{world.dataset} is ready to go")
#
#     @property
#     def n_users(self):
#         return self.n_user
#
#     @property
#     def m_items(self):
#         return self.m_item
#
#     @property
#     def trainDataSize(self):
#         return self.traindataSize
#
#     @property
#     def testDict(self):
#         return self.__testDict
#
#     @property
#     def allPos(self):
#         return self._allPos
#
#     def _split_A_hat(self,A):
#         A_fold = []
#         fold_len = (self.n_users + self.m_items) // self.folds
#         for i_fold in range(self.folds):
#             start = i_fold*fold_len
#             if i_fold == self.folds - 1:
#                 end = self.n_users + self.m_items
#             else:
#                 end = (i_fold + 1) * fold_len
#             A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
#         return A_fold
#
#     def _convert_sp_mat_to_sp_tensor(self, X):
#         coo = X.tocoo().astype(np.float32)
#         row = torch.Tensor(coo.row).long()
#         col = torch.Tensor(coo.col).long()
#         index = torch.stack([row, col])
#         data = torch.FloatTensor(coo.data)
#         return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
#
#     def getSparseGraph(self):
#         print("loading adjacency matrix")
#         if self.Graph is None:
#             try:
#                 pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
#                 print("successfully loaded...")
#                 norm_adj = pre_adj_mat
#             except :
#                 print("generating adjacency matrix")
#                 s = time()
#                 adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
#                 adj_mat = adj_mat.tolil()
#                 R = self.UserItemNet.tolil()
#                 adj_mat[:self.n_users, self.n_users:] = R
#                 adj_mat[self.n_users:, :self.n_users] = R.T
#                 adj_mat = adj_mat.todok()
#                 # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
#
#                 rowsum = np.array(adj_mat.sum(axis=1))
#                 d_inv = np.power(rowsum, -0.5).flatten()
#                 d_inv[np.isinf(d_inv)] = 0.
#                 d_mat = sp.diags(d_inv)
#
#                 norm_adj = d_mat.dot(adj_mat)
#                 norm_adj = norm_adj.dot(d_mat)
#                 norm_adj = norm_adj.tocsr()
#                 end = time()
#                 print(f"costing {end-s}s, saved norm_mat...")
#                 sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
#
#             if self.split == True:
#                 self.Graph = self._split_A_hat(norm_adj)
#                 print("done split matrix")
#             else:
#                 self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
#                 self.Graph = self.Graph.coalesce().to(world.device)
#                 print("don't split the matrix")
#         return self.Graph
#
#     def __build_test(self):
#         """
#         return:
#             dict: {user: [items]}
#         """
#         test_data = {}
#         for i, item in enumerate(self.testItem):
#             user = self.testUser[i]
#             if test_data.get(user):
#                 test_data[user].append(item)
#             else:
#                 test_data[user] = [item]
#         return test_data
#
#     def getUserItemFeedback(self, users, items):
#         """
#         users:
#             shape [-1]
#         items:
#             shape [-1]
#         return:
#             feedback [-1]
#         """
#         # print(self.UserItemNet[users, items])
#         return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))
#
#     def getUserPosItems(self, users):
#         posItems = []
#         for user in users:
#             posItems.append(self.UserItemNet[user].nonzero()[1])
#         return posItems
#
#     # def getUserNegItems(self, users):
#     #     negItems = []
#     #     for user in users:
#     #         negItems.append(self.allNeg[user])
#     #     return negItems

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Include graph information
    gowalla dataset with noise labels
    """

    def __init__(self, config=world.config, path="../data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0

        # ======================== 确保路径正确 ========================
        # 如果path不存在，尝试创建完整路径
        if not os.path.exists(path):
            # 尝试从相对于当前工作目录的路径
            potential_paths = [
                path,
                os.path.join(os.getcwd(), path),
            ]
            for p in potential_paths:
                if os.path.exists(p):
                    path = p
                    break

        self.data_path = path  # 保存数据路径，用于后续输出

        train_file = os.path.join(path, 'train.txt')
        test_file = os.path.join(path, 'test.txt')
        noise_file = os.path.join(path, 'noise_data.txt')  # 噪声数据文件

        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        # ======================== 读取干净的训练数据 ========================
        cprint(f"Loading train data from {train_file}")
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)

        # ======================== 读取噪声数据 ========================
        noise_user = []
        noise_item = []
        noise_size = 0

        if os.path.exists(noise_file):
            cprint(f"Loading noise data from {noise_file}")
            with open(noise_file) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        items = [int(i) for i in l[1:]]
                        uid = int(l[0])
                        noise_user.extend([uid] * len(items))
                        noise_item.extend(items)
                        self.m_item = max(self.m_item, max(items))
                        self.n_user = max(self.n_user, uid)
                        noise_size += len(items)
            cprint(f"Loaded {noise_size} noise interactions")
        else:
            cprint(f"Warning: {noise_file} not found!")

        # ======================== 合并干净数据和噪声数据作为混合训练集 ========================
        self.trainUser = np.array(trainUser + noise_user)
        self.trainItem = np.array(trainItem + noise_item)
        # 构建噪声标签：0表示干净，1表示噪声
        self.noise_labels = np.array(
            [0] * len(trainUser) + [1] * len(noise_user)
        )
        self.traindataSize = len(self.trainUser)

        # 获取唯一用户（用于采样）
        self.trainUniqueUsers = np.unique(
            np.concatenate([np.array(trainUniqueUsers), np.array(noise_user)])
        )

        cprint(f"Total training interactions: {self.traindataSize}")
        cprint(f"Clean interactions: {len(trainUser)}, Noise interactions: {noise_size}")
        cprint(f"Noise label distribution: {np.sum(self.noise_labels == 0)} clean, "
               f"{np.sum(self.noise_labels == 1)} noisy")

        # ======================== 读取测试数据 ========================
        cprint(f"Loading test data from {test_file}")
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)

        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.Graph = None
        print(f"{len(trainUser)} clean interactions for training")
        print(f"{noise_size} noise interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.traindataSize + self.testDataSize) / self.n_users / self.m_items}")

        print(f"\n创建用户-物品交互矩阵...")
        print(f"  总交互数: {len(self.trainUser)}")
        print(f"  用户数: {self.n_user}")
        print(f"  物品数: {self.m_item}")

        # 确保索引在有效范围内
        assert self.trainUser.min() >= 0 and self.trainUser.max() < self.n_user
        assert self.trainItem.min() >= 0 and self.trainItem.max() < self.m_item

        # 使用 coo_matrix，它会自动处理重复项
        data = np.ones(len(self.trainUser), dtype=np.float32)
        row_indices = self.trainUser.astype(np.int32)
        col_indices = self.trainItem.astype(np.int32)

        # 先创建 COO 矩阵
        from scipy.sparse import coo_matrix
        coo = coo_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.n_user, self.m_item),
            dtype=np.float32
        )

        # 转换为 CSR（会自动合并重复项）
        self.UserItemNet = coo.tocsr()

        # # (users,items), bipartite graph - 基于干净数据构建
        # self.UserItemNet = csr_matrix(
        #     (np.ones(len(trainUser)), (self.trainUser, self.trainItem)),
        #     shape=(self.n_user, self.m_item)
        # )
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    # ======================== 新增：获取噪声标签 ========================
    def getNoiseLabels(self):
        """
        返回噪声标签数组
        0: 干净交互
        1: 噪声交互
        """
        return self.noise_labels

    # ... 其他方法保持不变 ...

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        """
        构建图的邻接矩阵（归一化）
        """
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                # 尝试加载预先保存的邻接矩阵
                pre_adj_mat = sp.load_npz(os.path.join(self.path, 's_pre_adj_mat.npz'))
                print("successfully loaded pre-computed adjacency matrix...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()

                # 构建二部图邻接矩阵
                adj_mat = sp.dok_matrix(
                    (self.n_users + self.m_items, self.n_users + self.m_items),
                    dtype=np.float32
                )
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()

                # R 在左上角，R^T 在右下角
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()

                # 计算度矩阵的逆平方根用于对称归一化
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                # 对称归一化: D^{-1/2} A D^{-1/2}
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                end = time()
                print(f"costing {end - s}s, saving adjacency matrix...")

                # 保存以供下次使用
                os.makedirs(self.path, exist_ok=True)
                sp.save_npz(os.path.join(self.path, 's_pre_adj_mat.npz'), norm_adj)

            # 转换为 PyTorch 稀疏张量
            if self.split:
                # 如果启用图分割
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                # 不分割，直接使用
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("adjacency matrix loaded to device")

        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # 在 Loader 类中添加方法
    def getNoiseLabelByUserItem(self, users, items):
        """
        根据 (user, item) 对获取噪声标签
        users: array of user ids
        items: array of item ids
        return: array of noise labels
        """
        noise_labels = []
        for user, item in zip(users, items):
            # 找到该 (user, item) 在 trainUser, trainItem 中的索引
            mask = (self.trainUser == user) & (self.trainItem == item)
            idx = np.where(mask)[0]
            if len(idx) > 0:
                noise_labels.append(self.noise_labels[idx[0]])
            else:
                # 如果找不到（不应该发生），默认为干净
                noise_labels.append(0)
        return np.array(noise_labels)

