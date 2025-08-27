import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import scipy.sparse as sp


class MultiViewDataset(Dataset):
    def __init__(self, data_name, data_X, data_Y):
        super(MultiViewDataset, self).__init__()
        self.data_name = data_name

        self.X = dict()
        self.num_views = data_X.shape[0]

        for v in range(self.num_views):
            self.X[v] = self.normalize(data_X[v])

        self.Y = data_Y
        self.Y = np.squeeze(self.Y)
        if np.min(self.Y) == 1:
            self.Y = self.Y - 1
        self.Y = self.Y.astype(dtype=np.int64)
        self.num_classes = len(np.unique(self.Y))
        self.dims = self.get_dims()

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.Y[index]
        return data, target, index

    def __len__(self):
        return len(self.X[0])

    def get_dims(self):
        dims = []
        for view in range(self.num_views):
            dims.append([self.X[view].shape[1]])
        return np.array(dims)

    @staticmethod
    def normalize(x, min=0):
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def postprocessing(self, index, addNoise=False, sigma=0, ratio_noise=0.5, addConflict=False, ratio_conflict=0.5):
        if addNoise:
            self.addNoise(index, ratio_noise, sigma=sigma)
            print('addNoise')
        if addConflict:
            self.addConflict(index, ratio_conflict)
            print('addConflict')
        pass

    def addNoise(self, index, ratio, sigma):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            views = np.random.choice(np.array(self.num_views), size=np.random.randint(self.num_views), replace=False)
            for v in views:
                self.X[v][i] = np.random.normal(self.X[v][i], sigma)
        pass

    def addConflict(self, index, ratio):
        records = dict()
        for c in range(self.num_classes):
            i = np.where(self.Y == c)[0][0]
            temp = dict()
            for v in range(self.num_views):
                temp[v] = self.X[v][i]
            records[c] = temp
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            v = np.random.randint(self.num_views)
            self.X[v][i] = records[(self.Y[i] + 1) % self.num_classes][v]
        pass

class Multi_view_data2(Dataset):

    def __init__(self, data_path, name=None):

        super(Multi_view_data2, self).__init__()

        self.data_name = name
        dataset = sio.loadmat(data_path)

        view_number = int((len(dataset) - 4))

        self.X = dict()
        
        if name == 'RGBD':
            self.Y = dataset['gt'].reshape(-1).astype(int)
            view_number = 2
            for v_num in range(view_number):
                X = dataset['X'][0][v_num]
                self.X[v_num] = self.normalize(X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]))
        elif name == 'Fashion':
            view_number = 3
            self.Y = dataset['Y'].reshape(-1).astype(int) 
            for v_num in range(view_number):
                X = dataset['X' + str(v_num + 1)]
                self.X[v_num] = self.normalize(X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]))
        else:
            if 'gt' in dataset.keys() and 'X' in dataset.keys():
                self.Y = dataset['gt'].reshape(-1).astype(int)
                view_number = int(dataset['X'].shape[1])
                for v_num in range(view_number):
                    self.X[v_num] = self.normalize(dataset['X'][0][v_num].T )
            elif 'truelabel' in dataset.keys(): # BBC
                self.Y = dataset['truelabel'][0][0].reshape(-1).astype(int)
                view_number = int(dataset['data'].shape[1])
                for v_num in range(view_number):
                    self.X[v_num] = self.normalize(dataset['data'][0][v_num].toarray().T)
            elif 'X' in dataset.keys() and 'Y' in dataset.keys(): # 100Leaves Catech20 LandUse
                self.Y = dataset['Y'].reshape(-1).astype(int)
                view_number = dataset['X'].shape[1]
                for v_num in range(view_number):
                    x = dataset['X'][0][v_num]
                    if sp.issparse(x):
                        x = x.toarray()
                    self.X[v_num] = self.normalize(x) 
            elif 'Y' in dataset.keys() and 'X1' in dataset.keys():
                self.Y = dataset['Y'].reshape(-1).astype(int) 
                for v_num in range(view_number):
                    self.X[v_num] = self.normalize(dataset['X' + str(v_num + 1) ])  # x1_train x2_train x3_train
            elif 'gt' in dataset.keys() and 'x1' in dataset.keys(): # MSRC # XMediaNet
                self.Y = dataset['gt'].reshape(-1).astype(int) 
                for v_num in range(view_number):
                    self.X[v_num] = self.normalize(dataset['x' + str(v_num + 1) ]) 

        if self.Y.min() == 1:
            self.Y -= 1

        self.num_classes = len(np.unique(self.Y))
        self.num_views = view_number
        self.dims = self.get_dims()

    """
    Gets the data and categories for the corresponding index
    """
    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.Y[index]
        return data, target, index

    def __len__(self):
        return len(self.X[0])
    
    def get_dims(self):
        dims = []
        for view in range(self.num_views):
            dims.append([self.X[view].shape[1]])

        return np.array(dims)
    
    @staticmethod
    def normalize(x, min=0):
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:  
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def postprocessing(self, index, addNoise=False, sigma=0, ratio_noise=0.5, addConflict=False, ratio_conflict=0.5):
        if addNoise:
            self.addNoise(index, ratio_noise, sigma=sigma)
            print("addNoise")
        if addConflict:
            self.addConflict(index, ratio_conflict)
            print("addConflict")
        pass

    def addNoise(self, index, ratio, sigma):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            views = np.random.choice(np.array(self.num_views), size=np.random.randint(self.num_views), replace=False)
            for v in views:
                self.X[v][i] = np.random.normal(self.X[v][i], sigma)

    def addConflict(self, index, ratio):
        records = dict()
        for c in range(self.num_classes):
            i = np.where(self.Y == c)[0][0]
            temp = dict()
            for v in range(self.num_views):
                temp[v] = self.X[v][i]
            records[c] = temp
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            v = np.random.randint(self.num_views)
            self.X[v][i] = records[(self.Y[i] + 1) % self.num_classes][v]

def HandWritten():
    # dims of views: 240 76 216 47 64 6
    data_path = "dataset/handwritten.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['Y']
    return MultiViewDataset("HandWritten", data_X, data_Y)


def Scene():
    # dims of views: 20 59 40
    data_path = "dataset/Scene15.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['gt']
    for v in range(len(data_X)):
        data_X[v] = data_X[v].T
    return MultiViewDataset("Scene", data_X, data_Y)


def PIE():
    # dims of views: 484 256 279
    data_path = "dataset/PIE_face_10.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['gt']
    for v in range(len(data_X)):
        data_X[v] = data_X[v].T
    return MultiViewDataset("PIE", data_X, data_Y)

def ANIMAL():
    # dims of views: 4096 4096
    data_path = "dataset/animal.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['gt']
    for v in range(len(data_X)):
        data_X[v] = data_X[v].T
    return MultiViewDataset("ANIMAL", data_X, data_Y)

def MSRC():
    data_path = "dataset/MSRCV1_6views.mat"
    return Multi_view_data2(data_path,name='MSRC')

def Caltech(): 
    data_path = "dataset/2view-caltech101-8677sample.mat"
    return Multi_view_data2(data_path)

def HMDB():
    data_path = "dataset/HMDB.mat"
    return Multi_view_data2(data_path)

def Leaves():
    data_path = "dataset/100Leaves.mat"
    return Multi_view_data2(data_path,name='Leaves')

def RGBD():
    data_path = "dataset/rgbd_mtv.mat"
    return Multi_view_data2(data_path, name="RGBD")

def Fashion():
    data_path = "dataset/3V_Fashion_MV.mat"
    return Multi_view_data2(data_path, name="Fashion")

def NUSWIDEOBJ():
    data_path = "dataset/NUSWIDEOBJ.mat"
    return Multi_view_data2(data_path, name="NUSWIDEOBJ")

def LandUse():
    data_path = "dataset/LandUse_21.mat"
    return Multi_view_data2(data_path, name="LandUse")

def Reuters():
    data_path = "dataset/Reuters.mat"
    return Multi_view_data2(data_path, name="Reuters")