from scipy.io import loadmat
def split_data(datas,lables,train,test,val):
    pass
#从.mat文件加载Indian_pines数据集
def load_data():
    Indian_pines_corrected=loadmat("databases/Indian_pines_corrected.mat")['indian_pines_corrected']
    Indian_pines_gt=loadmat("databases/Indian_pines_gt.mat")['indian_pines_gt']
    return Indian_pines_corrected,Indian_pines_gt
def get_data():
    Indian_pines,Indian_pines_gt=load_data()
    #查看数据的结构
    #Indian_pines.shape is (145, 145, 200),Indian_pines_gt.shape is (145, 145)
    # print(f"Indian_pines.shape is {Indian_pines.shape},Indian_pines_gt.shape is {Indian_pines_gt.shape}")
    #转化为(145*145,200) (145*145,)
    Indian_pines=Indian_pines.reshape(-1,200)
    Indian_pines_gt=Indian_pines_gt.reshape(-1,1)
    # Indian_pines.shape is (21025, 200),Indian_pines_gt.shape is (21025, 1)
    # print(f"Indian_pines.shape is {Indian_pines.shape},Indian_pines_gt.shape is {Indian_pines_gt.shape}")

    #划分数据集为训练集、测试集、验证集 (6:3:1)
    split_data(Indian_pines,Indian_pines_gt,6,3,1)

get_data()