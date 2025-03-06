import numpy as np
import pandas as pd
data = pd.read_csv(r"Iris.csv")
#删除不需要的ID与Species列,因为现在进行回归预测，类别信息没有用处
data.drop(["Id","Species"],axis=1,inplace=True)
#删除重复记录
data.drop_duplicates(inplace=True)

class KNN:
    """
    K近邻算法
    该算法用于回归预测，根据前3个特征属性，寻找最近的k个邻居
    然后再根据k个邻居的第四个特征属性，去预测当前样本的第四个特征值
    """
    def __init__(self,k):
        self.k = k

    def fit(self,X,y):
        #将X与y转换成ndarray数组的形式，方便统一进行操作
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self,X):
        #转换成数组类型
        X = np.asarray(X)
        #保存预测的结果值
        result = []
        for x in X:
            #计算距离（计算与训练集中每个X的距离）
            dis = np.sqrt(np.sum((x-self.X)**2,axis=1))
            #返回数组排序后，每个元素在原数组中的索引
            index = dis.argsort()
            #取前k个距离最近的索引
            index = index[:self.k]
            #计算均值，然后加入到结果列表当中
            result.append(np.mean(self.y[index]))
        return np.array(result)

    #考虑权重的预测函数
    def predict2(self,X):
        #转换成数组类型
        X = np.asarray(X)
        #保存预测的结果值
        result = []
        for x in X:
            #计算距离（计算与训练集中每个X的距离）
            dis = np.sqrt(np.sum((x-self.X)**2,axis=1))
            #返回数组排序后，每个元素在原数组中的索引
            index = dis.argsort()
            #取前k个距离最近的索引
            index = index[:self.k]
            #求所有邻居节点距离倒数之和
            #最后加上一个很小的值，就是为了避免除数（距离）为0的情况
            s = np.sum(1/(dis[index] + 0.001))
            #使用每个结点距离的倒数，除以倒数之和，得到权重
            weight = (1 / (dis[index] + 0.001)) / s
            #使用邻居节点的标签值，乘以对应的权重，然后相加，得到最终的预测结果
            result.append(np.sum(self.y[index]*weight))
        return np.array(result)

t = data.sample(len(data),random_state=0)
train_X = t.iloc[:120,:-1]
train_y = t.iloc[:120,-1]
test_X = t.iloc[120:,:-1]
test_y = t.iloc[120:,-1]
knn = KNN(k=3)
knn.fit(train_X,train_y)
result = knn.predict(test_X)
np.mean(np.sum((result - test_y) ** 2))

#可视化
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(10,10))
#绘制预测值
plt.plot(result,"ro-",label="预测值")
#绘制真实值
plt.plot(test_y.values,"go--",label="真实值")
plt.title("KNN连续值预测展示")
plt.xlabel("节点序号")
plt.ylabel("花瓣宽度")
plt.legend()
plt.show()