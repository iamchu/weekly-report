import numpy as np
import pandas as pd

#处理数据集
#读取鸢尾花数据集，header参数来制定标题的行。默认为0。如果没有标题，则使用None。
data = pd.read_csv(r"Iris.csv",header=0)
#将类别文本映射成为数值类型，
data["Species"] = data["Species"].map({"Iris-virginica":0,"Iris-setosa":1,"Iris-versicolor":2})
#删除不需要的id列
data = data.drop("Id",axis=1)
#删除重复的记录
data.drop_duplicates(inplace=True)

#算法模型
class KNN:
    """K近邻算法实现分类"""
    def __init__(self,k):
        """k:邻居的个数"""
        self.k = k

    def fit(self,X,y):
        """
        训练方法
        X:类数组类型，形状为：[样本数量，特征数量]
        待训练的样本特征（属性）
        y:类数组类型，形状为：[样本数量]
        每个样本的目标值（标签）
        """
        #将X转换成ndarray数组类型
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self,X):
        """
        根据参数传递的样本，对样本数据进行预测
        :param X:类数组类型，形状为：[样本数量，特征数量]
        待训练的样本特征（属性）
        :return:result 预测的结果，数组类型
        """
        X = np.asarray(X)
        result = []
        #对ndarry数组进行遍历，每次取数组中的一行
        for x in X:
            #对于测试集中的每一个样本，依次与训练集中的所有样本求距离
            dis = np.sqrt(np.sum((x - self.X) ** 2,axis=1))
            #返回数组排序后，每个元素在原数组中的索引
            index = dis.argsort()
            #进行截断，只取前k个元素（取距离最近的k个元素的索引）
            index = index[:self.k]
            #返回数组中每个元素出现的次数，元素必须是非负的整数
            count = np.bincount(self.y[index])
            #返回ndarray数组中，最大值的元素对应的索引，该索引就是我们判定的类别
            #最大元素索引，就是出现次数最多的元素
            result.append(count.argmax())
        return np.asarray(result)

    def predict2(self,X):
        """
        根据参数传递的样本，对样本数据进行预测(考虑权重，使用距离的倒数作为权重)
        :param X:类数组类型，形状为：[样本数量，特征数量]
        待训练的样本特征（属性）
        :return:result 预测的结果，数组类型
        """
        X = np.asarray(X)
        result = []
        #对ndarry数组进行遍历，每次取数组中的一行
        for x in X:
            #对于测试集中的每一个样本，依次与训练集中的所有样本求距离
            dis = np.sqrt(np.sum((x - self.X) ** 2,axis=1))
            #返回数组排序后，每个元素在原数组中的索引
            index = dis.argsort()
            #进行截断，只取前k个元素（取距离最近的k个元素的索引）
            index = index[:self.k]
            #返回数组中每个元素出现的次数，元素必须是非负的整数
            count = np.bincount(self.y[index],weights=1/dis[index])
            #返回ndarray数组中，最大值的元素对应的索引，该索引就是我们判定的类别
            #最大元素索引，就是出现次数最多的元素
            result.append(count.argmax())
        return np.asarray(result)

#训练与测试
#提取出每个类别鸢尾花的数据
t0 = data[data["Species"] == 0]
t1 = data[data["Species"] == 1]
t2 = data[data["Species"] == 2]
#对每个类别的数据进行洗牌
t0 = t0.sample(len(t0),random_state=0)
t1 = t1.sample(len(t1),random_state=0)
t2 = t2.sample(len(t2),random_state=0)
#构建训练集与测试集
train_X = pd.concat([t0.iloc[:40,:-1],t1.iloc[:40,:-1],t2.iloc[:40,:-1]],axis=0)
train_y = pd.concat([t0.iloc[:40,-1],t1.iloc[:40,-1],t2.iloc[:40,-1]],axis=0)
test_X = pd.concat([t0.iloc[40:,:-1],t1.iloc[40:,:-1],t2.iloc[40:,:-1]],axis=0)
test_y = pd.concat([t0.iloc[40:,-1],t1.iloc[40:,-1],t2.iloc[40:,-1]],axis=0)
#创建KNN对象，进行训练与测试
knn = KNN(k=3)
#进行训练
knn.fit(train_X,train_y)
#进行测试，获得测试的结果
result = knn.predict(test_X)
print(np.sum(result == test_y)/len(result))

#可视化
import matplotlib as mpl
import matplotlib.pyplot as plt
#默认情况下，matplotlib不支持中文显示，我们需要进行一下设置
#设置字体为黑体，以支持中文显示
mpl.rcParams["font.family"] = "SimHei"
#设置在中文字体时，能够正常显示符号（-）
mpl.rcParams["axes.unicode_minus"] = False

#设置画布大小
plt.figure(figsize=(10,10))
#绘制训练集数据
plt.scatter(x=t0["SepalLengthCm"][:40],y=t0["PetalLengthCm"][:40],color="r",label="Iris-virginica")
plt.scatter(x=t1["SepalLengthCm"][:40],y=t1["PetalLengthCm"][:40],color="g",label="Iris-setosa")
plt.scatter(x=t2["SepalLengthCm"][:40],y=t2["PetalLengthCm"][:40],color="b",label="Iris-versicolor")
plt.title("训练集数据")
#绘制测试集数据
right = test_X[result == test_y]
wrong = test_X[result != test_y]
plt.scatter(x=right["SepalLengthCm"],y=right["PetalLengthCm"],color="c",marker="x",label="right")
plt.scatter(x=wrong["SepalLengthCm"],y=wrong["PetalLengthCm"],color="m",marker=">",label="wrong")
plt.xlabel("花萼长度")
plt.ylabel("花瓣长度")
plt.title("KNN分类结果显示")
#将图例展示在最合适的地方
plt.legend(loc="best")
plt.show()