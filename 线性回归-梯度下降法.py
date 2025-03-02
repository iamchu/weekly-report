import numpy as np
import pandas as pd

data = pd.read_csv(r"boston_housing.csv")

class LinearRegression:
    """
    梯度下降法实现线性回归
    """
    def __init__(self,alpha,times):
        """
        :param alpha:学习率，用来控制步长(权重调整幅度)
        :param times:循环迭代的次数
        """
        self.alpha = alpha
        self.times = times

    def fit(self,X,y):
        """
        训练
        :param X:类数组类型 [样本数量，特征数量]
                 待训练的样本特征属性（特征矩阵）
        :param y:类数组类型。形状 [样本数量]
                 目标值（标签信息）
        """
        X = np.asarray(X)
        y = np.asarray(y)
        #创建权重的向量，初始值为0（或任何其他的值），长度比特征数量多1，多出的一个值是截距
        self.w_ = np.zeros(1+X.shape[1])
        #创建损失列表，用来保存每次迭代后的损失值。损失值计算：（预测值-真实值）的平方和除以2
        self.loss_ = []

        #进行循环，多次迭代。在每次迭代中，不断的去调整权重值，使得损失值不断减小
        for i in range(self.times):
            #计算预测值
            y_hat = np.dot(X,self.w_[1:]) + self.w_[0]
            #计算真实值与预测值之间的差距
            error = y - y_hat
            #将损失值加入到损失列表中
            self.loss_.append(np.sum(error ** 2) / 2)
            #根据差距调整权重w_,根据公式：调整为 权重（j） = 权重（j）+ 学习率 * sum((y-y_hat)*x(j))
            self.w_[0] += self.alpha * np.sum(error)
            self.w_[1:] += self.alpha * np.dot(X.T,error)

    def predict(self,X):
        """
        预测
        :param X: 类数组类型，形状[样本数量，特征数量]
        :return: result数组类型，预测的结果
        """
        X = np.asarray(X)
        result = np.dot(X,self.w_[1:])+self.w_[0]
        return result

class StandardScaler:
    """
    该类对数据进行标准化处理
    """
    def fit(self,X):
        """
        根据传递的样本，计算每个特征列的均值与标准差
        :param X:训练数据，用来计算均值与标准差
        """
        X = np.asarray(X)
        self.std_ = np.std(X,axis=0)   #标准差
        self.mean_ = np.mean(X,axis=0)   #均值

    def transform(self,X):
        """
        对给定的数据X，进行标准化处理。（将X的每一列都变成标准正态分布的数据）
        :param X:类数组类型，待转换数据
        :return:result类数组类型，参数转换成标准正态分布后的结果
        """
        return (X - self.mean_) / self.std_

    def fit_transform(self,X):
        """
        对数据进行训练，并转换，返回转换之后的结果
        :param X:类数组类型，待转换的数据
        :return:result类数组类型，参数转换成标准正态分布后的结果
        """
        self.fit(X)
        return self.transform(X)

#为了避免每个特征数量级不同，从而在梯度下降过程中带来影响
#我们现在考虑对每个特征进行标准化处理
lr = LinearRegression(alpha=0.0005,times=20)
t = data.sample(len(data),random_state=0)
train_X = t.iloc[:400,:-1]
train_y = t.iloc[:400,-1]
test_X = t.iloc[400:,:-1]
test_y = t.iloc[400:,-1]

#对数据进行标准化处理
s = StandardScaler()
train_X = s.fit_transform(train_X)
test_X = s.transform(test_X)

s2 = StandardScaler()
train_y = s2.fit_transform(train_y)
test_y = s2.transform(test_y)

lr.fit(train_X,train_y)
result = lr.predict(test_X)
print(np.mean(result - test_y)**2)

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
plt.title("线性回归预测-梯度下降")
plt.xlabel("样本序号")
plt.ylabel("房价")
plt.legend()
plt.show()

#绘制累计误差值
plt.plot(range(1,lr.times + 1),lr.loss_,"o-")
plt.show()

#因为房价涉及多个维度，不方便进行可视化显示，为了实现可视化
#我们只选取其中的一个维度（RM），并画出直线，进行拟合
lr = LinearRegression(alpha=0.0005,times=50)
t = data.sample(len(data),random_state=0)
train_X = t.iloc[:400,5:6]
train_y = t.iloc[:400,-1]
test_X = t.iloc[400:,5:6]
test_y = t.iloc[400:,-1]

#对数据进行标准化处理
s = StandardScaler()
train_X = s.fit_transform(train_X)
test_X = s.transform(test_X)
s2 = StandardScaler()
train_y = s2.fit_transform(train_y)
test_y = s2.transform(test_y)

lr.fit(train_X,train_y)
result = lr.predict(test_X)
print(np.mean(result - test_y)**2)

#绘制散点图
plt.scatter(train_X["rm"],train_y)
#查看方程系数
#print(lr.w_)
#构建方程 y = -2.77333712e-16 + 6.54984608e-01 * x
x = np.arange(-5,5,0.1)
y = -2.77333712e-16 + 6.54984608e-01 * x
plt.plot(x,y,"r")
plt.show()