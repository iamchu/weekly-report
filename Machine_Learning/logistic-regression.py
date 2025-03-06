import numpy as np
import pandas as pd

#处理数据集
#读取鸢尾花数据集，header参数来制定标题的行。默认为0。如果没有标题，则使用None。
data = pd.read_csv(r"Iris.csv",header=0)
#将类别文本映射成为数值类型，
data["Species"] = data["Species"].map({"Iris-virginica":0,"Iris-setosa":1,"Iris-versicolor":2})
#只选取类别为0与1的鸢尾花数据，进行逻辑回归二分类
data = data[data["Species"]!=2]

class LogisticRegression:
    """
    实现逻辑回归算法
    """
    def __init__(self,alpha,times):
        self.alpha = alpha  #学习率
        self.times = times  #迭代次数

    def sigmoid(self,z):
        """
        :param z:float类型，值为 z=w.T*x
        :return:p值在[0,1]之间
                返回样本属于类别1的概率值，用来作为结果预测
                当z>=0.5(z>=0)时，判定为类别1，否则判定为类别0
        """
        z = np.clip(z, -500, 500)  # 限制 z 的范围，防止溢出
        return 1.0 / (1.0+np.exp(-z))

    def fit(self,X,y):
        X = np.asarray(X)
        y = np.asarray(y)
        #创建权重向量，初始值为0，长度比特征多1（截距）
        self.w_ = np.zeros(1+X.shape[1])
        #创建损失列表，用来保存每次迭代后的损失值
        self.loss_ = []

        for i in range(self.times):
            z = np.dot(X,self.w_[1:]) + self.w_[0]
            #计算概率值（结果判定为1的概率值）
            p = self.sigmoid(z)
            #根据逻辑回归的代价函数（损失函数），计算损失值
            #逻辑回归的代价函数：
            #J（w） = -sum(yi * log(s(zi)) + (1-yi) * log(1-s(zi))) [i从1到n，n为样本数量]
            epsilon = 1e-10  # 避免 log(0)
            cost = -np.sum(y * np.log(p + epsilon) + (1 - y) * np.log(1 - p + epsilon))
            #cost = -np.sum(y * np.log(p) + (1-y) * np.log(1-p))
            self.loss_.append(cost)

            #调整权重值
            #根据公式：权重[j列] = 权重[j列] + 学习率 * sum((y-s(z))*x(j))
            self.w_[0] += self.alpha * np.sum(y-p)
            self.w_[1:] += self.alpha * np.dot(X.T,y-p)

    def predict_proba(self,X):
        """
        对样本数据进行预测
        :return:预测结果，概率值
        """
        X = np.asarray(X)
        z = np.dot(X,self.w_[1:] + self.w_[0])
        p = self.sigmoid(z)
        #将预测结果变成二维数组结构便于后续的拼接
        p = p.reshape(-1,1)
        #将两个数组进行拼接，方向为横向拼接
        return np.concatenate([1-p,p],axis=1)

    def predict(self,X):
        """
        根据参数传递的样本，对样本数据进行预测
        :return:数组类型，预测的结果分类值
        """
        return np.argmax(self.predict_proba(X),axis=1)

#提取出每个类别鸢尾花的数据
t1 = data[data["Species"] == 0]
t2 = data[data["Species"] == 1]
#对每个类别的数据进行洗牌
t1 = t1.sample(len(t1),random_state=0)
t2 = t2.sample(len(t2),random_state=0)
#构建训练集与测试集
train_X = pd.concat([t1.iloc[:40,:-1],t2.iloc[:40,:-1]],axis=0)
train_y = pd.concat([t1.iloc[:40,-1],t2.iloc[:40,-1]],axis=0)
test_X = pd.concat([t1.iloc[40:,:-1],t2.iloc[40:,:-1]],axis=0)
test_y = pd.concat([t1.iloc[40:,-1],t2.iloc[40:,-1]],axis=0)

#鸢尾花的特征列都在同一个数量级，我们这里可以不用标准化处理
lr = LogisticRegression(alpha=0.01,times=20)
lr.fit(train_X,train_y)
#预测的概率值
lr.predict_proba(test_X)
#预测类别
lr.predict(test_X)
result = lr.predict(test_X)
#计算准确性
print(np.sum((result == test_y) / len(test_y)))

#可视化
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False
#绘制预测值
plt.plot(result,"ro",ms=15,label="预测值")
#绘制真实值
plt.plot(test_y.values,"go",label="真实值")
plt.title("逻辑回归")
plt.xlabel("样本序号")
plt.ylabel("类别")
plt.legend()
plt.show()