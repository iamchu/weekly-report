import numpy as np
import pandas as pd

"""
波士顿房价数据集字段说明:
crim:房屋所在镇的犯罪率
zn:面积大于25000平方英尺住宅所占的比例
indus:房屋所在镇非零售区域所占比例
chas:房屋是否位于河边，如果位于河边，则值为1，否则值为0
nox:一氧化氮的浓度
rm:平均房间数量
age:1940年前建成房屋所占比例
dis:房屋距离波士顿五大就业中心的加权距离
rad:距离房屋最近的公路
tax:财产税额度
ptratio:房屋所在镇师生比例
black:计算公式:1000*(房屋所在镇非美籍人口所占比例 - 0.63) ** 2
lstat:弱势群体人口所占比例
medv:房屋的平均价格
"""
data = pd.read_csv(r"boston_housing.csv")
#查看各个特征列是否存在缺失值
#data.info()
#查看是否具有重复值
#data.duplicated().any()

class LinearRegression:
    """
    最小二乘法实现线性回归
    """
    def fit(self,X,y):
        """
        训练模型
        :param X: 类数组类型[样本数量，特征数量]
                  特征矩阵，用来对模型进行训练
        :param y:类数组类型，形状：[样本数量]
        """
        #说明：如果X是数组对象的一部分，而不是完整的对象数据（例如，X是由其他对象通过切片传递过来）
        #则无法完成矩阵的切换
        #这里创建X的拷贝对象，避免转换成矩阵的时候失败
        X = np.asmatrix(X.copy())
        #y是一维结构（行向量或列向量)，一维结构可以不用拷贝
        #注意：我们现在要进行矩阵的运算，因此需要是二维的结构，我们通过reshape方法进行转换
        y = np.asmatrix(y).reshape(-1,1)    #一列，行数随内容匹配
        #通过最小二乘公式，求解出最佳的权重值。
        self.w_ = (X.T * X).I *X.T * y

    def predict(self,X):
        """
        对样本数据进行预测
        :param X: 待预测的样本特征
        :return: result：数组类型，预测的结果
        """
        #将X进行转换，注意，需要对X进行拷贝
        X = np.asmatrix(X.copy())
        result = X*self.w_
        #将矩阵转换为ndarray数组，进行扁平化处理
        #使用ravel可以将数组进行扁平化处理
        return np.array(result).ravel()

#不考虑截距的情况
t = data.sample(len(data),random_state=0)
train_X = t.iloc[:400,:-1]
train_y = t.iloc[:400,-1]
test_X = t.iloc[400:,:-1]
test_y = t.iloc[400:,-1]

lr = LinearRegression()
lr.fit(train_X,train_y)
result = lr.predict(test_X)
np.mean((result-test_y)**2)

#考虑截距，增加一列，该列的所有值都是1（本身是b，为了方便矩阵运算作为W0）
t = data.sample(len(data),random_state=0)
#可以这样增加一列
#按照习惯，截距作为W0，我们为之配上一个X0，X0放在最前面
new_columns = t.columns.insert(0,"Intercept")
#重新安排列的顺序，如果值为空，则使用fill_value参数指定的值进行填充
t = t.reindex(columns=new_columns,fill_value=1)
train_X = t.iloc[:400,:-1]
train_y = t.iloc[:400,-1]
test_X = t.iloc[400:,:-1]
test_y = t.iloc[400:,-1]

lr = LinearRegression()
lr.fit(train_X,train_y)
result = lr.predict(test_X)
print(np.mean((result-test_y)**2))

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
plt.title("线性回归预测-最小二乘法")
plt.xlabel("样本序号")
plt.ylabel("房价")
plt.legend()
plt.show()
