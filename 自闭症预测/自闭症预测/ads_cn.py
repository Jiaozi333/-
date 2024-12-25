import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

# 读取数据集1和数据集2
df1 = pd.read_csv('Autism_Data.arff', na_values='?')
df2 = pd.read_csv('Toddler Autism dataset July 2018.csv', na_values='?')

sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 提取ASD类别为YES的数据（成年人）
data1 = df1[df1['Class/ASD'] == 'YES']

# 提取ASD Traits为Yes的数据（幼儿）
data2 = df2[df2['Class/ASD Traits '] == 'Yes']

# 计算ASD阳性成年人的比例
print("成年人阳性比例: {:.2f}%".format(len(data1) / len(df1) * 100))

# 计算ASD阳性幼儿的比例
print("幼儿阳性比例: {:.2f}%".format(len(data2) / len(df2) * 100))

# 缺失值可视化
fig, ax = plt.subplots(1, 2, figsize=(20, 6))

sns.heatmap(data1.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax[0])
ax[0].set_title('成年人数据集缺失值情况')
ax[0].set_ylabel('样本索引')

sns.heatmap(data2.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax[1])
ax[1].set_title('幼儿数据集缺失值情况')
ax[1].set_ylabel('样本索引')

plt.show()

# 出生时黄疸与性别分布
fig, ax = plt.subplots(1, 2, figsize=(20, 6))
sns.countplot(x='jundice', data=data1, hue='gender', ax=ax[0])
ax[0].set_title('成年人出生时黄疸与性别分布')
ax[0].set_xlabel('出生时是否有黄疸')

sns.countplot(x='Jaundice', data=data2, hue='Sex', ax=ax[1])
ax[1].set_title('幼儿出生时黄疸与性别分布')
ax[1].set_xlabel('出生时是否有黄疸')

plt.show()

# 年龄分布
fig, ax = plt.subplots(1, 2, figsize=(20, 6))
sns.histplot(data1['age'], kde=False, bins=45, color='darkred', ax=ax[0])
ax[0].set_xlabel('年龄（岁）')
ax[0].set_title('ASD阳性成年人年龄分布')

sns.histplot(data2['Age_Mons'], kde=False, bins=30, color='darkred', ax=ax[1])
ax[1].set_xlabel('年龄（月）')
ax[1].set_title('ASD阳性幼儿年龄分布')

plt.show()

# 国家分布
plt.figure(figsize=(20, 6))
sns.countplot(
    x='contry_of_res',
    data=data1,
    order=data1['contry_of_res'].value_counts().index[:15],
    hue='gender',
    palette='viridis'
)
plt.title('ASD阳性成年人的国家分布')
plt.xlabel('国家')
plt.ylabel('人数')
plt.tight_layout()
plt.show()

# 数据预处理与特征工程
within24_36 = pd.get_dummies(df2['Age_Mons'] > 24, drop_first=True)
within0_12 = pd.get_dummies(df2['Age_Mons'] < 13, drop_first=True)
male = pd.get_dummies(df2['Sex'], drop_first=True)
ethnics = pd.get_dummies(df2['Ethnicity'], drop_first=True)
jaundice = pd.get_dummies(df2['Jaundice'], drop_first=True)
ASD_genes = pd.get_dummies(df2['Family_mem_with_ASD'], drop_first=True)
ASD_traits = pd.get_dummies(df2['Class/ASD Traits '], drop_first=True)

final_data = pd.concat([within0_12, within24_36, male, ethnics, jaundice, ASD_genes, ASD_traits], axis=1)
final_data.columns = [
    'within0_12', 'within24_36', 'male', 'Latino', 'Native Indian', 'Others', 'Pacifica',
    'White European', 'asian', 'black', 'middle eastern', 'mixed', 'south asian', 'jaundice',
    'ASD_genes', 'ASD_traits'
]

X = final_data.iloc[:, :-1]
y = final_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

# 定义分类器
models = {
    "逻辑回归": LogisticRegression(),
    "随机森林": RandomForestClassifier(),
    "K近邻算法": KNeighborsClassifier(),
    "支持向量机": SVC(),
    "决策树": DecisionTreeClassifier()
}

# 存储准确率
accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    accuracies[name] = acc
    print(f"{name} 准确率: {acc:.2f}")

# 绘制准确率柱状图
plt.figure(figsize=(10, 6))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette="viridis")
plt.title("模型准确率比较")
plt.ylabel("准确率")
plt.xlabel("模型")
plt.show()
