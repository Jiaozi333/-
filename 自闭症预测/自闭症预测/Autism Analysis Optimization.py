import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 设置绘图风格
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用于显示负号

# 读取数据
df1 = pd.read_csv('Autism_Data.arff', na_values='?')
df2 = pd.read_csv('Toddler Autism dataset July 2018.csv', na_values='?')

# 数据筛选
data1 = df1[df1['Class/ASD'] == 'YES']  # 筛选出成年人ASD阳性数据
data2 = df2[df2['Class/ASD Traits '] == 'Yes']  # 筛选出幼儿ASD阳性数据

# 数据预处理
within24_36 = pd.get_dummies(df2['Age_Mons'] > 24, drop_first=True)  # 大于24个月的为1
within0_12 = pd.get_dummies(df2['Age_Mons'] < 13, drop_first=True)  # 小于13个月的为1
male = pd.get_dummies(df2['Sex'], drop_first=True)  # 性别为男性的为1
ethnics = pd.get_dummies(df2['Ethnicity'], drop_first=True)  # 独热编码种族
jaundice = pd.get_dummies(df2['Jaundice'], drop_first=True)  # 是否有黄疸
ASD_genes = pd.get_dummies(df2['Family_mem_with_ASD'], drop_first=True)  # 亲属中是否有自闭症
ASD_traits = pd.get_dummies(df2['Class/ASD Traits '], drop_first=True)  # ASD 特征

# 合并数据
final_data = pd.concat([within0_12, within24_36, male, ethnics, jaundice, ASD_genes, ASD_traits], axis=1)
final_data.columns = [
    'within0_12', 'within24_36', 'male', 'Latino', 'Native Indian', 'Others', 'Pacifica', 'White European',
    'asian', 'black', 'middle eastern', 'mixed', 'south asian', 'jaundice', 'ASD_genes', 'ASD_traits'
]

# 划分特征和标签
X = final_data.iloc[:, :-1]
y = final_data.iloc[:, -1]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# 逻辑回归
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

# 逻辑回归网格搜索
param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000]}
grid_log = GridSearchCV(LogisticRegression(), param_grid, refit=True)
grid_log.fit(X_train, y_train)
pred_log = grid_log.predict(X_test)

# 随机森林
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

# 标准化数据并应用于 KNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=101)

# KNN
error_rate = []
for i in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)

# 支持向量机（SVM）
svm = SVC(probability=True)
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)

# 决策树
dtc = DecisionTreeClassifier(random_state=101)
dtc.fit(X_train, y_train)
pred_dtc = dtc.predict(X_test)

# 1. ROC 曲线与 AUC 分数
log_probs = grid_log.predict_proba(X_test)[:, 1]
rfc_probs = rfc.predict_proba(X_test)[:, 1]
knn_probs = knn.predict_proba(X_test)[:, 1]
svm_probs = svm.predict_proba(X_test)[:, 1]
dtc_probs = dtc.predict_proba(X_test)[:, 1]

log_fpr, log_tpr, _ = roc_curve(y_test, log_probs)
rfc_fpr, rfc_tpr, _ = roc_curve(y_test, rfc_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)
dtc_fpr, dtc_tpr, _ = roc_curve(y_test, dtc_probs)

log_auc = roc_auc_score(y_test, log_probs)
rfc_auc = roc_auc_score(y_test, rfc_probs)
knn_auc = roc_auc_score(y_test, knn_probs)
svm_auc = roc_auc_score(y_test, svm_probs)
dtc_auc = roc_auc_score(y_test, dtc_probs)

plt.figure(figsize=(10, 6))
plt.plot(log_fpr, log_tpr, label=f'逻辑回归 (AUC = {log_auc:.2f})')
plt.plot(rfc_fpr, rfc_tpr, label=f'随机森林 (AUC = {rfc_auc:.2f})')
plt.plot(knn_fpr, knn_tpr, label=f'KNN (AUC = {knn_auc:.2f})')
plt.plot(svm_fpr, svm_tpr, label=f'SVM (AUC = {svm_auc:.2f})')
plt.plot(dtc_fpr, dtc_tpr, label=f'决策树 (AUC = {dtc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
plt.title('ROC 曲线')
plt.xlabel('假阳性率 (FPR)')
plt.ylabel('真正率 (TPR)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# 2. 混淆矩阵热力图
def plot_conf_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['否', '是'], yticklabels=['否', '是'])
    plt.title(f'{model_name} 混淆矩阵')
    plt.xlabel('预测值')
    plt.ylabel('实际值')
    plt.show()

plot_conf_matrix(y_test, pred_log, '逻辑回归')
plot_conf_matrix(y_test, pred_rfc, '随机森林')
plot_conf_matrix(y_test, pred_knn, 'KNN')
plot_conf_matrix(y_test, pred_svm, '支持向量机')
plot_conf_matrix(y_test, pred_dtc, '决策树')

# 3. 分类报告表格化
log_report = classification_report(y_test, pred_log, output_dict=True)
rfc_report = classification_report(y_test, pred_rfc, output_dict=True)
knn_report = classification_report(y_test, pred_knn, output_dict=True)
svm_report = classification_report(y_test, pred_svm, output_dict=True)
dtc_report = classification_report(y_test, pred_dtc, output_dict=True)

log_df = pd.DataFrame(log_report).transpose()
rfc_df = pd.DataFrame(rfc_report).transpose()
knn_df = pd.DataFrame(knn_report).transpose()
svm_df = pd.DataFrame(svm_report).transpose()
dtc_df = pd.DataFrame(dtc_report).transpose()

print("逻辑回归分类报告")
print(log_df)
print("\n随机森林分类报告")
print(rfc_df)
print("\nKNN分类报告")
print(knn_df)
print("\n支持向量机分类报告")
print(svm_df)
print("\n决策树分类报告")
print(dtc_df)

# 4. 参数调优效果展示
results = pd.DataFrame(grid_log.cv_results_)
plt.figure(figsize=(10, 6))
sns.lineplot(x=results['param_C'], y=results['mean_test_score'], marker='o', label='平均测试得分')
plt.xscale('log')
plt.title('逻辑回归超参数调优')
plt.xlabel('C (正则化参数)')
plt.ylabel('平均测试得分')
plt.legend()
plt.tight_layout()
plt.show()

# 5. 错误率对比
log_error = 1 - grid_log.best_estimator_.score(X_test, y_test)
rfc_error = 1 - rfc.score(X_test, y_test)
knn_error = 1 - knn.score(X_test, y_test)
svm_error = 1 - svm.score(X_test, y_test)
dtc_error = 1 - dtc.score(X_test, y_test)

model_names = ['逻辑回归', '随机森林', 'KNN', '支持向量机', '决策树']
error_rates = [log_error, rfc_error, min(error_rate), svm_error, dtc_error]

plt.figure(figsize=(8, 6))
sns.barplot(x=model_names, y=error_rates, palette='coolwarm')
plt.title('错误率对比')
plt.ylabel('错误率')
plt.xlabel('模型')
plt.tight_layout()
plt.show()
