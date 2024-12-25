import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# 获取文件的相对路径
file_path = os.path.join(os.path.dirname(__file__), 'Toddler Autism dataset July 2018.csv')

# 读取数据集并进行预处理
df2 = pd.read_csv(file_path, na_values='?')

within24_36 = pd.get_dummies(df2['Age_Mons'] > 24, drop_first=True)
within0_12 = pd.get_dummies(df2['Age_Mons'] < 13, drop_first=True)
male = pd.get_dummies(df2['Sex'], drop_first=True)
ethnics = pd.get_dummies(df2['Ethnicity'], drop_first=True)
jaundice = pd.get_dummies(df2['Jaundice'], drop_first=True)
ASD_genes = pd.get_dummies(df2['Family_mem_with_ASD'], drop_first=True)
ASD_traits = pd.get_dummies(df2['Class/ASD Traits '], drop_first=True)

final_data = pd.concat([within0_12, within24_36, male, ethnics, jaundice, ASD_genes, ASD_traits], axis=1)
final_data.columns = ['within0_12', 'within24_36', 'male', 'Latino', 'Native Indian', 'Others', 'Pacifica', 'White European', 'asian', 'black', 'middle eastern', 'mixed', 'south asian', 'jaundice', 'ASD_genes', 'ASD_traits']

X = final_data.iloc[:, :-1]
y = final_data.iloc[:, -1]

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# 训练逻辑回归模型
logmodel = LogisticRegression(C=0.01)
logmodel.fit(X_scaled, y)
logmodel_accuracy = round(logmodel.score(X_scaled, y), 2)

# 训练随机森林分类器
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_scaled, y)
rfc_accuracy = round(rfc.score(X_scaled, y), 2)

# 训练K近邻分类器
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_scaled, y)
knn_accuracy = round(knn.score(X_scaled, y), 2)

# 训练支持向量机（SVM）
svm_model = SVC(C=1, gamma=0.1)
svm_model.fit(X_scaled, y)
svm_accuracy = round(svm_model.score(X_scaled, y), 2)

# 训练决策树分类器
dtc = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
dtc.fit(X_scaled, y)
dtc_accuracy = round(dtc.score(X_scaled, y), 2)

def predict_asd(model, input_data):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return "有自闭症特征" if prediction[0] == 1 else "没有自闭症特征"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = {
            'within0_12': int(request.form['age']) < 13,
            'within24_36': int(request.form['age']) > 24,
            'male': 1 if request.form['gender'] == '男' else 0,
            'Latino': 1 if request.form['ethnicity'] == 'Latino' else 0,
            'Native Indian': 1 if request.form['ethnicity'] == 'Native Indian' else 0,
            'Others': 1 if request.form['ethnicity'] == 'Others' else 0,
            'Pacifica': 1 if request.form['ethnicity'] == 'Pacifica' else 0,
            'White European': 1 if request.form['ethnicity'] == 'White European' else 0,
            'asian': 1 if request.form['ethnicity'] == 'asian' else 0,
            'black': 1 if request.form['ethnicity'] == 'black' else 0,
            'middle eastern': 1 if request.form['ethnicity'] == 'middle eastern' else 0,
            'mixed': 1 if request.form['ethnicity'] == 'mixed' else 0,
            'south asian': 1 if request.form['ethnicity'] == 'south asian' else 0,
            'jaundice': 1 if request.form['jaundice'] == '是' else 0,
            'ASD_genes': 1 if request.form['family_autism'] == '是' else 0,
        }
        logmodel_pred = predict_asd(logmodel, input_data)
        rfc_pred = predict_asd(rfc, input_data)
        knn_pred = predict_asd(knn, input_data)
        svm_pred = predict_asd(svm_model, input_data)
        dtc_pred = predict_asd(dtc, input_data)
        return render_template('index.html',
                               logmodel_pred=logmodel_pred, logmodel_accuracy=logmodel_accuracy,
                               rfc_pred=rfc_pred, rfc_accuracy=rfc_accuracy,
                               knn_pred=knn_pred, knn_accuracy=knn_accuracy,
                               svm_pred=svm_pred, svm_accuracy=svm_accuracy,
                               dtc_pred=dtc_pred, dtc_accuracy=dtc_accuracy)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
