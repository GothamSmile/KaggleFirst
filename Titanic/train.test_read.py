import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # 绘图函数
import seaborn as sns  # 基于matplotlib的可视化库
import warnings

warnings.filterwarnings('ignore')
'''
1.导入数据
'''
train = pd.read_csv(r'E:\python数据\Titanic\train.csv')
test = pd.read_csv(r'E:\python数据\Titanic\test.csv')
PassengerId = test['PassengerId']
all_data = pd.concat([train, test], ignore_index=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(train.head(10))
train.info()
# PassengerID（ID）
# •Survived(存活与否)
# •Pclass（客舱等级，较为重要）
# •Name（姓名，可提取出更多信息）
# •Sex（性别，较为重要）
# •Age（年龄，较为重要）
# •Parch（直系亲友）
# •SibSp（旁系）
# •Ticket（票编号）
# •Fare（票价）
# •Cabin（客舱编号）
# •Embarked（上船的港口编号）
'''
2.数据初步分析
'''

print(train['Survived'].value_counts())
# 女性的生存概率比男性大很多
sns.barplot(x="Sex", y="Survived", data=train)
# 乘客社会等级越高，幸存率越高
sns.barplot(x="Pclass", y="Survived", data=train)
# 配偶及兄弟姐妹数适中的乘客幸存率更高
sns.barplot(x="SibSp", y="Survived", data=train)
# 父母与子女数适中的乘客幸存率更高
sns.barplot(x="Parch", y="Survived", data=train)
'''
从不同生还情况的密度图可以看出，在年龄15岁的左侧，生还率有明显差别，
密度图非交叉区域面积非常大，但在其他年龄段，则差别不是很明显，认为是随机所致，
因此可以考虑将此年龄偏小的区域分离出来
'''
facet = sns.FacetGrid(train, hue="Survived", aspect=2)
# data:数据  row:行 col:列 hue:颜色语义
'''
class seaborn.FacetGrid(data, row=None, col=None, hue=None, col_wrap=None,
 sharex=True, sharey=True, height=3, aspect=1, palette=None, row_order=None, 
col_order=None, hue_order=None, hue_kws=None, dropna=True, legend_out=True, despine=True, 
margin_titles=False, xlim=None, ylim=None, subplot_kws=None, gridspec_kws=None, size=None)

'''
facet.map(sns.kdeplot, 'Age', shade=True)  # kdeplot(核密度估计图)，一个或多个序列，阴影填满
facet.set(xlim=(0, train['Age'].max()))  # 0到最大年龄
facet.add_legend()  # 增加标签
plt.xlabel('Age')  # 设置x轴为年龄
plt.ylabel('density')  # 设置y轴为密度

# Embarked登港港口与生存情况的分析
# 结果分析:C地的生存率更高,这个也应该保留为模型特征.
sns.countplot('Embarked', hue='Survived', data=train)

# Title Feature(New)：不同称呼的乘客幸存率不同
# 新增Title特征，从姓名中提取乘客的称呼，归纳为六类。
all_data['Title'] = all_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))

all_data['Title'] = all_data['Title'].map(Title_Dict)
sns.barplot(x="Title", y="Survived", data=all_data)

# FamilyLabel Feature(New)：家庭人数为2到4的乘客幸存率较高
# 新增FamilyLabel特征，先计算FamilySize=Parch+SibSp+1，然后把FamilySize分为三类
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
sns.barplot(x="FamilySize", y="Survived", data=all_data)


# 按生存率把FamilySize分为三类，构成FamilyLabel特征。

def Fam_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif s > 7:
        return 0


all_data['FamilyLabel'] = all_data['FamilySize'].apply(Fam_label)
sns.barplot(x="FamilyLabel", y="Survived", data=all_data)

# Deck Feature(New)：不同甲板的乘客幸存率不同
# 新增Deck特征，先把Cabin空缺值填充为'Unknown'，再提取Cabin中的首字母构成乘客的甲板号。

all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
all_data['Deck'] = all_data['Cabin'].str.get(0)
sns.barplot(x="Deck", y="Survived", data=all_data)

# TicketGroup Feature(New)：与2至4人共票号的乘客幸存率较高
# 新增TicketGroup特征，统计每个乘客的共票号数。
Ticket_Count = dict(all_data['Ticket'].value_counts())
all_data['TicketGroup'] = all_data['Ticket'].apply(lambda x: Ticket_Count[x])
sns.barplot(x='TicketGroup', y='Survived', data=all_data)


# 按生存率把TicketGroup分为三类。
def Ticket_Label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0


all_data['TicketGroup'] = all_data['TicketGroup'].apply(Ticket_Label)
sns.barplot(x='TicketGroup', y='Survived', data=all_data)

'''
3.进入数据清洗阶段
'''

# 缺失值填充
# Age Feature：Age缺失量为263，缺失量较大，用Sex, Title, Pclass三个特征构建随机森林模型，填充年龄缺失值
from sklearn.ensemble import RandomForestRegressor  # 导入随机森林模型（回归）
import numpy.ndarray
age_df = all_data[['Age', 'Pclass', 'Sex']]
age_df = pd.get_dummies(age_df)
known_age = age_df[age_df.Age.notnull()].values
unknown_age = age_df[age_df.Age.isnull()].values
y = known_age[:, 0]
X = known_age[:, 1:]
rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
rfr.fit(X, y)
predictedAges = rfr.predict(unknown_age[:, 1::])
all_data.loc[(all_data.Age.isnull()), 'Age'] = predictedAges

# Embarked Feature：Embarked缺失量为2，缺失Embarked信息的乘客的Pclass均为1，且Fare均为80，
all_data[all_data['Embarked'].isnull()]
# 因为Embarked为C且Pclass为1的乘客的Fare中位数为80，所以缺失值填充为C。
all_data.groupby(by=["Pclass", "Embarked"]).Fare.median()
all_data['Embarked'] = all_data['Embarked'].fillna('C')

# Fare缺失量为1，缺失Fare信息的乘客的Embarked为S，Pclass为3，
all_data[all_data['Fare'].isnull()]
# 所以用Embarked为S，Pclass为3的乘客的Fare中位数填充。
fare = all_data[(all_data['Embarked'] == "S") & (all_data['Pclass'] == 3)].Fare.median()
all_data['Fare'] = all_data['Fare'].fillna(fare)

# 同组识别(看不懂)
# 把姓氏相同的乘客划分为同一组，从人数大于一的组中分别提取出每组的妇女儿童和成年男性。
all_data['Surname'] = all_data['Name'].apply(lambda x: x.split(',')[0].strip())
Surname_Count = dict(all_data['Surname'].value_counts())
all_data['FamilyGroup'] = all_data['Surname'].apply(lambda x: Surname_Count[x])
Female_Child_Group = all_data.loc[
    (all_data['FamilyGroup'] >= 2) & ((all_data['Age'] <= 12) | (all_data['Sex'] == 'female'))]
Male_Adult_Group = all_data.loc[(all_data['FamilyGroup'] >= 2) & (all_data['Age'] > 12) & (all_data['Sex'] == 'male')]

# 发现绝大部分女性和儿童组的平均存活率都为1或0，即同组的女性和儿童要么全部幸存，要么全部遇难。
Female_Child = pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
Female_Child.columns = ['GroupCount']
Female_Child
sns.barplot(x=Female_Child.index, y=Female_Child["GroupCount"]).set_xlabel('AverageSurvived')

# 绝大部分成年男性组的平均存活率也为1或0。
Male_Adult = pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())
Male_Adult.columns = ['GroupCount']
Male_Adult
sns.barplot(x=Male_Adult.index, y=Male_Adult["GroupCount"]).set_xlabel('AverageSurvived')

# 因为普遍规律是女性和儿童幸存率高，成年男性幸存较低，所以我们把不符合普遍规律的反常组选出来单独处理。
# 把女性和儿童组中幸存率为0的组设置为遇难组，把成年男性组中存活率为1的设置为幸存组，
# 推测处于遇难组的女性和儿童幸存的可能性较低，处于幸存组的成年男性幸存的可能性较高。
Female_Child_Group = Female_Child_Group.groupby('Surname')['Survived'].mean()
Dead_List = set(Female_Child_Group[Female_Child_Group.apply(lambda x: x == 0)].index)
print(Dead_List)
Male_Adult_List = Male_Adult_Group.groupby('Surname')['Survived'].mean()
Survived_List = set(Male_Adult_List[Male_Adult_List.apply(lambda x: x == 1)].index)
print(Survived_List)

# 为了使处于这两种反常组中的样本能够被正确分类，对测试集中处于反常组中的样本的Age，Title，Sex进行惩罚修改。
train = all_data.loc[all_data['Survived'].notnull()]
test = all_data.loc[all_data['Survived'].isnull()]
test.loc[(test['Surname'].apply(lambda x: x in Dead_List)), 'Sex'] = 'male'
test.loc[(test['Surname'].apply(lambda x: x in Dead_List)), 'Age'] = 60
test.loc[(test['Surname'].apply(lambda x: x in Dead_List)), 'Title'] = 'Mr'
test.loc[(test['Surname'].apply(lambda x: x in Survived_List)), 'Sex'] = 'female'
test.loc[(test['Surname'].apply(lambda x: x in Survived_List)), 'Age'] = 5
test.loc[(test['Surname'].apply(lambda x: x in Survived_List)), 'Title'] = 'Miss'

# 3)特征转换
# 选取特征，转换为数值变量，划分训练集和测试集
all_data = pd.concat([train, test])
all_data = all_data[
    ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title',  'TicketGroup']]
all_data = pd.get_dummies(all_data)
train = all_data[all_data['Survived'].notnull()]
test = all_data[all_data['Survived'].isnull()].drop('Survived', axis=1)
X = train.values[:, 1:]
y = train.values[:, 0]#ar...maxt变成了values

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest

pipe = Pipeline([('select', SelectKBest(k=10)),
                 ('classify', RandomForestClassifier(random_state=10, max_features='sqrt'))])

param_test = {'classify__n_estimators': list(range(20, 50, 2)),
              'classify__max_depth': list(range(3, 60, 3))}
gsearch = GridSearchCV(estimator=pipe, param_grid=param_test, scoring='roc_auc', cv=10)
gsearch.fit(X, y)
print(gsearch.best_params_, gsearch.best_score_)



from sklearn.pipeline import make_pipeline

select = SelectKBest(k=11)
clf = RandomForestClassifier(random_state=10, warm_start=True,
                             n_estimators=26,
                             max_depth=6,
                             max_features='sqrt')
pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)



from sklearn import model_selection, metrics    #cross_validation模块弃用，所有的包和方法都在model_selection中,包和方法名没有发生变化
cv_score = model_selection.cross_val_score(pipeline, X, y, cv= 10)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))

predictions = pipeline.predict(test)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
submission.to_csv(r"E:\python数据\submission1.csv", index=False)