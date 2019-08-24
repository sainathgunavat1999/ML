import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.preprocessing import LabelEncoder

columnNames = ['Size', 'Color', 'Shape', 'label']
var = pd.read_csv("SizeShape1.csv", header=None, names=columnNames)
#print(pandavar)
var.head()
featureColumn=['Size', 'Color', 'Shape']
X=var[featureColumn]
y=var.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
clf = DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = featureColumn,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree.png')
Image(graph.create_png())
