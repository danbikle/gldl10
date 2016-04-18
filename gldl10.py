# gldl10.py

# This script should predict iris data.
# ref:
# https://dato.com/products/create/docs/graphlab.toolkits.deeplearning.html

from __future__ import division

import graphlab as gl
import pdb

# In graphlab.SFrame how to read csv which has no header?
iris_sf = gl.SFrame.read_csv('iris.csv',header=False)
# In graphlab.SFrame how to name a column?
iris_sf.rename({'X1':'x1','X2':'x2','X3':'x3','X4':'x4','X5':'label'})
print(iris_sf.head())
train_sf = iris_sf[3:-3]
# In graphlab.SFrame how to write to CSV file?
train_sf.save('train.csv')
train_sf = gl.SFrame.read_csv('train.csv')
print(train_sf.head())

# In python, how to initialize a list with default values?
mylist = [False] * len(iris_sf)
# Perhaps above syntax is related to idea of broadcasting?

# I should use mylist to help pick first 3 and last 3 rows from iris_sf
mylist[:3]  = [True,True,True]
mylist[-3:] = [True,True,True]
mylist

# In graphlab.SFrame how to slice with list of Booleans?
my_sa   = gl.SArray(mylist)
test_sf = iris_sf[(my_sa)]
print(test_sf)
test_sf.save('test.csv')

# I should follow idea I see here:
# https://dato.com/products/create/docs/graphlab.toolkits.deeplearning.html
train2_sf, validation_sf = train_sf.random_split(0.8)
net = gl.deeplearning.create(train2_sf, target='label')
net.layers

m = gl.neuralnet_classifier.create(train2_sf, target='label',
  network = net,
  validation_set=validation_sf,
  metric=['accuracy', 'recall@2'],
  max_iterations=4096)

pred = m.classify(test_sf)
pred

pred_top2 = m.predict_topk(test_sf, k=2)
pred_top2

myeval = m.evaluate(test_sf)
print(myeval)

'bye'
