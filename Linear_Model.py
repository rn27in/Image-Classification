from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

X_vgg16 = joblib.load('../X_vgg16_3000')
X_vgg19 = joblib.load('../X_vgg19_3000')
X_resnet = joblib.load('../X_resnet_3000')
X_inception = joblib.load('../X_inception_3000')
X_xception = joblib.load('../X_xception_1500')

final = np.hstack((X_vgg16,X_vgg19,X_resnet,X_inception,X_xception))

################Building the model####################
clf = LogisticRegression(penalty = "l2", multi_class= 'ovr', C = 1, class_weight= 'balanced')
start = time.time()
model = clf.fit(final[:2251,], np.array(label_train))
end = time.time()

pred_test = clf.predict(final[2251:3215,])#######Pred on val set#######

#######Accuracy################################
sum(pred_train == np.array(label_train))/(1.0* len(pred_train))
np.sum(pred_test == np.array(label_test))/(1.0* len(pred_test))


pred_subs = clf.predict(final[3215:,])  ###72.06%# ########## Final result#####

######################################End#################################################
