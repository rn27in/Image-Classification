from sklearn.decomposition import PCA
from sklearn.externals import joblib
import numpy as np

path_vgg16 = "../VGG16/"
path_vgg19 = "../VGG19/"
path_resnet = "../Resnet50/"
path_inception = "../Inception/"
path_xception = "../Xception/"

list_pretrained = ['vgg16', 'VGG19', 'resnet','Inception', 'Xception']
path = [path_vgg16, path_vgg19,path_resnet, path_inception, path_xception]

###################For each model, we load all 3 datasets(train,test,val) one at a time#####################
list_of_features = []
for element in range(len(list_pretrained)):
    lst_temp = []
    array_path = path[element] + list_pretrained[element]
    temp_array1 = joblib.load(path[element]+ 'train_features_'+ list_pretrained[element])
    temp_array2 = joblib.load(path[element] + 'test_features_' + list_pretrained[element])
    temp_array3 = joblib.load(path[element] + 'subs_features_' + list_pretrained[element])

    lst_temp.append((temp_array1,temp_array2, temp_array3))
    list_of_features.append(lst_temp)

joblib.dump(list_of_features, '../list_of_features')

list_of_features = joblib.load('../list_of_features')


##########################Run a PCA on each set of features extracted by the Pre-trained model####################################
start = time.time()
feat_1 = np.vstack((list_of_features[0][0][0],list_of_features[0][0][1],list_of_features[0][0][2]))
feat_1 = np.reshape(feat_1, (4947, (7*7*512)))
pca = PCA(n_components= 3000,random_state= 42)
pca.fit(feat_1)
X_vgg16 = pca.transform(feat_1)
end = time.time()
joblib.dump(X_vgg16, "../X_vgg16_3000")



##############################################################
start = time.time()
feat_2 = np.vstack((list_of_features[1][0][0],list_of_features[1][0][1],list_of_features[1][0][2]))
feat_2 = np.reshape(feat_2, (4947, (7*7*512)))
pca = PCA(n_components= 3000,random_state= 42)
pca.fit(feat_2)
X_vgg19 = pca.transform(feat_2)

joblib.dump(X_vgg19, "../X_vgg19_3000")
end = time.time()


##########################################################
start = time.time()
feat_3 = np.vstack((list_of_features[2][0][0],list_of_features[2][0][1],list_of_features[2][0][2]))
feat_3 = np.reshape(feat_3, (4947, (1*1*2048)))
pca = PCA(n_components= 1500,random_state= 42)
pca.fit(feat_3)
X_resnet = pca.transform(feat_3)

joblib.dump(X_resnet, "../X_resnet_3000")
end = time.time()



###################################################

feat_4 = np.vstack((list_of_features[3][0][0],list_of_features[3][0][1],list_of_features[3][0][2]))
feat_4 = np.reshape(feat_4, (4947, (5*5*2048)))
pca = PCA(n_components= 3000,random_state= 42)
pca.fit(feat_4)
X_inception = pca.transform(feat_4)

joblib.dump(X_inception, "../X_inception_3000")

############################################

feat_5 = np.vstack((list_of_features[4][0][0],list_of_features[4][0][1],list_of_features[4][0][2]))
feat_5 = np.reshape(feat_5, (4947, (7*7*2048)))
pca = PCA(n_components= 1500,random_state= 42)
pca.fit(feat_5)
X_xception = pca.transform(feat_5)

joblib.dump(X_xception, "../X_xception_1500")
##################################################
