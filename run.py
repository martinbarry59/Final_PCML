from proj1_helpers import *
import implementations as im

DATA_TRAIN_PATH = r'../train.csv' 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# Ridge Regression
def ridge_regression_demo(y, tx,lamb,degree):
    # define parameter
    tX = im.build_poly(tx, degree)
    weight, loss =  im.ridge_regression(y , tX , lamb)
    
    print("Training RMSE={tr:.3f}".format(tr=loss))
    return weight ,loss

#Split the training set according to the jet number
index0 = tX[:,22] == 0
tX0 = tX[index0,:]
tX0 = np.delete(tX0,22,1)
index1 = tX[:,22] == 1
tX1 = tX[index1,:]
tX1 = np.delete(tX1,22,1)
index2 = tX[:,22] == 2
tX2 = tX[index2,:]
tX2 = np.delete(tX2,22,1)
index3 = tX[:,22] == 3
tX3 = tX[index3,:]
tX3 = np.delete(tX3,22,1)

#Formating the train data as explained in the pdf report
tX0_final, index_final_0 = im.formating(tX0) 
tX1_final, index_final_1 = im.formating(tX1) 
tX2_final, index_final_2 = im.formating(tX2) 
tX3_final, index_final_3 = im.formating(tX3) 

lambdas = np.logspace(-10,1,10)
degrees = [5,6,7,8,9,10,11,12,13,14]


# It's just to long to be efficient so only run one time, then parameters are put by hands
"""
losses0 = grid_search(y[index0],tX0_final,lambdas,degrees)
value0,lambda_0, degree_0 = get_best_parameters(lambdas,degrees,losses0)
print(value0,lambda_0,degree_0)

losses1 = grid_search(y[index1],tX1_final,lambdas,degrees)
value1,lambda_1, degree_1 = get_best_parameters(lambdas,degrees,losses1)
print(value1,lambda_1,degree_1)

losses2 = grid_search(y[index2],tX2_final,lambdas,degrees)
value2,lambda_2, degree_2= get_best_parameters(lambdas,degrees,losses2)
print(value2,lambda_2,degree_2)

losses3 = grid_search(y[index3],tX3_final,lambdas,degrees)
value3,lambda_3, degree_3 = get_best_parameters(lambdas,degrees,losses3)
print(value3,lambda_3,degree_3)
"""

lamb0 =1.6681005372e-09
degree0 = 6
weight0 , loss0 = ridge_regression_demo(y[index0], tX0_final,lamb0,degree0)
lamb1 = 7.74263682681e-06
degree1 = 10
weight1 , loss1 = ridge_regression_demo(y[index1], tX1_final,lamb1,degree1)
lamb2 = 2.78255940221e-08
degree2 = 11
weight2 , loss2 = ridge_regression_demo(y[index2], tX2_final,lamb2,degree2)
lamb3 = 4.64158883361e-07
degree3 = 11
weight3 , loss3  = ridge_regression_demo(y[index3], tX3_final,lamb3,degree3)


#Load test data
DATA_TEST_PATH = '../test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

#Preprocessing the test data the same way than the train data
index0 = tX_test[:,22] == 0
tX0 = tX_test[index0,:]
tX0 = np.delete(tX0,22,1)
index1 = tX_test[:,22] == 1
tX1 = tX_test[index1,:]
tX1 = np.delete(tX1,22,1)
index2 = tX_test[:,22] == 2
tX2 = tX_test[index2,:]
tX2 = np.delete(tX2,22,1)
index3 = tX_test[:,22] == 3
tX3 = tX_test[index3,:]
tX3 = np.delete(tX3,22,1)

tX0_final, index_final_0 = im.formating(tX0) 
tX1_final, index_final_1 = im.formating(tX1) 
tX2_final, index_final_2 = im.formating(tX2) 
tX3_final, index_final_3 = im.formating(tX3) 

#Building the polynomial basis for the test data and predicting results
tX0_final_test =  im.build_poly(tX0_final, degree0)
ypred0 = predict_labels(weight0, tX0_final_test)
tX1_final_test =  im.build_poly(tX1_final, degree1)
ypred1 = predict_labels(weight1, tX1_final_test)
tX2_final_test =  im.build_poly(tX2_final, degree2)
ypred2 = predict_labels(weight2, tX2_final_test)
tX3_final_test =  im.build_poly(tX3_final, degree3)
ypred3 = predict_labels(weight3, tX3_final_test)

#Assembling the predicted y
y_pred = np.zeros((tX_test.shape[0]))
y_pred[index0]=ypred0.reshape(ypred0.shape[0],1)
y_pred[index1]=ypred1.reshape(ypred1.shape[0],1)
y_pred[index2]=ypred2.reshape(ypred2.shape[0],1)
y_pred[index3]=ypred3.reshape(ypred3.shape[0],1)

#Generating the csv
OUTPUT_PATH = 'results.csv' 
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)