

fileName1 = 'transferData/alexNet2DTrainTest_current.npy'
fileName2 = 'transferData/alexNet2DValidation_current.npy'
alexNetTrainTest = np.load(fileName1)
alexNetValid = np.load(fileName2)
fileName = 'transferData/AlexNet2Ddata.mat'
sio.savemat(fileName,mdict={'alexNetTrainTest':alexNetTrainTest,'alexNetValid':alexNetValid})