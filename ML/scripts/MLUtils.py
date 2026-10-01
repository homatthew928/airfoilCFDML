import torch
import gpytorch
from scipy.stats.qmc import Sobol
import numpy as np
from itertools import product
from sklearn.cluster import DBSCAN
import sys

def trainTestData (X, yCl, yCd, folds, depVar, foldI, XMin, XMax):
    '''
    Generates the training and testing data sets

    Parameters:
        X: unnormalised independent variables data
        yCl: Cl data
        yCd: Cd data
        folds: fold split information
        depVar: dependent variable
        foldI: fold index
        XMin: lower bounds of independent variables
        XMax: upper bounds of independent variables

    Returns:
        XTrain: normalised independent variables data for training
        yTrain: dependent variable data for training
        XSTest: normalised independent variables data in symmetric domain for testing
        ySTest: dependent variable data in symmetric domain for testing
        XCTest: normalised independent variables data in cambered domain for testing
        yCTest: dependent variable data in cambered domain for testing
    '''
    if depVar == 'Cd':
        y = yCd
    elif depVar == 'Cl':
        y = yCl
    else:
        print('Invalid dependent variable')
        return

    trainIdx, testIdx = folds[foldI]

    XTrain = (torch.tensor(X[trainIdx], dtype = torch.float64) - XMin)/(XMax - XMin)
    yTrain = torch.tensor(y[trainIdx], dtype = torch.float64)

    XTest = X[testIdx]
    yTest = y[testIdx]

    # symmetric tests
    XSTest = (torch.tensor(XTest[XTest[:, 0] == 0], dtype = torch.float64) - XMin)/(XMax - XMin)
    ySTest = torch.tensor(yTest[XTest[:, 0] == 0], dtype = torch.float64)

    # cambered tests
    XCTest = (torch.tensor(XTest[XTest[:, 0] > 0], dtype = torch.float64) - XMin)/(XMax - XMin)
    yCTest = torch.tensor(yTest[XTest[:, 0] > 0], dtype = torch.float64)

    return XTrain, yTrain, XSTest, ySTest, XCTest, yCTest

class exactGPModel(gpytorch.models.ExactGP):
    def __init__(self, XTrain, yTrain, likelihood, kernel, ARD):
        super().__init__(XTrain, yTrain, likelihood)
        self.meanModule = gpytorch.means.ConstantMean()

        if kernel == "rbf":
            baseKernel = gpytorch.kernels.RBFKernel(ard_num_dims = ARD)
        elif kernel == "matern32":
            baseKernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims = ARD)
        elif kernel == "matern52":
            baseKernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims = ARD)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        self.covarModule = gpytorch.kernels.ScaleKernel(baseKernel)

    def forward(self, x):
        xMean = self.meanModule(x)
        xCovar = self.covarModule(x)
        return gpytorch.distributions.MultivariateNormal(xMean, xCovar)

def GPR(kernel, ARD, noiseFloor,
        XTrain, yTrain, XSTest=None, ySTest=None, XCTest=None, yCTest=None,
        returnModel=False, nIterMax=500, tol=1e-6, patience=10):
    '''
    Train and test a model by GPR

    Parameters:
        kernel: kernel choice
        ARD: ARD choice
        noiseFloor: minimum noise of training points
        XTrain: independent variables data for training
        yTrain: dependent variable data for training
        XSTest: independent variables data in symmetric domain for testing
        ySTest: dependent variable data in symmetric domain for testing
        XCTest: independent variables data in cambered domain for testing
        yCTest: dependent variable data in cambered domain for testing
        returnModel: return trained model without test scores evaluation if true
        nIterMax: maximum iteration to minimise Marginal Log-Likelihood
        tol: accpetable tolerance when minimising Marginal Log-Likelihood
        patience: number of consecutive iterations under tolerance required

    Returns:
        model: trained model state
        likelihood: trained likelihood state
        learnedNoise: optimised noise for trained model
        trainRMSE.numpy(): RMSE of training points
        testSRMSE.numpy(): RMSE of symmetric testing points
        testCRMSE.numpy(): RMSE of cambered testing points
        trainMAE.numpy(): MAE of training points
        testSMAE.numpy(): MAE of symmetric testing points
        testCMAE.numpy(): MAE of cambered testing points
        trainNLPD.numpy(): NLPD of training points
        testSNLPD.numpy(): NLPD of symmetric testing points
        testCNLPD.numpy(): NLPD of cambered testing points
    '''
    # training model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint = gpytorch.constraints.GreaterThan(noiseFloor))
    model = exactGPModel(XTrain, yTrain, likelihood, kernel, ARD)

    model = model.double()
    likelihood = likelihood.double()

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    prevLoss = None
    stallCount = 0

    for i in range(nIterMax):  # minimising Marginal Loss Likelihood
        optimizer.zero_grad()
        output = model(XTrain)
        loss = -mll(output, yTrain)
        loss.backward()
        optimizer.step()

        currentLoss = loss.item()
        if prevLoss is not None and abs(prevLoss - currentLoss) < tol:
            stallCount += 1
            if stallCount >= patience:
                break
        else:
            stallCount = 0
        prevLoss = currentLoss

    # evaluation mode using trained model
    model.eval()
    likelihood.eval()

    # return model information if returnModel is true
    learnedNoise = likelihood.noise.item()

    if returnModel:
        return model, likelihood, learnedNoise

    # calcualte traing and test scores with predictions
    with torch.no_grad():
        trainPred = likelihood(model(XTrain))
        testSPred  = likelihood(model(XSTest))
        testCPred = likelihood(model(XCTest))

    trainMean, trainVar = trainPred.mean, trainPred.variance
    testSMean,  testSVar  = testSPred.mean,  testSPred.variance
    testCMean,  testCVar  = testCPred.mean,  testCPred.variance

    def RMSE(predMean, yTrue):
        return torch.sqrt(torch.mean((predMean - yTrue) ** 2))

    def MAE(predMean, yTrue):
        return torch.mean(torch.abs(predMean - yTrue))

    def NLPD(predMean, predVar, yTrue):
        return torch.mean(0.5 * torch.log(2 * torch.pi * predVar) + 0.5 * ((yTrue - predMean) ** 2) / predVar)

    trainRMSE = RMSE(trainMean, yTrain)
    trainMAE  = MAE(trainMean, yTrain)
    testSRMSE  = RMSE(testSMean, ySTest)

    testSMAE   = MAE(testSMean, ySTest)
    testCRMSE  = RMSE(testCMean, yCTest)
    testCMAE   = MAE(testCMean, yCTest)

    trainNLPD = NLPD(trainMean, trainVar, yTrain)
    testSNLPD  = NLPD(testSMean, testSVar, ySTest)
    testCNLPD  = NLPD(testCMean, testCVar, yCTest)

    return trainRMSE.numpy(), testSRMSE.numpy(), testCRMSE.numpy(), trainMAE.numpy(), testSMAE.numpy(), testCMAE.numpy(), trainNLPD.numpy(), testSNLPD.numpy(), testCNLPD.numpy(), learnedNoise

def loadModel(depVar, modelsPath):
    '''
    Load trained model

    Parameters:
        depVar: dependent variable to determine which model to load
        modelsPath: path to stored models

    Returns:
        model: trained model
        likelihood: trained liklihood
        trainedModelDict: dictionary containing information on trained model
    '''
    # extract information on trained model
    trainedModelDict = torch.load(modelsPath / f'{depVar}GPRFinal.pth', weights_only = False)

    # input data used to train model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(trainedModelDict['noiseConstraintFloor'])
    )
    model = exactGPModel(trainedModelDict['XTrain'], trainedModelDict['yTrain'], likelihood, trainedModelDict['kernel'], trainedModelDict['ARD'])

    # use trained model/likelihood state to reconstruct model
    model.load_state_dict(trainedModelDict['modelStateDict'])
    likelihood.load_state_dict(trainedModelDict['likelihoodStateDict'])

    model = model.double()
    likelihood = likelihood.double()

    # swich to evaluation mode
    model.eval()
    likelihood.eval()

    return model, likelihood, trainedModelDict

def predict(M, P, T, AoA,
            CdModel, CdLikelihood, ClModel, ClLikelihood,
            XMin, XMax, requiresGrad = False, fastPredVar = False):
    '''
    Predict Cd and Cl with variances and L/D

    Parameters:
        M: M values
        P: P values
        T: T Values
        AoA: AoA values
        CdModel: trained model for Cd
        CdLikelihood: trained likelihood for Cd
        ClModel: trained model for Cl
        ClLikelihood: trained likelihood for Cl
        XMin: lower bound of independent variables
        XMax: upper bound of independent variables
        requiresGrad: track gradients if true
        fastPredVar: speeds up prediction if true

    Returns:
        CdMean: mean of Cd
        CdVar: variance of Cd
        ClMean: mean of Cl
        ClVar: variance of Cl
        LodMean: mean of L/D
        X: unnormalised independent variables data for gradient calculations
    '''
    # prepare indepedent variables data for input
    M = torch.atleast_1d(torch.tensor(M, dtype = torch.float64))
    P = torch.atleast_1d(torch.tensor(P, dtype = torch.float64))
    T = torch.atleast_1d(torch.tensor(T, dtype = torch.float64))
    AoA = torch.atleast_1d(torch.tensor(AoA, dtype = torch.float64))

    isSymmetric = (M==0).double()
    P = torch.where(M==0, torch.tensor(4, dtype = torch.float64), P)

    X = torch.stack([M, P, T, AoA, isSymmetric], dim=1)

    # turns on gradient tracking if requiresGrad is true
    if requiresGrad:
        X.requires_grad_(True)

    # normalise inputs
    XNorm = (X-XMin)/(XMax-XMin)

    # enable gradient tracking if requiresGrad is true
    enableGrad = torch.enable_grad() if requiresGrad else torch.no_grad()

    # prediction
    with enableGrad, gpytorch.settings.fast_pred_var(fastPredVar):
        CdPred = CdLikelihood(CdModel(XNorm))
        ClPred = ClLikelihood(ClModel(XNorm))

    CdMean, CdVar = CdPred.mean, CdPred.variance
    ClMean, ClVar = ClPred.mean, ClPred.variance

    LoDMean = ClMean/CdMean

    return CdMean, CdVar, ClMean, ClVar, LoDMean, X

def predictBatched(M, P, T, AoA, CdModel, CdLikelihood, ClModel, ClLikelihood, XMin, XMax, batchSize=2000):
    '''
    Prediction in batches to avoid crashing (no gradient tracking, with fast predictions)

    Parameters:
        M: M values
        P: P values
        T: T Values
        AoA: AoA values
        CdModel: trained model for Cd
        CdLikelihood: trained likelihood for Cd
        ClModel: trained model for Cl
        ClLikelihood: trained likelihood for Cl
        XMin: lower bound of independent variables
        XMax: upper bound of independent variables
        batchSize: number of points for each prediction

    Returns:
        torch.cat(CdMeanAll): means of Cd
        torch.cat(CdVarAll): variances of Cd
        torch.cat(ClMeanAll): means of Cl
        torch.cat(ClVarAll): variances of Cl
        torch.cat(LoDMeanAll): means of L/D
    '''
    n = len(M)
    CdMeanAll, CdVarAll, ClMeanAll, ClVarAll, LoDMeanAll = [], [], [], [], []

    for start in range(0, n, batchSize):
        end = min(start + batchSize, n)

        CdMean, CdVar, ClMean, ClVar, LoDMean, _ = predict(
            M[start:end], P[start:end], T[start:end], AoA[start:end],
            CdModel, CdLikelihood, ClModel, ClLikelihood,
            XMin, XMax, fastPredVar = True
        )

        CdMeanAll.append(CdMean)
        CdVarAll.append(CdVar)
        ClMeanAll.append(ClMean)
        ClVarAll.append(ClVar)
        LoDMeanAll.append(LoDMean)

    return torch.cat(CdMeanAll), torch.cat(CdVarAll), torch.cat(ClMeanAll), torch.cat(ClVarAll), torch.cat(LoDMeanAll)

def denseSampling(XMin, XMax, spacing = 0.1):
    '''
    Structured sampling of design space

    Parameters:
        XMin: lower bound of independent variables
        XMax: upper bound of independent variables
        spacing: node-to-node spacing

    Returns:
        candidates: sampled nodes
        spacing: selected spacing
    '''
    # convert to numpy
    XMin = XMin.numpy()
    XMax = XMax.numpy()

    # cambered domain
    MNodes = np.arange(spacing, XMax[0]+spacing, spacing)
    PNodes = np.arange(XMin[1], XMax[1]+spacing, spacing)
    TNodes = np.arange(XMin[2], XMax[2]+spacing, spacing)

    # stack cambered candidates and symmetric candidates
    candidates = np.vstack((
        np.column_stack((np.zeros(len(TNodes)), np.zeros(len(TNodes)), TNodes)),
        np.array(list(product(MNodes, PNodes, TNodes)))
    ))

    return candidates, spacing


def filterFeasible (candidates,
                    maxCd, minCl, minAoA, maxAoA,
                    CdModel, CdLikelihood, ClModel, ClLikelihood, XMin, XMax, k = 2, fullRange = True):
    '''
    Filter feasible candidates
        - operates under the assumption of monoticity for Cd and Cl in AoA range
        - if fullRange = True, CdUCB < maxCd at maxAoA and ClLCB > minCl at minAoA
            - CdUCB must be < maxCd and ClLCB must be > minCl at accpeted MPT across the AoA range assuming monoticity
        - if fullRange = False, CdUCB < maxCd at minAoA and ClLCB > minCl at maxAoA
            - CdUCB < maxCd and ClLCB > minCl must be satisfied by accepted MPT at some point in AoA range assuming monoticity

    Parameters:
        candidates: nodes to test
        maxCd: maximum Cd accepted
        minCl: minimum Cl accepted
        minAoA: minimum angle of attack
        maxAoA: maximum angle of attack
        CdModel: trained model for Cd
        CdLikelihood: trained likelihood for Cd
        ClModel: trained model for Cl
        ClLikelihood: trained likelihood for Cl
        XMin: lower bound of independent variables
        XMax: upper bound of independent variables
        k: multiple of standard deviation used to evaluate confidence bounds
        fullRange: candidates must be feasible across the whole AoA range if true

    Returns:
        feasible: data on accepted points
        k: multiple used when evaluating confidence bounds with standard deviation
    '''
    # indepedent variables data
    M = candidates[:, 0]
    P = candidates[:, 1]
    T = candidates[:, 2]

    if fullRange:
        # accept CdUCB < maxCd at maxAoA
        CdMean, CdVar, _, _, _ = predictBatched(M, P, T, np.full(len(candidates), maxAoA), CdModel, CdLikelihood, ClModel, ClLikelihood, XMin, XMax)

        CdUCB = CdMean + k*torch.sqrt(CdVar)

        feasibleMask = CdUCB < maxCd
        feasibleMask = feasibleMask.numpy()

        M = M[feasibleMask]
        P = P[feasibleMask]
        T = T[feasibleMask]

        if len(M) == 0:
            print ('No feasible points')
            sys.exit()

        CdMean = CdMean.detach().numpy()[feasibleMask]
        CdVar = CdVar.detach().numpy()[feasibleMask]

        # accept ClLCB > minCl at minAoA
        _, _, ClMean, ClVar, _ = predictBatched(M, P, T, np.full(len(M), minAoA), CdModel, CdLikelihood, ClModel, ClLikelihood, XMin, XMax)

        ClLCB = ClMean - k*torch.sqrt(ClVar)

        feasibleMask = ClLCB > minCl
        feasibleMask = feasibleMask.numpy()

    else:
        # accept CdUCB < maxCd at minAoA
        CdMean, CdVar, _, _, _ = predictBatched(M, P, T, np.full(len(candidates), minAoA), CdModel, CdLikelihood, ClModel, ClLikelihood, XMin, XMax)

        CdUCB = CdMean + k*torch.sqrt(CdVar)

        feasibleMask = CdUCB < maxCd
        feasibleMask = feasibleMask.numpy()

        M = M[feasibleMask]
        P = P[feasibleMask]
        T = T[feasibleMask]

        if len(M) == 0:
            print ('No feasible points')
            sys.exit()

        CdMean = CdMean.detach().numpy()[feasibleMask]
        CdVar = CdVar.detach().numpy()[feasibleMask]

        # accept ClLCB > minCl at maxAoA
        _, _, ClMean, ClVar, _ = predictBatched(M, P, T, np.full(len(M), maxAoA), CdModel, CdLikelihood, ClModel, ClLikelihood, XMin, XMax)

        ClLCB = ClMean - k*torch.sqrt(ClVar)

        feasibleMask = ClLCB > minCl
        feasibleMask = feasibleMask.numpy()

    # accepted points data
    feasible = {
        'M' : M[feasibleMask],
        'P' : P [feasibleMask],
        'T' : T[feasibleMask],
        'CdMean' : CdMean[feasibleMask],
        'CdVar' : CdVar[feasibleMask],
        'ClMean' : ClMean.detach().numpy()[feasibleMask],
        'ClVar' : ClVar.detach().numpy()[feasibleMask],
    }

    return feasible, k


def clustering(feasible, XMin, XMax, spacing):
    '''
    Identify clusters

    Parameters:
        feasible: feasible configurations data
        XMin: lower bound of independent variables
        XMax: upper bound of independent variables
        spacing: node spacing

    Returns:
        symmetricClusters: list of symmetric cluster data
        nSymmetricClusters: number of symmetric clusters
        nSymmetricNoise: number of noise points in symmetric domain
        camberedClusters: list of cambered cluster data
        nCamberedClusters: number of cambered clusters
        nCamberedNoise: number of noise points in cambered domain
    '''
    # separate symmetric and cambered configurations
    symmetricMask = feasible['M'] == 0

    camberedFeasible = {k: v[~symmetricMask] for k, v in feasible.items()}
    symmetricFeasible = {k: v[symmetricMask] for k, v in feasible.items()}

    noCambered = camberedFeasible['M'].size == 0
    noSymmetric = symmetricFeasible['M'].size == 0

    # range of M, P and T for normalisation
    MPTMin = XMin[:3].numpy()
    MPTMax = XMax[:3].numpy()

    normSpacingM = spacing/(MPTMax[0] - MPTMin[0]) + 1e-3
    normSpacingP = spacing/(MPTMax[1] - MPTMin[1]) + 1e-3
    normSpacingT = spacing/(MPTMax[2] - MPTMin[2]) + 1e-3

    maxNormSpacing = np.linalg.norm([normSpacingM, normSpacingP, normSpacingT])

    # clustering in cambered domain
    camberedClusters = []
    nCamberedClusters = 0
    nCamberedNoise = 0

    if not noCambered:
        camberedMPT = np.column_stack([camberedFeasible['M'], camberedFeasible['P'], camberedFeasible['T']])
        camberedMPTNorm = (camberedMPT - MPTMin) / (MPTMax - MPTMin)


        dbscanCambered = DBSCAN(eps=maxNormSpacing, min_samples=1)
        camberedLabels = dbscanCambered.fit_predict(camberedMPTNorm)

        nCamberedClusters = len(set(camberedLabels)) - (1 if -1 in camberedLabels else 0)
        nCamberedNoise = np.sum(camberedLabels == -1)

        for clusterId in range (nCamberedClusters):
            mask = camberedLabels == clusterId

            camberedClusters.append({k: v[mask] for k, v in camberedFeasible.items()})

    # clustering in symmetric domain
    symmetricClusters = []
    nSymmetricClusters = 0
    nSymmetricNoise = 0

    if not noSymmetric:
        symmetricMPT = np.column_stack([symmetricFeasible['M'], symmetricFeasible['P'], symmetricFeasible['T']])
        symmetricMPTNorm = (symmetricMPT - MPTMin) / (MPTMax - MPTMin)

        dbscanSymmetric = DBSCAN(eps=normSpacingT, min_samples=1)
        symmetricLabels = dbscanSymmetric.fit_predict(symmetricMPTNorm)

        nSymmetricClusters = len(set(symmetricLabels)) - (1 if -1 in symmetricLabels else 0)
        nSymmetricNoise = np.sum(symmetricLabels == -1)
        symmetricClusters = []

        for clusterId in range (nSymmetricClusters):
            mask = symmetricLabels == clusterId

            symmetricClusters.append({k: v[mask] for k, v in symmetricFeasible.items()})

    return symmetricClusters, nSymmetricClusters, nSymmetricNoise, camberedClusters, nCamberedClusters, nCamberedNoise
