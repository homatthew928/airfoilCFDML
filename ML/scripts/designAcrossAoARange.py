from pathlib import Path
import importlib
import MLUtils
importlib.reload(MLUtils)
from MLUtils import filterFeasible, denseSampling, loadModel, clustering, predictBatched, predict
import sys
import numpy as np
import pandas as pd

# user inputs
maxCd = float(input('Maximum Cd: '))
minCl = float(input('Minimum Cl: '))
operatingAoA = float(input('Operating AoA: '))
minAoA = float(input('Minimum AoA: '))
maxAoA = float(input('Maximum AoA: '))

# project directory
projectDir = Path(__file__).resolve().parents[2]

# loading models and model information
modelsPath = projectDir/'ML'/'models'
CdModel, CdLikelihood, trainedModelDict = loadModel('Cd', modelsPath)
ClModel, ClLikelihood, _ = loadModel('Cl', modelsPath)

XMin, XMax = trainedModelDict['XMin'], trainedModelDict['XMax']

# sampling
candidates, spacing = denseSampling(XMin, XMax)

# filtering
print ('Initiating filtering ...')
feasibleData, _ = filterFeasible(candidates,
                          maxCd, minCl, minAoA, maxAoA,
                          CdModel, CdLikelihood, ClModel, ClLikelihood,
                          XMin, XMax)

if len(feasibleData['M']) == 0:
    sys.exit('No feasible candidates found')

print(f"{len(feasibleData['M'])} feasible candidates found out of {len(candidates)}")

# clustering
print ('Initiating clusters identification ...')

symmetricClusters, nSymmetricClusters, nSymmetricNoise, camberedClusters, nCamberedClusters, nCamberedNoise = clustering(feasibleData, XMin, XMax, spacing)

print (f'A total of {nSymmetricClusters + nCamberedClusters} cluster(s) was/were identified')

if nSymmetricClusters + nCamberedClusters == 0:
    sys.exit('No feasible points')

print (f'{nSymmetricClusters} cluster(s) is/are symmetric')
print (f'{nCamberedClusters} cluster(s) is/are cambered')

if nSymmetricClusters == 0:  # domain selection
    selectedDomain = 'cambered'
    print ('Cambered domain automatically selected')
elif nCamberedClusters == 0:
    selectedDomain = 'symmetric'
    print ('Symmetric domain automatically selected')
else:
    domainInput = input('domain (s: symmetric, c: cambered): ')
    if domainInput == 's':
        selectedDomain = 'symmetric'
    elif domainInput == 'c':
        selectedDomain = 'cambered'

if selectedDomain == 'symmetric':
    clusters = symmetricClusters
    nClusters = nSymmetricClusters
elif selectedDomain == 'cambered':
    clusters = camberedClusters
    nClusters = nCamberedClusters

if nClusters == 1:  # cluster selection from domain
    clusterIndex = 0
else:
    print (f'Select cluster from 0-{nClusters-1}')
    clusterIndex = int(input('Cluster: '))

selectedClusterData = clusters[clusterIndex]

# predicting configurations data at operating AoA
CdMean, CdVar, ClMean, ClVar, LoDMean = predictBatched(selectedClusterData['M'], selectedClusterData['P'], selectedClusterData['T'], np.full(len(selectedClusterData['M']), operatingAoA),
                                                        CdModel, CdLikelihood, ClModel, ClLikelihood, XMin, XMax)

CdMean = CdMean.detach().numpy()
CdVar = CdVar.detach().numpy()
ClMean = ClMean.detach().numpy()
ClVar = ClVar.detach().numpy()
LoDMean = LoDMean.detach().numpy()

operatingAoAData = {
    'M' : selectedClusterData['M'],
    'P' : selectedClusterData['P'],
    'T' : selectedClusterData['T'],
    'CdMean' : CdMean,
    'CdVar' : CdVar,
    'ClMean' : ClMean,
    'ClVar' : ClVar,
    'LoDMean' : LoDMean
}

# identifying maximum L/D configuration
maxLoDMean = max(operatingAoAData['LoDMean'])
LoDMask = operatingAoAData['LoDMean'] == maxLoDMean
maxLoDM = np.round(operatingAoAData['M'][LoDMask][0], 1)
maxLoDP = np.round(operatingAoAData['P'][LoDMask][0], 1)
maxLoDT = np.round(operatingAoAData['T'][LoDMask][0], 1)

print (f'Maximum L/D at {operatingAoA} degrees = {maxLoDMean}')
print (f'Configuration: M = {maxLoDM}, P = {maxLoDP}, T = {maxLoDT}')

# sequential parameter pinning and range updates
if selectedDomain == 'symmetric':
    TRange = [np.round(min(selectedClusterData['T']), 1), np.round(max(selectedClusterData['T']), 1)]  # range of T

    # automatic selection of T if only 1 T available
    if TRange[0] == TRange[1]:
        selectedT = TRange[0]
        print (f'Only one value of T available: {selectedT}')

    # user selection of T
    else:
        print (f"T range: {TRange[0]} to {TRange[1]}")

        unavailablePoints = [float(np.round(i, 1)) for i in np.arange(TRange[0], TRange[1] + spacing/2, spacing)  # identify unavailable Ts in range
                            if np.round(i, 1) not in np.round(selectedClusterData['T'], 1)]
        if len(unavailablePoints) != 0:
            print (f'Unavailable points: {unavailablePoints}')

        selectedT = float(input(f"T: "))

    # prediction at chosen configuration
    CdMean, CdVar, ClMean, ClVar, LoDMean, X = predict(0, 0, selectedT, operatingAoA,
                                                       CdModel, CdLikelihood, ClModel, ClLikelihood,
                                                       XMin, XMax, requiresGrad=True)

    # gradients
    CdMean.sum().backward(retain_graph=True)
    gradCd = X.grad.clone()
    X.grad.zero_()

    ClMean.sum().backward(retain_graph=True)
    gradCl = X.grad.clone()
    X.grad.zero_()

    LoDMean.sum().backward()
    gradLoD = X.grad.clone()

    # values
    CdMean = CdMean.detach().numpy()
    ClMean = ClMean.detach().numpy()
    LoDMean = LoDMean.detach().numpy()

    # presentation
    print (f'Cd = {CdMean[0]}')
    print (f'Cl = {ClMean[0]}')
    print (f'L/D = {LoDMean[0]}')

    gradCdDf = pd.DataFrame({'dCd/dT': gradCd[:,2].numpy(), 'dCd/dAoA': gradCd[:,3].numpy()})

    print (gradCdDf.to_string(index=False))

    gradClDf = pd.DataFrame({'dCl/dT': gradCl[:,2].numpy(), 'dCl/dAoA': gradCl[:,3].numpy()})

    print (gradClDf.to_string(index=False))

    gradLoDDf = pd.DataFrame({'dLoD/dT': gradLoD[:,2].numpy(), 'dLoD/dAoA': gradLoD[:,3].numpy()})

    print (gradLoDDf.to_string(index=False))

else:
    # initial range
    ranges0 = {
        'M' : [np.round(min(selectedClusterData['M']), 1), np.round(max(selectedClusterData['M']), 1)],
        'P' : [np.round(min(selectedClusterData['P']), 1), np.round(max(selectedClusterData['P']), 1)],
        'T' : [np.round(min(selectedClusterData['T']), 1), np.round(max(selectedClusterData['T']), 1)]
    }

    print (f"M range: {ranges0['M'][0]} to {ranges0['M'][1]}")
    print (f"P range: {ranges0['P'][0]} to {ranges0['P'][1]}")
    print (f"T range: {ranges0['T'][0]} to {ranges0['T'][1]}")

    # pinning first parameter
    pin1 = input('First parameter to pin (M, P, T): ')

    # automatic selection of pin1 if only 1 value available
    if ranges0[pin1][0] == ranges0[pin1][1]:
        pin1Val = ranges0[pin1][0]
        print (f'Only one value of {pin1} available: {pin1Val}')

    # user selection of value of pin1
    else:
        print (f'{pin1} range: {ranges0[pin1][0]} to {ranges0[pin1][1]}')

        unavailablePoints = [float(np.round(i, 1)) for i in np.arange(ranges0[pin1][0], ranges0[pin1][1] + spacing/2, spacing)
                             if np.round(i, 1) not in np.round(selectedClusterData[pin1], 1)]
        if len(unavailablePoints) != 0:
            print (f'Unavailable points: {unavailablePoints}')

        pin1Val = float(input(f'Pin value: '))

    # range after pinning 1 parameter
    mask1 = np.round(selectedClusterData[pin1], 1) == pin1Val

    ranges1 = {
        'M' : [np.round(min(selectedClusterData['M'][mask1]), 1), np.round(max(selectedClusterData['M'][mask1]), 1)],
        'P' : [np.round(min(selectedClusterData['P'][mask1]), 1), np.round(max(selectedClusterData['P'][mask1]), 1)],
        'T' : [np.round(min(selectedClusterData['T'][mask1]), 1), np.round(max(selectedClusterData['T'][mask1]), 1)]
    }

    # pinning second parameter
    remainingParameters = np.array(['M', 'P', 'T'])[np.array(['M', 'P', 'T']) != pin1]

    pin2 = input(f'Second parameter to pin ({remainingParameters[0]}, {remainingParameters[1]}): ')

    # automatic selection of pin2 if only 1 value available
    if ranges1[pin2][0] == ranges1[pin2][1]:
        pin2Val = ranges1[pin2][0]
        print (f'Only one value of {pin2} available: {pin2Val}')

    # user selection of value of pin2
    else:
        print (f'{pin2} range: {ranges1[pin2][0]} to {ranges1[pin2][1]}')

        unavailablePoints = [float(np.round(i, 1))  for i in np.arange(ranges1[pin2][0], ranges1[pin2][1] + spacing/2, spacing)
                             if np.round(i, 1) not in np.round(selectedClusterData[pin2][mask1], 1)]
        if len(unavailablePoints) != 0:
            print (f'Unavailable points: {unavailablePoints}')

        pin2Val = float(input(f'Pin value: '))

    # range after pinning 2 parameters
    mask2 = (np.round(selectedClusterData[pin1], 1) == pin1Val) & (np.round(selectedClusterData[pin2], 1) == pin2Val)

    ranges2 = {
        'M' : [np.round(min(selectedClusterData['M'][mask2]), 1), np.round(max(selectedClusterData['M'][mask2]), 1)],
        'P' : [np.round(min(selectedClusterData['P'][mask2]), 1), np.round(max(selectedClusterData['P'][mask2]), 1)],
        'T' : [np.round(min(selectedClusterData['T'][mask2]), 1), np.round(max(selectedClusterData['T'][mask2]), 1)]
    }

    # pinning third parameter
    pin3 = remainingParameters[remainingParameters != pin2][0]

    print (f'Remaining parameter: {pin3}')

    # automatic selection of pin3 if only 1 value available
    if ranges2[pin3][0] == ranges2[pin3][1]:
        pin3Val = ranges2[pin3][0]
        print (f'Only one value of {pin3} available: {pin3Val}')

    else:
        print (f'{pin3} range: {ranges2[pin3][0]} to {ranges2[pin3][1]}')

        unavailablePoints = [float(np.round(i, 1)) for i in np.arange(ranges2[pin3][0], ranges2[pin3][1] + spacing/2, spacing)
                             if np.round(i, 1) not in np.round(selectedClusterData[pin3][mask2], 1)]
        if len(unavailablePoints) != 0:
            print (f'Unavailable points: {unavailablePoints}')

        pin3Val = float(input(f'Pin value: '))

    # identify the pinned M, P and T
    pins = [pin1, pin2, pin3]
    pinVals = [pin1Val, pin2Val, pin3Val]

    pinDict = dict(zip(pins, pinVals))
    selectedM, selectedP, selectedT = pinDict['M'], pinDict['P'], pinDict['T']

    # prediction at chosen configuration
    CdMean, CdVar, ClMean, ClVar, LoDMean, X = predict(selectedM, selectedP, selectedT, operatingAoA,
                                                       CdModel, CdLikelihood, ClModel, ClLikelihood,
                                                       XMin, XMax, requiresGrad=True)

    # gradients
    CdMean.sum().backward(retain_graph=True)
    gradCd = X.grad.clone()
    X.grad.zero_()

    ClMean.sum().backward(retain_graph=True)
    gradCl = X.grad.clone()
    X.grad.zero_()

    LoDMean.sum().backward()
    gradLoD = X.grad.clone()

    # values
    CdMean = CdMean.detach().numpy()
    CdVar = CdVar.detach().numpy()
    ClMean = ClMean.detach().numpy()
    ClVar = ClVar.detach().numpy()
    LoDMean = LoDMean.detach().numpy()

    # presentation
    print (f'Cd = {CdMean[0]}')
    print (f'Cl = {ClMean[0]}')
    print (f'L/D = {LoDMean[0]}')

    gradCdDf = pd.DataFrame({'dCd/dM': gradCd[:,0].numpy(), 'dCd/dP': gradCd[:,1].numpy(), 'dCd/dT': gradCd[:,2].numpy(), 'dCd/dAoA': gradCd[:,3].numpy()})

    print (gradCdDf.to_string(index=False))

    gradClDf = pd.DataFrame({'dCl/dM': gradCl[:,0].numpy(), 'dCl/dP': gradCl[:,1].numpy(), 'dCl/dT': gradCl[:,2].numpy(), 'dCl/dAoA': gradCl[:,3].numpy()})

    print (gradClDf.to_string(index=False))

    gradLoDDf = pd.DataFrame({'dLoD/dM': gradLoD[:,0].numpy(), 'dLoD/dP': gradLoD[:,1].numpy(), 'dLoD/dT': gradLoD[:,2].numpy(), 'dLoD/dAoA': gradLoD[:,3].numpy()})

    print (gradLoDDf.to_string(index=False))