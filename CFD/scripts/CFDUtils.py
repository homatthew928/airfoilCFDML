import numpy as np
from scipy.optimize import brentq
import gmsh
import pandas as pd
import re
from datetime import datetime, UTC

def airfoilGeneration(M, P, T):
    '''
    Generates nodes of a NACA 4-digit airfoil

    Parameters:
        M: maximum camber (e.g. M = 6 camber is 6%)
        P: camber position (e.g. P = 4 camber position is 20%)
        T: thickness (e.g. T = 12 thickness is 12%)

    Return:
        nodes: airfoil nodes
        t: scaled thickness (e.g. t = 0.12 thickness is 12%)
    '''

    # rescale parameters
    m = M*0.01
    p = P*0.1
    t = T*0.01

    # cosine spacing to increase density near leading and trailing edge
    b = np.linspace (0, np.pi, 100)
    xc = (1-np.cos(b))/2

    # equations defined for a NACA 4-digit airfoil
    def camber (m, p, x):
        if m == 0:
            return np.zeros_like(x)

        yc = np.where (x<p, m/p**2 * (2*p*x-x**2), m/(1-p)**2 * (1-2*p+2*p*x-x**2))

        return yc

    def theta (m, p, x):
        if m == 0:
            return np.zeros_like(x)

        grad = np.where(x<p, 2*m/p**2 * (p-x), 2*m/(1-p)**2 * (p-x))

        return np.arctan(grad)

    a0 = 0.2969
    a1 = -0.126
    a2 = -0.3516
    a3 = 0.2843
    a4 = -0.1036

    def thickness (t, x):
        yt = t/0.2 * (a0*x**0.5 + a1*x + a2*x**2 + a3*x**3 + a4*x**4)

        return yt

    xu = xc - thickness(t, xc)*np.sin(theta(m, p, xc))
    yu = camber(m, p, xc) + thickness(t, xc)*np.cos(theta(m, p, xc))

    xl = xc + thickness(t, xc)*np.sin(theta(m, p, xc))
    yl = camber(m, p, xc) - thickness(t, xc)*np.cos(theta(m, p, xc))

    xn = np.concatenate((xl[::-1], xu[1:-1]))
    yn = np.concatenate((yl[::-1], yu[1:-1]))

    nodes = np.column_stack((xn, yn))

    return nodes, t

def nearfieldControl(meshLevel):
    '''
    Generates parameters to control mesh refinement for different mesh level

    Parameter:
        meshLevel: mesh level (1-4)

    Returns:
        r: boundary layer expansion ratio
        h0: first layer height on airfoil surface
        H: boundary layer thickness
        r1: expansion ratio from boundary layer to farfield
    '''
    if meshLevel == 1:
        r = 1.3
        h0 = 5e-6
        H = 0.0167994
        r1 = 1.3

    elif meshLevel == 2:
        r = 1.2
        h0 = 5e-6
        H = 0.0167994
        r1 = 1.2

    elif meshLevel == 3:
        r = 1.12
        h0 = 5e-6
        H = 0.0167994
        r1 = 1.15

    elif meshLevel == 4:
        r = 1.08
        h0 = 5e-6
        H = 0.0167994
        r1 = 1.1

    return r, h0, H, r1

def expansionRatio(H, h0, n):
    '''
    Calculates the expansion ratio required

    Parameters:
        H: total height
        h0: first layer height
        n: number of layers

    Returns:
        r: expansion ratio
    '''
    def residual(r):
        return h0*(r**n - 1) /(r - 1) - H

    r = brentq(residual, 1.000001, 2)

    return r

def tangent(nodes):
    '''
    Calculates the tangents along a set of nodes of an airfoil
        - note that len(tangents) = len(nodes)+2 because trailing edge has 3 tangents
            - along the lower surface
            - along the upper surface
            - normal to the trailing edge direction

    Parameters:
        nodes: airfoil nodes

    Returns:
        tangents: list of tangents at nodes
    '''
    vector = np.zeros_like(nodes)

    vector[0] = nodes[1] - nodes[0]
    vector[-1] = nodes[0] - nodes[-2]

    vector[1:-1] = nodes[2:] - nodes[:-2]

    vector = np.vstack((vector,
                         nodes[0] - nodes[-1],
                         nodes[1] - nodes[-1]
                         ))

    vectorLengths = np.linalg.norm(vector,axis=1)

    tangents = vector / vectorLengths[:, None]

    return tangents

def offset(nodes, H):
    '''
    Calculate the nodes of an airfoil with a surface offset

    Parameters:
        nodes: airfoil nodes
        H: offset distance

    returns
        offsetNodes: nodes of the offset surface of an airfoil
    '''
    tangents = tangent(nodes)
    nodes = np.vstack((nodes,
                        nodes[0],
                        nodes[0]
                        ))
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))


    offsetNodes = (nodes + H * normals)

    return offsetNodes

def normal(nodes):
    '''
    Calculates the normals along a set of nodes

    Parameters:
        nodes: nodes

    Returns:
        normals: list of normals at nodes
    '''
    tangents = tangent(nodes)

    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

    return normals

def distance (nodes):
    '''
    Calculates the distance along a set of nodes

    Parameters:
        nodes: nodes

    Returns:
        d: distance
    '''
    d = 0
    for i in range (len(nodes)-1):
        d += np.linalg.norm (nodes[i] - nodes[i+1])

    return d

def intersection(p1, p2, d1, d2):
    '''
    Calculates the point of intersection of two lines

    Parameters:
        p1: coordinates of point 1
        p2: coordinates of point 2
        d1: direction of line 1 from point1
        d2: direction of line 2 from point 2

    Return:
        intersection: coordinate of intersection
    '''
    intersection = (d2[1]*(p2[0]-p1[0])-d2[0]*(p2[1]-p1[1]))/(d1[0]*d2[1]-d2[0]*d1[1])*d1 + p1

    return intersection

def layers(H, h0, r):
    '''
    Calculates the number of intervals along a line with progression in spacing

    Parameters:
        H: total thickness
        h0: first layer thickness
        r: progression expansion ratio

    Return:
        layers: number of intervals
    '''
    layers = np.ceil(np.log((H/h0)*(r-1)+1)/np.log(r))
    return layers

def meshGeneration(M, P, T, meshLevel, AoADeg, farfield, caseName, geoPathCreate, aRefinement=10, bRefinement=50, cRefinement=50):
    '''
    Generates a 2D mesh around an airfoil

    Parameters:
        M: maximum camber (e.g. M = 6 camber is 6%)
        P: camber position (e.g. P = 4 camber position is 20%)
        T: thickness (e.g. T = 12 thickness is 12%)
        meshLevel: mesh level (1-4)
        AoADeg: angle of attack in degrees
        farfield: farfield extension distance
        caseName: case name for saving
        geoPathCreate: function to create windows path for saving
    '''
    nodes, t = airfoilGeneration(M, P, T)  # generate airfoil nodes

    r, h0, H, r1 = nearfieldControl(meshLevel)  # obtain mesh refinement control parameters

    AoA = np.deg2rad(AoADeg)  # angle of attack in radians

    # unit vectors parallel and perpendicular to flow direction
    et = np.array([np.cos(AoA), np.sin(AoA)])
    en = np.array([-np.sin(AoA), np.cos(AoA)])

    l = layers(H, h0, r)  # number of layers in BL
    hf = h0 * (r**l-1)  # thickeness of outermost layer

    R = t**2 * 2  # a proportional to the radius of curvature of leading edge

    airfoilOffset = offset(nodes, H)  # offset nodes of airfoil

    # points generation
    ref0 = -R*normal(nodes)[99]

    ref1Idx = np.argmin(np.abs([np.dot(nodes[i]-ref0, tangent(nodes)[i]) for i in range (0, 99)]))
    ref2Idx = np.argmin(np.abs([np.dot(nodes[i]-ref0, tangent(nodes)[i]) for i in range (100, 198)])) + 100

    farfieldDirection1 = (-et-en)/np.linalg.norm(-et-en)
    farfieldDirection2 = (-et+en)/np.linalg.norm(-et+en)

    ref1 = ref0 + farfield*farfieldDirection1
    ref2 = ref0 + farfield*farfieldDirection2

    ref3 = intersection(ref2, airfoilOffset[-1], et, en)
    ref5 = airfoilOffset[-1] + farfield*et
    ref4 = intersection(ref3, ref5, et, en)
    ref7 = intersection(ref1, airfoilOffset[-1], et, -en)
    ref6 = intersection(ref7, ref5, et, -en)

    ref = np.vstack([ref0, ref1, ref2, ref3, ref4, ref5, ref6, ref7])

    # mesh refinement controls
    a = np.ceil(aRefinement*(np.sqrt(2)**(meshLevel)))
    b = np.ceil(bRefinement*(np.sqrt(2)**(meshLevel)))
    c = np.ceil(cRefinement*(np.sqrt(2)**(meshLevel)))
    d = np.ceil(H/(distance(nodes[0:ref1Idx+1])/c))
    e = layers(farfield-H, hf, r1)
    r2 = expansionRatio(np.linalg.norm(ref4-ref6)/2, hf, e)
    f = np.ceil(farfield/(np.linalg.norm(ref7-ref1)/(c+1)))
    r3 = expansionRatio(farfield, distance(airfoilOffset[0:ref1Idx+1])/c/2, f)

    # generate path for saving
    geoPath = geoPathCreate(caseName)
    geoPath.parent.mkdir(parents=True, exist_ok=True)

    # geo script generation and saving
    with geoPath.open("w", encoding="utf-8") as file:
        n, _ = np.shape(nodes)

        file.write('a = ' + str(a) + ';' + '\n')
        file.write('b = ' + str(b) + ';' + '\n')
        file.write('c = ' + str(c) + ';' + '\n')
        file.write('d = ' + str(d) + ';' + '\n')
        file.write('e = ' + str(e) + ';' + '\n')
        file.write('f = ' + str(f) + ';' + '\n')
        file.write('r = ' + str(r) + ';' + '\n')
        file.write('r1 = ' + str(r1) + ';' + '\n')
        file.write('r2 = ' + str(r2) + ';' + '\n')
        file.write('r3 = ' + str(r3) + ';' + '\n')
        file.write('l = ' + str(l) + ';' + '\n')
        file.write('n = ' + str(n) + ';' + '\n')

        file.write('ref1Idx = ' + str(ref1Idx+1) + ';' + '\n')
        file.write('ref2Idx = ' + str(ref2Idx+1) + ';' + '\n')

        for i in range (n):
            points = 'Point(' + str(i+1) + ') = '
            coords = '{' + str(nodes[i, 0]) + ', ' + str(nodes[i, 1]) + ', ' + '0};' + '\n'

            file.write(points+coords)

        file.write(r'''
        Spline(1) = {1:ref1Idx};
        Spline(2) = {ref1Idx:ref2Idx};
        Spline(3) = {ref2Idx:n, 1};

        ''')

        for i in range (n+2):
            points = 'Point(' + str(1000+i+1) + ') = '
            coords = '{' + str(airfoilOffset[i, 0]) + ', ' + str(airfoilOffset[i, 1]) + ', ' + '0};' + '\n'

            file.write(points+coords)

        file.write(r'''
        Line(4) = {1000+n+2, 1001};
        Spline(5) = {1001:1000+ref1Idx};
        Spline(6) = {1000+ref1Idx:1000+ref2Idx};
        Spline(7) = {1000+ref2Idx:1000+n+1};
        Line(8) = {1000+n+1, 1000+n+2};

        ''')


        file.write(r'''
        Line(9) = {1, 1000+1};
        Line(10) = {ref1Idx, 1000+ref1Idx};
        Line(11) = {ref2Idx, 1000+ref2Idx};
        Line(12) = {1, 1000+n+1};
        Line(13) = {1, 1000+n+2};
        ''')

        for i in range (len(ref)):
            file.write('Point(' + str(i+2000) + ') = {' + str(ref[i][0]) + ', ' + str(ref[i][1]) + ', 0};' + '\n')

        file.write(r'''
        Circle(14) = {2001, 2000, 2002};
        Line(15) = {2002, 2003};
        Line(16) = {2003, 2004};
        Line(17) = {2004, 2005};
        Line(18) = {2005, 2006};
        Line(19) = {2006, 2007};
        Line(20) = {2007, 2001};
        Line(21) = {1000+ref1Idx, 2001};
        Line(22) = {1000+ref2Idx, 2002};
        Line(23) = {1000+n+2, 2003};
        Line(24) = {1000+n+2, 2005};
        Line(25) = {1000+n+2, 2007};

        Curve Loop (1) = {-9, 13, 4};
        Plane Surface(1) = {1};

        Curve Loop (2) = {-1, 9, 5, -10};
        Plane Surface(2) = {2};

        Curve Loop (3) = {-2, 10, 6, -11};
        Plane Surface(3) = {3};

        Curve Loop (4) = {-3, 11, 7, -12};
        Plane Surface(4) = {4};

        Curve Loop (5) = {12, 8, -13};
        Plane Surface(5) = {5};

        Curve Loop (6) = {-6, 21, 14, -22};
        Plane Surface(6) = {6};

        Curve Loop (7) = {-7, 22, 15, -23, -8};
        Plane Surface(7) = {7};

        Curve Loop (8) = {23, 16, 17, -24};
        Plane Surface(8) = {8};

        Curve Loop (9) = {24, 18, 19, -25};
        Plane Surface(9) = {9};

        Curve Loop (10) = {25, 20, -21, -5, -4};
        Plane Surface(10) = {10};

        Transfinite Curve {2, 6, 14} = a+1;
        Transfinite Curve {3, 7} = b+1;
        Transfinite Curve {8} = d+1;
        Transfinite Curve {15} = b+d+1;
        Transfinite Curve {1, 5} = c+1;
        Transfinite Curve {4} = d+1;
        Transfinite Curve {20} = c+d+1;
        Transfinite Curve {13} = d+1;
        Transfinite Curve {9, 10, 11, 12} = l+1 Using Progression r;
        Transfinite Curve {21, 22} = e+1 Using Progression r1;
        Transfinite Curve {23, -17, 18, 25} = e+1 Using Progression r2;
        Transfinite Curve {16, -19} = f+1;
        Transfinite Curve {24} = f+1 Using Progression r3;

        Transfinite Surface {1} = {1000+n+2, 1, 1001};
        Transfinite Surface {2} = {1, ref1Idx, 1000+ref1Idx, 1001};
        Transfinite Surface {3} = {ref1Idx, ref2Idx, 1000+ref2Idx, 1000+ref1Idx};
        Transfinite Surface {4} = {ref2Idx, 1, 1000+n+1, 1000+ref2Idx};
        Transfinite Surface {5} = {1000+n+2, 1, 1000+n+1};
        Transfinite Surface {6} = {1000+ref1Idx, 2001, 2002, 1000+ref2Idx};
        Transfinite Surface {7} = {1000+ref2Idx, 2002, 2003, 1000+n+2};
        Transfinite Surface {8} = {1000+n+2, 2003, 2004, 2005};
        Transfinite Surface {9} = {1000+n+2, 2005, 2006, 2007};
        Transfinite Surface {10} = {1000+n+2, 2007, 2001, 1000+ref1Idx};

        Recombine Surface {1:10};
        allSurfaces[] = {1:10};
        ext[] = Extrude {0, 0, 0.0001}
            {
                Surface {allSurfaces[]};
                Layers {1};
                Recombine;
            };
        Physical Volume('fluid') = {1:10};
        Physical Surface('frontAndBack') = {1:10, 42, 64, 86, 108, 125, 147, 174, 196, 218, 245};
        Physical Surface('airfoil') = {51, 73, 95};
        Physical Surface('inlet') = {142};
        Physical Surface('top') = {165, 187};
        Physical Surface('outlet') = {191, 209};
        Physical Surface('bottom') = {213, 232};
        ''')

    # mesh generation and saving
    meshPath = geoPath.parent/'mesh.msh'

    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 1)
        gmsh.open(str(geoPath))
        gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
        gmsh.option.setNumber("Geometry.Tolerance", 1e-16)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(meshPath))

    finally:
        gmsh.finalize()

def dataExtraction (dataPath, setPath, CFDPath):
    '''
    Extracts data from OpenFOAM CFD simulations

    Parameters:
        dataPath: path to storing CFD solutions
        setPath: path to information on simulation parameters
        CFDPath: path to directory of CFD runs
    '''
    # create or read existing data file
    if not dataPath.exists():
        data = pd.DataFrame({
            'config' : [],
            'M' : [],
            'P' : [],
            'T' : [],
            'AoA' : [],
            'Cd' : [],
            'CdCoV' : [],
            'Cl' : [],
            'ClCoV' : [],
            'time' : [],
            'iteration' : [],
            'cells' : [],
            'maxNonOrtho' : [],
            'avgNonOrtho' : [],
            'maxSkewness' : [],
            'maxAR' : [],
            'minyPlus' : [],
            'maxyPlus' : [],
            'avgyPlus' : [],
            'timestamp' : []
        })
    else:
        data = pd.read_csv(dataPath, sep = '\t')

    # identify the next unstored configuration
    if data.empty:
        lastConfig = 0
    else:
        lastConfig = data['config'].iloc[-1]

    config = int(lastConfig+1)

    # extract information on CFD runs
    while True:
        # forces and coefficient of variance
        forcePath = next(CFDPath.glob(f'{config}_*/postProcessing/forceCoeffs/0/coefficient.dat'), None)

        if forcePath is None:
            break

        df = pd.read_csv(forcePath, skiprows= 13, sep = r'\s+', header = None).to_numpy()
        forceData = np.column_stack((df[:, 0], df[:, 1], df[:, 4]))

        Cd = np.average(forceData[-200:, 1])
        CdCoV = np.std(forceData[-200:, 1])/Cd
        Cl = np.average(forceData[-200:, 2])
        ClCoV = np.std(forceData[-200:, 2])/Cl

        # run time
        simpleFoamLog = next(CFDPath.glob(f'{config}_*/log.simpleFoam'), None)
        matches = re.findall(r"ExecutionTime\s*=\s*[0-9.eE+-]+\s*s\s+ClockTime\s*=\s*([0-9.eE+-]+)\s*s", simpleFoamLog.read_text(encoding="utf-8",errors="ignore"))
        time = float(matches[-1])

        # iteration count
        matches = re.findall(r"^Time\s*=\s*([0-9.eE+-]+)", simpleFoamLog.read_text(encoding="utf-8",errors="ignore"), flags = re.MULTILINE)
        iteration = int(matches[-1])

        # number of cells
        checkMeshLog = next(CFDPath.glob(f'{config}_*/log.checkMesh'), None)
        matches = re.findall(r"cells:\s*(\d+)", checkMeshLog.read_text(encoding="utf-8",errors="ignore"))
        cells = int(matches[-1])

        # maximum and average mesh non-orthogonality
        matches = re.findall(r"Mesh\s+non-orthogonality\s+Max:\s*([0-9.eE+-]+)", checkMeshLog.read_text(encoding="utf-8",errors="ignore"))
        maxNonOrtho = float(matches[-1])

        matches = re.findall(r"Mesh\s+non-orthogonality.*?average:\s*([0-9.eE+-]+)", checkMeshLog.read_text(encoding="utf-8",errors="ignore"))
        avgNonOrtho = float(matches[-1])

        # maximum mesh skewness
        matches = re.findall(r"Max\s+skewness\s*=\s*([0-9.eE+-]+)", checkMeshLog.read_text(encoding="utf-8",errors="ignore"))
        maxSkewness = float(matches[-1])

        # maximum aspect ratio
        matches = re.findall(r"Max\s+aspect\s+ratio\s*[:=]\s*([0-9.eE+-]+)", checkMeshLog.read_text(encoding="utf-8",errors="ignore"))
        maxAR = float(matches[-1])

        # minimum, maximum and average y+
        yPlusLog = next(CFDPath.glob(f'{config}_*/log.simpleFoam.yPlus'), None)
        matches = re.findall(r"y\+ :\s*min\s*=\s*([0-9.eE+-]+)", yPlusLog.read_text(encoding="utf-8",errors="ignore"))
        minyPlus = float(matches[-1])

        matches = re.findall(r"y\+ :.*?max\s*=\s*([0-9.eE+-]+)", yPlusLog.read_text(encoding="utf-8",errors="ignore"))
        maxyPlus = float(matches[-1])

        matches = re.findall(r"y\+ :.*?average\s*=\s*([0-9.eE+-]+)", yPlusLog.read_text(encoding="utf-8",errors="ignore"))
        avgyPlus = float(matches[-1])

        # time log of storing data
        timestamp = datetime.now(UTC)

        # add simulation input and ouput information to dataframe
        set = pd.read_csv(setPath, sep = '\t')

        newRow = pd.DataFrame(
            [[int(config)] +
            list(set.loc[set['config'] == config].iloc[0, 1:5]) +
            [Cd, CdCoV, Cl, ClCoV, time, iteration, cells, maxNonOrtho, avgNonOrtho, maxSkewness, maxAR, minyPlus, maxyPlus, avgyPlus, timestamp]],
            columns=data.columns
            )

        if data.empty:
            data = newRow.copy()
        else:
            data = pd.concat([data, newRow], ignore_index=True)

        config = int(config+1)

    # save dataframe
    data.to_csv(dataPath, sep="\t", index=False)