import numpy as np
import cv2
import Lib.models as models
from dlib import rectangle
import Lib.NonLinearLeastSquares as NonLinearLeastSquares

def getNormal(Tri_Dlib):
    a = Tri_Dlib[:, 0]
    b = Tri_Dlib[:, 1]
    c = Tri_Dlib[:, 2]

    axisX = b - a
    axisX = axisX / np.linalg.norm(axisX)
    axisY = c - a
    axisY = axisY / np.linalg.norm(axisY)
    axisZ = np.cross(axisX, axisY)
    axisZ = axisZ / np.linalg.norm(axisZ)

    return axisZ

def flipWinding(Tri_Dlib):
    return [Tri_Dlib[1], Tri_Dlib[0], Tri_Dlib[2]]

def fixMeshWinding(Dlib_mesh, vertices):
    for i in range(Dlib_mesh.shape[0]):
        Tri_Dlib = Dlib_mesh[i]
        normal = getNormal(vertices[:, Tri_Dlib])
        if normal[2] > 0:
            Dlib_mesh[i] = flipWinding(Tri_Dlib)

    return Dlib_mesh

def getShape3D(Dlib_Mean3dshape, Dlib_shape, params):

    s = params[0]

    r = params[1:4]

    t = params[4:6]
    w = params[6:]


    R = cv2.Rodrigues(r)[0]
    shape3D = Dlib_Mean3dshape + np.sum(w[:, np.newaxis, np.newaxis] * Dlib_shape, axis=0)

    shape3D = s * np.dot(R, shape3D)
    shape3D[:2, :] = shape3D[:2, :] + t[:, np.newaxis]

    return shape3D

def getMask(renderedImg):
    mask = np.zeros(renderedImg.shape[:2], dtype=np.uint8)

def load3DFaceModel(filename):
    faceModelFile = np.load(filename)
    Dlib_Mean3dshape = faceModelFile["mean3DShape"]
    Dlib_mesh = faceModelFile["mesh"]
    Dlib_idxs3D = faceModelFile["idxs3D"]
    Dlib_idxs2D = faceModelFile["idxs2D"]
    Dlib_shape = faceModelFile["blendshapes"]
    Dlib_mesh = fixMeshWinding(Dlib_mesh, Dlib_Mean3dshape)

    return Dlib_Mean3dshape, Dlib_shape, Dlib_mesh, Dlib_idxs3D, Dlib_idxs2D

def getFaceKeypoints(image, detector, predictor, maxImgSizeForDetection=640):
    imgScale = 1
    scaledImg = image
    if max(image.shape) > maxImgSizeForDetection:
        imgScale = maxImgSizeForDetection / float(max(image.shape))
        scaledImg = cv2.resize(image, (int(image.shape[1] * imgScale), int(image.shape[0] * imgScale)))



    dets = detector(scaledImg, 1)

    if len(dets) == 0:
        return None

    shapes2D = []
    for det in dets:
        faceRectangle = rectangle(int(det.left() / imgScale), int(det.top() / imgScale), int(det.right() / imgScale), int(det.bottom() / imgScale))


        dlibShape = predictor(image, faceRectangle)
        
        shape2D = np.array([[p.x, p.y] for p in dlibShape.parts()])

        shape2D = shape2D.T

        shapes2D.append(shape2D)

    return shapes2D
    

def getFaceTextureCoords(image, Dlib_Mean3dshape, Dlib_shape, Dlib_idxs2D, Dlib_idxs3D, detector, predictor):
    projectionModel = models.OrthographicProjectionBlendshapes(Dlib_shape.shape[0])

    keypoints = getFaceKeypoints(image, detector, predictor)[0]
    modelParams = projectionModel.getInitialParameters(Dlib_Mean3dshape[:, Dlib_idxs3D], keypoints[:, Dlib_idxs2D])
    modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([Dlib_Mean3dshape[:, Dlib_idxs3D], Dlib_shape[:, :, Dlib_idxs3D]], keypoints[:, Dlib_idxs2D]), verbose=0)
    textureCoords = projectionModel.fun([Dlib_Mean3dshape, Dlib_shape], modelParams)

    return textureCoords
