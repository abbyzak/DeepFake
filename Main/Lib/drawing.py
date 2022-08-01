import cv2 as ComputerVision
import numpy

def drawPoints(img, dlib_coords, color=(0, 255, 0)):
    for dlib_coord in dlib_coords:
        ComputerVision.circle(img, (int(dlib_coord[0]), int(dlib_coord[1])), 2, color)

def drawCross(img, params, center=(100, 100), scale=30.0):
    R = ComputerVision.Rodrigues(params[1:4])[0]

    dlib_coords = numpy.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    dlib_coords = numpy.dot(dlib_coords, R.T)
    dlib_coords2D = dlib_coords[:, :2]

    dlib_coords2D = (pints2D * scale + center).astype(numpy.int32)
    
    ComputerVision.line(img, (center[0], center[1]), (dlib_coords2D[0, 0], dlib_coords2D[0, 1]), (255, 0, 0), 3)
    ComputerVision.line(img, (center[0], center[1]), (dlib_coords2D[1, 0], dlib_coords2D[1, 1]), (0, 255, 0), 3)
    ComputerVision.line(img, (center[0], center[1]), (dlib_coords2D[2, 0], dlib_coords2D[2, 1]), (0, 0, 255), 3)

def drawMesh(img, shape, mesh, color=(255, 0, 0)):
    for triangle in mesh:
        dlib_coord1 = shape[triangle[0]].astype(numpy.int32)
        dlib_coord2 = shape[triangle[1]].astype(numpy.int32)
        dlib_coord3 = shape[triangle[2]].astype(numpy.int32)

        ComputerVision.line(img, (dlib_coord1[0], dlib_coord1[1]), (dlib_coord2[0], dlib_coord2[1]), (255, 0, 0), 1)
        ComputerVision.line(img, (dlib_coord2[0], dlib_coord2[1]), (dlib_coord3[0], dlib_coord3[1]), (255, 0, 0), 1)
        ComputerVision.line(img, (dlib_coord3[0], dlib_coord3[1]), (dlib_coord1[0], dlib_coord1[1]), (255, 0, 0), 1)

def drawProjectedShape(img, x, projection, mesh, params, lockedTranslation=False):
    localParams = numpy.copy(params)

    if lockedTranslation:
        localParams[4] = 100
        localParams[5] = 200

    projectedShape = projection.fun(x, localParams)

    drawPoints(img, projectedShape.T, (0, 0, 255))
    drawMesh(img, projectedShape.T, mesh)
    drawCross(img, params)
