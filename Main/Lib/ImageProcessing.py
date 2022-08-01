import numpy
import cv2

def blendImages(src, dst, Dlib_Mask, featherAmount=0.2):

    Dlib_MaskIndices = numpy.where(Dlib_Mask != 0)

    Dlib_MaskPts = numpy.hstack((Dlib_MaskIndices[1][:, numpy.newaxis], Dlib_MaskIndices[0][:, numpy.newaxis]))
    faceSize = numpy.max(Dlib_MaskPts, axis=0) - numpy.min(Dlib_MaskPts, axis=0)
    featherAmount = featherAmount * numpy.max(faceSize)

    hull = cv2.convexHull(Dlib_MaskPts)
    dists = numpy.zeros(Dlib_MaskPts.shape[0])
    for i in range(Dlib_MaskPts.shape[0]):
        dists[i] = cv2.pointPolygonTest(hull, (int(Dlib_MaskPts[i, 0]), int(Dlib_MaskPts[i, 1])), True)

    Dlib_Weigh = numpy.clip(dists / featherAmount, 0, 1)

    composedImg = numpy.copy(dst)
    composedImg[Dlib_MaskIndices[0], Dlib_MaskIndices[1]] = Dlib_Weigh[:, numpy.newaxis] * src[Dlib_MaskIndices[0], Dlib_MaskIndices[1]] + (1 - Dlib_Weigh[:, numpy.newaxis]) * dst[Dlib_MaskIndices[0], Dlib_MaskIndices[1]]

    return composedImg


def colorTransfer(src, dst, Dlib_Mask):
    transferredDst = numpy.copy(dst)

    Dlib_MaskIndices = numpy.where(Dlib_Mask != 0)


    Dlib_MaskedSrc = src[Dlib_MaskIndices[0], Dlib_MaskIndices[1]].astype(numpy.int32)
    Dlib_MaskedDst = dst[Dlib_MaskIndices[0], Dlib_MaskIndices[1]].astype(numpy.int32)

    meanSrc = numpy.mean(Dlib_MaskedSrc, axis=0)
    meanDst = numpy.mean(Dlib_MaskedDst, axis=0)

    Dlib_MaskedDst = Dlib_MaskedDst - meanDst
    Dlib_MaskedDst = Dlib_MaskedDst + meanSrc
    Dlib_MaskedDst = numpy.clip(Dlib_MaskedDst, 0, 255)

    transferredDst[Dlib_MaskIndices[0], Dlib_MaskIndices[1]] = Dlib_MaskedDst

    return transferredDst

