import os as filer
import dlib
import cv2 as CompVisi
import numpy

import Lib.models as models
import Lib.NonLinearLeastSquares as NonLinearLeastSquares
import Lib.ImageProcessing as ImageProcessing

from Lib.drawing import *

import FaceRendering
import Lib.utils as utils


predictor_path = filer.path.join(filer.path.dirname(__file__), "..", "shape_predictor_68_face_landmarks.dat")
image_name = filer.path.join(filer.path.dirname(__file__), "..", "data", "baby.webp")
cap = CompVisi.VideoCapture('sampleVid.webm')
#video file to out.avi


maxImageSizeForDetection = 200

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel( filer.path.join(filer.path.dirname(__file__), "..", "candide.npz"))

projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

modelParams = None
drawOverlay = False
#cap = CompVisi.VideoCapture(0)
writer = None
cameraImg = cap.read()[1]

textureImg = CompVisi.imread(image_name)
textureCoords = utils.getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)
renderer = FaceRendering.FaceRenderer(cameraImg, textureImg, textureCoords, mesh)



# Create a VideoCapture object and read from inumpyut file
# If the inumpyut is the camera, pass 0 instead of the video file name


# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ended,cameraImg = cap.read()
    if ended == False:
        break
	#process here
    shapes2D = utils.getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)
    if shapes2D is not None:
    	for shape2D in shapes2D:
		    #3D model parameter initialization
		    modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])

		    #3D model parameter optimization
		    modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]), verbose=0)

		    #rendering the model to an image
		    shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)
		    renderedImg = renderer.render(shape3D)

		    #blending of the rendered face with the image
		    mask = numpy.copy(renderedImg[:, :, 0])
		    renderedImg = ImageProcessing.colorTransfer(cameraImg, renderedImg, mask)
		    cameraImg = ImageProcessing.blendImages(renderedImg, cameraImg, mask)

    if writer is not None:
	    writer.write(cameraImg)
	    writer.write(cameraImg)
    if writer is None:
	    print("Starting video writer")
	    writer = CompVisi.VideoWriter(filer.path.join(filer.path.dirname(__file__), "..", "out.avi"),CompVisi.VideoWriter_fourcc('X', 'V', 'I', 'D'),25,(cameraImg.shape[1], cameraImg.shape[0]))




# When everything done, release the video capture object
cap.release()

# Clfileres all the frames
CompVisi.destroyAllWindows()





'''

while True:
    cameraImg = cap.read()[1]
    shapes2D = utils.getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)

    if shapes2D is not None:
        for shape2D in shapes2D:
            #3D model parameter initialization
            modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])

            #3D model parameter optimization
            modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]), verbfilere=0)

            #rendering the model to an image
            shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)
            renderedImg = renderer.render(shape3D)

            #blending of the rendered face with the image
            mask = numpy.copy(renderedImg[:, :, 0])
            renderedImg = ImageProcessing.colorTransfer(cameraImg, renderedImg, mask)
            cameraImg = ImageProcessing.blendImages(renderedImg, cameraImg, mask)

    if writer is not None:
	    writer.write(cameraImg)
    if writer is None:
	    print("Starting video writer")
	    writer = CompVisi.VideoWriter(filer.path.join(filer.path.dirname(__file__), "..", "out.avi"),
		                     CompVisi.VideoWriter_fourcc('X', 'V', 'I', 'D'),
		                     25,
		                     (cameraImg.shape[1], cameraImg.shape[0]))


'''

