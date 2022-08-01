## How to use it ##
0. Download preTrained model from my google drive (https://drive.google.com/file/d/1pxiAR5M4PMOEnLEhLHgCIuYrDt9zuTet/view?usp=sharing) and place it alognside ReadMe.md in parent folder
1. You should be able to automatically install the libraries by running `pip install -r requirements.txt`
2. Copy the picture of face in data folder and change the name of image in main.py line 16 AS-> image_name = filer.path.join(filer.path.dirname(__file__), "..", "data", "baby.webp") #change baby.webp to your image name 
3. Copy sample video in Main folder and change the name of video in main.py line 17 AS-> cap = CompVisi.VideoCapture('sampleVid.webm') ##change sampleVid.webp to your video
4. To start the program you will have to run a Python script named main.py from terminal by typing 'ipython main.py'


##Once that is finished and everything is initialized the Video start converting. For each captured Frame from video the following steps are taken:

1. The face region is detected and the facial landmarks are located.
2. The 3D models is fitted to the located landmarks.
3. The 3D models is rendered using pygame with the texture obtained during initialization.
4. The image of the rendered model is saved in out.avi in main folder

