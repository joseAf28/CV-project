# CV-Project

The project was done in the context of the Computer Vision class. It was decomposed into two parts. 

The first part involved computing a reliable coordinate transformation (homography) $H_R$ for each frame $F_i$ in a video sequence relative to a reference frame $F_R$.

The second one builds upon the logic developed in the previous problem, adding 3D point filtering and rigid transformation estimation instead of homography. 

The detailed solution proposed for both problems is presented in the pdf file. Furthermore, this project had the specific requirement of not allowing import algorithms of the usual Computer Vision libraries (e.g., OpenCV)