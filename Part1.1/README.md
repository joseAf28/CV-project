## Core Concepts of the Project

1. ## Object detection and tracking

   1. Identify and track moving objects (vehicles and pedestrians) in video frames.
   2. Using YOLO, we'll have a set of trajectories for detected objects, represented as seqeunces of pixel coordinates in the camera frame

2. ## Coordinate Mapping

   1. Map object trajectories from the camera's pixel coordinates to a georeferenced Google Maps image
   2. Camera captures a perspective view of the real world, while Google Maps uses an orthographic (top-down) view.
   3. The **homography**, the transformation that relates two planes, to achieve this mapping 