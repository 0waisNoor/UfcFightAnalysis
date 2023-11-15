<h1>Ufc punch counter</h1>
How the code works in a nutshell is it finetunes a R-CNN nueral network called detectron2 using a custom image dataset I made with images of fighters with labelled bounding boxes around each fighter.
A MoveNet pose detection nueral network is then used to detect the fighter's left hip and extract the color of their shorts which is then used for maintaining seperate punch counters for each fighter using shorts color for fighter classification.

<h2>How to run the code</h2>
1. Train the model using train.ipynb <br>
2. Add in a video of a ufc match called input.mp4 <br>
3. Run videoPunchCounter.py to output a video that counts the number of punches in the match <br>
4. Upload 2 cropped image of both the fighters from the match named fighter1.png and fighter2.png. Run videoPunchCounterV2.ipynb to output a video that counter the number of punches by <b>each fighter</b>.  <br>
