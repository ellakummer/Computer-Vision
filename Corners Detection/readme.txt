4 functions : 

1) rgb2gray : 

We use the Y channel of YIQ model for every pixel of the image:
0.299*r + 0.587*g + 0.114*b

2) smooth1D:

We form a 1D horizontal gaussian filter of size 3*sigma :
For acceptable accuracy, kernels are generally truncated so that the
discarded samples are less than 1/1000 of the peak value. So ater calculating this boundary with the gaussian formula, we find a size of 3*sigma is sufficient. 

Then we convolve the 1D filter with the image, using the function convolve1D.

To finish, we normalize.
We convolve a matrix made of ones with the gaussian filter : matrix of weights.
Then we divide our convolved image by our matrix of weights. 

3) smooth2D :

We smooth the image along the vertical direction:
we apply the smooth1D function to our image. 

We smooth the image along the horizontal direction:
we apply the smooth1D function to the output of the last step, but transposed. 

Then we just transpose again our image to have it in the "right direction".

4) harris :

We compute Ix and Iy by finite differences:
For Ix, we convolve the image with the array [0.5 0 -0.5], and we do the same for Iy but with the transposed image. 

We compute Ix2, Iy2, or Ixy:
For Ix2 and Iy2, we square all the elements of the matrixes.
for Ixy, we create a matrix, where each component is the results of the multiplication of this components in the matrixes Ix and Iy. 

We smooth the squared derivatives:
We just apply the function Smooth2D to Ix2, Iy2, Ixy.
We have sIx2, sIy2, sIxy as output.

We compute the cornerness function R:
R = ((sIx2*sIy2)-(sIxy*sIxy)) - (k*((sIx2+sIy2)**2)), with k = 0.04
(apply the formula to every component of the matrixes, to create this new matrix R)

We mark local maxima as corner candidates and we perform quadratic approximation to local corners up to subpixel accuracy:
For every component of the matrix R, we compared its value to its neighbors values. If it is the maximum, it is a corner candidate and we perform the quadratic approximation up to subpixel accuracy. 

We perform thresholding and discard weak corners:
If a component is a corner candidate and its R value is bigger than the threshold, we add it to the corners.
In reality we had the triple (x-value of quadratic approximation to local corners upto sub-pixel accuracy, y-value of quadratic approximation to local corners upto sub-pixel accuracy,R value of the component).  


To finish ) 

I find 233 corners instead of 239.
The x- and y-values are corrects, but the R-values are slightly different (slightly under the right one), so this could explain why I have 6 pixels missing. 
Unfortunately, I couldn't find where the mistake for the R value really comes from. 

