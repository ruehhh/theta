# theta

Tools for creating absolute value histograms for the Siegel theta functions of rank 2
See e.g. https://mathworld.wolfram.com/SiegelThetaFunction.html

Created for a paper in preparation with Jens Marklof and Tariq Osman concerning the tails of these distributions.

The function takes the form of a sum over the integer lattice Z^2 and takes as arguments:
1) A complex valued symmetric matrix, whose imaginary part is positive definite (the Siegel Upper-Half Plane)
2) Two vectors x and y in R^2
3) A real number t

The Siegel theta function satisfies a number of invariance properties and can be viewed as a periodic function on a smaller fundamental domain.

This fundamental domain has an explicit formulation in rank 2. See https://link.springer.com/article/10.1007/BF01342938.

The functions generate_theta and generate_theta_0 generate random elements of this fundamental domain by sampling the Siegel upper-half plane and then applying the criteria in the paper above.
In the former, the sample is uniform on the fundamental domain, in the second the sample is uniform on the orbit of x = (0,0), y = (0,0).
