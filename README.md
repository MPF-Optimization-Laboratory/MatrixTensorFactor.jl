# MatrixTensorFactor.jl

Factorize a 3rd order tensor Y into a matrix A and 3rd order tensor B: Y=AB. Componentwise, this is:

Y[i,j,k] = sum_{r=1}^R ( A[i,r] * B[r,j,k] )
