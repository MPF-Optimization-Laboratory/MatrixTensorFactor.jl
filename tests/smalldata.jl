using Pkg

Pkg.add("HDF5")
Pkg.add("SparseArrays")

using HDF5
using SparseArrays
using KernelDensity
using Plots
using MatrixTensorFactor

fid2 = h5open("testdata\\smalldata.h5ad","r")
coordinates = fid2["coordinates"] |> read
M = fid2["count_matrix"] |> read
close(fid2)

M = sparse(M) #cannot store a sparse matrix with h5ad (?)

m,n = 4180,4021 # number of cells x number of genes (features)
# Note we use n,m becuase these are stored in CSR format

# Each row of M is the values for a gene
# Ex. M[1,:] gives the values for the first gene
# The different columns corrispond to the coordinates the genes are sampled at:

xs, ys = coordinates[1,:], -coordinates[2,:]

begin
gene = 1 #181
# Plot the coordinates with darker points indicating larger values,
scatter(xs, ys, markeralpha=M[gene,:] ./ maximum(M[gene,:])) |> display

f = kde2d((xs, ys), M[gene,:])

heatmap(f.x, f.y, f.density) |> display
end

# Extract all genes and compile into a tensor
n_genes = 100 #n
J = K = 2^5 # Number of samples in each dimention
I = n_genes

Y = zeros(I, J, K) # Data tensor

xs_resample = range(f.x[begin], f.x[end], length=J)
ys_resample = range(f.y[begin], f.y[end], length=K)

for gene in 1:n_genes #number of genes
    f = kde2d((xs, ys), M[gene,:])
    Y[gene, :, :] = pdf(f, xs_resample, ys_resample)
end

heatmap(xs_resample, ys_resample, Y[7, :, :]) |> display

# Normalize the sum of each gene (horizontal) slice
Y_slices = eachslice(Y, dims=1)
slice_sums = sum.(Y_slices) # all 0.666 for some reason...
Y_slices ./= slice_sums

#using Statistics: median
#Y ./= median(Y[:])
###########

# Decomposition
R = 10
C, F, rel_errors, norm_grad, dist_Ncone = nnmtf(Y, R;tol=1e-6,maxiter=150, rescale_Y=false);

plot(rel_errors,yaxis=:log10) |> display
plot(norm_grad,yaxis=:log10) |> display
plot(dist_Ncone,yaxis=:log10) |> display

Y_hat = C*F
rel_error(Y,Y_hat)
mean_rel_error(Y, Y_hat, dims=1)

heatmap(C) |> display
for r in 1:R
    heatmap(F[r,:,:], title="cell type $r") |> display
end

for r1 in 1:R
    for r2 in 1:r1-1
        scatter(C[:,r1], C[:,r2], title="cell type $r1 vs. $r2",
            xlabel="type $r1",
            ylabel="type $r2",
            ) |> display
    end
end

begin
    gene = 50#3807
    heatmap(Y_hat[gene,:,:]) |> display
    heatmap!(Y[gene,:,:]) |> display
    scatter(Y[gene,:,:][:],Y_hat[gene,:,:][:])
    plot!([0,maximum(Y[gene,:,:])],[0,maximum(Y[gene,:,:])]) |> display
    rel_error(Y_hat[gene,:,:],Y[gene,:,:])
end
