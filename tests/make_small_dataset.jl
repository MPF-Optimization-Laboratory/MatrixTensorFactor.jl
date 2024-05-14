using Pkg

Pkg.add("HDF5")
Pkg.add("SparseArrays")

using HDF5
using SparseArrays

fid = h5open("testdata\\very_small_9_5.h5ad", "r")
coordinates = fid["obsm"]["spatial"] |> read
count = fid["layers"]["count"] |> read
close(fid)

m,n = 4180,4021 # number of cells x number of genes (features)
# Note we use n,m becuase these are stored in CSR format
sparse_M = SparseMatrixCSC(n,m,count["indptr"] .+ 1, count["indices"] .+ 1, count["data"])

dense_M = collect(M) # Dense
dense_Int16_M = convert.(Int16, dense_M) # convert Int64 to Int16

fid2 = h5open("testdata\\smalldata.h5ad","cw")
fid2["coordinates"] = coordinates
fid2["count_matrix"] = dense_Int16_M
close(fid2)
