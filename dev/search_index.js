var documenterSearchIndex = {"docs":
[{"location":"MatrixTensorFactor/#Exported-Terms","page":"Exported Terms","title":"Exported Terms","text":"","category":"section"},{"location":"MatrixTensorFactor/","page":"Exported Terms","title":"Exported Terms","text":"MatrixTensorFactor","category":"page"},{"location":"MatrixTensorFactor/#MatrixTensorFactor","page":"Exported Terms","title":"MatrixTensorFactor","text":"Matrix-Tensor Factorization\n\n\n\n\n\n","category":"module"},{"location":"MatrixTensorFactor/#Types","page":"Exported Terms","title":"Types","text":"","category":"section"},{"location":"MatrixTensorFactor/","page":"Exported Terms","title":"Exported Terms","text":"Abstract3Tensor","category":"page"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.Abstract3Tensor","page":"Exported Terms","title":"MatrixTensorFactor.Abstract3Tensor","text":"Alias for an AbstractArray{T, 3}.\n\n\n\n\n\n","category":"type"},{"location":"MatrixTensorFactor/#Nonnegative-Matrix-Tensor-Factorization","page":"Exported Terms","title":"Nonnegative Matrix-Tensor Factorization","text":"","category":"section"},{"location":"MatrixTensorFactor/","page":"Exported Terms","title":"Exported Terms","text":"nnmtf\nnnmtf_proxgrad_online","category":"page"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.nnmtf","page":"Exported Terms","title":"MatrixTensorFactor.nnmtf","text":"nnmtf(Y::Abstract3Tensor, R::Integer; kwargs...)\n\nNon-negatively matrix-tensor factorizes an order 3 tensor Y with a given \"rank\" R.\n\nFactorizes Y approx A B where displaystyle Yijk approx sum_r=1^R Air*Brjk and the factors A B geq 0 are nonnegative.\n\nNote there may NOT be a unique optimal solution\n\nArguments\n\nY::Abstract3Tensor: tensor to factorize\nR::Integer: rank to factorize Y (size(A)[2] and size(B)[1])\n\nKeywords\n\nmaxiter::Integer=100: maxmimum number of iterations\ntol::Real=1e-3: desiered tolerance for the convergence criterion\nrescale_AB::Bool=true: scale B at each iteration so that the factors (horizontal slices) have similar 3-fiber sums.\nrescale_Y::Bool=true: Preprocesses the input Y to have normalized 3-fiber sums (on average), and rescales the final B so Y=A*B.\nnormalize::Symbol=:fibres: part of B that should be normalized (must be in IMPLIMENTED_NORMALIZATIONS)\nprojection::Symbol=:nnscale: constraint to use and method for enforcing it (must be in IMPLIMENTED_PROJECTIONS)\ncriterion::Symbol=:ncone: how to determine if the algorithm has converged (must be in IMPLIMENTED_CRITERIA)\nstepsize::Symbol=:lipshitz: used for the gradient decent step (must be in IMPLIMENTED_STEPSIZES)\nmomentum::Bool=false: use momentum updates\ndelta::Real=0.9999: safeguard for maximum amount of momentum (see eq 3.5 Xu & Yin 2013)\nR_max::Integer=size(Y)[1]: maximum rank to try if R is not given\nprojectionA::Symbol=projection: projection to use on factor A (must be in IMPLIMENTED_PROJECTIONS)\nprojectionB::Symbol=projection: projection to use on factor B (must be in IMPLIMENTED_PROJECTIONS)\n\nReturns\n\nA::Matrix{Float64}: the matrix A in the factorization Y ≈ A * B\nB::Array{Float64, 3}: the tensor B in the factorization Y ≈ A * B\nrel_errors::Vector{Float64}: relative errors at each iteration\nnorm_grad::Vector{Float64}: norm of the full gradient at each iteration\ndist_Ncone::Vector{Float64}: distance of the -gradient to the normal cone at each iteration\nIf R was estimated, also returns the optimal R::Integer\n\nImplimentation of block coordinate decent updates\n\nWe calculate the partial gradients and corresponding Lipshitz constants like so:\n\nbeginalign\n  boldsymbolP^tqr =textstylesum_jk boldsymbolmathscrB^nqjk boldsymbolmathscrB^nrjk\n  boldsymbolQ^tir =textstylesum_jkboldsymbolmathscrYijk boldsymbolmathscrB^nrjk \n  nabla_A f(boldsymbolA^tboldsymbolmathscrB^t) = boldsymbolA^t boldsymbolP^t - boldsymbolQ^t \n  L_A = leftlVert boldsymbolP^t rightrVert_2\nendalign\n\nSimilarly for boldsymbolmathscrB:\n\nbeginalign\n  boldsymbolT^t+1=(boldsymbolA^t+frac12)^top boldsymbolA^t+frac12\n  boldsymbolmathscrU^t+1=(boldsymbolA^t+frac12)^top boldsymbolmathscrY \n  nabla_boldsymbolmathscrB f(boldsymbolA^t+frac12boldsymbolmathscrB^t) =  boldsymbolT^t+1 boldsymbolmathscrB^t - boldsymbolmathscrU^t+1 \n  L_B = leftlVert boldsymbolT^t+1 rightrVert_2\nendalign\n\nTo ensure the iterates stay \"close\" to normalized, we introduce a renormalization step after the projected gradient updates:\n\nbeginalign\n    boldsymbolC rr=frac1Jtextstylesum_jk boldsymbolmathscrB^t+frac12rjk\n    boldsymbolA^t+1= boldsymbolA^t+frac12 boldsymbolC\n    boldsymbolmathscrB^t+1= (boldsymbolC^t+1)^-1boldsymbolmathscrB^t+frac12\nendalign\n\nWe typicaly use the following convergence criterion:\n\nd(-nabla ell(boldsymbolA^tboldsymbolmathscrB^t) N_mathcalC(boldsymbolA^tboldsymbolmathscrB^t))^2leqdelta^2 R(I+JK)\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.nnmtf_proxgrad_online","page":"Exported Terms","title":"MatrixTensorFactor.nnmtf_proxgrad_online","text":"nnmtf using Online proximal (projected) gradient decent alternating through blocks (BCD)\n\nUpdates Y each iteration with a new sample\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#Implimented-Factorization-Options","page":"Exported Terms","title":"Implimented Factorization Options","text":"","category":"section"},{"location":"MatrixTensorFactor/","page":"Exported Terms","title":"Exported Terms","text":"IMPLIMENTED_OPTIONS\nIMPLIMENTED_NORMALIZATIONS\nIMPLIMENTED_PROJECTIONS\nIMPLIMENTED_CRITERIA\nIMPLIMENTED_STEPSIZES","category":"page"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.IMPLIMENTED_OPTIONS","page":"Exported Terms","title":"MatrixTensorFactor.IMPLIMENTED_OPTIONS","text":"Lists all implimented options\n\n\n\n\n\n","category":"constant"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.IMPLIMENTED_NORMALIZATIONS","page":"Exported Terms","title":"MatrixTensorFactor.IMPLIMENTED_NORMALIZATIONS","text":"IMPLIMENTED_NORMALIZATIONS::Set{Symbol}\n\n:fibre: set sum_k=1^K Brjk = 1 for all r j, or when projection==:nnscale,   set sum_j=1^Jsum_k=1^K Brjk = J for all r\n:slice: set sum_j=1^Jsum_k=1^K Brjk = 1 for all r\n:nothing: does not enforce any normalization of B\n\n\n\n\n\n","category":"constant"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.IMPLIMENTED_PROJECTIONS","page":"Exported Terms","title":"MatrixTensorFactor.IMPLIMENTED_PROJECTIONS","text":"IMPLIMENTED_PROJECTIONS::Set{Symbol}\n\n:nnscale: Two stage block coordinate decent; 1) projected gradient decent onto nonnegative   orthant, 2) shift any weight from B to A according to normalization. Equivilent to   :nonnegative when normalization==:nothing.\n:simplex: Euclidian projection onto the simplex blocks accoring to normalization\n:nonnegative: zero out negative entries\n\n\n\n\n\n","category":"constant"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.IMPLIMENTED_CRITERIA","page":"Exported Terms","title":"MatrixTensorFactor.IMPLIMENTED_CRITERIA","text":"IMPLIMENTED_CRITERIA::Set{Symbol}\n\n:ncone: vector-set distance between the -gradient of the objective and the normal cone\n:iterates: A,B before and after one iteration are close in L2 norm\n:objective: objective before and after one iteration is close\n\n\n\n\n\n","category":"constant"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.IMPLIMENTED_STEPSIZES","page":"Exported Terms","title":"MatrixTensorFactor.IMPLIMENTED_STEPSIZES","text":"IMPLIMENTED_STEPSIZES::Set{Symbol}\n\n:lipshitz: gradient step 1/L for lipshitz constant L\n:spg: spectral projected gradient stepsize\n\n\n\n\n\n","category":"constant"},{"location":"MatrixTensorFactor/#Constants","page":"Exported Terms","title":"Constants","text":"","category":"section"},{"location":"MatrixTensorFactor/","page":"Exported Terms","title":"Exported Terms","text":"MAX_STEP\nMIN_STEP","category":"page"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.MAX_STEP","page":"Exported Terms","title":"MatrixTensorFactor.MAX_STEP","text":"MAX_STEP = 1e10\n\nMaximum step size allowed for spg stepsize method.\n\n\n\n\n\n","category":"constant"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.MIN_STEP","page":"Exported Terms","title":"MatrixTensorFactor.MIN_STEP","text":"MIN_STEP = 1e-10\n\nMinimum step size allowed for spg stepsize method.\n\n\n\n\n\n","category":"constant"},{"location":"MatrixTensorFactor/#Kernel-Density-Estimation","page":"Exported Terms","title":"Kernel Density Estimation","text":"","category":"section"},{"location":"MatrixTensorFactor/#Constants-2","page":"Exported Terms","title":"Constants","text":"","category":"section"},{"location":"MatrixTensorFactor/","page":"Exported Terms","title":"Exported Terms","text":"DEFAULT_N_SAMPLES\nDEFAULT_ALPHA","category":"page"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.DEFAULT_N_SAMPLES","page":"Exported Terms","title":"MatrixTensorFactor.DEFAULT_N_SAMPLES","text":"DEFAULT_N_SAMPLES = 64::Integer\n\nNumber of samples to use when standardizing a vector of density estimates.\n\n\n\n\n\n","category":"constant"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.DEFAULT_ALPHA","page":"Exported Terms","title":"MatrixTensorFactor.DEFAULT_ALPHA","text":"DEFAULT_ALPHA = 0.9::Real\n\nSmoothing parameter for calculating a kernel's bandwidth.\n\n\n\n\n\n","category":"constant"},{"location":"MatrixTensorFactor/#1D","page":"Exported Terms","title":"1D","text":"","category":"section"},{"location":"MatrixTensorFactor/","page":"Exported Terms","title":"Exported Terms","text":"default_bandwidth\nmake_densities\nmake_densities2d\nstandardize_KDEs\nstandardize_2d_KDEs\nfilter_inner_percentile\nfilter_2d_inner_percentile","category":"page"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.default_bandwidth","page":"Exported Terms","title":"MatrixTensorFactor.default_bandwidth","text":"default_bandwidth(data; alpha=0.9, inner_percentile=100)\n\nCoppied from KernelDensity since this function is not exported. I want access to it so that the same bandwidth can be used for different densities for the same measurements.\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.make_densities","page":"Exported Terms","title":"MatrixTensorFactor.make_densities","text":"make_densities(s::Sink; kwargs...)\nmake_densities(s::Sink, domains::AbstractVector{<:AbstractVector}; kwargs...)\n\nEstimates the densities for each measurement in a Sink.\n\nWhen given domains, a list where each entry is a domain for a different measurement, resample the kernel on this domain.\n\nParameters\n\nbandwidths::AbstractVector{<:Real}: list of bandwidths used for each measurement's\n\ndensity estimation\n\ninner_percentile::Integer=100: value between 0 and 100 that filters out each measurement\n\nby using the inner percentile range. This can help remove outliers and focus in on where the bulk of the data is.\n\nReturns\n\ndensity_estimates::Vector{UnivariateKDE}\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.make_densities2d","page":"Exported Terms","title":"MatrixTensorFactor.make_densities2d","text":"makedensities2d(s::Sink; kwargs...) makedensities2d(s::Sink, domains::AbstractVector{<:AbstractVector}; kwargs...)\n\nSimilar to make_densities but performs the KDE on 2 measurements jointly.\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.standardize_KDEs","page":"Exported Terms","title":"MatrixTensorFactor.standardize_KDEs","text":"standardize_KDEs(KDEs::AbstractVector{UnivariateKDE}; n_samples=DEFAULT_N_SAMPLES,)\n\nResample the densities so they all are sampled from the same domain.\n\n\n\n\n\nResample the densities within each sink/source so that like-measurements use the same scale.\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.standardize_2d_KDEs","page":"Exported Terms","title":"MatrixTensorFactor.standardize_2d_KDEs","text":"standardize_2d_KDEs(KDEs::AbstractVector{BivariateKDE}; n_samples=DEFAULT_N_SAMPLES,)\n\nResample the densities so they all are sampled from the same x and y coordinates.\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.filter_inner_percentile","page":"Exported Terms","title":"MatrixTensorFactor.filter_inner_percentile","text":"Filters elements so only the ones in the inner P percentile remain. See filter_2d_inner_percentile.\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.filter_2d_inner_percentile","page":"Exported Terms","title":"MatrixTensorFactor.filter_2d_inner_percentile","text":"Filters 2d elements so only the ones in the inner P percentile remain. See filter_inner_percentile.\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#2D","page":"Exported Terms","title":"2D","text":"","category":"section"},{"location":"MatrixTensorFactor/","page":"Exported Terms","title":"Exported Terms","text":"repeatcoord\nkde2d\ncoordzip","category":"page"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.repeatcoord","page":"Exported Terms","title":"MatrixTensorFactor.repeatcoord","text":"repeatcoord(coordinates, values)\n\nRepeates coordinates the number of times given by values.\n\nBoth lists should be the same length.\n\nExample\n\ncoordinates = [(0,0), (1,1), (1,2)] values = [1, 3, 2] repeatcoord(coordinates, values)\n\n[(0,0), (1,1), (1,1), (1,1), (1,2), (1,2)]\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.kde2d","page":"Exported Terms","title":"MatrixTensorFactor.kde2d","text":"kde2d((xs, ys), values)\n\nPerforms a 2d KDE based on two lists of coordinates, and the value at those coordinates. Input ––-\n\nxs, ys::Vector{Real}: coordinates/locations of samples\nvalues::Vector{Integer}: value of the sample\n\nReturns\n\nf::BivariateKDE use f.x, f.y for the location of the (re)sampled KDE,\n\nand f.density for the sample values of the KDE\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.coordzip","page":"Exported Terms","title":"MatrixTensorFactor.coordzip","text":"coordzip(rcoords)\n\nZips the \"x\" and \"y\" values together into a list of x coords and y coords. Example –––- coordzip([(0,0), (1,1), (1,1), (1,1), (1,2), (1,2)])\n\n[[0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 2, 2]]\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#Approximations","page":"Exported Terms","title":"Approximations","text":"","category":"section"},{"location":"MatrixTensorFactor/","page":"Exported Terms","title":"Exported Terms","text":"d_dx\nd2_dx2\ncurvature\nstandard_curvature","category":"page"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.d_dx","page":"Exported Terms","title":"MatrixTensorFactor.d_dx","text":"d_dx(y::AbstractVector{<:Real})\n\nApproximates the 1nd derivative of a function using only given samples y of that function.\n\nAssumes y came from f(x) where x was an evenly sampled, unit intervel grid. Note the approximation uses centered three point finite differences for the next-to-end points, and foward/backward three point differences for the begining/end points respectively. The remaining interior points use five point differences.\n\nWill use the largest order method possible by defult (currently 5 points), but can force a specific order method with the keyword order. See d2_dx2.\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.d2_dx2","page":"Exported Terms","title":"MatrixTensorFactor.d2_dx2","text":"d2_dx2(y::AbstractVector{<:Real}; order::Integer=length(y))\n\nApproximates the 2nd derivative of a function using only given samples y of that function.\n\nAssumes y came from f(x) where x was an evenly sampled, unit intervel grid. Note the approximation uses centered three point finite differences for the next-to-end points, and foward/backward three point differences for the begining/end points respectively. The remaining interior points use five point differences.\n\nWill use the largest order method possible by defult (currently 5 points), but can force a specific order method with the keyword order. See d_dx.\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.curvature","page":"Exported Terms","title":"MatrixTensorFactor.curvature","text":"curvature(y::AbstractVector{<:Real})\n\nApproximates the signed curvature of a function given evenly spaced samples.\n\nUses d_dx and d2_dx2 to approximate the first two derivatives.\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.standard_curvature","page":"Exported Terms","title":"MatrixTensorFactor.standard_curvature","text":"standard_curvature(y::AbstractVector{<:Real})\n\nApproximates the signed curvature of a function, scaled to the unit box 01^2.\n\nSee curvature.\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#Other-Functions","page":"Exported Terms","title":"Other Functions","text":"","category":"section"},{"location":"MatrixTensorFactor/","page":"Exported Terms","title":"Exported Terms","text":"*(::AbstractMatrix, ::Abstract3Tensor)\ncombined_norm\ndist_to_Ncone\nrel_error\nmean_rel_error\nresidual","category":"page"},{"location":"MatrixTensorFactor/#Base.:*-Tuple{AbstractMatrix, AbstractArray{T, 3} where T}","page":"Exported Terms","title":"Base.:*","text":"Base.*(A::AbstractMatrix, B::Abstract3Tensor)\n\nComputes the Abstract3Tensor C where C_ijk = sum_l=1^L A_il * B_ljk.\n\nWhen the third dimention of B has length 1, this is equivilent to the usual matrix-matrix multiplication. For this reason, we resuse the same symbol.\n\nThis is equivilent to the 1-mode product B times_1 A.\n\n\n\n\n\n","category":"method"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.combined_norm","page":"Exported Terms","title":"MatrixTensorFactor.combined_norm","text":"combined_norm(u, v, ...)\n\nCompute the combined norm of the arguments as if all arguments were part of one large array.\n\nThis is equivilent to norm(cat(u, v, ...)), but this implimentation avoids creating an intermediate array.\n\nu = [3 0]\nv = [0 4 0]\ncombined_norm(u, v)\n\n# output\n\n5.0\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.dist_to_Ncone","page":"Exported Terms","title":"MatrixTensorFactor.dist_to_Ncone","text":"dist_to_Ncone(grad_A, grad_B, A, B)\n\nCalculate the distance of the -gradient to the normal cone of the positive orthant.\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.rel_error","page":"Exported Terms","title":"MatrixTensorFactor.rel_error","text":"rel_error(x, xhat)\n\nCompute the relative error between x (true value) and xhat (its approximation).\n\nThe relative error is given by:\n\nfraclVert hatx - x rVertlVert x rVert\n\nSee also mean_rel_error.\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.mean_rel_error","page":"Exported Terms","title":"MatrixTensorFactor.mean_rel_error","text":"mean_rel_error(X, Xhat; dims=(1,2))\n\nCompute the mean relative error between the dims-order slices of X and Xhat.\n\nThe mean relative error is given by:\n\nfrac1Nsum_j=1^NfraclVert hatX_j - X_j rVertlVert X_j rVert\n\nSee also rel_error.\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#MatrixTensorFactor.residual","page":"Exported Terms","title":"MatrixTensorFactor.residual","text":"residual(Yhat, Y; normalize=:nothing)\n\nWrapper to use the relative error calculation according to the normalization used.\n\nnormalize==:nothing: entry-wise L2 relative error between the two arrays\nnormalize==:fibres: average L2 relative error between all 3-fibres\nnormalize==:slices: average L2 relative error between all 1-mode slices\n\nSee also rel_error, mean_rel_error.\n\n\n\n\n\n","category":"function"},{"location":"MatrixTensorFactor/#Index","page":"Exported Terms","title":"Index","text":"","category":"section"},{"location":"MatrixTensorFactor/","page":"Exported Terms","title":"Exported Terms","text":"","category":"page"},{"location":"#Matrix-Tensor-Factorization","page":"Matrix Tensor Factorization","title":"Matrix Tensor Factorization","text":"","category":"section"},{"location":"","page":"Matrix Tensor Factorization","title":"Matrix Tensor Factorization","text":"Depth = 3","category":"page"},{"location":"#How-setup-the-environment","page":"Matrix Tensor Factorization","title":"How setup the environment","text":"","category":"section"},{"location":"#Recomended-Method","page":"Matrix Tensor Factorization","title":"Recomended Method","text":"","category":"section"},{"location":"","page":"Matrix Tensor Factorization","title":"Matrix Tensor Factorization","text":"Run julia\nAdd the package with pkg> add https://github.com/MPF-Optimization-Laboratory/MatrixTensorFactor.jl.git (use julia> ] to get to the package manager)\nImport with using MatrixTensorFactor","category":"page"},{"location":"","page":"Matrix Tensor Factorization","title":"Matrix Tensor Factorization","text":"OR","category":"page"},{"location":"#In-Browser","page":"Matrix Tensor Factorization","title":"In Browser","text":"","category":"section"},{"location":"","page":"Matrix Tensor Factorization","title":"Matrix Tensor Factorization","text":"Go to https://github.com/MPF-Optimization-Laboratory/MatrixTensorFactor.jl\nClick \"<> Code\" and press \"+\" to \"Create a codespace on main\". It make take a few moments to set up.\nOpen the command palett with Ctrl+Shift+P (Windows) or Cmd+Shift+P (Mac)\nEnter >Julia: Start REPL\nIn the REPL, resolve any dependency issues with pkg> resolve and pkg> instantiate (use julia> ] to get to the package manager). It may take a few minutes to download dependencies.","category":"page"},{"location":"","page":"Matrix Tensor Factorization","title":"Matrix Tensor Factorization","text":"Run one of the example files by opening the file and pressing the triangular \"run\" button, or >Julia: Execute active File in REPL.","category":"page"},{"location":"","page":"Matrix Tensor Factorization","title":"Matrix Tensor Factorization","text":"OR","category":"page"},{"location":"#On-your-own-device","page":"Matrix Tensor Factorization","title":"On your own device","text":"","category":"section"},{"location":"","page":"Matrix Tensor Factorization","title":"Matrix Tensor Factorization","text":"Clone the repo at https://github.com/MPF-Optimization-Laboratory/MatrixTensorFactor.jl\nNavigate to the root of the repository in a terminal and run julia\nActivate the project with pkg> activate . (use julia> ] to get to the package manager)\nresolve any dependency issues with pkg> resolve","category":"page"},{"location":"#Importing-the-package","page":"Matrix Tensor Factorization","title":"Importing the package","text":"","category":"section"},{"location":"","page":"Matrix Tensor Factorization","title":"Matrix Tensor Factorization","text":"Type julia> using MatrixTensorFactor","category":"page"},{"location":"#Examples","page":"Matrix Tensor Factorization","title":"Examples","text":"","category":"section"},{"location":"","page":"Matrix Tensor Factorization","title":"Matrix Tensor Factorization","text":"smalldata: decomposes a subset of genomic data to identify gene profiles for learned cell types syntheticdata1d.jl: generate multiple mixtures of 3, 1d probability distributions syntheticdata2d.jl: generate multiple mixtures of 3, 2d probability distributions","category":"page"},{"location":"#MatrixTensorFactor","page":"Matrix Tensor Factorization","title":"MatrixTensorFactor","text":"","category":"section"},{"location":"","page":"Matrix Tensor Factorization","title":"Matrix Tensor Factorization","text":"Defines the main factorization function nnmtf and related mathematical functions. See the full list of Exported Terms.","category":"page"},{"location":"#Index","page":"Matrix Tensor Factorization","title":"Index","text":"","category":"section"},{"location":"","page":"Matrix Tensor Factorization","title":"Matrix Tensor Factorization","text":"","category":"page"}]
}
