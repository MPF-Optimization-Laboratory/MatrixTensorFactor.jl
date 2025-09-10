using BenchmarkTools
using Logging
using BlockTensorFactorization


global_logger(SimpleLogger(Warn))

fact = BlockTensorFactorization.factorize

function nnrankr_matrix((I, J), R)
    A = randn(I, R) .|> abs
    B = randn(R, J) .|> abs
    return A*B
end

options = (
    :rank => (2,3,4),
    :tolerance => (0.01),
    :converged => (RelativeError),
    :δ => 0.9999,
    :model => Tucker,
    #:constraints => nonnegative!,
)

n_subblock_n_momentum(Y) = fact(Y;
    do_subblock_updates=false,
    momentum=false,
    options...
)

y_subblock_n_momentum(Y) = fact(Y;
    do_subblock_updates=true,
    momentum=false,
    options...
)

n_subblock_y_momentum(Y) = fact(Y;
    do_subblock_updates=false,
    momentum=true,
    options...
)

y_subblock_y_momentum(Y) = fact(Y;
    do_subblock_updates=true,
    momentum=true,
    options...
)

I, J = 10, 10
R = 3

@benchmark n_subblock_n_momentum(Y) setup=(Y=array(Tucker1((I, J), R)))

@benchmark y_subblock_n_momentum(Y) setup=(Y=nnrankr_matrix((I, J), R))

@benchmark n_subblock_y_momentum(Y) setup=(Y=nnrankr_matrix((I, J), R))

@benchmark y_subblock_y_momentum(Y) setup=(Y=nnrankr_matrix((I, J), R))

performance_increase(old, new) = (old - new) / new * 100

time_decrease(old, new) = (old - new) / old * 100


###################


I, J, K = 10, 10, 10
R1, R2, R3 = 2, 3, 4

b = @benchmark n_subblock_n_momentum(Y) setup=(Y=Tucker((I, J, K), (R1, R2, R3))|>array)
display(b)
b = @benchmark y_subblock_n_momentum(Y) setup=(Y=Tucker((I, J, K), (R1, R2, R3))|>array)
display(b)
b = @benchmark n_subblock_y_momentum(Y) setup=(Y=Tucker((I, J, K), (R1, R2, R3))|>array)
display(b)
b = @benchmark y_subblock_y_momentum(Y) setup=(Y=Tucker((I, J, K), (R1, R2, R3))|>array)
display(b)

##########


I, J, K = 6, 6, 6
R = 3

b = @benchmark n_subblock_n_momentum(Y) setup=(Y=CPDecomposition((I, J, K), R)|>array)
display(b)
b = @benchmark y_subblock_n_momentum(Y) setup=(Y=CPDecomposition((I, J, K), R)|>array)
display(b)
b = @benchmark n_subblock_y_momentum(Y) setup=(Y=CPDecomposition((I, J, K), R)|>array)
display(b)
b = @benchmark y_subblock_y_momentum(Y) setup=(Y=CPDecomposition((I, J, K), R)|>array)
display(b)
