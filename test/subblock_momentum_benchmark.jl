using BenchmarkTools
using BlockTensorDecomposition

fact = BlockTensorDecomposition.factorize

options = (
    :rank => 3,
    :tolerence => (1, 0.03),
    :converged => (GradientNNCone, RelativeError),
    :δ => 0.9,
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

@benchmark n_subblock_n_momentum(Y) setup=(Y=Tucker1((I, J), R))

@benchmark y_subblock_n_momentum(Y) setup=(Y=Tucker1((I, J), R))

@benchmark n_subblock_y_momentum(Y) setup=(Y=Tucker1((I, J), R))

@benchmark y_subblock_y_momentum(Y) setup=(Y=Tucker1((I, J), R))

performance_increase(old, new) = (old - new) / new * 100

time_decrease(old, new) = (old - new) / old * 100
