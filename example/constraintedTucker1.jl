using BlockTensorDecomposition

fact = BlockTensorDecomposition.factorize

C = abs_randn(5, 11, 12)
A = abs_randn(10, 5)
Y = Tucker1((C, A))
Y = array(Y)

@time decomposition, stats_data, kwargs = fact(Y;
    rank=5,
    momentum=true,
    tolerence=(1, 0.03),
    converged=(GradientNNCone, RelativeError),
    constrain_init=true,
    constraints=nnegative!,
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError]
);

display(stats_data)

display(kwargs[:update])
