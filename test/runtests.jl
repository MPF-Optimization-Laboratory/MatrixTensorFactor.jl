"""
Unit tests
"""

using Test

using Random
using LinearAlgebra

using BlockTensorDecomposition

const VERBOSE = true

@testset verbose=VERBOSE "BlockTensorDecomposition" begin

    @testset "Utils" begin

        @testset "interlace" begin
        @test interlace(1:10,10:15) == [1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7, 8, 9, 10]
        @test interlace(1:5,10:20) == [1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 15, 16, 17, 18, 19, 20]
        @test interlace(1:3, ("this", "that", "other")) == Any[1, "this", 2, "that", 3, "other"]
        end

        @testset "multifoldl" begin
            @test multifoldl((+,*,-), (2,3,4,5)) == 15 # ((2 + 3) * 4) - 5
            @test multifoldl([+,*,-], [2,3,4,5]) == 15
            @test multifoldl((*,), [2,3]) == 6
        end

        @testset "norm2" begin
            @test norm2([3, 4]) == 25
            @test norm2(1:10) == sum((1:10) .^ 2)
        end

        @testset "geomean" begin
            @test geomean(5) == 5
            @test geomean(2, 1/2) == 1
            @test geomean((1,2,3)) ≈ 1.8171205928321397
        end

        @testset "getnotindex" begin
            @test getnotindex(1:10, 5) == [1, 2, 3, 4, 6, 7, 8, 9, 10]
            @test getnotindex(1:10, (5, 8)) == [1, 2, 3, 4, 6, 7, 9, 10]
            @test getnotindex(1:10, [5, 8]) == [1, 2, 3, 4, 6, 7, 9, 10]
            @test getnotindex(1:10, 4:6) == [1, 2, 3, 7, 8, 9, 10]
        end

        @testset "proj_one_hot" begin
            @test proj_one_hot([-6,-4,0,2]) == [0, 0, 0, 1]
            @test proj_one_hot([-6,-4,-1,-2]) == [0, 0, 1, 0]
            @test proj_one_hot([-6.1,-4.2,-1.3,-2.4]) == [0., 0., 1., 0.]
        end

        @testset "all_recursive" begin
            @test all_recursive(x -> typeof(x) <: Function, nonnegative!) == true
            @test all_recursive(x -> typeof(x) <: Function, [nonnegative!, sin]) == true
            @test all_recursive(x -> typeof(x) <: Function, [(nonnegative!,), sin]) == true
            @test all_recursive(x -> x > 1, []) == true
            @test all_recursive(x -> x > 1, [[],[],[[]]]) == true
            @test all_recursive(x -> x > 0, [[[[]]], [], [1], [[2]]]) == true
            @test all_recursive(x -> x > 0, [[[[]]], [], [1], [[-1]]]) == false
            @test all_recursive(x -> x > 0, [1:10, 2:11, 3:5, [1:3, 2]]) == true
            @test all_recursive(x -> x > 0, ((1,), ((2,),), ((3,),(4,)))) == true
            @test all_recursive(x -> x > 0, (1:10, 2:11, 3:5, [1:3, 2])) == true
            @test all_recursive(x -> x > 0, [1:10, 2:11, 3:5, (1:3, 2)]) == true
            @test all_recursive(x -> x > 0, [1:10, 2:11, [1:3, 2], 3:5]) == true
            @test all_recursive(x -> x > 0, collect.([1:10, 2:11, 3:5, [1:3, 2]])) == true
            @test all_recursive(x -> x < 11, [1:10, 2:11, 3:5, [1:3, 2]]) == false
        end

        @testset "eachfibre" begin
            A = reshape(collect(1:60), 3,4,5)
            @test eachfibre(A; n=1) == eachslice(A; dims=(2,3))
            @test eachfibre(A; n=2) == eachslice(A; dims=(1,3))
            @test eachfibre(A; n=3) == eachslice(A; dims=(1,2))
        end
    end

    @testset "Products" begin
        A = randn(10, 20)
        B = randn(10, 20)
        @test all(slicewise_dot(A, A) .≈ A*A') # test this separately since it uses a different routine when the argument is the same
        @test all(slicewise_dot(A, B) .≈ A*B')
    end

@testset "Constraints" begin
    @testset "L1" begin
        v = collect(1:5)
        l1normalize!(v)
        @test v == [0, 0, 0, 0, 1]

        v = Vector{Float64}([1, -1, 1, 1])
        l1normalize!(v)
        @test v == [1, -1, 1, 1] / 4

        v = [1, -1, 1, 1]
        @test_broken l1normalize!(v) # v can only hold Int, so there is an error here

        A = Array{Float64}(reshape(1:12, 3,4))
        l1normalize_rows!(A)
        @test all(A .≈ [0 0 0 1; 0 0 0 1; 0 0 0 1])

        A = Array{Float64}(reshape(1:12, 3,4))
        l1normalize_cols!(A)
        @test all(A .≈ [0 0 0 0; 0 0 0 0; 1 1 1 1])

        A = Array{Float64}(ones(2, 3))
        l1normalize_rows!(A)
        @test all(A .≈ ones(2, 3) / 3)

        A = Array{Float64}(ones(2, 3))
        l1normalize_cols!(A)
        @test all(A .≈ ones(2, 3) / 2)

        A = Array{Float64}(reshape(1:12, 3,2,2))
        l1scale_1slices!(A)
        @test A ≈ [0.045454545454545456 0.18181818181818182; 0.07692307692307693 0.19230769230769232; 0.1 0.2;;; 0.3181818181818182 0.45454545454545453; 0.3076923076923077 0.4230769230769231; 0.3 0.4]

        A = randn(3,5,10) .|> abs
        l1scale_average12slices!(A)
        @test all(sum(A; dims=(2,3)) .≈ 5)
        @test check(l1scale_average12slices!, A)
    end

    @testset "L2" begin
        v = Vector{Float64}([1, 1])
        l2normalize!(v)
        @test v == [1, 1] / sqrt(2)

        v= Vector{Float64}([1, -1, 1, 1])
        l2normalize!(v)
        @test v == [1, -1, 1, 1] / 2

        A = Array{Float64}(ones(2, 3))
        l2normalize_rows!(A)
        @test all(A .≈ ones(2, 3) / √3)

        A = Array{Float64}(ones(2, 3))
        l2normalize_cols!(A)
        @test all(A .≈ ones(2, 3) / √2)
    end

    @testset "Linfinity" begin
        v = collect(1:5)
        linftynormalize!(v)
        @test v == [1, 1, 1, 1, 1]

        v = collect(1:5) / 5
        linftynormalize!(v)
        @test v == (1:5) / 5

        v = [0, 0.1, 0.8, 0.2]
        linftynormalize!(v)
        @test v == [0, 0.1, 1, 0.2]

        v = [0, 0.1, -0.8, 0.2]
        linftynormalize!(v)
        @test v == [0, 0.1, -1, 0.2]

        v = [0, 0.1, -0.8, 0.2]
        linftynormalize!(v)
        @test v == [0, 0.1, -1, 0.2]

        A = Array{Float64}(reshape(1:12, 3,4))
        l1normalize_rows!(A)
        @test A == [0 0 0 1; 0 0 0 1; 0 0 0 1]

        A = Array{Float64}(reshape(1:12, 3,4))
        l1normalize_cols!(A)
        @test A == [0 0 0 0; 0 0 0 0; 1 1 1 1]
    end

    @testset "Composition" begin
        c! = ∘(l1scale_cols!, nonnegative!)
        @test typeof(c!) <: Function
        @test typeof(c!) <: AbstractConstraint
        @test typeof(c!) <: ComposedConstraint

        v = [-1., -1., 2., 1., 2., 3.]
        c!(v)
        @test v ≈ [0.0, 0.0, 0.25, 0.125, 0.25, 0.375]
        @test check(c!, v)

        c! = ∘(nonnegative!, l1scale_cols!)
        @test typeof(c!) <: Function
        @test typeof(c!) <: AbstractConstraint
        @test typeof(c!) <: ComposedConstraint

        v = [-1., -1., 2., 1., 2., 3.]
        c!(v)
        @test v ≈ [0., 0., .2, .1, .2, .3]
        @test_broken check(c!, v) # Only nonnegativity is satisfied. Entries do not sum to 1
    end

    @testset "Convertion" begin
        @test ProjectedNormalization(l1scale!) == l1normalize!
        @test ScaledNormalization(l1normalize!) == l1scale!
    end
end

@testset "SuperDiagonal" begin
    v = 1:10
    S = SuperDiagonal(v, 2)
    @test diag(S) === v
    @test size(S) == (10, 10)
    @test ndims(S) == 2
    @test eltype(S) == Int
    @test_throws BoundsError S[10, 11]
    @test_throws BoundsError S[10, 0]
    @test S[9,9] == 9
    @test S[4,6] == 0
    @test array(S) == Diagonal(v)
    @test eltype(array(S)) == Int

    S = SuperDiagonal(v, 3)
    @test diag(S) === v
    @test size(S) == (10, 10, 10)
    @test ndims(S) == 3

    @test size(array(S)) == (10, 10, 10)

end

@testset "AbstractDecomposition" begin
    A = randn(3,3);
    B = randn(4,3);
    C = randn(5,3);

    G = CPDecomposition((A, B, C))

    @testset "Copying" begin
        G_copy = copy(G)
        G_deepcopy = deepcopy(G)
        @test A == matrix_factor(G,1)
        @test A === matrix_factor(G,1)
        @test A == matrix_factor(G_copy, 1)
        @test !(A === matrix_factor(G_copy,1))
        @test A == matrix_factor(G_deepcopy, 1)
        @test !(A === matrix_factor(G_deepcopy,1))
        copy(Tucker((2,2,2), (1,1,1)))
        copy(Tucker1((2,2,2), 1))
        deepcopy(Tucker((2,2,2), (1,1,1)))
        deepcopy(Tucker1((2,2,2), 1))
    end

    G = SingletonDecomposition(A)

    G = CPDecomposition((A, B))

    G = Tucker((randn(3,3,3), A, B, C))

    @test rankof(G) == (3,3,3)

    G = Tucker1(([1 2]', [3 4]))
    @test size(G) == (1, 1)
    @test_throws MethodError Tucker1{Int, 3}(([1 2]', [3 4])) # missing the freeze argument
    @test_throws ArgumentError Tucker1{Int, 3}(([1 2]', [3 4]), (false, false)) # the 3 in Tucker1{Int, 3} should be 2 since this combines to a 1×1 matrix

    G = Tucker1((randn(3,3,3), A))

    G = Tucker1((B', A)) # Can it handle types that are an abstract matrix like Ajoint

    @test_throws ArgumentError Tucker((G, A)) # Can handle auto conversion to TuckerN in the future??

    G = Tucker1((10,11,12), 5);
    Y = Tucker1((10,11,12), 5; init=abs_randn); # check if other initilizations work

    @test isfrozen(G, 0) == false # the core is not frozen
    @test isfrozen(G, 1) == false # the matrix factor A is not frozen

    @test G[1] ≈ array(G)[1] # vectorized indexing, check approx because it calculates in a slightly different way
    @test G[1,1,1] ≈ array(G)[1] # regular indexing, check approx because it calculates in a slightly different way

    # Tucker test

    Y = [776 1008 1240 1472; 1072 1392 1712 2032; 1368 1776 2184 2592;;; 1104 1432 1760 2088; 1524 1976 2428 2880; 1944 2520 3096 3672]

    A = reshape(1:6, 3, 2)
    B = reshape(1:8, 4, 2)
    C = reshape(1:4, 2, 2)
    G = reshape(1:8, 2, 2, 2)

    T = Tucker((G, A, B, C))

    @test array(T) == Y
    @test T[2] == 1072
    @test typeof(T[2]) == typeof(1072)
    @test eltype(T) == typeof(1072)
    @test eltype(T) == typeof(T[2])

    # CPDecomposition test
    A = reshape(1:6, 3, 2)
    B = reshape(1:8, 4, 2)
    C = reshape(1:4, 2, 2)
    D = reshape(1:10, 5, 2)

    G = CPDecomposition((A, B, C, D))

    @test isfrozen(G, 0) == true # the core is frozen
    @test isfrozen(G, 1) == false # the matrix factor A is not frozen

    T = [361 434 507 580; 452 544 636 728; 543 654 765 876;;; 482 580 678 776; 604 728 852 976; 726 876 1026 1176;;;; 422 508 594 680; 529 638 747 856; 636 768 900 1032;;; 564 680 796 912; 708 856 1004 1152; 852 1032 1212 1392;;;; 483 582 681 780; 606 732 858 984; 729 882 1035 1188;;; 646 780 914 1048; 812 984 1156 1328; 978 1188 1398 1608;;;; 544 656 768 880; 683 826 969 1112; 822 996 1170 1344;;; 728 880 1032 1184; 916 1112 1308 1504; 1104 1344 1584 1824;;;; 605 730 855 980; 760 920 1080 1240; 915 1110 1305 1500;;; 810 980 1150 1320; 1020 1240 1460 1680; 1230 1500 1770 2040]

    @test all(G[1] .== G[begin:1])
    @test G[1] == 361
    @test G[begin:1] == [361]
    @test array(G) == T
    @test ndims(G) == 4
    @test rankof(G) == 2
    @test all(diag(core(G)) .== 1)
    @test length(diag(core(G))) == 2 # same as rankof(G)

    @test_throws ArgumentError Tucker((A, B, C))

    frozen_factors = (false, true, false, false)
    G = CPDecomposition((A, B, C, D), frozen_factors)
    @test frozen(G) == frozen_factors

    @testset "TuckerGradient" begin
        T = Tucker((10,11,12), (3,4,5))
        Y = randn(10,11,12)
        C = core(T)
        matrices = matrix_factors(T)

        for n in 1:3
            An = factor(T, n)
            CM = tuckerproduct(C, matrices; exclude=n)

            # two ways of calculating the block gradient of ||T - Y||_F^2 w.r.t. An
            grad1 = slicewise_dot(T - Y, CM; dims=n)
            grad2 = slicewise_dot(T, CM; dims=n) - slicewise_dot(Y, CM; dims=n)
            # grad2 is slower to compute than grad1

            # another way, but treat it like the Tucker1 gradient
            # this is like grad2 but takes advantage of the symetric slicewise_dot
            # so grad3 is the fastest to compute
            grad3 = An*slicewise_dot(CM, CM; dims=n) - slicewise_dot(Y, CM; dims=n)

            # should all give the same answer
            @test all(grad1 .≈ grad2)
            @test all(grad1 .≈ grad3)
            @test all(grad2 .≈ grad3)
        end
    end

end

@testset "BlockUpdates" begin
    G1 = CPDecomposition((3,3,3), 2)
    G2 = deepcopy(G1)
    U = ConstraintUpdate(2, l2normalize_cols!)

    U(G1)
    l2normalize_cols!(factor(G2, 2))

    @test G1 ≈ G2

    U = ConstraintUpdate(1, l2scale! ∘ nonnegative!)
    @test U.n == 1

    U = BlockedUpdate(ConstraintUpdate(1, l2normalize_cols!), ConstraintUpdate(2, nonnegative!))
    @test isnothing(U.n)

    A = [-1.8  2.0  0.5;
          3.0 -4.0 -2.0;
         -7.2 -5.1  6.3]
    B = [0.0 2.0 0.5;
         3.0 0.0 0.0;
         0.0 0.0 6.3]
    G = SingletonDecomposition(A)
    H = SingletonDecomposition(A)

    U = NNProjection(1)
    U(G)
    @test all(G .≈ H)
    @test all(G .≈ B)
    @test all(A .≈ B)

    A = [-1.8  2.0  0.5;
    3.0 4.0 -0.2;
   -7.2 -5.1  0.6]
    G = SingletonDecomposition(A)
    U = ConstraintUpdate(1, l1scale_cols! ∘ nonnegative!)
    U(G)
    @test all(G .≈ [0.0 0.33333333 0.45454545; 1.0 0.66666667 0.0; 0.0 0.0 0.54545455])
    @test check(U, G)

    A = [-1.8  2.0  0.5;
    3.0 4.0 -0.2;
   -7.2 -5.1  0.6]
    G = SingletonDecomposition(A)
    U = ConstraintUpdate(1, simplex_cols!)
    U(G)
    @test all(G .≈ [0.0 0.0 0.45; 1.0 1.0 0.0; 0.0 0.0 0.55])
    @test check(U, G)

    one = ConstraintUpdate(1, simplex!)
    two = ConstraintUpdate(2, simplex!)
    three = ConstraintUpdate(3, simplex!)
    four = ConstraintUpdate(4, l1scale! ∘ nonnegative!)
    other = BlockedUpdate([ConstraintUpdate(3, l2normalize!), ConstraintUpdate(1, l2normalize!)])
    before_grouped = BlockedUpdate([two, one, two, four, three, one, other, three])

    grouped_updates = group_by_factor(before_grouped)
    @test [U.n for U in grouped_updates] == [2, 1, 4, 3, nothing]

    X = Tucker1((10,11,64), 3)
    core_constraint_update! = ConstraintUpdate(0, l1scale_average12slices! ∘ nonnegative!; whats_rescaled=(x -> eachcol(matrix_factor(x, 1))))
    core_constraint_update!(X)

    @test check(core_constraint_update!, X)
end

@testset "BlockUpdatedDecomposition" begin
    G = Tucker1((10,11,12), 5);
    Y = Tucker1((10,11,12), 5);
    Y = array(Y);

    fact = BlockTensorDecomposition.factorize

    # check hitting maximum number of iterations
    decomposition, stats_data = fact(Y; rank=5, momentum=false, maxiter=2);
    # check convergence on first iteration
    decomposition, stats_data = fact(Y; rank=5, momentum=false, tolerence=Inf);
    # check momentum
    decomposition, stats_data = fact(Y; rank=5, momentum=true, tolerence=Inf);
    # check constraints
    ## a single constraint, to be applied on every block
    decomposition, stats_data = fact(Y; rank=5, constraints=nonnegative!, tolerence=Inf);
    ## a collection of constraints
    decomposition, stats_data = fact(Y; rank=5, tolerence=Inf,
        constraints=[ConstraintUpdate(0, nonnegative!), ConstraintUpdate(0, l1scale_12slices!), ConstraintUpdate(1, nonnegative!)],
    );
    ## check if you can constrain the initialization
    decomposition, stats_data = fact(Y; rank=5, tolerence=Inf, constrain_init=true,
        constraints=[ConstraintUpdate(0, nonnegative!), ConstraintUpdate(0, l1scale_12slices!), ConstraintUpdate(1, nonnegative!)],
    );

    fact(Y; rank=5, tolerence=Inf, constrain_init=true,
        constraints=[l1scale_12slices! ∘ nonnegative!, nonnegative!],
    ); # constraints are ok since l1scale_12slices! can be applied to the core (the 0th factor)

    @test_broken fact(Y; rank=5, tolerence=Inf, constrain_init=true,
        constraints=[nonnegative!, l1scale_12slices! ∘ nonnegative!],
    ); # the constraint l1scale_12slices! cannot be applied to a matrix (the 1st factor)

    # Quick test to make sure Tucker works
    Y = Tucker((10,11,12), (2,3,4))
    Y = array(Y)
    decomposition, stats_data = fact(Y; model=Tucker, rank=(2,3,4), maxiter=2)

    # Quick test to make sure CP works
    Y = CPDecomposition((10,11,12), 3)
    Y = array(Y)
    decomposition, stats_data = fact(Y; model=CPDecomposition, rank=3, maxiter=2)

    # Quick test for Sub Block Updates Tucker1
    Y = Tucker1((10,10,10), 3)
    Y = array(Y)
    decomposition, stats_data = fact(Y; model=Tucker1, rank=3, maxiter=2, do_subblock_updates=true)

    # Quick test for Sub Block Updates Tucker
    Y = Tucker((10,11,12), (2,3,4))
    Y = array(Y)
    decomposition, stats_data = fact(Y; model=Tucker, rank=(2,3,4), maxiter=2, do_subblock_updates=true)

    # Quick test for Sub Block Updates CPDecomposition
    Y = CPDecomposition((10,11,12), 3)
    Y = array(Y)
    decomposition, stats_data = fact(Y; model=CPDecomposition, rank=3, maxiter=2, do_subblock_updates=true)

    # Regular run of Tucker1
    C = abs_randn(5, 11, 12)
    A = abs_randn(10, 5)
    Y = Tucker1((C, A))
    Y = array(Y)

    decomposition, stats_data = fact(Y;
        rank=5,
        tolerence=(2, 0.05),
        converged=(GradientNNCone, RelativeError),
        constrain_init=true,
        constraints=nonnegative!,
        stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError]
    );


end

@testset "nnmtf" begin
    R = 3
    Y = Tucker1((10,11,64), R);

    A = matrix_factor(Y, 1)
    B = core(Y)
    A .= abs.(A)
    B .= abs.(B)
    A_rows = eachrow(A)
    A_rows ./= sum.(A_rows)
    B_1slices = eachslice(B; dims=(1,2))
    B_1slices ./= sum.(B_1slices)

    Y1 = array(Y);
    Y2 = copy(Y1);

    core_constraint_update! = ConstraintUpdate(0, l1scale_average12slices! ∘ nonnegative!; whats_rescaled=(x -> eachcol(matrix_factor(x, 1))))

    X, stats, kwargs = BlockTensorDecomposition.factorize(Y2;
        model=Tucker1, rank=R, tolerence=0.001, converged=RelativeError,
        constraints=[core_constraint_update!, ConstraintUpdate(1, nonnegative!)],
        )
    A2 = matrix_factor(X, 1)
    B2 = core(X)

    @test_broken norm(X-Y1)/norm(Y1) <= 0.001 # could be a bit worse than tolerence=0.001
        # since we auto-apply constraints at the last step

    @test norm(X-Y1)/norm(Y1) <= 0.005 # should still be reasonably good

    A1, B1, rel_errors, norm_grad, dist_Ncone = nnmtf(Y1, R)

    @test norm((B1 ×₁ A1) - Y1)/ norm(Y1) <= 0.01 # expect less than 1% relative error (convergence criterion is GradientNNCone)

    # @test all(A1 .≈ A2)
    # @test all(B1 .≈ B2)
end

end
