using Images
using ImageView


function MultiFactorize(Y; kwargs...)

	scale = 2
	jump = div(size(Y)[3], scale)
	decomposition = get(kwargs, :decomposition, nothing)
	final_size = size(Y, 3)
	while jump > 1

		Y_prime = Y[:, :, 1:Int(jump):end]
		decomposition, stats_data, output_kwargs = factorize(Y_prime; decomposition=decomposition, kwargs...)
		decomposition = resize_decomp_const(decomposition, scale, final_size)
		jump = Int(jump // scale)

	end

	return factorize(Y; decomposition=decomposition, kwargs...)
end

function MultiFactorizeSimplex(Y; kwargs...)

	scale = 2
	jump = div(size(Y)[3], scale)
	println("This function ran")
	decomposition = get(kwargs, :decomposition, nothing)
	final_size = size(Y, 3)
	while jump > 1

		Y_prime = Y[:, :, 1:Int(jump):end]
		println("Size Y': $(size(Y_prime, 3))")
		println("scale: $jump")

		# Parse new constraint for simplex
		# constraint_matrix = nonnegative!
		# new_core_constraint! = ScaledNormalization(l1norm; whats_normalized=(x -> eachslice(x; dims=1)), scale=(A -> size(A)[2]/jump))
		# core_constraint_update! = ConstraintUpdate(0, new_core_constraint!; whats_rescaled=(x -> eachcol(matrix_factor(x, 1))))
		# constraints=[core_constraint_update!, ConstraintUpdate(1, constraint_matrix)]

		# constraint_matrix = nonnegative!
		# new_core_constraint! = ScaledNormalization(l1norm; whats_normalized=(x -> eachslice(x; dims=1)), scale=(size(Y[2])*size(Y)[3]/final_size))
		# core_constraint_update! = ConstraintUpdate(0, new_core_constraint!; whats_rescaled=(x -> eachcol(matrix_factor(x, 1))))
		# constraints=[core_constraint_update!, ConstraintUpdate(1, constraint_matrix)]

		# IDEA: Maybe resize each fiber individually to sum to 1 (should be very slow but if this doesn't converge idk what will)
		scale_factor = final_size / size(Y_prime)[3]
		Y_prime .*= scale_factor
		decomposition, stats_data, output_kwargs = factorize(Y_prime; decomposition=decomposition, kwargs...)
		decomposition = resize_decomp_const(decomposition, scale, final_size)
		jump = round(Int, jump / scale)
	end

	return factorize(Y; decomposition=decomposition, kwargs...)
end



function resize_decomp_const(decomp, scale, final_size)
	# TODO: make robust sizes
	tensor = factors(decomp)[1]
	factor_matrix = factors(decomp)[2]
	

	# Initialize a new tensor with double the size in the third dimension
    new_tensor = Array{eltype(tensor)}(undef, size(tensor, 1), size(tensor, 2), scale * size(tensor, 3))

    # Iterate over each slice in the third dimension and copy it twice
	i = 1
    for slice in eachslice(tensor, dims=3)
		for j in i:(i+scale-1)
        	new_tensor[:, :, j] = slice    # Place the original slice
		end
		i += scale
    end
	new_tensor .*= 1/scale

	return Tucker1((new_tensor, factor_matrix))
end

function resize_decomp_linear(decomp, scale, final_size)
	# TODO make this helper
end

function resize_decomp_imresize(decomp, scale, final_size)

	tensor = factors(decomp)[1]
	factor_matrix = factors(decomp)[2]
	
	new_dim = min(scale*size(tensor, 3), final_size)

	new_tensor = imresize(tensor, (size(tensor, 1), size(tensor, 2), new_dim))

	# new_tensor = restrict(tensor, dims=3)

	return Tucker1((new_tensor , factor_matrix))
end


