# Decomposiiton Models

```docs
AbstractDecomposition
```

The following common tensor models are avalible as valid arguments to `model` and built into this package.

```
Tucker
Tucker1
CPDecomposition
```

### Tucker types

```docs
Tucker
Tucker1
CPDecomposition
```

Note these are all subtypes of `AbstractTucker`.

```docs
AbstractTucker
```

### Other types

```docs
GenericDecomposition
SingletonDecomposition
```

## How Julia treats an AbstractDecomposition

`AbstractDecomposition` is an abstract subtype of `AbstractArray`. `AbstractDecomposition` will keep track of the element type and number of dimentions like other `AbstractArray`. This is the `T` and `N` in the type `Array{T,N}`. To make `AbstractDecomposition` behave like other array types, Julia only needs to know how to access/compute indexes of the array through `getindex`. These indecies are computed on the fly when a particular index is requested, or the whole tensor is computed from its factors through `array(X)`. This has the advantage of minimizing the memory used, and allows for the most flexibility since any operation that is supported by `AbstractArray` will work on `AbstractDecomposition` types. The drawback is that repeated requests to entries must recompute the entry each time. In these cases, it is best to "flatten" the array with `array(X)` first before making these repeated calls.

Some basic operations like `+`, `-`, `*`, `\`, and `/` will either compute the operation is some optimized way, or call `array(X)` function to first flatten the decomposition into a regular `Array` type in some optimized way. Operations that don't have an optimized method (becuase I can only do so much), will instead call Julia's `Array{T,N}(X)` to convert the model into a regular `Array{T,N}` type. This is usualy slower and less memory effecient since it calls `getindex` on every index individually, instead of computing the whole array at once.
