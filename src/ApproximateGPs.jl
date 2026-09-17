module ApproximateGPs

using Reexport

@reexport using AbstractGPs
@reexport using GPLikelihoods

include("API.jl")
@reexport using .API: approx_lml

include("utils.jl")

include("SparseVariationalApproximationModule.jl")
@reexport using .SparseVariationalApproximationModule:
    SparseVariationalApproximation, Centered, NonCentered

include("LaplaceApproximationModule.jl")
@reexport using .LaplaceApproximationModule: LaplaceApproximation
@reexport using .LaplaceApproximationModule:
    build_laplace_objective, build_laplace_objective!

include("NearestNeighborsModule.jl")
@reexport using .NearestNeighborsModule: NearestNeighbors

include("HilbertSpaceApproximationModule.jl")
@reexport using .HilbertSpaceApproximationModule:
    HilbertSpaceApproximation, HilbertSpaceBasis, spectral_density, spectral_weights

include("deprecations.jl")

include("TestUtils.jl")

end
