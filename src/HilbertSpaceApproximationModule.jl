module HilbertSpaceApproximationModule

using ..API

export HilbertSpaceApproximation, HilbertSpaceBasis, spectral_density, spectral_weights

using LinearAlgebra
using Statistics
using StatsBase

using ChainRulesCore: ChainRulesCore
using Distances: Euclidean
using IrrationalConstants: log2π
using SpecialFunctions: gamma

using AbstractGPs: AbstractGPs, GP, FiniteGP, ApproxPosteriorGP, At_A, diag_At_A
using KernelFunctions:
    Kernel,
    KernelSum,
    ScaledKernel,
    TransformedKernel,
    ScaleTransform,
    ARDTransform,
    SqExponentialKernel,
    MaternKernel,
    Matern32Kernel,
    Matern52Kernel,
    ExponentialKernel,
    dim

using ..ApproximateGPs: _chol_lower

#
# Spectral densities.
#

@doc raw"""
    spectral_density(kernel::Kernel, ω::AbstractVector{<:Real})

Compute the spectral density ``S(\omega)`` of the stationary `kernel` at the angular
frequency `ω`, i.e. the Fourier transform of ``k(\tau)``.

Methods are defined for `SqExponentialKernel` and the Matérn kernels with the Euclidean
metric, and for `ScaledKernel`, `KernelSum` and kernels transformed by `ScaleTransform` or
`ARDTransform`. Other stationary kernels can be used with
[`HilbertSpaceApproximation`](@ref) by adding a method for them.
"""
function spectral_density(kernel::Kernel, ω::AbstractVector{<:Real})
    return throw(
        ArgumentError(
            "no `spectral_density` method for $(nameof(typeof(kernel))), so it cannot be " *
            "used with a Hilbert-space approximation; define " *
            "`ApproximateGPs.spectral_density(::$(nameof(typeof(kernel))), ω)` to add one",
        ),
    )
end

function spectral_density(::SqExponentialKernel{Euclidean}, ω::AbstractVector{<:Real})
    D = length(ω)
    return (2π)^(D / 2) * exp(-sum(abs2, ω) / 2)
end

# Rasmussen & Williams (2006), eq. (4.15), in angular frequency.
function _matern_spectral_density(ν::Real, ω::AbstractVector{<:Real})
    D = length(ω)
    return 2^D * π^(D / 2) * gamma(ν + D / 2) * (2ν)^ν / gamma(ν) *
           (2ν + sum(abs2, ω))^(-(ν + D / 2))
end

function spectral_density(k::MaternKernel{<:Real,Euclidean}, ω::AbstractVector{<:Real})
    return _matern_spectral_density(only(k.ν), ω)
end
function spectral_density(::ExponentialKernel{Euclidean}, ω::AbstractVector{<:Real})
    return _matern_spectral_density(1//2, ω)
end
function spectral_density(::Matern32Kernel{Euclidean}, ω::AbstractVector{<:Real})
    return _matern_spectral_density(3//2, ω)
end
function spectral_density(::Matern52Kernel{Euclidean}, ω::AbstractVector{<:Real})
    return _matern_spectral_density(5//2, ω)
end

function spectral_density(k::ScaledKernel, ω::AbstractVector{<:Real})
    return only(k.σ²) * spectral_density(k.kernel, ω)
end

function spectral_density(k::KernelSum, ω::AbstractVector{<:Real})
    return sum(kᵢ -> spectral_density(kᵢ, ω), k.kernels)
end

# k(x) = k₀(Ax)  ⟹  S(ω) = |det A|⁻¹ S₀(A⁻ᵀ ω)
# A = Diagonal(inverse lengthscales) for both transforms below
function spectral_density(
    k::TransformedKernel{<:Kernel,<:ScaleTransform}, ω::AbstractVector{<:Real}
)
    s = only(k.transform.s)
    return s^(-length(ω)) * spectral_density(k.kernel, ω ./ s)
end
function spectral_density(
    k::TransformedKernel{<:Kernel,<:ARDTransform}, ω::AbstractVector{<:Real}
)
    v = k.transform.v
    length(v) == length(ω) || throw(
        DimensionMismatch(
            "kernel has $(length(v)) lengthscales but the frequency has $(length(ω)) components",
        ),
    )
    return spectral_density(k.kernel, ω ./ v) / prod(v)
end

#
# The approximation.
#

@doc raw"""
    HilbertSpaceApproximation(; m, c=1.5)
    HilbertSpaceApproximation(; m, L)

Construct a reduced-rank approximation [1] to a GP with a stationary kernel. The kernel is
replaced by its expansion in the eigenfunctions ``\phi_j`` of the Laplacian on the domain
`centre ± L`, where `centre` is the midpoint of the training inputs:

```math
k(x, x') \approx \sum_j S(\omega_j) \phi_j(x) \phi_j(x'),
```

where ``S`` is the [`spectral_density`](@ref) of the kernel. Conditioning on Gaussian
observations then costs ``O(n \prod(m)^2)`` rather than ``O(n^3)``.

`m` is the number of basis functions per input dimension, given either as an integer or as
one count per dimension. The half-widths `L` are either given directly or set to `c` times
the half-range of the inputs, in which case `c` must exceed 1. See [2] for guidance on
choosing `m` and `c`.

N.B. `prod(m)` grows exponentially with the input dimension, so the approximation is only
practical in low dimensions. Also, meaningful predictions can only be made inside the domain.

[1] - Solin, Arno, and Simo Särkkä. "Hilbert space methods for reduced-rank Gaussian
process regression." Statistics and Computing 30.2 (2020): 419-446.

[2] - Riutort-Mayol, Gabriel, et al. "Practical Hilbert space approximate Bayesian
Gaussian processes for probabilistic programming." Statistics and Computing 33.1 (2023).
"""
struct HilbertSpaceApproximation{
    Tm,Tc<:Union{Nothing,Real},TL<:Union{Nothing,AbstractVector{<:Real}}
}
    m::Tm
    c::Tc
    L::TL
end

function HilbertSpaceApproximation(; m, c=nothing, L=nothing)
    c === nothing && L === nothing && (c = 1.5)
    c === nothing ||
        L === nothing ||
        throw(
            ArgumentError(
                "give either the boundary factor `c` or the half-widths `L`, not both"
            ),
        )
    all(>(0), m) || throw(ArgumentError("the basis counts m must be positive, got $m"))
    if c !== nothing
        c > 1 || throw(
            ArgumentError(
                "the boundary factor c must exceed 1 so that the domain contains the inputs, got $c",
            ),
        )
    else
        L = collect(L)
        all(>(0), L) || throw(ArgumentError("the half-widths L must be positive, got $L"))
    end
    return HilbertSpaceApproximation(m, c, L)
end

_basis_counts(m::Integer, D::Int) = ntuple(_ -> Int(m), D)
function _basis_counts(m, D::Int)
    length(m) == D ||
        throw(DimensionMismatch("got $(length(m)) basis counts for $D input dimensions"))
    return ntuple(d -> Int(m[d]), D)
end

function _half_widths(approx::HilbertSpaceApproximation, ranges::AbstractVector, D::Int)
    approx.L === nothing && return approx.c .* ranges ./ 2
    length(approx.L) == D || throw(
        DimensionMismatch("got $(length(approx.L)) half-widths for $D input dimensions")
    )
    return approx.L
end

#
# The basis.
#

@doc raw"""
    HilbertSpaceBasis(approx::HilbertSpaceApproximation, x::AbstractVector)

Construct the Laplacian eigenbasis that `approx` places on the domain around the inputs
`x`. Its eigenfunctions are

```math
\phi_j(x) = \prod_{d=1}^{D} \frac{1}{\sqrt{L_d}}
            \sin\!\big(\omega_{jd} (x_d - c_d + L_d)\big),
\qquad
\omega_{jd} = \frac{\pi j_d}{2 L_d},
```

where ``c_d`` is the midpoint of `x` in dimension ``d`` and ``1 \le j_d \le m_d``.

`basis(x)` returns the `length(x) × prod(m)` matrix with entries ``\phi_j(x_i)``. To
evaluate the same approximate process at new inputs, reuse `basis` rather than
constructing a new one, which would lie on a different domain.

See also [`spectral_weights`](@ref).
"""
struct HilbertSpaceBasis{
    TL<:AbstractVector{<:Real},TC<:AbstractVector{<:Real},TΩ<:AbstractMatrix{<:Real}
}
    L::TL
    centre::TC
    Ω::TΩ
end

Base.length(basis::HilbertSpaceBasis) = size(basis.Ω, 1)

input_dimension(basis::HilbertSpaceBasis) = length(basis.L)

_coordinates(xᵢ::Real) = (xᵢ,)
_coordinates(xᵢ::AbstractVector{<:Real}) = xᵢ

# prod(m) × D matrix, rows are the frequencies ωⱼ
function _frequencies(m::NTuple{D,Int}, L::AbstractVector{<:Real}) where {D}
    return _multi_indices(m) .* π ./ (2 .* transpose(L))
end

# prod(m) × D integer matrix, rows are the multi-indices j
function _multi_indices(m::NTuple{D,Int}) where {D}
    return [Tuple(j)[d] for j in vec(CartesianIndices(m)), d in 1:D]
end
ChainRulesCore.@non_differentiable _multi_indices(::Any)

function HilbertSpaceBasis(approx::HilbertSpaceApproximation, x::AbstractVector)
    D = dim(x) # returns 0 if `dim` doesn't have a method for `eltype(x)`
    D > 0 || throw(
        ArgumentError(
            "unknown `dim` of input type $(typeof(x)); define " *
            "`KernelFunctions.dim`",
        ),
    )
    m = _basis_counts(approx.m, D)
    lo = [minimum(xᵢ -> _coordinates(xᵢ)[d], x) for d in 1:D]
    hi = [maximum(xᵢ -> _coordinates(xᵢ)[d], x) for d in 1:D]
    L = _half_widths(approx, hi .- lo, D)
    return HilbertSpaceBasis(L, (lo .+ hi) ./ 2, _frequencies(m, L))
end

function (basis::HilbertSpaceBasis)(x::AbstractVector)
    D = input_dimension(basis)
    dim(x) == D || throw(
        DimensionMismatch(
            "basis is $(D)-dimensional but the inputs are $(dim(x))-dimensional"
        ),
    )
    L, centre, Ω = basis.L, basis.centre, basis.Ω
    # speeds up reverse AD compared to a comprehension over `x`
    # by keeping the number of pullbacks small
    return mapreduce((a, b) -> a .* b, 1:D) do d
        xd = map(xᵢ -> _coordinates(xᵢ)[d], x)
        return sin.((xd .- centre[d] .+ L[d]) .* transpose(view(Ω, :, d))) ./ sqrt(L[d])
    end
end

"""
    spectral_weights(basis::HilbertSpaceBasis, kernel::Kernel)

Compute the square roots of the [`spectral_density`](@ref) of `kernel` at each frequency
of `basis`.
"""
function spectral_weights(basis::HilbertSpaceBasis, kernel::Kernel)
    return [sqrt(spectral_density(kernel, ω)) for ω in eachrow(basis.Ω)]
end

#
# Posterior.
#

# Ψ = Φ diag(w), so that the approximate prior covariance is Ψ Ψᵀ
function _whitened_basis(basis::HilbertSpaceBasis, w::AbstractVector, x::AbstractVector)
    return basis(x) .* w'
end

@doc raw"""
    posterior(approx::HilbertSpaceApproximation, fx::FiniteGP, y::AbstractVector{<:Real})

Compute the exact posterior of the reduced-rank model that `approx` substitutes for
`fx.f`, given observations `y` of `fx` with Gaussian noise `fx.Σy`. The approximate prior
is ``f = \Psi \varepsilon`` with ``\varepsilon \sim N(0, I)``, where ``\Psi`` is the
[`HilbertSpaceBasis`](@ref) scaled by the [`spectral_weights`](@ref), so the posterior can
be found in closed form.
"""
function AbstractGPs.posterior(
    approx::HilbertSpaceApproximation, fx::FiniteGP{<:GP}, y::AbstractVector{<:Real}
)
    return first(_posterior_and_whitened_basis(approx, fx, y))
end

# Also returns Ψ at the training inputs so we don't need to recalculate Ψ in `approx_lml`
function _posterior_and_whitened_basis(
    approx::HilbertSpaceApproximation, fx::FiniteGP{<:GP}, y::AbstractVector{<:Real}
)
    basis = HilbertSpaceBasis(approx, fx.x)
    w = spectral_weights(basis, fx.f.kernel)
    Ψ = _whitened_basis(basis, w, fx.x)
    Σy_inv_Ψ = fx.Σy \ Ψ
    A = cholesky(Symmetric(I + Ψ' * Σy_inv_Ψ))
    α = A \ (Σy_inv_Ψ' * (y - mean(fx)))
    return ApproxPosteriorGP(approx, fx.f, (basis=basis, w=w, A=A, α=α)), Ψ
end

#
# Various methods implementing the Internal AbstractGPs API.
# See AbstractGPs.jl API docs for more info.
#

# V** = Ψ* A⁻¹ Ψ*ᵀ
#     = VᵀV
# V = Lᴬ⁻¹ Ψ*ᵀ
function _V_and_Ψ(f::ApproxPosteriorGP{<:HilbertSpaceApproximation}, x::AbstractVector)
    Ψ = _whitened_basis(f.data.basis, f.data.w, x)
    return _chol_lower(f.data.A) \ Ψ', Ψ
end

function _V(f::ApproxPosteriorGP{<:HilbertSpaceApproximation}, x::AbstractVector)
    return first(_V_and_Ψ(f, x))
end

function Statistics.mean(
    f::ApproxPosteriorGP{<:HilbertSpaceApproximation}, x::AbstractVector
)
    return mean(f.prior, x) + _whitened_basis(f.data.basis, f.data.w, x) * f.data.α
end

function Statistics.cov(
    f::ApproxPosteriorGP{<:HilbertSpaceApproximation}, x::AbstractVector
)
    return At_A(_V(f, x))
end

function Statistics.var(
    f::ApproxPosteriorGP{<:HilbertSpaceApproximation}, x::AbstractVector
)
    return diag_At_A(_V(f, x))
end

function Statistics.cov(
    f::ApproxPosteriorGP{<:HilbertSpaceApproximation}, x::AbstractVector, y::AbstractVector
)
    return _V(f, x)' * _V(f, y)
end

function StatsBase.mean_and_cov(
    f::ApproxPosteriorGP{<:HilbertSpaceApproximation}, x::AbstractVector
)
    V, Ψ = _V_and_Ψ(f, x)
    return mean(f.prior, x) + Ψ * f.data.α, At_A(V)
end

function StatsBase.mean_and_var(
    f::ApproxPosteriorGP{<:HilbertSpaceApproximation}, x::AbstractVector
)
    V, Ψ = _V_and_Ψ(f, x)
    return mean(f.prior, x) + Ψ * f.data.α, diag_At_A(V)
end

#
# Marginal likelihood.
#

"""
    approx_lml(approx::HilbertSpaceApproximation, fx::FiniteGP, y::AbstractVector{<:Real})

Compute an approximation to the log of the marginal likelihood, which can be used to optimise
the hyperparameters of `fx`. This is the exact log marginal likelihood of the reduced-rank model
that `approx` substitutes for `fx.f`.
"""
function API.approx_lml(
    approx::HilbertSpaceApproximation, fx::FiniteGP{<:GP}, y::AbstractVector{<:Real}
)
    f_post, Ψ = _posterior_and_whitened_basis(approx, fx, y)
    δ = y - mean(fx)
    Σy_inv_δ = fx.Σy \ δ
    quadform = δ'Σy_inv_δ - (Ψ'Σy_inv_δ)'f_post.data.α
    ld = logdet(fx.Σy) + logdet(f_post.data.A)
    return -(ld + length(y) * oftype(quadform, log2π) + quadform) / 2
end

end
