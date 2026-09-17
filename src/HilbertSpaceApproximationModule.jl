module HilbertSpaceApproximationModule

using ..API

export HilbertSpaceApproximation, HilbertSpaceBasis, spectral_density, spectral_weights

using LinearAlgebra
using Statistics
using StatsBase

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

Power spectral density of a stationary `kernel` at the angular frequency vector `ω`:

```math
S(\omega) = \int k(\tau) e^{-i \omega^\top \tau} \, \mathrm{d}\tau,
\qquad
k(\tau) = (2\pi)^{-D} \int S(\omega) e^{i \omega^\top \tau} \, \mathrm{d}\omega,
```

where ``D`` is the input dimension, `length(ω)`.

[`HilbertSpaceApproximation`](@ref) evaluates this at each basis frequency, so a kernel
can only be used with it if it has a `spectral_density` method. Methods are defined for
`SqExponentialKernel` and the Matérn kernels under the Euclidean metric, and compose
through `ScaledKernel`, `KernelSum`, and the `ScaleTransform`/`ARDTransform` that
`with_lengthscale` produces. To use another stationary kernel, add a method for it.

```jldoctest
julia> spectral_density(SqExponentialKernel(), [0.0])
2.5066282746310002

julia> spectral_density(2.0 * SqExponentialKernel(), [0.0])
5.0132565492620005
```
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

# For k(x) = k₀(Ax), the spectral density is S(ω) = |det A|⁻¹ S₀(A⁻ᵀ ω). Both transforms
# below are diagonal, with A's diagonal holding the inverse lengthscales.
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

A Hilbert-space (reduced-rank) approximation [1] to a Gaussian process with a stationary
kernel, which replaces the kernel by its truncated expansion in the Laplace eigenbasis of
a bounded box:

```math
k(x, x') \approx \sum_{j} S(\omega_j) \, \phi_j(x) \, \phi_j(x'),
```

with eigenfunctions ``\phi_j`` and frequencies ``\omega_j`` as in
[`HilbertSpaceBasis`](@ref) and ``S`` the kernel's [`spectral_density`](@ref). The
approximate prior is therefore a Bayesian linear model in `prod(m)` fixed basis
functions, so conditioning on Gaussian observations costs ``O(n \prod(m)^2)`` rather than
``O(n^3)``.

`m` is the number of basis functions per input dimension, either an integer used for
every dimension or one count per dimension. The total `prod(m)` grows exponentially with
the input dimension, which limits the method to a few dimensions.

The box is `centre ± L`, where `centre` is the midpoint of the training inputs in each
dimension. Give either `c`, which sets `L` to `c` times the inputs' half-range in each
dimension, or `L` directly. The approximation degrades near the boundary, so `c` must
exceed 1; [2] gives guidance on choosing `c` and `m` together against the kernel's
lengthscale.

Predictions are those of the reduced-rank model rather than of the original GP. Outside
the box they are meaningless: the basis functions there are a periodic continuation of
their behaviour inside it.

```jldoctest
julia> x = range(-3, 3; length=50); y = sin.(x);

julia> fx = GP(with_lengthscale(Matern52Kernel(), 1.0))(x, 0.01);

julia> f_post = posterior(HilbertSpaceApproximation(; m=40), fx, y);

julia> isapprox(mean(f_post([0.4])), mean(posterior(fx, y)([0.4])); atol=1e-3)
true
```

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
                "the boundary factor c must exceed 1 so that the box contains the inputs, got $c",
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

The Laplace eigenbasis that [`HilbertSpaceApproximation`](@ref) `approx` induces on the
box spanned by the inputs `x`, held separately from any kernel.

The box is `centre ± L`, where `centre` is the midpoint of `x` in each dimension and `L`
comes from `approx`. On it, the Dirichlet eigenfunctions of the Laplacian are

```math
\phi_j(x) = \prod_{d=1}^{D} \frac{1}{\sqrt{L_d}}
            \sin\!\big(\omega_{jd} (x_d - c_d + L_d)\big),
\qquad
\omega_{jd} = \frac{\pi j_d}{2 L_d},
```

indexed by the multi-index ``j \in \{1, \dots, m_1\} \times \dots \times \{1, \dots, m_D\}``.

Calling a basis, `basis(x)`, gives the `length(x) × prod(m)` matrix whose `(i, j)` entry
is ``\phi_j(x_i)``. Building the basis once and calling it on several input vectors is
the only way to evaluate the same approximate process at new points: a basis built afresh
from new inputs sits on a different box, which changes what every coefficient means.

```jldoctest
julia> basis = HilbertSpaceBasis(HilbertSpaceApproximation(; m=5), range(-1, 1; length=20));

julia> size(basis(rand(7) .- 0.5))
(7, 5)
```

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

# Frequency vectors for the Cartesian product of 1..mᵈ over the dimensions, as a
# prod(m) × D matrix whose rows are the ωⱼ.
function _frequencies(m::NTuple{D,Int}, L::AbstractVector{<:Real}) where {D}
    js = vec(collect(Iterators.product(map(Base.OneTo, m)...)))
    return [j[d] * π / (2 * L[d]) for j in js, d in 1:D]
end

function HilbertSpaceBasis(approx::HilbertSpaceApproximation, x::AbstractVector)
    # `dim` returns 0 for input types that have not opted in, which would silently give
    # an empty basis rather than an error.
    D = dim(x)
    D > 0 || throw(
        ArgumentError(
            "cannot infer the dimension of inputs of type $(typeof(x)); define " *
            "`KernelFunctions.dim` for it",
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
    return [
        prod(
            sin(Ω[j, d] * (_coordinates(xᵢ)[d] - centre[d] + L[d])) / sqrt(L[d]) for
            d in 1:D
        ) for xᵢ in x, j in axes(Ω, 1)
    ]
end

"""
    spectral_weights(basis::HilbertSpaceBasis, kernel::Kernel)

The square roots of `kernel`'s [`spectral_density`](@ref) at each frequency of `basis`.

With `Φ = basis(x)` and `w = spectral_weights(basis, kernel)`, `Φ * (w .* z)` for
`z ~ Normal(0, I)` is a draw from the Hilbert-space approximation to `GP(kernel)` at `x`.
This is the form to use inside a probabilistic program, where `z` is sampled rather than
marginalised out; `posterior` marginalises it for a Gaussian likelihood.
"""
function spectral_weights(basis::HilbertSpaceBasis, kernel::Kernel)
    return [sqrt(spectral_density(kernel, ω)) for ω in eachrow(basis.Ω)]
end

#
# Posterior.
#

# The whitened design matrix Ψ = Φ diag(w), for which the approximate prior covariance
# is Ψ Ψᵀ.
function _whitened_basis(basis::HilbertSpaceBasis, w::AbstractVector, x::AbstractVector)
    return basis(x) .* w'
end

@doc raw"""
    posterior(approx::HilbertSpaceApproximation, fx::FiniteGP, y::AbstractVector{<:Real})

Compute the exact posterior of the reduced-rank model that `approx` puts in place of
`fx.f`, given observations `y` of `fx` under the Gaussian noise `fx.Σy`.

Writing ``\Psi`` for the [`HilbertSpaceBasis`](@ref) scaled by the
[`spectral_weights`](@ref), the approximate prior is ``f = \Psi \varepsilon`` with
``\varepsilon \sim N(0, I)``, and the posterior over ``\varepsilon`` has precision
``A = I + \Psi^\top \Sigma_y^{-1} \Psi``, which is what this factorises.
"""
function AbstractGPs.posterior(
    approx::HilbertSpaceApproximation, fx::FiniteGP{<:GP}, y::AbstractVector{<:Real}
)
    basis = HilbertSpaceBasis(approx, fx.x)
    w = spectral_weights(basis, fx.f.kernel)
    Ψ = _whitened_basis(basis, w, fx.x)
    Σy_inv_Ψ = fx.Σy \ Ψ
    A = cholesky(Symmetric(I + Ψ' * Σy_inv_Ψ))
    α = A \ (Σy_inv_Ψ' * (y - mean(fx)))
    return ApproxPosteriorGP(approx, fx.f, (basis=basis, w=w, A=A, α=α))
end

#
# Various methods implementing the Internal AbstractGPs API.
# See AbstractGPs.jl API docs for more info.
#

# V = Lᴬ⁻¹ Ψ*ᵀ, for which the posterior covariance Ψ* A⁻¹ Ψ*ᵀ is VᵀV.
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

The exact log marginal likelihood of the reduced-rank model that `approx` puts in place
of `fx.f`, which approximates the log marginal likelihood of `fx` itself.

Under that model `y ~ N(mean(fx), Ψ Ψᵀ + fx.Σy)`, and the matrix determinant lemma turns
both the log determinant and the quadratic form into `prod(m)`-sized problems.
"""
function API.approx_lml(
    approx::HilbertSpaceApproximation, fx::FiniteGP{<:GP}, y::AbstractVector{<:Real}
)
    f_post = AbstractGPs.posterior(approx, fx, y)
    Ψ = _whitened_basis(f_post.data.basis, f_post.data.w, fx.x)
    δ = y - mean(fx)
    Σy_inv_δ = fx.Σy \ δ
    quadform = δ'Σy_inv_δ - (Ψ'Σy_inv_δ)'f_post.data.α
    ld = logdet(fx.Σy) + logdet(f_post.data.A)
    return -(ld + length(y) * oftype(quadform, log2π) + quadform) / 2
end

end
