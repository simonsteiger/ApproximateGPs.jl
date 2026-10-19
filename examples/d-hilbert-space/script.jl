# # Regression with the Hilbert Space Approximation
#
# This example demonstrates how to carry out Gaussian process regression using the Hilbert space approximation (HSGP).
# An HSGP approximates a GP by replacing the kernel with a finite set of basis functions.
#
# For a basic introduction to the functionality of this library, please refer to the [User Guide](@ref).
#
# ## Setup

using ApproximateGPs
using Distributions
using LogExpFunctions: softplus, invsoftplus
using Zygote
using Optim
using LineSearches

using Plots
## We output plots as PNG to reduce their size
default(; fmt=:png, palette=:seaborn_colorblind, legend=:outertopright, size=(700, 400))
#nb ## Regardless of the `fmt` setting, by default Literate saves also the SVG output
#nb ## in notebooks: https://github.com/fredrikekre/Literate.jl/issues/61
#nb ## We use DisplayAs to ensure that only the PNG output is saved.
#nb using DisplayAs

using Random
Random.seed!(1234);

# ## Generate some training data
#
# We define a data-generating function `g`:

g(x) = sin(3π * x) + 0.3 * cos(9π * x) + 0.5 * sin(7π * x)

N = 2000 # Number of training points
x = rand(Uniform(-1, 1), N)
y = g.(x) + 0.3 * randn(N)

xgrid = -1:0.001:1 # for visualization

scatter(x, y; xlabel="x", ylabel="y", markershape=:xcross, markeralpha=0.3, legend=false)
#nb DisplayAs.PNG(current())

# ## Create the approximation
#
# Here we write a function that creates our GP prior at the training inputs `x` given the hyperparameter vector `theta`.
# Since the observation noise is Gaussian, we pass its variance directly to the GP, and we treat it as a third hyperparameter alongside the kernel variance and lengthscale.

function build_gp(theta)
    ## all hyperparameters must be positive
    variance = softplus(theta[1])
    lengthscale = softplus(theta[2])
    noise_variance = softplus(theta[3])

    kernel = variance * with_lengthscale(SqExponentialKernel(), lengthscale)
    return GP(kernel)(x, noise_variance)
end;

# We define the GP at our initial hyperparameter values, here with variance 1.0, lengthscale 0.2 and noise variance 0.5:

theta0 = [invsoftplus(1.0), invsoftplus(0.2), invsoftplus(0.5)]
fx = build_gp(theta0)
fx.f.kernel

# The HSGP replaces the kernel by a weighted sum of `m` fixed basis functions, which are sine waves on a domain around the training inputs.
# This means that the HSGP prior is a linear model that scales `O(N m^2)` rather than the exact GP's `O(N^3)` for `N` data points.
# Since the weights are determined by the kernel's spectral density, only stationary kernels that define `spectral_density` can be used.
#
# !!! note "Boundary factor"
#     It is crucial that we define the basis functions on a domain that extends past the training data.
#     We therefore stretch this domain using the boundary factor `c`; for example, with `c = 1.5`, the domain runs from -1.5 to 1.5 (our input data's domain is -1 to 1).
#     The basis functions are all zero at the edges of the domain, so our approximation starts breaking at these limits.
#     That's why we always have to choose `c > 1`!
#
# !!! note "Choosing `m` and `c`"
#     A basic rule of thumb is: larger lengthscales need a larger domain, and shorter lengthscales need more basis functions.
#
#     Riutort-Mayol et al. (2023) give a specific heuristic for choosing `m` and `c` for the squared exponential kernel: for `S` and lengthscales between `ℓ_min` and `ℓ_max`, choose `c ≥ max(3.2 ℓ_max / S, 1.2)` and `m ≥ 1.75 c S / ℓ_min`, where `S` is half of the range spanned by the input values.
#
# In our data, half of the range spanned by the input data is `S = 1`, and we expect the lengthscale to move from its initial value of 0.2 down to about 0.1.
# A range of 0.05 to 0.2 requires `c ≥ 1.2`; with `c = 1.5`, it requires `m ≥ 53`, so we use `m = 60`:

approx = HilbertSpaceApproximation(; m=60, c=1.5)

# We can now compute the posterior of the approximate model:

f_post = posterior(approx, fx, y)

# Let's have a look at the posterior with these initial hyperparameter settings.
# It shouldn't come as a surprise that these are a poor fit for the data:

scatter(
    x,
    y;
    xlabel="x",
    ylabel="y",
    markershape=:xcross,
    markeralpha=0.3,
    label="Training Data",
)
plot!(xgrid, f_post; label="Initial Posterior", color=4)
#nb DisplayAs.PNG(current())

# ## Optimise the hyperparameters
#
# Since the HSGP is a linear model with Gaussian noise, we can compute the log marginal likelihood exactly using `approx_lml`.

objective(theta) = -approx_lml(approx, build_gp(theta), y);

# !!! warning "Numerical stability"
#     For the squared exponential kernel, weights at some lengthscale settings can shrink so quickly that the weights of the fastest-oscillating basis functions underflow to zero.
#     The gradient of `approx_lml` then becomes `NaN`, and the optimiser fails.
#     With `m = 60` and `c = 1.5`, this happens for lengthscales above about 0.6.
#
#     If you choose `m` and `c` with the heuristic above, this may happen when the range of lengthscales gets very large.
#     Should you need to work with such a large range of lengthscales, we recommend using a Matérn kernel instead because it is less prone to this issue.
#
# Next, we pass this objective to Optim.jl's LBFBS optimiser.
# To avoid a large initial gradient that breaks computation, we scale the first step `alphaguess` to unit length:

training_results = Optim.optimize(
    objective,
    θ -> only(Zygote.gradient(objective, θ)),
    theta0,
    LBFGS(; alphaguess=LineSearches.InitialStatic(; scaled=true));
    inplace=false,
)

# Having learned about the hyperparameter values from the data now, we construct the HSGP again with the new values:

fx_opt = build_gp(training_results.minimizer)
fx_opt.f.kernel

# Finally, we once again construct the posterior of the HSGP given the observations:

f_post_opt = posterior(approx, fx_opt, y)

# ## Compare with the exact GP
#
# In this example, our data still allow for computing the exact GP posterior and marginal likelihood for the optimised hyperparameters.
# This allows us to check how much of an approximation our HSGP actually is:

approx_lml(approx, fx_opt, y), logpdf(fx_opt, y)

#-

f_exact = posterior(fx_opt, y);

# Finally, we plot both posteriors to see the results.

scatter(
    x,
    y;
    markershape=:xcross,
    markeralpha=0.3,
    xlim=(-1, 1),
    xlabel="x",
    ylabel="y",
    title="posterior (HSGP)",
    label="Training Data",
    color=1,
)
plot!(xgrid, f_post_opt; label="Approximate Posterior", color=4)
plot!(xgrid, mean(f_exact(xgrid)); label="Exact Posterior Mean", color=3, linestyle=:dash)
#nb DisplayAs.PNG(current())

# It seems that we were able to approximate the exact GP very well with this HSGP!
