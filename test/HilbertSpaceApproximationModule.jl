@testset "hilbert_space_approximation" begin
    rng = MersenneTwister(654321)

    x = collect(range(-3.0, 3.0; length=40))
    y = sin.(x) .+ 0.05 .* randn(rng, 40)
    σ² = 0.05^2

    @testset "spectral_density" begin
        # k(0) = (2π)⁻ᴰ ∫ S(ω) dω, and for the SqExponentialKernel S integrates to
        # (2π)^{D/2} × (2π)^{D/2} = (2π)^D, so S(0) = (2π)^{D/2}.
        for D in 1:3
            @test spectral_density(SqExponentialKernel(), zeros(D)) ≈ (2π)^(D / 2)
        end

        # The spectral densities are only meaningful under the Euclidean metric.
        @test_throws "no `spectral_density` method for SqExponentialKernel" spectral_density(
            SqExponentialKernel(; metric=Cityblock()), [0.5]
        )
        @test_throws "no `spectral_density` method for PeriodicKernel" spectral_density(
            PeriodicKernel(), [0.5]
        )

        # A kernel with ARD lengthscales needs one frequency component per dimension.
        @test_throws DimensionMismatch spectral_density(
            with_lengthscale(SqExponentialKernel(), [1.0, 2.0]), [0.5]
        )
    end

    @testset "basis reconstructs the kernel" begin
        # Σⱼ S(ωⱼ) φⱼ(x) φⱼ(x') → k(x, x'), which is the defining property of the
        # approximation and pins down the normalisation of every spectral density.
        approx = HilbertSpaceApproximation(; m=400, c=2.5)
        basis = HilbertSpaceBasis(approx, x)

        # The tolerances differ because the truncation error is set by how fast the
        # spectral density decays: Matern12's falls off as ‖ω‖⁻², the slowest here, so a
        # basis of a given size reproduces it least well.
        kernels = [
            SqExponentialKernel() => 1e-10,
            with_lengthscale(SqExponentialKernel(), 2.0) => 1e-3,
            3.0 * with_lengthscale(SqExponentialKernel(), 0.7) => 1e-10,
            Matern12Kernel() => 1e-1,
            with_lengthscale(Matern32Kernel(), 1.3) => 1e-3,
            Matern52Kernel() => 1e-5,
            MaternKernel(; ν=1.7) => 1e-4,
            SqExponentialKernel() + 2.0 * Matern32Kernel() => 1e-3,
        ]
        for (kernel, atol) in kernels
            Ψ = basis(x) .* spectral_weights(basis, kernel)'
            @test Ψ * Ψ' ≈ kernelmatrix(kernel, x) atol = atol
        end
    end

    @testset "approximates the exact GP" begin
        f = GP(with_lengthscale(Matern52Kernel(), 0.8))
        fx = f(x, σ²)
        xt = collect(range(-2.5, 2.5; length=11))

        f_exact = posterior(fx, y)
        m_exact, c_exact = mean_and_cov(f_exact(xt))

        # The error has two sources: truncating the basis, controlled by m, and the
        # boundary, controlled by c. Increasing m alone therefore stalls at a floor set
        # by c, so both are raised together here.
        errors = map([(20, 1.5), (60, 2.0), (200, 2.5)]) do (m, c)
            approx = HilbertSpaceApproximation(; m, c)
            m_approx, c_approx = mean_and_cov(posterior(approx, fx, y)(xt))
            return (
                mean=maximum(abs, m_approx - m_exact),
                cov=maximum(abs, c_approx - c_exact),
                lml=abs(approx_lml(approx, fx, y) - logpdf(fx, y)),
            )
        end

        for quantity in (:mean, :cov, :lml)
            @test issorted(getfield.(errors, quantity); rev=true)
            @test getfield(last(errors), quantity) < 1e-3
        end
    end

    @testset "AbstractGPs API" begin
        f = GP(with_lengthscale(Matern52Kernel(), 0.8))
        f_approx_post = posterior(HilbertSpaceApproximation(; m=60, c=2.0), f(x, σ²), y)
        AbstractGPs.TestUtils.test_internal_abstractgps_interface(
            rng, f_approx_post, collect(range(-1.2, 1.2; length=6)), randn(rng, 7)
        )
    end

    @testset "basis is tied to the inputs it was built from" begin
        approx = HilbertSpaceApproximation(; m=10, c=1.5)
        basis = HilbertSpaceBasis(approx, x)
        @test length(basis) == 10
        @test basis.centre ≈ [0.0]
        @test basis.L ≈ [1.5 * 3.0]

        # Rebuilding the basis from the new inputs moves the box, so the coefficients
        # would no longer mean the same thing.
        xt = collect(range(-1.0, 1.0; length=5))
        @test basis(xt) != HilbertSpaceBasis(approx, xt)(xt)

        @test_throws DimensionMismatch basis(ColVecs(rand(rng, 2, 5)))
    end

    @testset "explicit half-widths" begin
        approx = HilbertSpaceApproximation(; m=10, L=[4.5])
        @test HilbertSpaceBasis(approx, x).L == [4.5]
        @test_throws DimensionMismatch HilbertSpaceBasis(approx, ColVecs(rand(rng, 2, 5)))
    end

    @testset "multiple input dimensions" begin
        X = ColVecs(4 .* rand(rng, 2, 40) .- 2)
        Xt = ColVecs(2 .* rand(rng, 2, 6) .- 1)
        f = GP(3.0, 1.5 * with_lengthscale(SqExponentialKernel(), [0.9, 1.4]))
        fX = f(X, 0.01)
        yX = rand(rng, fX)

        # 30 basis functions per dimension resolve wavelengths well below both
        # lengthscales, so the truncation error is already negligible here and c alone
        # sets how close the approximation gets.
        approx = HilbertSpaceApproximation(; m=(30, 30), c=3.0)
        m_approx, v_approx = mean_and_var(posterior(approx, fX, yX)(Xt))
        m_exact, v_exact = mean_and_var(posterior(fX, yX)(Xt))

        @test m_approx ≈ m_exact atol = 1e-5
        @test v_approx ≈ v_exact atol = 1e-6
        @test approx_lml(approx, fX, yX) ≈ logpdf(fX, yX) atol = 1e-3

        # RowVecs describes the same inputs as ColVecs, so it must give the same answer.
        approx_rows = posterior(approx, f(RowVecs(collect(X.X')), 0.01), yX)
        @test mean(approx_rows(RowVecs(collect(Xt.X')))) ≈ m_approx
    end

    @testset "Zygote can take gradients of the log marginal likelihood" begin
        function objective(θ)
            kernel = exp(θ[1]) * with_lengthscale(Matern52Kernel(), exp(θ[2]))
            return approx_lml(
                HilbertSpaceApproximation(; m=40, c=2.0), GP(kernel)(x, σ²), y
            )
        end

        θ = [0.0, -0.2]
        lml, (grad,) = Zygote.withgradient(objective, θ)

        @test lml ≈ objective(θ)
        @test grad ≈ only(FiniteDifferences.grad(central_fdm(5, 1), objective, θ)) rtol =
            1e-5
    end

    @testset "constructor validation" begin
        @test_throws "not both" HilbertSpaceApproximation(; m=10, c=1.5, L=[1.0])
        @test_throws "must exceed 1" HilbertSpaceApproximation(; m=10, c=1.0)
        @test_throws "must be positive" HilbertSpaceApproximation(; m=0)
        @test_throws "must be positive" HilbertSpaceApproximation(; m=10, L=[-1.0])
        @test_throws DimensionMismatch HilbertSpaceBasis(
            HilbertSpaceApproximation(; m=(5, 5)), x
        )
    end
end
