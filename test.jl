using PEPSKit, OptimKit, LinearAlgebra
using JLD2, TensorKit, MPSKitModels, KrylovKit, ChainRulesCore

include("util.jl")
include("chain_crap.jl")

# prepare PEPS anzatz
D_phys = 2;
D_virt = 3;
peps_init = randn(ComplexF64, ℂ^D_phys ← ℂ^D_virt ⊗ ℂ^D_virt ⊗ (ℂ^D_virt)' ⊗ (ℂ^D_virt)');
space(peps_init)

bond_matrix = zeros(ComplexF64, ℂ^D_phys ← ℂ^D_virt ⊗ ℂ^D_virt);
bond_matrix[1, 1:D_virt ÷2, 1:D_virt ÷2] = Diagonal(ones(D_virt ÷2));
bond_matrix[2, D_virt ÷2+1:D_virt, D_virt ÷2+1:D_virt] = Diagonal(ones(D_virt ÷2));
fuser = isometry(ℂ^32 ← ℂ^D_phys ⊗ ℂ^D_phys ⊗ ℂ^D_phys ⊗ ℂ^D_phys ⊗ ℂ^D_phys);

X = σˣ(spin = 1//2);
Y = σʸ(spin = 1//2);
Z = σᶻ(spin = 1//2);
my_id = id(space(X,1));

lattice = fill(ℂ^32, (2, 2));
t = 1.0;
U = 1.0;

O_on_site = U/4 * on_site_terms(Z, fuser); # Sum the tuple of 3 operators into a single operator
O1_two_site, O2_two_site = -t/2 .* two_site_terms(X, Y, Z, my_id, fuser);

# Combined Hamiltonian with both single-site and two-site terms
ham = PEPSKit.LocalOperator(lattice, 
        CartesianIndex((1, 1)) => O_on_site,
        CartesianIndex((1, 2)) => O_on_site,
        CartesianIndex((2, 1)) => O_on_site,
        CartesianIndex((2, 2)) => O_on_site,
        (CartesianIndex((1, 1)), CartesianIndex((1, 2))) => (O1_two_site, O2_two_site),
        (CartesianIndex((1, 1)), CartesianIndex((2, 1))) => (O1_two_site, O2_two_site),
        (CartesianIndex((1, 2)), CartesianIndex((2, 2))) => (O1_two_site, O2_two_site),
        (CartesianIndex((1, 2)), CartesianIndex((1, 3))) => (O1_two_site, O2_two_site),
        (CartesianIndex((2, 1)), CartesianIndex((2, 2))) => (O1_two_site, O2_two_site),
        (CartesianIndex((2, 1)), CartesianIndex((3, 1))) => (O1_two_site, O2_two_site),
        (CartesianIndex((2, 2)), CartesianIndex((2, 3))) => (O1_two_site, O2_two_site),
        (CartesianIndex((2, 2)), CartesianIndex((3, 2))) => (O1_two_site, O2_two_site),
);   


function PEPSKit.cost_function(peps::InfinitePEPS, env::CTMRGEnv, O::PEPSKit.LocalOperator)
    @show " I am computing the cost function! "
    E = expectation_value_combined(peps, O, peps, env)
    ignore_derivatives() do
        isapprox(imag(E), 0; atol = sqrt(eps(real(E)))) ||
            @warn "Expectation value is not real: $E."
    end
    return real(E)
end


# peps_check = InfinitePEPS(fill(convert_peps(peps_init, bond_matrix, fuser), (2, 2)));
# space(peps_check[1, 1])
# env_space = ℂ^20;
# env0 = CTMRGEnv(randn, Float64, peps_check, env_space);
# cost = PEPSKit.cost_function(peps_check, env0, ham)

boundary_alg = SimultaneousCTMRG(;
    tol = 1.0e-9, # 1e-8
    trunc = truncrank(10),
    maxiter = 150,
    verbosity = 3,
);

gradient_alg = LinSolver(;
    solver_alg = KrylovKit.GMRES(;
        tol = 1e-9, maxiter = 1, verbosity = 3, krylovdim = 50,
    ),
    iterscheme = :fixed,
);

optimizer_alg = LBFGS(32; maxiter=50, gradtol=1e-5, verbosity=3, linesearch = HagerZhangLineSearch(;c₁ = 1e-4, verbosity = 2, maxiter=10, maxfg = 10))

algs = PEPSOptimize(;
    boundary_alg = boundary_alg,
    optimizer_alg = optimizer_alg,
    gradient_alg = gradient_alg,
    reuse_env=true,
);


function my_hummble_fixedpoint(
        operator, peps₀, env₀, alg::PEPSOptimize, bond_matrix, fuser; (finalize!) = OptimKit._finalize!,
    )

    # initialize info collection vectors
    T = promote_type(real(scalartype(peps₀)), real(scalartype(env₀)))
    contraction_metrics = Vector{NamedTuple}()
    gradnorms_unitcell = Vector{Matrix{T}}()
    times = Vector{Float64}()

    # normalize the initial guess
    # peps₀ = PEPSKit.peps_normalize(peps₀)

    # optimize operator cost function
    (peps_final, env_final), cost_final, ∂cost, numfg, convergence_history = optimize(
        (peps₀, env₀), alg.optimizer_alg;
        retract = PEPSKit.peps_retract, inner = PEPSKit.real_inner, finalize!, (transport!) = (PEPSKit.peps_transport!),
    ) do (peps, env)
        start_time = time_ns()
        E, gs = PEPSKit.withgradient(peps) do ψ
            # Apply conversion inside withgradient so gradients flow through
            ψ_fused = InfinitePEPS([convert_peps(ψ[r,c], bond_matrix, fuser) for r in 1:size(ψ, 1), c in 1:size(ψ, 2)])
            env′, info = PEPSKit.hook_pullback(
                leading_boundary, env, ψ_fused, alg.boundary_alg;
                alg_rrule = alg.gradient_alg,
            )
            ignore_derivatives() do
                alg.reuse_env && PEPSKit.update!(env, env′)
                push!(contraction_metrics, info.contraction_metrics)
            end
            return cost_function(ψ_fused, env′, operator)
        end
        g = only(gs)  # `withgradient` returns tuple of gradients `gs`
        push!(gradnorms_unitcell, norm.(g.A))
        push!(times, (time_ns() - start_time) * 1.0e-9)
        # Return gradient tuple: (gradient for peps, gradient for env = NoTangent)
        return E, g
    end

    info = (;
        last_gradient = ∂cost,
        fg_evaluations = numfg,
        costs = convergence_history[:, 1],
        gradnorms = convergence_history[:, 2],
        contraction_metrics,
        gradnorms_unitcell,
        times,
    )
    return peps_final, env_final, cost_final, info
end


println(">>> Starting fixedpoint...")
peps_cell = fill(peps_init, (2, 2));
peps_merged = InfinitePEPS(fill(convert_peps(peps_init, bond_matrix, fuser), (2, 2)));
env_space = ℂ^16;
env0 = CTMRGEnv(randn, ComplexF64, peps_merged, env_space);
env, _ = leading_boundary(env0, peps_merged, algs.boundary_alg);
cost_function(peps_merged, env, ham)

peps, env, E, info = my_hummble_fixedpoint(
    ham, InfinitePEPS(peps_cell), env0, algs, bond_matrix, fuser;
);




