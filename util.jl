using ChainRulesCore
using PEPSKit

function convert_peps(peps_init, bond_m, fuser)
    # absorb bond matrix into PEPS tensors
    @tensor peps_mod[-1 -2 -3 -4 -5; -6 -7 -8 -9] := peps_init[-1; 1 2 3 4] * flip(bond_m, 2)[-2; 1 -6] * flip(bond_m, 2)[-3; 2 -7] * 
            flip(bond_m, 2)[-4; -8 3] * flip(bond_m, 2)[-5; -9 4];

    @tensor peps_fused[-1; -2 -3 -4 -5] := peps_mod[1 2 3 4 5; -2 -3 -4 -5] * fuser[-1; 1 2 3 4 5];
    return peps_fused
end

function local_ipeps_update(peps::InfinitePEPS, gate, inds::Vector{CartesianIndex{2}})
    # Functionally update the PEPS tensors at specified indices
    peps_A = deepcopy(peps.A)
    for ind in inds
        r, c = Tuple(ind)
        @tensor peps_mod[-1; -2 -3 -4 -5] := gate[-1; 1] * peps_A[r,c][1; -2 -3 -4 -5]
        peps_A[r,c] = peps_mod
    end
    return InfinitePEPS(peps_A)
end

# Custom ChainRules rrule to handle the mutation
function ChainRulesCore.rrule(::typeof(local_ipeps_update), peps::InfinitePEPS, gate, inds::Vector{CartesianIndex{2}})
    result = local_ipeps_update(peps, gate, inds)
    
    function local_ipeps_update_pullback(ȳ)
        # ȳ is the incoming gradient (an InfinitePEPS or tangent)
        if isa(ȳ, InfinitePEPS)
            ΔA = ȳ.A
        else
            ΔA = ȳ.A
        end
        
        # Return tangent wrapped as Tangent{InfinitePEPS}
        Δpeps = Tangent{InfinitePEPS}(; A = ΔA)
        
        return (NoTangent(), Δpeps, NoTangent(), NoTangent())
    end
    
    return result, local_ipeps_update_pullback
end

function network_overlap(bra::InfinitePEPS, ket::InfinitePEPS, env::CTMRGEnv)
    # Create a sandwich network where each site contains (ket, bra) tuple
    # Let gradients flow through the data
    data = map((b, k) -> (k, b), bra.A, ket.A)
    sandwich_network = InfiniteSquareNetwork(data)
    return network_value(sandwich_network, env)
end

function on_site_terms(site_gate, fuser)
    @tensor on_site_term_1[-1; -2] := site_gate[1; 2] * fuser[-1; 1 3 4 5 6] * conj(fuser[-2; 2 3 4 5 6]);

    @tensor on_site_term_2[-1; -2] := fuser[-1; 1 2 3 4 5] * conj(fuser[-2; 6 7 8 9 10]) * 
                                      site_gate[1; 6] * site_gate[2; 7] * site_gate[3; 8] * site_gate[4; 9] * site_gate[5; 10];

    @tensor on_site_term_3[-1; -2] := fuser[-1; 5 1 2 3 4] * conj(fuser[-2; 5 6 7 8 9]) * 
          
    site_gate[1; 6] * site_gate[2; 7] * site_gate[3; 8] * site_gate[4; 9];
        
    return -on_site_term_1 - on_site_term_2 + on_site_term_3
end

function two_site_terms(Xgate, Ygate, Zgate, my_id, fuser)

    two_site_terms_1 = []
    two_site_terms_2 = []

    ###### 1st gates #######
    gate_list = [Ygate, my_id, Ygate, my_id, my_id]
    @tensor h1[-1; -2] := fuser[-1; 1 2 3 4 5] * conj(fuser[-2; 6 7 8 9 10]) * gate_list[1][1; 6] * gate_list[2][2; 7] * 
                                        gate_list[3][3; 8] * gate_list[4][4; 9] * gate_list[5][5; 10];

    gate_list = [Xgate, Zgate, Zgate, Zgate, my_id]
    @tensor h2[-1; -2] := fuser[-1; 1 2 3 4 5] * conj(fuser[-2; 6 7 8 9 10]) * gate_list[1][1; 6] * gate_list[2][2; 7] * 
                                        gate_list[3][3; 8] * gate_list[4][4; 9] * gate_list[5][5; 10];

    push!(two_site_terms_1, h1)
    push!(two_site_terms_2, h2)

    ###### 2nd gates #######
    gate_list = [Xgate, my_id, Ygate, my_id, my_id]
    @tensor h1[-1; -2] := fuser[-1; 1 2 3 4 5] * conj(fuser[-2; 6 7 8 9 10]) * gate_list[1][1; 6] * gate_list[2][2; 7] * 
                                        gate_list[3][3; 8] * gate_list[4][4; 9] * gate_list[5][5; 10];

    gate_list = [Ygate, Zgate, Zgate, Zgate, my_id]
    @tensor h2[-1; -2] := fuser[-1; 1 2 3 4 5] * conj(fuser[-2; 6 7 8 9 10]) * gate_list[1][1; 6] * gate_list[2][2; 7] * 
                                        gate_list[3][3; 8] * gate_list[4][4; 9] * gate_list[5][5; 10];

    push!(two_site_terms_1, h1)
    push!(two_site_terms_2, h2)

    ###### 3rd gates #######
    gate_list = [Zgate, my_id, Xgate, my_id, my_id]
    @tensor h1[-1; -2] := fuser[-1; 1 2 3 4 5] * conj(fuser[-2; 6 7 8 9 10]) * gate_list[1][1; 6] * gate_list[2][2; 7] * 
                                        gate_list[3][3; 8] * gate_list[4][4; 9] * gate_list[5][5; 10];

    gate_list = [my_id, my_id, my_id, my_id, my_id]
    @tensor h2[-1; -2] := fuser[-1; 1 2 3 4 5] * conj(fuser[-2; 6 7 8 9 10]) * gate_list[1][1; 6] * gate_list[2][2; 7] *
                                        gate_list[3][3; 8] * gate_list[4][4; 9] * gate_list[5][5; 10];
                            
    push!(two_site_terms_1, h1)
    push!(two_site_terms_2, h2)


    ####### 4th gates #######
    gate_list = [my_id, Zgate, Xgate, Zgate, Zgate]
    @tensor h1[-1; -2] := fuser[-1; 1 2 3 4 5] * conj(fuser[-2; 6 7 8 9 10]) * gate_list[1][1; 6] * gate_list[2][2; 7] * 
                                        gate_list[3][3; 8] * gate_list[4][4; 9] * gate_list[5][5; 10];

    gate_list = [Zgate, Zgate, Zgate, Zgate, my_id]
    @tensor h2[-1; -2] := fuser[-1; 1 2 3 4 5] * conj(fuser[-2; 6 7 8 9 10]) * gate_list[1][1; 6] * gate_list[2][2; 7] *
                                        gate_list[3][3; 8] * gate_list[4][4; 9] * gate_list[5][5; 10];
    
    push!(two_site_terms_1, h1)
    push!(two_site_terms_2, h2)

    ####### 5th gates ####### 
    gate_list = [Ygate, my_id, Ygate, Zgate, Zgate]
    @tensor h1[-1; -2] := fuser[-1; 1 2 3 4 5] * conj(fuser[-2; 6 7 8 9 10]) * gate_list[1][1; 6] * gate_list[2][2; 7] * 
                                        gate_list[3][3; 8] * gate_list[4][4; 9] * gate_list[5][5; 10];

    gate_list = [Xgate, Zgate, my_id, my_id, my_id]
    @tensor h2[-1; -2] := fuser[-1; 1 2 3 4 5] * conj(fuser[-2; 6 7 8 9 10]) * gate_list[1][1; 6] * gate_list[2][2; 7] *
                                        gate_list[3][3; 8] * gate_list[4][4; 9] * gate_list[5][5; 10];

    push!(two_site_terms_1, h1)
    push!(two_site_terms_2, h2)

    ####### 6th gates #######
    gate_list = [Xgate, my_id, Ygate, Zgate, Zgate]
    @tensor h1[-1; -2] := fuser[-1; 1 2 3 4 5] * conj(fuser[-2; 6 7 8 9 10]) * gate_list[1][1; 6] * gate_list[2][2; 7] * 
                                        gate_list[3][3; 8] * gate_list[4][4; 9] * gate_list[5][5; 10];

    gate_list = [Ygate, Zgate, my_id, my_id, my_id]
    @tensor h2[-1; -2] := fuser[-1; 1 2 3 4 5] * conj(fuser[-2; 6 7 8 9 10]) * gate_list[1][1; 6] * gate_list[2][2; 7] *
                                        gate_list[3][3; 8] * gate_list[4][4; 9] * gate_list[5][5; 10];

    push!(two_site_terms_1, h1)
    push!(two_site_terms_2, h2)

    ####### 7th gates #######
    gate_list = [Zgate, my_id, Xgate, Zgate, Zgate]
    @tensor h1[-1; -2] := fuser[-1; 1 2 3 4 5] * conj(fuser[-2; 6 7 8 9 10]) * gate_list[1][1; 6] * gate_list[2][2; 7] * 
                                        gate_list[3][3; 8] * gate_list[4][4; 9] * gate_list[5][5; 10];

    gate_list = [my_id, my_id, Zgate, Zgate, my_id]
    @tensor h2[-1; -2] := fuser[-1; 1 2 3 4 5] * conj(fuser[-2; 6 7 8 9 10]) * gate_list[1][1; 6] * gate_list[2][2; 7] *
                                        gate_list[3][3; 8] * gate_list[4][4; 9] * gate_list[5][5; 10];

    push!(two_site_terms_1, h1)
    push!(two_site_terms_2, h2)

    ####### 8th gates #######
    gate_list = [my_id, Zgate, Xgate, my_id, my_id]
    @tensor h1[-1; -2] := fuser[-1; 1 2 3 4 5] * conj(fuser[-2; 6 7 8 9 10]) * gate_list[1][1; 6] * gate_list[2][2; 7] * 
                                        gate_list[3][3; 8] * gate_list[4][4; 9] * gate_list[5][5; 10];

    gate_list = [Zgate, Zgate, my_id, my_id, my_id]
    @tensor h2[-1; -2] := fuser[-1; 1 2 3 4 5] * conj(fuser[-2; 6 7 8 9 10]) * gate_list[1][1; 6] * gate_list[2][2; 7] *
                                        gate_list[3][3; 8] * gate_list[4][4; 9] * gate_list[5][5; 10];

    push!(two_site_terms_1, h1)
    push!(two_site_terms_2, h2)

    return sum(two_site_terms_1), sum(two_site_terms_2)
end

function PEPSKit.add_term!(
        operator::PEPSKit.LocalOperator, inds::Vector{CartesianIndex{2}}, term::Tuple{T1, T2};
        atol = zero(real(scalartype(term))),
    ) where {T1 <: AbstractTensorMap, T2 <: AbstractTensorMap}
    
    operator.terms[inds] = term
    

    return operator
end

"""
    normalize_term_indices(inds::Vector{CartesianIndex{2}}, peps_size::Tuple{Int,Int})

Check if interaction term indices are within the unit cell. If not, calculate the required
shift and return the shifted indices and the shift amount.

The shift_tuple represents the displacement needed to bring all indices into [1, Lr] x [1, Lc].
When shift_tuple is applied to PEPS via circshift(peps.A, shift_tuple), sites are rotated
in that direction, so the original indices get mapped to new positions.

Returns: (normalized_inds, shift_tuple) where shift_tuple is (Δr, Δc) to shift the PEPS.
"""
function normalize_term_indices(inds::Vector{CartesianIndex{2}}, peps_size::Tuple{Int,Int})
    Lr, Lc = peps_size
    
    # Find which indices are out of bounds
    min_r = minimum(i[1] for i in inds)
    max_r = maximum(i[1] for i in inds)
    min_c = minimum(i[2] for i in inds)
    max_c = maximum(i[2] for i in inds)
    
    # Calculate shift needed to bring indices within [1, Lr] x [1, Lc]
    # Shift is applied to the lattice coordinates via circshift
    shift_r = 0
    shift_c = 0
    
    if min_r < 1
        shift_r = 1 - min_r  # Positive shift moves indices down, wrapping low indices to high
    elseif max_r > Lr
        shift_r = Lr - max_r  # Negative shift moves indices up, wrapping high indices to low
    end
    
    if min_c < 1
        shift_c = 1 - min_c  # Positive shift moves indices right, wrapping low indices to high
    elseif max_c > Lc
        shift_c = Lc - max_c  # Negative shift moves indices left, wrapping high indices to low
    end
    
    # Normalize indices to be within unit cell using modular arithmetic
    normalized_inds = [CartesianIndex(
        mod(i[1] - 1 + shift_r, Lr) + 1,
        mod(i[2] - 1 + shift_c, Lc) + 1
    ) for i in inds]
    
    return normalized_inds, (shift_r, shift_c)
end

"""
    shift_peps_env(peps::InfinitePEPS, env::CTMRGEnv, shift_tuple::Tuple{Int,Int})

Shift PEPS and environment by the given (row_shift, col_shift) amount.
Positive shifts move the lattice down and right; negative shifts move it up and left.
"""
function shift_peps_env(peps::InfinitePEPS, env::CTMRGEnv, shift_tuple::Tuple{Int,Int})
    shifted_peps_A = ChainRulesCore.ignore_derivatives(circshift(peps.A, shift_tuple))
    shifted_peps = InfinitePEPS(shifted_peps_A)
    
    shifted_corners = ChainRulesCore.ignore_derivatives(circshift(env.corners, (0, shift_tuple...)))
    shifted_edges = ChainRulesCore.ignore_derivatives(circshift(env.edges, (0, shift_tuple...)))
    shifted_env = PEPSKit.CTMRGEnv(shifted_corners, shifted_edges)
    
    return shifted_peps, shifted_env
end


"""
    bring_term_inunitcell(peps::InfinitePEPS, env::CTMRGEnv, inds::Vector{CartesianIndex{2}})

Check if interaction term indices are within the unit cell. If not, shift both PEPS and 
environment so the term becomes within the cell, and return the modified indices.

Returns: (normalized_inds, shifted_peps, shifted_env, shift_tuple)
"""
function bring_term_inunitcell(peps::InfinitePEPS, env::CTMRGEnv, inds::Vector{CartesianIndex{2}})
    peps_size = size(peps.A)
    normalized_inds, shift_tuple = normalize_term_indices(inds, peps_size)
    
    if shift_tuple == (0, 0)
        # No shift needed
        return normalized_inds, peps, env, shift_tuple
    else
        # Shift PEPS and environment
        shifted_peps, shifted_env = shift_peps_env(peps, env, shift_tuple)
        return normalized_inds, shifted_peps, shifted_env, shift_tuple
    end
end

function PEPSKit.add_term!(
        operator::PEPSKit.LocalOperator, inds::CartesianIndex{2}, term::T1;
        atol = zero(real(scalartype(term))),
    ) where {T1 <: AbstractTensorMap}
    
    operator.terms[[inds]] = term
    

    return operator
end

"""
    expectation_value_combined(bra::InfinitePEPS, ham::LocalOperator, ket::InfinitePEPS, env::CTMRGEnv)

Compute expectation value for a combined Hamiltonian containing both single-site and two-site terms.
Automatically detects single-site (length(inds)==1) vs multi-site (length(inds)>1) terms and processes them accordingly.

Returns: sum of all term contributions, normalized by ⟨bra|ket⟩
"""
function expectation_value_combined(bra::InfinitePEPS, ham::PEPSKit.LocalOperator, ket::InfinitePEPS, env::CTMRGEnv)
    denominator = network_overlap(bra, ket, env)
    
    # Use regular map
    term_vals = map(collect(ham.terms)) do (inds, operator)
        # Convert inds to Vector{CartesianIndex{2}}
        inds_vec = inds
        # inds_vec = if inds isa Vector
        #     # Already a vector - could be Vector{CartesianIndex{2}} or Vector{Vector{Int}}
        #     if length(inds) > 0 && inds[1] isa CartesianIndex
        #         inds
        #     elseif length(inds) > 0 && inds[1] isa Vector
        #         # Vector of vectors - convert to CartesianIndex
        #         [CartesianIndex(Tuple(v)) for v in inds]
        #     else
        #         [CartesianIndex(Tuple(inds))]
        #     end
        # elseif inds isa CartesianIndex
        #     [inds]
        # elseif inds isa Tuple
        #     collect(inds)
        # else
        #     error("Unknown inds format: $(typeof(inds))")
        # end
        
        # Bring term within unit cell - let gradients flow through
        normalized_inds, shifted_ket, shifted_env, shift_tuple = bring_term_inunitcell(ket, env, inds_vec)
        @show inds_vec
        @show normalized_inds
        @show shift_tuple
        
        # Shift bra by the same amount
        shifted_bra, _ = shift_peps_env(bra, env, shift_tuple)
        
        # Determine if single-site or multi-site based on number of indices
        if length(normalized_inds) == 1
            # Single-site term: operator is just the gate
            ket_new = local_ipeps_update(shifted_ket, operator, normalized_inds)
        else
            # Multi-site term: operator is a tuple of gates
            ket_new = shifted_ket
            for (i, gate) in enumerate(operator)
                ket_new = local_ipeps_update(ket_new, gate, [normalized_inds[i]])
            end
        end
        
        return network_overlap(shifted_bra, ket_new, shifted_env) / denominator
    end
    
    return sum(term_vals)
end

PEPSKit.add_term!(operator::PEPSKit.LocalOperator, inds::Tuple, term) = PEPSKit.add_term!(operator, collect(inds), term)
PEPSKit.add_term!(operator::PEPSKit.LocalOperator, inds::Vector, term) = PEPSKit.add_term!(operator, map(CartesianIndex{2}, inds), term)