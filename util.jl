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
    peps_A = peps.A
    inds_set = Set(inds)  # Convert to Set for O(1) lookup
    
    # Create new array without mutation
    peps_A_new = [if CartesianIndex(r, c) in inds_set
                      @tensor peps_mod[-1; -2 -3 -4 -5] := gate[-1; 1] * peps_A[r,c][1; -2 -3 -4 -5]
                      peps_mod
                  else
                      peps_A[r, c]
                  end
                  for r in 1:size(peps_A, 1), c in 1:size(peps_A, 2)]
    
    return InfinitePEPS(peps_A_new)
end

# Custom ChainRules rrule to handle the mutation
# function ChainRulesCore.rrule(::typeof(local_ipeps_update), peps::InfinitePEPS, gate, inds::Vector{CartesianIndex{2}})
#     result = local_ipeps_update(peps, gate, inds)
    
#     function local_ipeps_update_pullback(ȳ)
#         ΔA = ȳ.A
#         Δpeps = Tangent{InfinitePEPS}(; A = ΔA)
        
#         # Compute gradient for gate by contracting backwards
#         peps_A = peps.A
#         Δgate = zero(gate)
#         for ind in inds
#             r, c = Tuple(ind)
#             r = mod1(r, size(peps_A, 1))
#             c = mod1(c, size(peps_A, 2))
#             @tensor Δgate[-1; 1] += ΔA[-1; -2 -3 -4 -5] * peps_A[r,c][1; -2 -3 -4 -5]
#         end
        
#         return (NoTangent(), Δpeps, Δgate, NoTangent())
#     end
    
#     return result, local_ipeps_update_pullback
# end


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

function PEPSKit.add_term!(
        operator::PEPSKit.LocalOperator, inds::CartesianIndex{2}, term::T1;
        atol = zero(real(scalartype(term))),
    ) where {T1 <: AbstractTensorMap}
    
    operator.terms[[inds]] = term
    

    return operator
end


function my_contract_local_norm(
        inds::Vector{CartesianIndex{2}}, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
    )
    static_inds = Tuple(Val.(inds))
    return PEPSKit._contract_local_norm(static_inds, (ket, bra), env)
end

"""
    expectation_value_combined(bra::InfinitePEPS, ham::LocalOperator, ket::InfinitePEPS, env::CTMRGEnv)

Compute expectation value for a combined Hamiltonian containing both single-site and two-site terms.
Automatically detects single-site (length(inds)==1) vs multi-site (length(inds)>1) terms and processes them accordingly.

Returns: sum of all term contributions, normalized by ⟨bra|ket⟩
"""
function expectation_value_combined(bra::InfinitePEPS, ham::PEPSKit.LocalOperator, ket::InfinitePEPS, env::CTMRGEnv)
    
    
    # Use regular map
    term_vals = map(collect(ham.terms)) do (inds, operator)
        denominator = my_contract_local_norm(inds, ket, bra, env)
        
        # Determine if single-site or multi-site based on number of indices
        if length(inds) == 1
            # Single-site term: operator is just the gate
            ket_new = local_ipeps_update(ket, operator, inds)
        else
            # Multi-site term: operator is a tuple of gates
            ket_new = ket
            for (i, gate) in enumerate(operator)
                ket_new = local_ipeps_update(ket_new, gate, [inds[i]])
            end
        end
        
        return my_contract_local_norm(inds, ket_new, bra, env) / denominator
    end
    
    return sum(term_vals)
end

PEPSKit.add_term!(operator::PEPSKit.LocalOperator, inds::Tuple, term) = PEPSKit.add_term!(operator, collect(inds), term)
PEPSKit.add_term!(operator::PEPSKit.LocalOperator, inds::Vector, term) = PEPSKit.add_term!(operator, map(CartesianIndex{2}, inds), term)