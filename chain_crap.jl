function ChainRulesCore.rrule(::typeof(PEPSKit.unitcell), state::InfinitePEPS)
    tensors = PEPSKit.unitcell(state)
    function unitcell_pullback(Δtensors_)
        Δtensors = unthunk(Δtensors_)
        return NoTangent(), InfinitePEPS(Δtensors)
    end
    return tensors, unitcell_pullback
end

function ChainRulesCore.rrule(::Type{InfinitePEPS}, A::Matrix{<:PEPSKit.PEPSTensor})
    network = InfinitePEPS(A)
    function InfinitePEPS_pullback(Δnetwork_)
        Δnetwork = unthunk(Δnetwork_)
        return NoTangent(), PEPSKit.unitcell(Δnetwork)
    end
    return network, InfinitePEPS_pullback
end

# function ChainRulesCore.rrule(::Type{InfiniteSquareNetwork}, A::Matrix)
#     network = InfiniteSquareNetwork(A)
#     function InfiniteSquareNetwork_pullback(Δnetwork_)
#         Δnetwork = unthunk(Δnetwork_)
#         return NoTangent(), PEPSKit.unitcell(Δnetwork)
#     end
#     return network, InfiniteSquareNetwork_pullback
# end

# Add rrules for constructors to ensure consistent tangent types
# function ChainRulesCore.rrule(::Type{InfinitePEPS}, A::Matrix)
#     peps = InfinitePEPS(A)
#     function InfinitePEPS_pullback(ȳ)
#         if isa(ȳ, InfinitePEPS)
#             ΔA = ȳ.A
#         else
#             ΔA = ȳ.A
#         end
#         return (NoTangent(), ΔA)
#     end
#     return peps, InfinitePEPS_pullback
# end

# function ChainRulesCore.rrule(::Type{InfiniteSquareNetwork}, A::Matrix)
#     network = InfiniteSquareNetwork(A)
#     function InfiniteSquareNetwork_pullback(ȳ)
#         if isa(ȳ, InfiniteSquareNetwork)
#             ΔA = ȳ.A
#         else
#             ΔA = ȳ.A
#         end
#         return (NoTangent(), ΔA)
#     end
#     return network, InfiniteSquareNetwork_pullback
# end
