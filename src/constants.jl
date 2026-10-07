#=
Shared ACO defaults for src/ and bin/.
Edit the values here; callers should reference these names instead of
hardcoding the same numbers.
=#

const __CONSTANTS_JL__ = true

# Exponent on heuristic desirability η in τ · η^ETA_EXP.
const ETA_EXP = 2
