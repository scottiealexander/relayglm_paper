module Progress

export show_progress, clear_lines_above

# ============================================================================ #
function clear_lines_above(numlinesup::Int)
    for _ in 1:numlinesup
        print(stdout, "\u1b[1G\u1b[K\u1b[A")
    end
end
# ============================================================================ #
function show_progress(pcent::Float64, depth::Int=0, prefix::AbstractString="",
        suffix::AbstractString="")

    iocolored = IOContext(stdout, :color => true)

    prefix = "  "^depth * prefix
    print(stdout, "\u1b[1G")   # go to first column
    printstyled(iocolored, prefix * "$(round(Int, 100.0*pcent))% " *
        suffix, color=:yellow)
    print(stdout, "\u1b[K")    # clear the rest of the line
end
# ============================================================================ #
end
