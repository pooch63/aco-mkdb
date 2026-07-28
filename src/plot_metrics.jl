using GLMakie

const METRICS_PLOT_PATH = "ga_metrics.png"

function plot_metrics(diversity_history::Vector{Float64}, best_fitness_history::Vector{Float64};
    path::AbstractString=METRICS_PLOT_PATH)
    # Scale best fitness to [0, 1] so it shares an axis with diversity.
    lo, hi = extrema(best_fitness_history)
    span = hi - lo
    scaled_fitness = span == 0 ? fill(1.0, length(best_fitness_history)) :
        [(f - lo) / span for f in best_fitness_history]

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="Generation", ylabel="Value", title="GA Metrics")
    gens = eachindex(diversity_history)

    lines!(ax, gens, diversity_history; label="Diversity", color=:steelblue)
    lines!(ax, gens, scaled_fitness; label="Best fitness (scaled)", color=:darkorange)

    axislegend(ax; position=:rb)

    save(path, fig)
    println("Saved metrics plot to $(abspath(path))")

    display(fig)
    print("Press Enter to close the plot and continue... ")
    readline()

    return fig
end
