# Bipartite Graph Lab

Next.js port of `../visualizer.html` — a client-side visual debugger for
`graph.jl` / `reduction.jl` / opponent search.

## Run

```bash
cd visualizer
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Layout

```
app/                  Next.js app router
components/           UI (Header, Sidebar, GraphCanvas, …)
hooks/useGraphLab.ts  Graph editor state + actions
lib/
  graph.ts            Graph helpers + demo seed
  reduction.ts        Port of common_neighbor_reduction!
  search.ts           k/θ-MDBB branch & bound
  layout.ts           Column layout constants
  io.ts               JSON/CSV load & save
  types.ts            Shared types
```

Algorithms live in `lib/` so you can unit-test or extend them without touching UI.
