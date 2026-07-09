# Spec: Migrate Magrathea.jl docs to Documenter.jl (BiGSTARS-style)

**Date:** 2026-06-17
**Goal:** Replace the current MkDocs (Material) documentation with a Documenter.jl
site styled to match BiGSTARS.jl, so Magrathea.jl's docs share the same framework,
chrome, navigation, API rendering, and visual design as the author's BiGSTARS.jl.

## Motivation

- BiGSTARS.jl (same author) uses **Documenter.jl** + `bigstars.css`; Magrathea.jl uses
  **MkDocs Material** (Python). The two render fundamentally differently (sidebar,
  nav, typography), so CSS alone cannot make MkDocs look like a Documenter site.
- Documenter is the idiomatic choice for a Julia package and gives native API docs
  (`@docs`/`@autodocs` from docstrings), consistent with BiGSTARS.
- Magrathea's current `extra.css` already uses the BiGSTARS palette and a hero/card/path
  concept, so the visual intent is established — this migration makes it faithful.

## Constraints

- **No SLEPc/MPI in docs CI.** Magrathea's solvers require `backend=:slepc` (the only
  backend; non-slepc paths throw). So any example that *solves* must render as
  **static** code (`execute=false`, plain ` ```julia ` fences) — identical to how
  BiGSTARS handles its SLEPc examples. `using Magrathea` itself loads fine (PetscWrap/
  SlepcWrap are weakdeps), so `@docs` autodocs and non-solving snippets are safe.
- Preserve all existing documentation content; do not drop pages.
- Keep the existing color palette (navy #10233f / blue #226f9f / teal #18a39b /
  gold #d59a26) — already shared between `extra.css` and `bigstars.css`.

## Design

### 1. Build & CI

- **`docs/make.jl`** — mirror BiGSTARS' structure:
  ```julia
  using Documenter, Magrathea
  # (DocumenterCitations only if we add a bibliography; see Open Questions)
  makedocs(
      sitename = "Magrathea.jl",
      modules  = [Magrathea],
      format   = Documenter.HTML(
          prettyurls = get(ENV, "CI", nothing) == "true",
          canonical  = "https://subhk.github.io/Magrathea.jl/stable",
          assets     = ["assets/magrathea.css"],
          collapselevel = 2,
      ),
      pages = PAGES,   # see §2
      warnonly = [:missing_docs, :cross_references],  # keep first migration green
  )
  deploydocs(repo = "github.com/subhk/Magrathea.jl", devbranch = "main", push_preview = false)
  ```
- **`docs/Project.toml`** — `Documenter`; `Magrathea` added via `make.jl` dev path or a
  `docs/Project.toml` with `[deps]` Documenter + a `Pkg.develop(path="..")` step in CI.
- **`.github/workflows/docs.yml`** — replace the MkDocs job with the standard Julia
  Documenter job: setup-julia → instantiate `docs/` → `julia --project=docs docs/make.jl`
  (Documenter's `deploydocs` pushes to `gh-pages`). Set `GITHUB_TOKEN`/`DOCUMENTER_KEY`
  and `permissions: contents: write`.
- **Remove** `mkdocs.yml` and `docs/requirements.txt`.

### 2. Structure

`docs/` content moves to `docs/src/`. `PAGES` in `make.jl` reproduces the current
MkDocs nav:

```julia
PAGES = [
  "Home" => "index.md",
  "Getting Started" => [
    "Installation" => "getting_started.md",
    "First Problem" => "problem_setup.md",
    "Examples" => "examples.md",
  ],
  "Analysis Modes" => [
    "Overview" => "analysis/index.md",
    "Onset Convection" => "analysis/onset_convection.md",
    "Biglobal (Axisymmetric Mean Flow)" => "analysis/biglobal_stability.md",
    "Triglobal (Non-Axisymmetric)" => "analysis/triglobal_stability.md",
  ],
  "User Guide" => [
    "Basic States" => "basic_states.md",
    "Tri-Global Analysis" => "triglobal.md",
    "MHD Extension" => "mhd_extension.md",
  ],
  "Theory" => [
    "Mathematical Foundations" => "theory/mathematical_foundations.md",
    "Spectral Methods" => "theory/spectral_methods.md",
  ],
  "Reference" => [
    "API Reference" => "reference.md",
    "Codebase Structure" => "codebase_structure.md",
    "Migration Guide (v2.0)" => "migration-v2.md",
    "FAQ" => "faq.md",
  ],
]
```

- `docs/MHD_IMPLEMENTATION.md` and `docs/MHD_USER_GUIDE.md` are currently NOT in the
  MkDocs nav (orphans). Decision: fold their content into `mhd_extension.md` / the MHD
  user-guide page, or add them under "User Guide". Default: add `MHD_USER_GUIDE.md`
  under User Guide; merge `MHD_IMPLEMENTATION.md` notes into `codebase_structure.md`.

### 3. Styling

- Port `bigstars.css` → **`docs/src/assets/magrathea.css`**: same `:root` palette; rename
  `--bigstars-*` → `--magrathea-*`; keep the Documenter selectors (`.docs-sidebar`,
  `.docs-menu .tocitem`, `#documenter-page h1/h2/h3`, `pre`/`code`, and the dark-theme
  variants `html.theme--documenter-dark` / catppuccin). Add the `magrathea-hero`,
  `magrathea-card`, `magrathea-card-grid`, `magrathea-path`, `magrathea-button` rules (ported from
  BiGSTARS' `bigstars-*` block).
- **`index.md`** landing page: hero (eyebrow + headline + action buttons) + card-grid +
  learning-path via ` ```@raw html `, mirroring BiGSTARS' `index.md`, with Magrathea's
  content (onset/biglobal/triglobal/MHD feature cards; "what to read first" path).
- Logo/favicon: reuse `docs/assets/magrathea-banner.svg` → `docs/src/assets/`.

### 4. API reference

`reference.md` uses Documenter autodocs from Magrathea's docstrings. Either:
- `@autodocs` with `Modules = [Magrathea]` (everything with a docstring), or
- curated `@docs` blocks grouped by theme (Params, Problems, solve, BasicStates, MHD,
  Operators), matching the export groups in `src/Magrathea.jl`.

Default: curated `@docs` blocks grouped by theme (cleaner, avoids dumping internals);
fall back to `@autodocs` per submodule if coverage gaps appear. `warnonly=[:missing_docs]`
during the first migration so the build stays green while docstring coverage is filled.

### 5. Content conversion rules (per page)

- **Math:** MkDocs `arithmatex` `$…$` / `$$…$$` → Documenter inline `` ``x`` `` and
  block ` ```math `. Documenter ships KaTeX.
- **Admonitions:** `!!! note` / `!!! warning` carry over unchanged (Documenter supports
  the same syntax). Material-specific `??? details` collapsibles → `!!! details`.
- **Material-only HTML / `pymdownx` features** (tabs, content tabs, annotations) →
  `@raw html` or plain markdown equivalents.
- **Links:** `[x](page.md)` relative links keep working in Documenter; section anchors
  re-checked (Documenter slugifies differently). `repo_url`/edit links handled by
  `deploydocs`/`HTML(repolink=…)`.
- **Code copy / highlight:** native in Documenter; drop the `content.code.copy` etc.
  MkDocs feature flags.

### 6. Examples

`examples.md` (and any solving snippet in getting_started/problem_setup) renders as
**static** ` ```julia ` blocks (no `@example`/`@repl` execution), because solving needs
SLEPc. A short note states how to run them locally (`mpiexec … julia …`), mirroring
BiGSTARS. No Literate dependency (Magrathea has no `examples/*.jl`).

## Out of scope

- No new documentation content beyond what exists (only conversion + the restyled
  landing page).
- No change to package source (`src/`).
- The unrelated CPU-correctness audit (separate branch/PR `audit/cpu-correctness-fixes`).

## Success criteria

- `julia --project=docs docs/make.jl` builds locally with no errors (warnings allowed
  via `warnonly` initially).
- Site renders with the BiGSTARS-style hero/cards/path landing page and `magrathea.css`
  palette in both light and dark Documenter themes.
- All current pages present and navigable; API reference shows Magrathea docstrings.
- CI `docs.yml` builds and deploys via `deploydocs` to GitHub Pages.
- MkDocs files (`mkdocs.yml`, `docs/requirements.txt`) removed.

## Open questions

1. **Bibliography:** BiGSTARS uses `DocumenterCitations` + `references.bib`. Does Magrathea
   want citations? Default: NO (Magrathea docs currently have none) — skip DocumenterCitations
   unless a `references.bib` is desired.
2. **API style:** curated `@docs` groups (default) vs blanket `@autodocs`.
3. **Orphan MHD pages:** fold-in vs add-to-nav (default: add USER_GUIDE, merge IMPLEMENTATION).
4. **Versioned docs:** keep `/stable` + `/dev` (Documenter default via `deploydocs`)?
   Default: yes (standard).
