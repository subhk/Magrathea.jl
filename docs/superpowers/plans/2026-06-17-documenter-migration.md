# Documenter.jl Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Magrathea.jl's MkDocs (Material) documentation with a Documenter.jl site styled to match BiGSTARS.jl.

**Architecture:** Standard Julia `docs/` Documenter setup (`make.jl` + `Project.toml`, content under `docs/src/`, `deploydocs` to `gh-pages`). Visual parity via a ported `magrathea.css` (= BiGSTARS' `bigstars.css` with the palette-variable prefix renamed) and a hero/card/learning-path landing page built with ` ```@raw html `. API docs via curated `@docs` blocks over Magrathea's docstrings.

**Tech Stack:** Julia 1.12, Documenter.jl, KaTeX (Documenter built-in), GitHub Pages.

**Conventions:**
- Local build uses the versioned Julia binary (juliaup launcher is broken on this machine):
  `JULIA=/Users/subha/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/Julia-1.12.app/Contents/Resources/julia/bin/julia`
- Build command (run from repo root):
  `"$JULIA" --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'` (once)
  then `"$JULIA" --project=docs docs/make.jl`
- "Build is green" = `make.jl` exits 0 and prints no `Error:`; warnings tolerated initially via `warnonly`.
- `docs/superpowers/` is NOT part of the site (plans/specs) — never moved into `docs/src/`.

---

## Task 1: Scaffold Documenter and get a green build with existing content

**Files:**
- Create: `docs/Project.toml`
- Create: `docs/make.jl`
- Move (git mv): `docs/*.md`, `docs/analysis/`, `docs/theory/`, `docs/assets/` → under `docs/src/`
  (NOT `docs/superpowers/`, NOT `docs/MHD_IMPLEMENTATION.md`/`docs/MHD_USER_GUIDE.md` yet — Task 6)

- [ ] **Step 1: Create `docs/Project.toml`**

```toml
[deps]
Documenter = "e30172f5-a6a5-5a46-863b-614d45cd2de4"

[compat]
Documenter = "1"
```

- [ ] **Step 2: Move existing content into `docs/src/`**

```bash
mkdir -p docs/src
git mv docs/index.md docs/getting_started.md docs/problem_setup.md docs/examples.md \
       docs/basic_states.md docs/triglobal.md docs/mhd_extension.md \
       docs/codebase_structure.md docs/migration-v2.md docs/faq.md docs/reference.md docs/src/
git mv docs/analysis docs/theory docs/assets docs/src/
```
(Leave `docs/superpowers/`, `docs/MHD_IMPLEMENTATION.md`, `docs/MHD_USER_GUIDE.md` in place.)

- [ ] **Step 3: Create `docs/make.jl`**

```julia
using Documenter
using Magrathea

makedocs(
    sitename = "Magrathea.jl",
    authors  = "Subhajit Kar",
    modules  = [Magrathea],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical  = "https://subhk.github.io/Magrathea.jl/stable",
        assets     = ["assets/magrathea.css"],
        collapselevel = 2,
        sidebar_sitename = false,
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "Installation" => "getting_started.md",
            "First Problem" => "problem_setup.md",
            "Examples" => "examples.md",
        ],
        "Analysis Modes" => [
            "Overview" => "analysis/index.md",
            "Onset Convection (No Mean Flow)" => "analysis/onset_convection.md",
            "Biglobal (Axisymmetric Mean Flow)" => "analysis/biglobal_stability.md",
            "Triglobal (Non-Axisymmetric Mean Flow)" => "analysis/triglobal_stability.md",
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
    ],
    # First migration: keep the build green while content/docstrings are converted.
    # These are tightened (removed) in Task 7 once conversion is complete.
    warnonly = [:missing_docs, :cross_references, :docs_block],
)

deploydocs(
    repo = "github.com/subhk/Magrathea.jl",
    devbranch = "main",
    push_preview = false,
)
```

- [ ] **Step 4: Instantiate the docs environment**

Run:
```bash
JULIA=/Users/subha/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/Julia-1.12.app/Contents/Resources/julia/bin/julia
"$JULIA" --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
```
Expected: resolves Documenter + Magrathea, no error.

- [ ] **Step 5: Build and verify it is green**

Run: `"$JULIA" --project=docs docs/make.jl`
Expected: exits 0; creates `docs/build/index.html`; no lines starting with `Error:`. (Warnings from `@raw`/Material leftovers are tolerated by `warnonly` and fixed in Tasks 3/5.)

- [ ] **Step 6: Commit**

```bash
git add docs/Project.toml docs/make.jl docs/src
git commit -m "docs: scaffold Documenter build, move content to docs/src"
```

---

## Task 2: Port `bigstars.css` → `magrathea.css`

**Files:**
- Create: `docs/src/assets/magrathea.css`

- [ ] **Step 1: Create `docs/src/assets/magrathea.css`** — this is BiGSTARS' `bigstars.css` verbatim with every `bigstars` token renamed to `magrathea` (palette values unchanged; they already match Magrathea's `extra.css`):

```css
/* Magrathea.jl documentation polish.
   Layers on top of Documenter's default themes (ported from BiGSTARS bigstars.css). */

:root {
  --magrathea-navy: #10233f;
  --magrathea-blue: #226f9f;
  --magrathea-teal: #18a39b;
  --magrathea-gold: #d59a26;
  --magrathea-ink: #142033;
  --magrathea-muted: #5c6c80;
  --magrathea-panel: #f7fafc;
  --magrathea-border: rgba(34, 111, 159, 0.18);
  --magrathea-shadow: 0 18px 45px rgba(16, 35, 63, 0.12);
}

html.theme--documenter-dark,
html.theme--catppuccin-mocha,
html.theme--catppuccin-macchiato,
html.theme--catppuccin-frappe {
  --magrathea-navy: #dbeafe;
  --magrathea-blue: #6cc8ff;
  --magrathea-teal: #5eead4;
  --magrathea-gold: #f8d477;
  --magrathea-ink: #e9f2ff;
  --magrathea-muted: #b7c5d8;
  --magrathea-panel: rgba(255, 255, 255, 0.06);
  --magrathea-border: rgba(117, 211, 255, 0.22);
  --magrathea-shadow: 0 18px 45px rgba(0, 0, 0, 0.32);
}

.docs-sidebar { border-right: 1px solid var(--magrathea-border); }
.docs-package-name { letter-spacing: 0.02em; }
.docs-menu .tocitem { border-radius: 9px; }
.docs-menu .is-active > .tocitem,
.docs-menu .tocitem:hover {
  background: linear-gradient(90deg, rgba(34, 111, 159, 0.16), rgba(24, 163, 155, 0.10));
}

#documenter-page h1,
#documenter-page h2,
#documenter-page h3 { color: var(--magrathea-ink); letter-spacing: -0.015em; }
#documenter-page h2 { border-bottom: 1px solid var(--magrathea-border); padding-bottom: 0.28rem; }
#documenter-page p,
#documenter-page li { line-height: 1.72; }
#documenter-page code { border-radius: 6px; }
#documenter-page pre {
  border: 1px solid var(--magrathea-border);
  border-radius: 14px;
  box-shadow: 0 10px 24px rgba(16, 35, 63, 0.08);
}

.magrathea-hero {
  position: relative; overflow: hidden; margin: 0.6rem 0 2rem; padding: 2.2rem;
  border: 1px solid var(--magrathea-border); border-radius: 24px;
  background:
    radial-gradient(circle at 14% 18%, rgba(24, 163, 155, 0.26), transparent 28%),
    radial-gradient(circle at 86% 12%, rgba(213, 154, 38, 0.24), transparent 30%),
    linear-gradient(135deg, rgba(34, 111, 159, 0.14), rgba(24, 163, 155, 0.08) 52%, rgba(213, 154, 38, 0.08));
  box-shadow: var(--magrathea-shadow);
}
.magrathea-hero::after {
  content: ""; position: absolute; right: -6rem; bottom: -7rem; width: 18rem; height: 18rem;
  border: 1px solid rgba(34, 111, 159, 0.22); border-radius: 50%;
  background: repeating-linear-gradient(35deg, rgba(34,111,159,0.12), rgba(34,111,159,0.12) 2px, transparent 2px, transparent 12px);
  pointer-events: none;
}
.magrathea-eyebrow {
  margin-bottom: 0.7rem; color: var(--magrathea-blue); font-size: 0.78rem; font-weight: 800;
  letter-spacing: 0.14em; text-transform: uppercase;
}
.magrathea-hero h1 { margin: 0; max-width: 760px; color: var(--magrathea-ink); font-size: clamp(2rem, 5vw, 3.8rem); line-height: 1.04; }
.magrathea-hero p { position: relative; max-width: 720px; margin-top: 1rem; color: var(--magrathea-muted); font-size: 1.08rem; }

.magrathea-actions, .magrathea-card-grid, .magrathea-path { display: grid; gap: 1rem; }
.magrathea-actions { grid-template-columns: repeat(auto-fit, minmax(180px, max-content)); margin-top: 1.35rem; }
.magrathea-button {
  display: inline-flex; align-items: center; justify-content: center; min-height: 2.75rem;
  padding: 0.75rem 1rem; border-radius: 999px; border: 1px solid var(--magrathea-border);
  font-weight: 800; text-decoration: none !important;
}
.magrathea-button.primary {
  color: white !important;
  background: linear-gradient(135deg, var(--magrathea-blue), var(--magrathea-teal));
  box-shadow: 0 10px 25px rgba(34, 111, 159, 0.26);
}
.magrathea-button.secondary { color: var(--magrathea-blue) !important; background: rgba(255, 255, 255, 0.58); }

.magrathea-card-grid { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin: 1.4rem 0 2rem; }
.magrathea-card {
  padding: 1.1rem; border: 1px solid var(--magrathea-border); border-radius: 18px;
  background: var(--magrathea-panel); box-shadow: 0 10px 24px rgba(16, 35, 63, 0.06);
}
.magrathea-card strong { display: block; margin-bottom: 0.35rem; color: var(--magrathea-ink); font-size: 1.02rem; }
.magrathea-card p { margin: 0; color: var(--magrathea-muted); font-size: 0.95rem; }

.magrathea-path { grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); counter-reset: docs-path; margin: 1.1rem 0 2rem; }
.magrathea-step {
  position: relative; padding: 1rem 1rem 1rem 3.2rem; border: 1px solid var(--magrathea-border);
  border-radius: 16px; background: linear-gradient(180deg, var(--magrathea-panel), transparent);
}
.magrathea-step::before {
  counter-increment: docs-path; content: counter(docs-path); position: absolute; top: 1rem; left: 1rem;
  display: grid; width: 1.7rem; height: 1.7rem; place-items: center; border-radius: 50%;
  color: white; background: var(--magrathea-blue); font-weight: 800;
}
.magrathea-step a { font-weight: 800; }
.magrathea-step p { margin: 0.25rem 0 0; color: var(--magrathea-muted); font-size: 0.94rem; }

@media (max-width: 640px) {
  .magrathea-hero { padding: 1.4rem; border-radius: 18px; }
  .magrathea-actions { grid-template-columns: 1fr; }
}
```

- [ ] **Step 2: Build and verify the asset is bundled**

Run: `"$JULIA" --project=docs docs/make.jl && ls docs/build/assets/magrathea.css`
Expected: green build; `docs/build/assets/magrathea.css` exists.

- [ ] **Step 3: Commit**

```bash
git add docs/src/assets/magrathea.css
git commit -m "docs: port BiGSTARS bigstars.css to magrathea.css"
```

---

## Task 3: BiGSTARS-style landing page (`docs/src/index.md`)

**Files:**
- Modify: `docs/src/index.md` (replace the head with hero + cards + path; keep the existing prose body below)

- [ ] **Step 1: Replace the top of `docs/src/index.md`** with the hero/cards/path (Magrathea content), mirroring BiGSTARS' index. Put this ABOVE the existing overview prose:

````markdown
# Magrathea.jl Documentation

```@raw html
<div class="magrathea-hero">
  <div class="magrathea-eyebrow">Linear stability in rotating spherical shells</div>
  <h1>Spectral eigenvalue problems for rotating convection &amp; MHD.</h1>
  <p>
    Magrathea.jl uses the Olver&ndash;Townsend ultraspherical method to build ultra-sparse,
    spurious-free generalized eigenvalue problems for onset, biglobal, and triglobal
    stability of rotating (magneto)convection in spherical shells.
  </p>
  <div class="magrathea-actions">
    <a class="magrathea-button primary" href="getting_started.html">Get started</a>
    <a class="magrathea-button secondary" href="examples.html">See examples</a>
  </div>
</div>
```

```@raw html
<div class="magrathea-card-grid">
  <div class="magrathea-card">
    <strong>Onset convection</strong>
    <p>Conductive background, single azimuthal wavenumber; find critical Rayleigh numbers.</p>
  </div>
  <div class="magrathea-card">
    <strong>Biglobal (axisymmetric mean flow)</strong>
    <p>Thermal-wind / meridional basic states with an axisymmetric background.</p>
  </div>
  <div class="magrathea-card">
    <strong>Triglobal (non-axisymmetric)</strong>
    <p>Mode-coupled stability for non-axisymmetric basic states across azimuthal wavenumbers.</p>
  </div>
  <div class="magrathea-card">
    <strong>MHD extension</strong>
    <p>Magnetoconvection and kinematic-dynamo problems with no_field, axial, and dipole fields.</p>
  </div>
  <div class="magrathea-card">
    <strong>Spurious-free Galerkin</strong>
    <p>Banded BC-recombined discretization removes the tau spurious-mode swarm; matches collocation to ~1e-12.</p>
  </div>
  <div class="magrathea-card">
    <strong>Unified solver API</strong>
    <p>One <code>solve(problem)</code> entry point across all problem types, returning a <code>StabilityResult</code>.</p>
  </div>
</div>
```

## What To Read First

```@raw html
<div class="magrathea-path">
  <div class="magrathea-step">
    <a href="getting_started.html">Installation</a>
    <p>Install Magrathea.jl and verify your setup.</p>
  </div>
  <div class="magrathea-step">
    <a href="problem_setup.html">First Problem</a>
    <p>Define an OnsetProblem and solve for leading eigenvalues.</p>
  </div>
  <div class="magrathea-step">
    <a href="analysis/index.html">Analysis Modes</a>
    <p>Pick onset, biglobal, or triglobal for your background state.</p>
  </div>
  <div class="magrathea-step">
    <a href="reference.html">API Reference</a>
    <p>Full parameter, problem, and solver reference.</p>
  </div>
</div>
```
````

(Keep the existing "## Overview" and the rest of the current index.md prose below this block. Remove any MkDocs-Material-specific landing HTML that was previously there.)

- [ ] **Step 2: Build and eyeball**

Run: `"$JULIA" --project=docs docs/make.jl`
Expected: green; open `docs/build/index.html` in a browser — hero, 6 cards, and 4-step path render with the navy/teal/gold palette in both light and dark themes (toggle top-right).

- [ ] **Step 3: Commit**

```bash
git add docs/src/index.md
git commit -m "docs: BiGSTARS-style hero/cards/path landing page"
```

---

## Task 4: API reference via `@docs` (`docs/src/reference.md`)

**Files:**
- Modify: `docs/src/reference.md`

- [ ] **Step 1: Replace `docs/src/reference.md`** with curated `@docs` blocks grouped by theme. Use the exported names from `src/Magrathea.jl`'s `export` block. (Any name whose docstring is missing will surface as a `:missing_docs` warning, tolerated for now by `warnonly`; drop those names or add docstrings in Task 7.)

````markdown
# API Reference

```@meta
CurrentModule = Magrathea
```

## Parameters

```@docs
OnsetParams
OnsetConvectionParams
BiglobalParams
TriglobalParams
MHDParams
```

## Problems & solve

```@docs
OnsetProblem
BiglobalProblem
TriglobalProblem
MHDProblem
solve
estimate_size
StabilityResult
```

## Critical-parameter search

```@docs
find_critical_Ra
find_critical_Ra_onset
find_critical_Ra_biglobal
find_critical_rayleigh_triglobal
```

## Basic states

```@docs
basic_state
conduction_basic_state
meridional_basic_state
nonaxisymmetric_basic_state
basic_state_selfconsistent
BasicState
BasicState3D
SphericalHarmonicBC
```

## Spectral & operators

```@docs
ChebyshevDiffn
LinearStabilityOperator
assemble_matrices
```
````

(If a listed name is not actually exported/defined, the build prints `Error: ... @docs ... no docs found`; with `warnonly=[:docs_block]` it stays green — remove that name. Verify against `src/Magrathea.jl` while editing.)

- [ ] **Step 2: Build and check the API renders**

Run: `"$JULIA" --project=docs docs/make.jl`
Expected: green; `docs/build/reference/index.html` shows rendered docstrings (signatures + bodies) grouped under the headings above.

- [ ] **Step 3: Commit**

```bash
git add docs/src/reference.md
git commit -m "docs: API reference via Documenter @docs blocks"
```

---

## Task 5: Convert MkDocs-specific content per page

**Files:**
- Modify: every `docs/src/**/*.md` that uses MkDocs-only syntax (math, content tabs, etc.)

Conversion rules (apply mechanically, then rebuild until clean):

| MkDocs / Material | Documenter |
|---|---|
| inline math `$x$` | `` ``x`` `` (double backtick) |
| block math `$$ … $$` | ` ```math ` fenced block |
| `!!! note "Title"` / `!!! warning` | unchanged (Documenter supports these) |
| `??? note` (collapsible) | `!!! details "Title"` |
| `=== "Tab A"` content tabs | split into sequential sections or `@raw html` |
| `{ .annotate }`, code annotations `# (1)!` | remove the annotation markup |
| `[[wikilinks]]`, `attr_list` `{: …}` | plain markdown / remove |
| emoji shortcodes `:rocket:` | literal emoji or remove |

- [ ] **Step 1: Find MkDocs-only markup**

Run:
```bash
grep -rnE '\$\$?|^\s*===|\{[.:]|\(1\)!|^\s*\?\?\?' docs/src --include='*.md' | grep -v assets
```
This lists the lines to convert. Work through each file.

- [ ] **Step 2: Convert math and admonitions** in each flagged file per the table above. For example, in `docs/src/theory/mathematical_foundations.md` and `theory/spectral_methods.md` (math-heavy), replace `$…$` → `` ``…`` `` and `$$…$$` → ` ```math ` blocks.

- [ ] **Step 3: Rebuild and fix cross-reference / parse errors**

Run: `"$JULIA" --project=docs docs/make.jl 2>&1 | grep -iE 'warning|error' | head -50`
Iterate: fix broken links (`[x](y.md)` → ensure target exists under `docs/src/`), malformed math, and any `@raw`/parse warnings until the list is empty (or only benign `:missing_docs`).

- [ ] **Step 4: Commit**

```bash
git add docs/src
git commit -m "docs: convert MkDocs math/admonitions/tabs to Documenter"
```

---

## Task 6: Orphan MHD pages, CI workflow, and MkDocs cleanup

**Files:**
- Move/merge: `docs/MHD_USER_GUIDE.md` → `docs/src/mhd_user_guide.md` (add to nav); fold `docs/MHD_IMPLEMENTATION.md` into `docs/src/codebase_structure.md`
- Modify: `docs/make.jl` (add the MHD user-guide page)
- Rewrite: `.github/workflows/docs.yml`
- Delete: `mkdocs.yml`, `docs/requirements.txt`

- [ ] **Step 1: Handle the orphan MHD pages**

```bash
git mv docs/MHD_USER_GUIDE.md docs/src/mhd_user_guide.md
```
Append the contents of `docs/MHD_IMPLEMENTATION.md` to the end of `docs/src/codebase_structure.md` (under a new `## MHD implementation notes` heading), then `git rm docs/MHD_IMPLEMENTATION.md`.

- [ ] **Step 2: Add the MHD user guide to `pages` in `docs/make.jl`** — under "User Guide", after `mhd_extension.md`:

```julia
            "MHD User Guide" => "mhd_user_guide.md",
```

- [ ] **Step 3: Replace `.github/workflows/docs.yml`** with the standard Documenter deploy:

```yaml
name: Documentation

on:
  push:
    branches: [main]
    tags: ['*']
  pull_request:

permissions:
  contents: write
  pull-requests: read
  statuses: write

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
        with:
          version: '1.10'
      - uses: julia-actions/cache@v2
      - name: Instantiate docs environment
        run: |
          julia --project=docs -e '
            using Pkg
            Pkg.develop(PackageSpec(path=pwd()))
            Pkg.instantiate()'
      - name: Build and deploy
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          DOCUMENTER_KEY: ${{ secrets.DOCUMENTER_KEY }}
        run: julia --project=docs docs/make.jl
```

- [ ] **Step 4: Remove MkDocs files**

```bash
git rm mkdocs.yml docs/requirements.txt
```

- [ ] **Step 5: Build green locally**

Run: `"$JULIA" --project=docs docs/make.jl`
Expected: green; `mhd_user_guide` appears in the sidebar; no reference to removed files.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "docs: Documenter CI workflow, MHD page wiring, remove MkDocs"
```

---

## Task 7: Tighten warnings and final verification

**Files:**
- Modify: `docs/make.jl` (reduce `warnonly`)
- Modify: any page with remaining `:missing_docs` / dead links

- [ ] **Step 1: List remaining warnings**

Run: `"$JULIA" --project=docs docs/make.jl 2>&1 | grep -iE 'warning|error'`

- [ ] **Step 2: Resolve them** — for each `@docs` name with no docstring, either remove it from `reference.md` or add a docstring in `src/`; fix any remaining dead cross-references.

- [ ] **Step 3: Tighten `warnonly` in `docs/make.jl`** — remove `:docs_block` and `:cross_references` (keep `:missing_docs` only if intentional coverage gaps remain, otherwise drop entirely):

```julia
    warnonly = false,
```

- [ ] **Step 4: Final clean build**

Run: `"$JULIA" --project=docs docs/make.jl`
Expected: exits 0 with `warnonly=false` and no warnings/errors. Open `docs/build/index.html` and confirm hero/cards/path + palette in light and dark.

- [ ] **Step 5: Commit**

```bash
git add docs
git commit -m "docs: resolve doc warnings, enforce strict build"
```

---

## Self-review notes (for the implementer)

- Spec coverage: Task 1 (build/CI scaffold + structure), Task 2 (magrathea.css), Task 3 (landing), Task 4 (API @docs), Task 5 (content conversion), Task 6 (CI + orphan pages + MkDocs removal), Task 7 (success criteria: strict green build). All spec sections covered.
- The `warnonly` escape hatch keeps every intermediate build green (bite-sized commits) and is removed in Task 7 — the final gate is a strict build.
- If `Pkg.develop(PackageSpec(path=pwd()))` is undesirable in CI, the equivalent is a `docs/Project.toml` listing Magrathea via a relative `[sources]`/`Pkg.develop` step — the workflow in Task 6 Step 3 already does this at build time.
