# Keep the documented source map synchronized with the checkout being built.
function check_source_map(root=normpath(joinpath(@__DIR__, "..")))
    source = read(joinpath(root, "docs", "src", "codebase_structure.md"), String)
    documented = Set(m.captures[1] for m in eachmatch(r"(?m)^\| `([^`]+)` \|", source))
    paths = filter(p -> startswith(p, "src/") || startswith(p, "ext/") ||
                        startswith(p, "test/") || startswith(p, "docs/") ||
                        startswith(p, "example/") || endswith(p, ".toml"), documented)
    # Manifest.toml may be absent in a fresh checkout before instantiation.
    missing = sort([p for p in paths if p != "Manifest.toml" && !ispath(joinpath(root, p))])
    isempty(missing) || error("Source map references missing paths: $(join(missing, ", "))")
    actual = Set{String}()
    for group in ("src", "ext")
        for (dir, _, files) in walkdir(joinpath(root, group)), file in files
            endswith(file, ".jl") || continue
            push!(actual, replace(relpath(joinpath(dir, file), root), '\\' => '/'))
        end
    end
    omitted = sort!(collect(setdiff(actual, documented)))
    isempty(omitted) || error("Source files missing from the documentation map: $(join(omitted, ", "))")
    @info "Source map checked" files=length(actual)
    return nothing
end
