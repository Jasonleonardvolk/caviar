# Find-All-Extensions.ps1
# Discovers all file extensions in the project (excluding node_modules, venv, etc.)

$excludePaths = @(
    "*node_modules*",
    "*venv*",
    "*.venv*",
    "*__pycache__*",
    "*.git*",
    "*_NONREPO_LARGE*",
    "*target*",
    "*.svelte-kit*",
    "*build*",
    "*dist*"
)

$extensions = @{}

Get-ChildItem -Path "D:\Dev\kha" -Recurse -File | Where-Object {
    $path = $_.FullName
    $skip = $false
    foreach ($exclude in $excludePaths) {
        if ($path -like $exclude) {
            $skip = $true
            break
        }
    }
    -not $skip
} | ForEach-Object {
    $ext = $_.Extension.ToLower()
    if ($ext) {
        if (-not $extensions.ContainsKey($ext)) {
            $extensions[$ext] = 0
        }
        $extensions[$ext]++
    }
}

Write-Host "`n=== FILE EXTENSIONS IN PROJECT ===" -ForegroundColor Cyan
$extensions.GetEnumerator() | Sort-Object Value -Descending | ForEach-Object {
    $category = switch -Wildcard ($_.Key) {
        ".ts" { "TypeScript" }
        ".tsx" { "TypeScript JSX" }
        ".js" { "JavaScript" }
        ".jsx" { "JavaScript JSX" }
        ".mjs" { "ES Module" }
        ".cjs" { "CommonJS" }
        ".svelte" { "Svelte" }
        ".vue" { "Vue" }
        ".py" { "Python" }
        ".pyx" { "Cython" }
        ".pyi" { "Python stub" }
        ".rs" { "Rust" }
        ".toml" { "TOML config" }
        ".wgsl" { "WebGPU Shader" }
        ".glsl" { "OpenGL Shader" }
        ".hlsl" { "DirectX Shader" }
        ".vert" { "Vertex Shader" }
        ".frag" { "Fragment Shader" }
        ".comp" { "Compute Shader" }
        ".spv" { "SPIR-V" }
        ".css" { "Stylesheet" }
        ".scss" { "Sass" }
        ".sass" { "Sass" }
        ".less" { "Less" }
        ".html" { "HTML" }
        ".md" { "Markdown" }
        ".mdx" { "MDX" }
        ".json" { "JSON" }
        ".jsonl" { "JSON Lines" }
        ".yaml" { "YAML" }
        ".yml" { "YAML" }
        ".xml" { "XML" }
        ".sh" { "Shell script" }
        ".bash" { "Bash script" }
        ".ps1" { "PowerShell" }
        ".psm1" { "PowerShell module" }
        ".bat" { "Batch file" }
        ".cmd" { "Command file" }
        ".c" { "C" }
        ".cpp" { "C++" }
        ".cc" { "C++" }
        ".cxx" { "C++" }
        ".h" { "C/C++ header" }
        ".hpp" { "C++ header" }
        ".hxx" { "C++ header" }
        ".cu" { "CUDA" }
        ".cuh" { "CUDA header" }
        ".cl" { "OpenCL" }
        ".metal" { "Metal shader" }
        ".java" { "Java" }
        ".kt" { "Kotlin" }
        ".go" { "Go" }
        ".dart" { "Dart" }
        ".swift" { "Swift" }
        ".m" { "Objective-C" }
        ".mm" { "Objective-C++" }
        ".cs" { "C#" }
        ".fs" { "F#" }
        ".vb" { "Visual Basic" }
        ".rb" { "Ruby" }
        ".php" { "PHP" }
        ".pl" { "Perl" }
        ".lua" { "Lua" }
        ".r" { "R" }
        ".jl" { "Julia" }
        ".scala" { "Scala" }
        ".clj" { "Clojure" }
        ".elm" { "Elm" }
        ".ex" { "Elixir" }
        ".exs" { "Elixir script" }
        ".erl" { "Erlang" }
        ".hrl" { "Erlang header" }
        ".zig" { "Zig" }
        ".nim" { "Nim" }
        ".v" { "V" }
        ".sol" { "Solidity" }
        ".asm" { "Assembly" }
        ".s" { "Assembly" }
        ".wat" { "WebAssembly Text" }
        ".wasm" { "WebAssembly" }
        ".proto" { "Protocol Buffers" }
        ".graphql" { "GraphQL" }
        ".gql" { "GraphQL" }
        ".sql" { "SQL" }
        ".prisma" { "Prisma schema" }
        ".env" { "Environment vars" }
        ".ini" { "INI config" }
        ".cfg" { "Config" }
        ".conf" { "Config" }
        ".properties" { "Properties" }
        ".gradle" { "Gradle" }
        ".cmake" { "CMake" }
        ".make" { "Makefile" }
        ".dockerfile" { "Dockerfile" }
        ".containerfile" { "Containerfile" }
        ".tf" { "Terraform" }
        ".tfvars" { "Terraform vars" }
        ".nix" { "Nix" }
        ".vim" { "Vim script" }
        ".el" { "Emacs Lisp" }
        default { "Other" }
    }
    
    Write-Host ("{0,-15} {1,6} files  ({2})" -f $_.Key, $_.Value, $category)
}

Write-Host "`nTotal unique extensions: $($extensions.Count)" -ForegroundColor Yellow