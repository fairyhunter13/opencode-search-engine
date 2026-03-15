# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for opencode-embedder

Build with:
    pyinstaller opencode-embedder.spec

The resulting binary will be at:
    dist/opencode-embedder
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all tokenizers data files (FastEmbed uses tokenizers)
datas = collect_data_files('tokenizers')

# Include the minimal ONNX test model for GPU provider verification
# This avoids requiring the heavy 'onnx' package at runtime
datas += [('opencode_embedder/test_model.onnx', 'opencode_embedder')]

# GPU provider support (ROCm/MIGraphX)
# IMPORTANT: We use system ROCm libraries at runtime, NOT bundled ones.
# This avoids: 1) 1.9GB+ binary size, 2) version mismatches, 3) slow builds
# The binary will use LD_LIBRARY_PATH=/opt/rocm/lib at runtime.
import os
import site

# No bundled ROCm libs needed - we use system libs at runtime via LD_LIBRARY_PATH

# Hidden imports for model server dependencies
hiddenimports = [
    'tokenizers',
    'msgpack',
    'fastembed',
    'chonkie',
    'langchain_text_splitters',
    'bs4',
    'lxml',
    'yaml',
]

# tree-sitter language pack: chonkie CodeChunker dynamically imports
# tree_sitter_language_pack.bindings.<lang> via importlib.import_module.
# We only bundle the languages listed in chunker._LANG_TO_TREESITTER to
# save ~100 MB vs bundling all 170+ languages.
_TS_LANGS = [
    'astro', 'bash', 'c', 'clojure', 'cmake', 'commonlisp', 'cpp', 'css',
    'd', 'dart', 'dockerfile', 'elixir', 'elm', 'erlang', 'fish', 'fsharp',
    'go', 'graphql', 'groovy', 'haskell', 'java', 'javascript', 'julia',
    'kotlin', 'latex', 'lua', 'make', 'nim', 'ocaml', 'perl', 'php',
    'powershell', 'proto', 'python', 'r', 'racket', 'ruby', 'rust', 'scala',
    'scheme', 'scss', 'sql', 'svelte', 'swift', 'tsx', 'typescript', 'v',
    'vue', 'zig',
]
hiddenimports += ['tree_sitter_language_pack']
hiddenimports += ['tree_sitter_language_pack.bindings']
hiddenimports += [f'tree_sitter_language_pack.bindings.{l}' for l in _TS_LANGS]
# csharp is handled via a separate top-level package
hiddenimports += ['tree_sitter_c_sharp', 'tree_sitter_embedded_template', 'tree_sitter_yaml']
hiddenimports += ['tree_sitter']

a = Analysis(
    ['opencode_embedder/__main__.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    # Runtime hook to set TOKENIZERS_PARALLELISM=false before any imports
    # This prevents deadlock on macOS with the HuggingFace tokenizers library
    runtime_hooks=['hook-pil-stub.py', 'hook-tokenizers.py'],
    excludes=[
        # Exclude heavy optional dependencies we don't use
        'matplotlib',
        'numpy.testing',
        'scipy',
        'PIL',
        'cv2',
        'torch',
        'tensorflow',
        'transformers',
    ],
    noarchive=False,
    optimize=2,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # Onedir mode: binaries collected separately
    name='opencode-embedder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Exclude ALL ROCm/HIP/MIGraphX libraries from bundle - use system libraries at runtime.
# This drastically reduces binary size (from 1.9GB+ to ~200MB) and build time (from 15min to <1min)
# At runtime, set LD_LIBRARY_PATH=/opt/rocm/lib to use system GPU libraries.
#
# IMPORTANT: We exclude EVERYTHING GPU-related including:
# - onnxruntime providers (libonnxruntime_providers_rocm.so, libonnxruntime_providers_migraphx.so)
# - All ROCm libraries (libamdhip64, librocblas, etc.)
# - All MIGraphX libraries (libmigraphx, etc.)
# The CPU provider will work without these, and GPU provider works via system libs.
ROCM_EXCLUDE_PATTERNS = [
    # ONNX Runtime GPU providers - use system libs
    'libonnxruntime_providers_rocm',
    'libonnxruntime_providers_migraphx',
    'libonnxruntime_providers_shared',
    # HIP runtime and libraries
    'libamdhip64',
    'libhipblas',       # All hipblas variants
    'libhipblaslt',
    'libhipfft',
    'libhiprtc',
    'libhipsparse',
    'libhipsolver',
    # MIGraphX libraries
    'libmigraphx',
    # MIOpen (AMD's deep learning library)
    'libMIOpen',
    # ROCm core libraries
    'libamd_comgr',
    'librocm',
    'librocblas',
    'libroctx',
    'libroctracer',
    'librocprofiler',
    'librocsolver',
    'librocsparse',
    'librocfft',
    # HSA runtime
    'libhsa-runtime',
    # DRM
    'libdrm_amdgpu',
    'libdrm',
]

def is_rocm_lib(name):
    """Check if a binary is a ROCm/MIGraphX library that should be excluded."""
    return any(pattern in name for pattern in ROCM_EXCLUDE_PATTERNS)

# Filter out ALL ROCm/MIGraphX libraries - use system libs at runtime
filtered_binaries = [(name, path, typ) for name, path, typ in a.binaries 
                     if not is_rocm_lib(name)]

# Onedir mode - files are pre-extracted for instant startup
# Avoids the 40+ second extraction delay on macOS with onefile mode
coll = COLLECT(
    exe,
    filtered_binaries,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name='opencode-embedder',
)
