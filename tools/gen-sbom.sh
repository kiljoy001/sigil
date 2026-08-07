#!/bin/sh
# Generate an SPDX 2.3 SBOM for sigil.
#
# sigil has no third-party runtime dependencies: no libc beyond the platform's
# own, no crypto library, no 9P library. SHA-1 is implemented in src/sigil.c
# precisely so the dependency list stays empty. That is the main thing this
# document records, and it is worth being able to prove rather than assert.
#
# The toolchain is captured as a build-time relationship because it affects the
# artifact (AVX2 codegen in particular) without being linked into it.

set -eu

cd "$(dirname "$0")/.."

# git prints partial output before failing on a repo with no commits, so
# squash whitespace rather than trusting the exit status alone.
version=$(git describe --tags --always --dirty 2>/dev/null | tr -d '\n\r "\\' )
commit=$(git rev-parse HEAD 2>/dev/null | tr -d '\n\r "\\' )
[ -n "$version" ] || version="unknown"
[ -n "$commit" ] || commit="unknown"
created=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cc_ver=$(${CC:-cc} --version 2>/dev/null | head -1 | tr -d '"\\' || echo "unknown")

# Hash every source file so the SBOM pins actual content, not just a version.
emit_files() {
	first=1
	for f in include/sigil.h include/sigil_embed.h src/sigil.c src/trit.c \
	         src/store.c src/scan_scalar.c src/scan_x86.c src/scan_neon.c src/scan_generic.c src/simhash.c \
	         src/embed_llama.c; do
		[ -f "$f" ] || continue
		sha=$(sha256sum "$f" | cut -d' ' -f1)
		# SPDXIDs allow only letters, digits, '.' and '-'.
		id=$(printf '%s' "$f" | tr '/_' '--' | tr -cd 'A-Za-z0-9.-')
		[ $first -eq 1 ] || printf ',\n'
		first=0
		cat <<EOF
    {
      "SPDXID": "SPDXRef-File$id",
      "fileName": "./$f",
      "checksums": [{ "algorithm": "SHA256", "checksumValue": "$sha" }],
      "licenseConcluded": "GPL-3.0-or-later",
      "copyrightText": "Copyright (c) 2026 Scott J Guyton"
    }
EOF
	done
}

cat <<EOF
{
  "spdxVersion": "SPDX-2.3",
  "dataLicense": "CC0-1.0",
  "SPDXID": "SPDXRef-DOCUMENT",
  "name": "sigil-$version",
  "documentNamespace": "https://github.com/kiljoy001/sigil/spdx/$commit",
  "creationInfo": {
    "created": "$created",
    "creators": ["Tool: sigil/tools/gen-sbom.sh"]
  },
  "packages": [
    {
      "SPDXID": "SPDXRef-Package-sigil",
      "name": "sigil",
      "versionInfo": "$version",
      "downloadLocation": "https://github.com/kiljoy001/sigil",
      "filesAnalyzed": false,
      "licenseConcluded": "GPL-3.0-or-later",
      "licenseDeclared": "GPL-3.0-or-later",
      "copyrightText": "Copyright (c) 2026 Scott J Guyton",
      "comment": "Core library has no third-party runtime dependencies: SHA-1 is implemented in-tree, with no link-time dependency on a crypto library. Semantic embedding is an optional backend requiring llama.cpp (see SPDXRef-Package-llama-cpp); without it the library builds and links, and sigil_embedder_llama() returns NULL."
    },
    {
      "SPDXID": "SPDXRef-Package-llama-cpp",
      "name": "llama.cpp",
      "versionInfo": "NOASSERTION",
      "downloadLocation": "https://github.com/ggml-org/llama.cpp",
      "filesAnalyzed": false,
      "licenseConcluded": "MIT",
      "licenseDeclared": "MIT",
      "copyrightText": "Copyright (c) 2023-2025 The ggml authors",
      "comment": "OPTIONAL build-time and runtime dependency, enabled by SIGIL_WITH_LLAMA. Dynamically linked (libllama.so, libggml.so, libggml-base.so). Not vendored; not required to build the core library."
    },
    {
      "SPDXID": "SPDXRef-Package-minilm",
      "name": "all-MiniLM-L6-v2",
      "versionInfo": "NOASSERTION",
      "downloadLocation": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
      "filesAnalyzed": false,
      "licenseConcluded": "Apache-2.0",
      "licenseDeclared": "Apache-2.0",
      "copyrightText": "NOASSERTION",
      "comment": "OPTIONAL model weights, not distributed with this source. Supplied by the user at runtime as a GGUF file and loaded by the llama.cpp backend. Recorded here because the LSH bits a store contains depend on which model produced them."
    }
  ],
  "files": [
$(emit_files)
  ],
  "relationships": [
    {
      "spdxElementId": "SPDXRef-DOCUMENT",
      "relationshipType": "DESCRIBES",
      "relatedSpdxElement": "SPDXRef-Package-sigil"
    },
    {
      "spdxElementId": "SPDXRef-Package-sigil",
      "relationshipType": "OPTIONAL_DEPENDENCY_OF",
      "relatedSpdxElement": "SPDXRef-Package-llama-cpp"
    },
    {
      "spdxElementId": "SPDXRef-Package-sigil",
      "relationshipType": "OPTIONAL_DEPENDENCY_OF",
      "relatedSpdxElement": "SPDXRef-Package-minilm"
    }
  ],
  "annotations": [
    {
      "annotationType": "OTHER",
      "annotator": "Tool: sigil/tools/gen-sbom.sh",
      "annotationDate": "$created",
      "comment": "Build toolchain: $cc_ver. Requires x86-64 with AVX2 for the SIMD kernels; scalar fallbacks build everywhere."
    }
  ]
}
EOF
