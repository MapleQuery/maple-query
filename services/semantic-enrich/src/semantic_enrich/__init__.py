"""MapleQuery semantic enrichment runtime.

Two functions and a smoke test: `generate_json` (outlines-backed
guided JSON generation) and `embed_batch` (sentence-transformers
L2-normalised 1024-dim vectors). Imported as a library by the
downstream enrichment pipeline; runs as a CLI for the smoke test.
"""
__version__ = "0.1.0"
