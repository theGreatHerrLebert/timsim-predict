"""timsim-predict — the CLI prediction jobs (CCS, RT, fragment intensity) for the timsim v2 pipeline.
Thin parquet-I/O wrappers over the `pepdl` inference predictors; the necroflow DAG calls the timsim-* entry
points. Depends only on pepdl (+ its mscorepy primitives) — no imspy-simulation/dia/search/vis."""
__all__ = []
