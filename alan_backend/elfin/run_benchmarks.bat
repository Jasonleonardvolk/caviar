@echo off
echo Running ELFIN Benchmark Suite...
python -m alan_backend.elfin.benchmarks.run %*
echo Benchmark suite completed.
