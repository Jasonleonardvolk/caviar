@echo off
echo == Activating venv ==
call .venv\Scripts\activate

echo == Import test ==
python -c "import sys, importlib.util, platform; print('Python:', sys.executable); spec = importlib.util.find_spec('penrose_engine_rs'); print('penrose_engine_rs spec:', spec); print('Platform:', platform.platform())"

pause
