.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch anri/ docs/source/ docs/build/

.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	ruff check .
	mypy .
	CUDA_VISIBLE_DEVICES='' pytest -v --color=yes --doctest-modules tests/ anri/

.PHONY : build
build :
	rm -rf *.egg-info/
	python -m build
