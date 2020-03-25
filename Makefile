
test:
	pytest -vv --pep8 --flakes slovnet

wheel:
	python setup.py bdist_wheel

version:
	bumpversion minor

upload:
	twine upload dist/*

clean:
	find . \
		-name '*.pyc' \
		-o -name __pycache__ \
		-o -name .DS_Store \
		| xargs rm -rf

	rm -rf dist/ build/ .pytest_cache/ .cache/ .ipynb_checkpoints/ \
		*.egg-info/ coverage.xml .coverage
