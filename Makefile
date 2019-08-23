
test:
	pytest -vv --pep8 --flakes --cov slovnet --cov-report term-missing

wheel:
	python setup.py bdist_wheel

version:
	bumpversion minor

upload:
	twine upload dist/*

clean:
	find . -name '*.pyc' -not -path '*/__pycache__/*' -o -name .DS_Store | xargs rm
	rm -rf dist build *.egg-info coverage.xml
