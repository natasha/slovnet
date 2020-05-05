
test:
	pytest -vv --pep8 --flakes slovnet \
		--cov-report term-missing --cov-report xml \
		--cov-config setup.cfg --cov slovnet/exec

package:
	python setup.py sdist bdist_wheel

version:
	bumpversion minor

publish:
	twine upload dist/*

clean:
	find . \
		-name '*.pyc' \
		-o -name __pycache__ \
		-o -name .ipynb_checkpoints \
		-o -name .DS_Store \
		| xargs rm -rf

	rm -rf dist/ build/ .pytest_cache/ .cache/ \
		*.egg-info/ coverage.xml .coverage
