
all: install

install:
	python3 setup.py install

dist:
	pip3 install --editable .

upload:
	pandoc --from=markdown --to=rst --output=README.rst README.md
	python3 setup.py sdist upload -r pypi

clean:
	rm -rf charrnn.egg-info dist build
