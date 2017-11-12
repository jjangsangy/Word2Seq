
all:
	@pip3 install --editable .

upload:
	@pandoc --from=markdown --to=rst --output=README.rst README.md
	@python3 setup.py sdist upload -r pypi

clean:
	@rm -rf *.egg dist build
