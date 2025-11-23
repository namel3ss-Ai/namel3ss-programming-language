PYTHON := python

.PHONY: test build release-testpypi validate-testpypi-install clean

test:
	$(PYTHON) -m pytest

build:
	$(PYTHON) -m build

release-testpypi: test build
	bash scripts/upload_testpypi.sh

validate-testpypi-install:
	bash scripts/validate_testpypi_install.sh

clean:
	rm -rf build dist *.egg-info .eggs
