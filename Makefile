.PHONY: *

VENV=.venv
PYTHON=$(VENV)/bin/python3
DEVICE=gpu
DATASET_FOLDER := dataset


# ================== LOCAL WORKSPACE SETUP ==================
venv:
	python3 -m venv $(VENV)
	@echo 'Path to Python executable $(shell pwd)/$(PYTHON)'


pre_commit_install:
	@echo "=== Installing pre-commit ==="
	$(PYTHON) -m pre_commit install


install_all: venv
	@echo "=== Installing common dependencies ==="
	$(PYTHON) -m pip install -r requirements/requirements-$(DEVICE).txt

	make pre_commit_install


fetch_dataset_from_dropbox:
	# Download dataset from Dropbox to local folder
	# Alternative option to get dataset if you don't have access to DVC
	wget "https://www.dropbox.com/scl/fi/nrn0y41dsfwqsrssav2eo/Classification_data.zip?rlkey=oieytodt749yzyippc6384tge&dl=0" -O $(DATASET_FOLDER)/dataset.zip
	unzip -q $(DATASET_FOLDER)/dataset.zip -d $(DATASET_FOLDER)
	rm $(DATASET_FOLDER)/dataset.zip
	find $(DATASET_FOLDER) -type f -name '.DS_Store' -delete
	rm $(DATASET_FOLDER)/dataset.zip
