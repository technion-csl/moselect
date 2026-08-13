Artifact submission notes
========================

To reproduce `make analysis/moselect/normalized_scatter.pdf`:

1. Install system and Python prerequisites:

   make install-prereqs

   This will attempt to install system packages (using `apt-get` or `yum`) and create a Python virtualenv at `.venv` and install Python packages from `requirements.txt`.

2. Activate the virtualenv:

   source .venv/bin/activate

3. Build the target:

   make analysis/moselect/normalized_scatter.pdf
