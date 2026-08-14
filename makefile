SHELL := /bin/bash
# run all lines of a recipe in a single invocation of the shell rather than each line being invoked separately
.ONESHELL:
# invoke recipes as if the shell had been passed the -e flag: the first failing command in a recipe will cause the recipe to fail immediately
.POSIX:

MODULE_NAME := all
$(MODULE_NAME):

export ROOT_DIR := $(PWD)
export HOST_NAME := $(shell hostname)

# global auxiliary functions
comma := ,
empty :=
space := $(empty) $(empty)

define array_to_comma_separated
$(subst $(space),$(comma),$(strip $1))
endef

SCRIPTS_ROOT_DIR := $(ROOT_DIR)/scripts

# the following list should preserve a topological ordering, i.e., if module B
# uses variables defined in module A, than module A should come before module B
SUBMODULES := experiments analysis

include benchmark.mk
include $(ROOT_DIR)/common.mk

# a top-level "clean" target, which calls all/clean
.PHONY: clean purge
purge: all/clean
clean: analysis/clean
	rm -rf results

# a generic pattern rule for deleting files
.PHONY: %/delete
%/delete:
	rm -rf $*

# empty recipes to prevent make from remaking the makefile and include files
# https://www.gnu.org/software/make/manual/html_node/Remaking-Makefiles.html
makefile: ;
$(ROOT_DIR)/common.mk: ;

.PHONY: install-prereqs
install-prereqs:
	@echo "Installing system packages and Python virtualenv..."
	@$(SCRIPTS_ROOT_DIR)/install_prereqs.sh

.PHONY: test_artificat moselect bayesian genetic short_moselect_test short_bayesian_test test_artificat
moselect: analysis/moselect/scatter.pdf
bayesian: analysis/bayesian_optimization/scatter.pdf
genetic: analysis/genetic_selector/scatter.pdf
short_moselect_test: install-prereqs
	source .venv/bin/activate
	$(MAKE) MOSELECT_NUM_OF_REPEATS=1 MOSELECT_NUM_LAYOUTS=25 MOSELECT_MAX_GAP=8 moselect
short_bayesian_test: install-prereqs
	source .venv/bin/activate
	$(MAKE) BAYESIAN_NUM_OF_REPEATS=1 BAYESIAN_NUM_LAYOUTS=25 BAYESIAN_INIT_METHOD=chebyshev_misses bayesian
test_artificat: short_moselect_test