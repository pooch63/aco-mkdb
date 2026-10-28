.PHONY: paper paper-emit paper-pdf paper-full paper-clean

paper:
	$(MAKE) -C paper all

paper-emit:
	$(MAKE) -C paper emit

paper-pdf:
	$(MAKE) -C paper pdf

paper-full:
	$(MAKE) -C paper full

paper-clean:
	$(MAKE) -C paper clean
