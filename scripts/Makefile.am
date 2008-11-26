bin_SCRIPTS = create_deformation

create_deformation: create_deformation.py
	sed < $< > $@-t \
		-e 's,xPKGDATADIRx,$(pkgdatadir),' &&\
	mv $@-t $@

EXTRA_DIST = $(bin_SCRIPTS)