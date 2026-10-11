CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11
LDLIBS   = -ldl

all: gpucheckpoint libgpucheckpoint.so demo/demo

gpucheckpoint: gpucheckpoint-cli.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
	strip $@ 2>/dev/null || true

libgpucheckpoint.so: gpucheckpoint.c gpucheckpoint.h
	$(CC) $(CFLAGS) -fPIC -shared -o $@ gpucheckpoint.c $(LDLIBS)
	strip $@ 2>/dev/null || true

demo/demo: demo/demo.c gpucheckpoint.h libgpucheckpoint.so
	$(CC) $(CFLAGS) -o $@ demo/demo.c -I. -L. -lgpucheckpoint \
		-Wl,-rpath,'$$ORIGIN/..' $(LDLIBS)

test: gpucheckpoint
	bash tests/run-tests.sh

clean:
	rm -f gpucheckpoint libgpucheckpoint.so demo/demo

.PHONY: all clean test
