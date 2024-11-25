CC=gcc
CXX=g++
RM=rm
CPPFLAGS=-g --std=c++20
LDFLAGS=
LDLIBS=

SRCS=main.cpp
OBJS= #$(subst .cc,.o,$(SRCS))
DEST=main

all: main

tool: $(OBJ)
		$(CXX) $(LDFLAGS) -o $(DEST) #$(OBJS) #$(LDLIBS)

clean:
		$(RM) $(OBJS) $(DEST)
