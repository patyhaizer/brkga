CC=g++
CFLAGS=-c -Wall
LDFLAGS=
SOURCES=main.cpp Instance.cpp Population.cpp Individual.cpp BRKGA.cpp
OBJECTS=$(SOURCES:.cpp=.o)
	EXECUTABLE=brkga

all: $(SOURCES) $(EXECUTABLE)
	    
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
clean:
	rm *o brkga
