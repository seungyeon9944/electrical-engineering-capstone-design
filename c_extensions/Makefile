CC      = gcc
CFLAGS  = -O2 -Wall -fPIC
TARGET  = fast_preprocess.so
SRC     = fast_preprocess.c

all: $(TARGET)

$(TARGET): $(SRC) fast_preprocess.h
	$(CC) $(CFLAGS) -shared -o $(TARGET) $(SRC)
	@echo "[OK] $(TARGET) 빌드 완료"

clean:
	rm -f $(TARGET)

.PHONY: all clean
