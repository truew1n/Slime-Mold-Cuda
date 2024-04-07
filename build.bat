@echo off

nvcc -o main main.cu -lcudart -lgdi32 -luser32

main