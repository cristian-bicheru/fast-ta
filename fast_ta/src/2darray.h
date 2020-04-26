#pragma once

double** double_malloc2d(int n, int len);
float** float_malloc2d(int n, int len);
void double_free2d(double** narr, int n);
void float_free2d(float** narr, int n);