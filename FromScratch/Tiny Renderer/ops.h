#include "tgaimage.h"
#include <cmath>
#include <iostream>
#include <vector>
#include "model.h"

void line(int x0, int y0, int x1, int y1, TGAImage &image, TGAColor color);
void wireframe_obj(Model &model, TGAImage &image, TGAColor color);