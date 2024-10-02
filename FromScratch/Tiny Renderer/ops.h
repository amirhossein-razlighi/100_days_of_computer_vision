#include "model.h"
#include "tgaimage.h"
#include <cmath>
#include <iostream>
#include <vector>

void line(int x0, int y0, int x1, int y1, TGAImage &image, TGAColor color);
void wireframe_obj(Model &model, TGAImage &image, TGAColor color);
void triangle_contour(Vec2f t0, Vec2f t1, Vec2f t2, TGAImage &image,
                      TGAColor color);
void triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, TGAColor color);