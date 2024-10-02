#include "ops.h"
#include "tgaimage.h"
#include <cmath>
#include <iostream>
#include <vector>

void line(int x0, int y0, int x1, int y1, TGAImage &image, TGAColor color) {
  bool steep = false;
  if (std::abs(x0 - x1) < std::abs(y0 - y1)) {
    std::swap(x0, y0);
    std::swap(x1, y1);
    steep = true;
  }
  if (x0 > x1) {
    std::swap(x0, x1);
    std::swap(y0, y1);
  }
  int dx = x1 - x0;
  int dy = y1 - y0;
  int derror = std::abs(dy) * 2;
  int error = 0;
  int y = y0;

  for (int x = x0; x <= x1; x++) {
    if (steep) {
      image.set(y, x, color);
    } else {
      image.set(x, y, color);
    }
    error += derror;
    if (error > dx) {
      y += (y1 > y0 ? 1 : -1);
      error -= dx * 2;
    }
  }
}

void wireframe_obj(Model &model, TGAImage &image, TGAColor color) {
  for (int i = 0; i < model.nfaces(); i++) {
    std::vector<int> face = model.face(i);
    for (int j = 0; j < 3; j++) {
      Vec3f v0 = model.vert(face[j]);
      Vec3f v1 = model.vert(face[(j + 1) % 3]);
      int x0 =
          (v0.x + 1.) * image.get_width() / 2.; // from [-1, 1] to [0, width]
      int y0 =
          (v0.y + 1.) * image.get_height() / 2.; // from [-1, 1] to [0, height]
      int x1 = (v1.x + 1.) * image.get_width() / 2.;
      int y1 = (v1.y + 1.) * image.get_height() / 2.;
      line(x0, y0, x1, y1, image, color);
    }
  }
}

void triangle_contour(Vec2f t0, Vec2f t1, Vec2f t2, TGAImage &image,
                      TGAColor color) {
  line(t0.x, t0.y, t1.x, t1.y, image, color);
  line(t1.x, t1.y, t2.x, t2.y, image, color);
  line(t2.x, t2.y, t0.x, t0.y, image, color);
}
bool compare_vec2i_y(const Vec2i &a, const Vec2i &b) { return a.y < b.y; }

void triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, TGAColor color) {
  std::vector<Vec2i> pts(3);
  pts[0] = t0;
  pts[1] = t1;
  pts[2] = t2;
  std::sort(pts.begin(), pts.end(), compare_vec2i_y);

  int total_height = pts[2].y - pts[0].y;
  int segment_height = pts[1].y - pts[0].y;
  for (int y = pts[0].y; y <= pts[1].y; y++) {
    float alpha = (float)(y - pts[0].y) / total_height;
    float beta = (float)(y - pts[0].y) / segment_height;
    Vec2i A = pts[0] + (pts[2] - pts[0]) * alpha;
    Vec2i B = pts[0] + (pts[1] - pts[0]) * beta;
    if (A.x > B.x) {
      std::swap(A, B);
    }
    for (int x = A.x; x <= B.x; x++) {
      image.set(x, y, color);
    }
  }
  segment_height = pts[2].y - pts[1].y;
  for (int y = pts[1].y; y <= pts[2].y; y++) {
    float alpha = (float)(y - pts[0].y) / total_height;
    float beta = (float)(y - pts[1].y) / segment_height;
    Vec2i A = pts[0] + (pts[2] - pts[0]) * alpha;
    Vec2i B = pts[1] + (pts[2] - pts[1]) * beta;
    if (A.x > B.x) {
      std::swap(A, B);
    }
    for (int x = A.x; x <= B.x; x++) {
      image.set(x, y, color);
    }
  }
}