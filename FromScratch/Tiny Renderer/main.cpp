#include "model.h"
#include "ops.h"
#include "tgaimage.h"

const TGAColor white = TGAColor(255, 255, 255, 255);
const TGAColor red = TGAColor(255, 0, 0, 255);
const TGAColor green = TGAColor(0, 255, 0, 255);

int main(int argc, char **argv) {
  Model model("Data/AfricanHead.obj");
  int width, height;
  width = 500;
  height = 500;
  TGAImage image(width, height, TGAImage::RGB);
  // wireframe_obj(model, image, white);
  // triangle_contour(Vec2f(100, 100), Vec2f(400, 100), Vec2f(200, 400), image,
  //                  red);
  // triangle_contour(Vec2f(50, 51), Vec2f(100, 120), Vec2f(400, 400), image,
  // red); 
  // triangle(Vec2i(50, 50), Vec2i(300, 150), Vec2i(400, 400), image,
  // white);
  // triangle(Vec2i(50, 50), Vec2i(150, 300), Vec2i(400, 400), image,
  // red);
  image.flip_vertically();
  image.write_tga_file("output.tga");
  return 0;
}
