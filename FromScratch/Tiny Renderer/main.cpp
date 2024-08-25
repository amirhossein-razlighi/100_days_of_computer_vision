#include "model.h"
#include "ops.h"
#include "tgaimage.h"

const TGAColor white = TGAColor(255, 255, 255, 255);
const TGAColor red = TGAColor(255, 0, 0, 255);

int main(int argc, char **argv) {
  Model model("Data/AfricanHead.obj");
  int width, height;
  width = 500;
  height = 500;
  TGAImage image(width, height, TGAImage::RGB);

  for (int i = 0; i < model.nfaces(); i++) {
    std::vector<int> face = model.face(i);
    for (int j = 0; j < 3; j++) {
      Vec3f v0 = model.vert(face[j]);
      Vec3f v1 = model.vert(face[(j + 1) % 3]);
      int x0 = (v0.x + 1.) * width / 2.;  // from [-1, 1] to [0, width]
      int y0 = (v0.y + 1.) * height / 2.; // from [-1, 1] to [0, height]
      int x1 = (v1.x + 1.) * width / 2.;
      int y1 = (v1.y + 1.) * height / 2.;
      line(x0, y0, x1, y1, image, white);
    }
  }
  image.flip_vertically();
  image.write_tga_file("output.tga");
  return 0;
}
