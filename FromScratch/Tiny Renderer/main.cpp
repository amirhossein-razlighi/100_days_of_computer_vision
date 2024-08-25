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
  wireframe_obj(model, image, white);
  image.flip_vertically();
  image.write_tga_file("output.tga");
  return 0;
}
