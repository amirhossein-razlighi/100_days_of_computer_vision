#include "ops.h"
#include "tgaimage.h"

const TGAColor white = TGAColor(255, 255, 255, 255);
const TGAColor red = TGAColor(255, 0, 0, 255);

int main(int argc, char **argv) {
  TGAImage image(200, 200, TGAImage::RGB);
  line(20, 13, 40, 80, image, white);
  line(80, 40, 13, 20, image, red);
  image.flip_vertically();
  image.write_tga_file("output.tga");
  return 0;
}
