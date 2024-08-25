#include "model.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

Model::Model(const char *filename) : verts_(), faces_() {
  std::ifstream in;
  in.open(filename, std::ifstream::in);
  if (in.fail()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  std::string line;
  while (!in.eof()) {
    std::getline(in, line);
    std::istringstream iss(line.c_str());
    char trash;
    if (!line.compare(0, 2, "v ")) { // line.compare returns 0 if equal
      iss >> trash;
      Vec3f v;
      for (int i = 0; i < 3; i++)
        iss >> v.raw[i];
      verts_.push_back(v);
    } else if (!line.compare(0, 2, "f ")) {
      std::vector<int> f;
      int itrash, idx;
      iss >> trash;
      while (iss >> idx >> trash >> itrash >> trash >>
             itrash) { // read vertex index, discrad /, discard texture index,
                       // and  discard normal index
        idx--;         // in wavefront obj all indices start at 1, not zero
        f.push_back(idx);
      }
      faces_.push_back(f);
    }
  }
}

Model::~Model() {}

int Model::nverts() { return (int)verts_.size(); }

int Model::nfaces() { return (int)faces_.size(); }

Vec3f Model::vert(int i) { return verts_[i]; }

std::vector<int> Model::face(int idx) { return faces_[idx]; }