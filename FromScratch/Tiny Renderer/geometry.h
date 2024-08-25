#include <vector>

template <class T> struct Vec2 {
  union {
    struct {
      T x, y;
    };
    struct {
      T u, v;
    };
    T raw[2];
  };

  Vec2() : x(T(0)), y(T(0)) {}
  Vec2(T xx, T yy) : x(xx), y(yy) {}

  Vec2 operator-(const Vec2 &v) const { return Vec2(x - v.x, y - v.y); }
  Vec2 operator+(const Vec2 &v) const { return Vec2(x + v.x, y + v.y); }
  Vec2 operator*(float f) const { return Vec2(x * f, y * f); }
  T operator*(const Vec2 &v) const { return x * v.x + y * v.y; }

  float norm() const { return std::sqrt(x * x + y * y); }
};

template <class T> struct Vec3 {
  union {
    struct {
      T x, y, z;
    };
    struct {
      T ivert, iuv, inorm;
    };
    T raw[3];
  };
  Vec3() : x(T(0)), y(T(0)), z(T(0)) {}
  Vec3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}

  Vec3 operator-(const Vec3 &v) const {
    return Vec3(x - v.x, y - v.y, z - v.z);
  }
  Vec3 operator+(const Vec3 &v) const {
    return Vec3(x + v.x, y + v.y, z + v.z);
  }
  Vec3 operator*(float f) const { return Vec3(x * f, y * f, z * f); }
  T operator*(const Vec3 &v) const { return x * v.x + y * v.y + z * v.z; }
  Vec3 operator^(const Vec3 &v) const {
    return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
  }

  float norm() const { return std::sqrt(x * x + y * y + z * z); }
};

typedef Vec2<float> Vec2f;
typedef Vec2<int> Vec2i;
typedef Vec3<float> Vec3f;
typedef Vec3<int> Vec3i;
