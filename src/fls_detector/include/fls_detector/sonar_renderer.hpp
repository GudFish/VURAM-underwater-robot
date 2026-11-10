#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <utility>

namespace fls {
constexpr double pi = 3.14159265358979323846;
struct Vec3 {
  double x = 0, y = 0, z = 0;
  Vec3() = default;
  Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
  double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
  Vec3 operator+(Vec3 b) const { return {x+b.x,y+b.y,z+b.z}; }
  Vec3 operator-(Vec3 b) const { return {x-b.x,y-b.y,z-b.z}; }
  Vec3 operator-() const { return {-x,-y,-z}; }
  Vec3 operator*(double s) const { return {x*s,y*s,z*s}; }
  Vec3 operator/(double s) const { return *this*(1/s); }
  double dot(Vec3 b) const { return x*b.x+y*b.y+z*b.z; }
  Vec3 cross(Vec3 b) const { return {y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x}; }
  double norm() const;
  Vec3 unit() const;
};
struct Pose {
  Vec3 position;
  std::array<double,9> rotation{{1,0,0,0,1,0,0,0,1}};
  static Pose quaternion(Vec3 p, double x, double y, double z, double w);
  static Pose rpy(Vec3 p, double roll, double pitch, double yaw);
  Vec3 rotate(Vec3 v) const;
  Vec3 inverseRotate(Vec3 v) const;
  Vec3 apply(Vec3 v) const { return position+rotate(v); }
  Pose compose(const Pose& b) const;
};
struct Material {
  double diffuse = 0.6, specular = 0.1, specular_exponent = 30;
  double backscatter = 1.0, backscatter_exponent = 20, incidence_exponent = 2;
};
struct Ray { Vec3 origin, direction; };
struct Hit {
  double distance = 0;
  Vec3 point, normal;
  uint16_t object_id = 0;
  int primitive_id = -1;
  Material material;
  bool planar = false;
};
struct Triangle { Vec3 a,b,c; };
class Mesh {
 public:
  explicit Mesh(std::vector<Triangle> triangles);
  static std::shared_ptr<Mesh> loadSTL(const std::string& path, double scale = 1);
  bool intersect(const Ray& ray, double limit, Hit& hit) const;
  size_t triangleCount() const { return triangles_.size(); }
  double boundRadiusSquared() const { return radius_squared_; }
 private:
  struct Node { Vec3 lo,hi; int begin=0,end=0,left=-1,right=-1; };
  std::vector<Triangle> triangles_;
  std::vector<int> indices_;
  std::vector<Node> nodes_;
  double radius_squared_ = 0;
  int build(int begin, int end);
};
enum class Shape { Mesh, Sphere, Box, Cylinder, Plane };
struct Object {
  Shape shape = Shape::Sphere;
  Pose pose;
  // Full dimensions for primitives. Planes are infinite local z=0 surfaces.
  Vec3 size{1,1,1};
  uint16_t id = 1;
  Material material;
  std::shared_ptr<const Mesh> mesh;
};
class Scene {
 public:
  std::vector<Object> objects;
  bool intersect(const Ray& ray, double limit, Hit& hit) const;
  bool visible(Vec3 from, Vec3 to) const;
};
struct Config {
  int num_beams = 512, num_ranges = 0, elevation_samples = 32;
  int max_rays_per_beam = 512, refinement_depth = 5, threads = 4;
  int width = 512, height = 256;
  double min_range = 0.2, max_range = 30, range_resolution = 0.02;
  double fov = 130*pi/180, vertical_fov = 20*pi/180;
  double angular_resolution = 0.6*pi/180, range_sigma = 0.0025/2.355;
  // If populated, these are the actual SDK bearing centres (radians), increasing.
  std::vector<double> bearings;
  double frequency_hz = 1.2e6, absorption_db_per_m = 0.4;
  double source_power = 1e8, tvg_db_per_decade = 20;
  double horizontal_edge_gain = 0.6, vertical_edge_gain = 0.5;
  double noise_floor_power = 1, noise_rayleigh_db = 1.5;
  std::vector<double> noise_mean_db, noise_std_db;
  // Physical distance of the last profile sample (first sample is at 0 m).
  // Beyond this distance, hold the endpoint statistics. Zero retains legacy
  // row-normalized profiles; a one-element profile is always constant.
  double noise_profile_range_m = 0;
  // Optional empirical display-domain bias in receive-beam index order. This
  // describes directional interference, not a calibrated acoustic noise source.
  std::vector<double> noise_beam_bias_db;
  double noise_correlation = 0.35;
  double display_min_db = 0, display_max_db = 75;
  double gain_db = 0, gamma = 150.0/255;
  bool gain_assist = false;
  double gain_assist_alpha = 0.05;
  bool multipath = false, wrt = true, prt = true;
  int random_seed = 7;
  void validate();
};
struct PathSignal {
  double range = 0, power = 0;
  uint16_t object_id = 0;
  // 0: direct; 1: WRT O-P2-P1-O; 2: PRT O-P1-P2-P1-O.
  int kind = 0;
  Vec3 normal;
  bool planar = false;
};
struct Frame {
  int rows = 0, cols = 0;
  std::vector<float> direct, wrt, prt;
  std::vector<uint16_t> instances;
  std::vector<uint8_t> raw, cartesian;
  std::vector<uint16_t> cartesian_instances;
  double render_ms = 0;
  uint64_t rays = 0;
};
class Renderer {
 public:
  explicit Renderer(Config config);
  const Config& config() const { return config_; }
  Frame render(const Scene& scene, const Pose& sensor, uint64_t frame_index = 0);
  std::vector<PathSignal> trace(const Scene& scene, const Pose& sensor,
                                double theta, double phi) const;
  bool cartesianSample(int row, int col, double& range, double& bearing) const;
  static double powerDb(double power);
  std::vector<uint8_t> display(const std::vector<float>& power, bool noise,
                               uint64_t frame_index = 0);
 private:
  Config config_;
  std::vector<std::vector<std::pair<int,double>>> angular_weights_;
  double assist_db_ = 0;
  double beamGain(Vec3 sensor_direction) const;
  double loss(double length) const;
  double scatter(const Material& material, Vec3 incoming, Vec3 outgoing, Vec3 normal) const;
  void blur(Frame& frame) const;
  void scanConvert(Frame& frame) const;
};
}  // namespace fls
