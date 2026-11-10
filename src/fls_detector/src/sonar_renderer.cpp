#include "fls_detector/sonar_renderer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace fls {
namespace {
constexpr double epsilon = 1e-5;
double clamp(double x, double lo, double hi) { return std::max(lo,std::min(x,hi)); }
Vec3 reflect(Vec3 d, Vec3 n) { return d-n*(2*d.dot(n)); }
bool boxHit(const Ray& r, Vec3 lo, Vec3 hi, double limit) {
  double near = 0, far = limit;
  for (int k=0;k<3;++k) {
    if (std::abs(r.direction[k])<1e-12) {
      if (r.origin[k]<lo[k] || r.origin[k]>hi[k]) return false;
    } else {
      double a=(lo[k]-r.origin[k])/r.direction[k], b=(hi[k]-r.origin[k])/r.direction[k];
      if (a>b) std::swap(a,b);
      near=std::max(near,a); far=std::min(far,b);
      if (near>far) return false;
    }
  }
  return far>epsilon;
}
float readFloat(const unsigned char* p) {
  uint32_t u=uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);
  float f; std::memcpy(&f,&u,4); return f;
}
bool primitiveHit(const Object& o, const Ray& r, double limit, Hit& h) {
  double t=limit; Vec3 normal;
  auto accept=[&](double candidate, Vec3 n) {
    if (candidate>epsilon && candidate<t) { t=candidate; normal=n; }
  };
  if (o.shape==Shape::Mesh) return o.mesh && o.mesh->intersect(r,limit,h);
  if (o.shape==Shape::Plane) {
    if (std::abs(r.direction.z)>1e-12) accept(-r.origin.z/r.direction.z,{0,0,1});
  } else if (o.shape==Shape::Box) {
    double near=-1e100,far=1e100; Vec3 nn,fn;
    for(int k=0;k<3;++k) {
      const double half=o.size[k]*0.5;
      if (std::abs(r.direction[k])<1e-12) {
        if(std::abs(r.origin[k])>half) return false;
        continue;
      }
      double a=(-half-r.origin[k])/r.direction[k],b=(half-r.origin[k])/r.direction[k];
      Vec3 na(k==0?-1:0,k==1?-1:0,k==2?-1:0),nb=-na;
      if(a>b) { std::swap(a,b); std::swap(na,nb); }
      if(a>near) { near=a; nn=na; }
      if(b<far) { far=b; fn=nb; }
      if(near>far) return false;
    }
    accept(near,nn); accept(far,fn);
  } else {
    const Vec3 radii=o.size*0.5;
    Vec3 p(r.origin.x/radii.x,r.origin.y/radii.y,r.origin.z/radii.z);
    Vec3 d(r.direction.x/radii.x,r.direction.y/radii.y,r.direction.z/radii.z);
    const bool sphere=o.shape==Shape::Sphere;
    const double a=d.x*d.x+d.y*d.y+(sphere?d.z*d.z:0);
    const double b=p.x*d.x+p.y*d.y+(sphere?p.z*d.z:0);
    const double c=p.x*p.x+p.y*p.y+(sphere?p.z*p.z:0)-1;
    const double disc=b*b-a*c;
    if(a>1e-20 && disc>=0) {
      for(double root : {(-b-std::sqrt(disc))/a,(-b+std::sqrt(disc))/a}) {
        Vec3 q=r.origin+r.direction*root;
        if(sphere || std::abs(q.z)<=radii.z) {
          accept(root,Vec3(q.x/(radii.x*radii.x),q.y/(radii.y*radii.y),
                           sphere?q.z/(radii.z*radii.z):0).unit());
        }
      }
    }
    if(!sphere && std::abs(r.direction.z)>1e-12) {
      for(double sign : {-1.,1.}) {
        double root=(sign*radii.z-r.origin.z)/r.direction.z;
        Vec3 q=r.origin+r.direction*root;
        if(q.x*q.x/(radii.x*radii.x)+q.y*q.y/(radii.y*radii.y)<=1)
          accept(root,{0,0,sign});
      }
    }
  }
  if(t>=limit) return false;
  h.distance=t; h.point=r.origin+r.direction*t; h.normal=normal; return true;
}
std::vector<double> gaussian(double sigma) {
  if(sigma<0.15) return {1};
  int radius=std::max(1,int(std::ceil(3*sigma)));
  std::vector<double> k(2*radius+1); double sum=0;
  for(int j=-radius;j<=radius;++j) sum+=(k[j+radius]=std::exp(-0.5*j*j/(sigma*sigma)));
  for(auto& v:k) v/=sum;
  return k;
}
}  // namespace

double Vec3::norm() const { return std::sqrt(dot(*this)); }
Vec3 Vec3::unit() const { const double n=norm(); return n>1e-15?*this/n:Vec3(); }
Pose Pose::quaternion(Vec3 p,double x,double y,double z,double w) {
  double n=std::sqrt(x*x+y*y+z*z+w*w);
  if(!std::isfinite(n) || n<1e-12) throw std::invalid_argument("invalid quaternion");
  x/=n;y/=n;z/=n;w/=n;
  Pose pose; pose.position=p;
  pose.rotation={{1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                  2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                  2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)}};
  return pose;
}
Pose Pose::rpy(Vec3 p,double r,double pitch,double yaw) {
  const double cr=std::cos(r/2),sr=std::sin(r/2),cp=std::cos(pitch/2),sp=std::sin(pitch/2);
  const double cy=std::cos(yaw/2),sy=std::sin(yaw/2);
  return quaternion(p,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy);
}
Vec3 Pose::rotate(Vec3 v) const {
  return {rotation[0]*v.x+rotation[1]*v.y+rotation[2]*v.z,
          rotation[3]*v.x+rotation[4]*v.y+rotation[5]*v.z,
          rotation[6]*v.x+rotation[7]*v.y+rotation[8]*v.z};
}
Vec3 Pose::inverseRotate(Vec3 v) const {
  return {rotation[0]*v.x+rotation[3]*v.y+rotation[6]*v.z,
          rotation[1]*v.x+rotation[4]*v.y+rotation[7]*v.z,
          rotation[2]*v.x+rotation[5]*v.y+rotation[8]*v.z};
}
Pose Pose::compose(const Pose& b) const {
  Pose out; out.position=apply(b.position);
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
    out.rotation[3*i+j]=0;
    for(int k=0;k<3;++k) out.rotation[3*i+j]+=rotation[3*i+k]*b.rotation[3*k+j];
  }
  return out;
}

Mesh::Mesh(std::vector<Triangle> ts) {
  for(const auto& t:ts) {
    if(std::isfinite(t.a.norm()+t.b.norm()+t.c.norm()) && (t.b-t.a).cross(t.c-t.a).norm()>1e-12)
      triangles_.push_back(t);
  }
  if(triangles_.empty()) throw std::invalid_argument("mesh has no nondegenerate triangles");
  indices_.resize(triangles_.size()); std::iota(indices_.begin(),indices_.end(),0);
  nodes_.reserve(triangles_.size()/2+1); build(0,indices_.size());
  const auto& root=nodes_.front();
  Vec3 bound{std::max(std::abs(root.lo.x),std::abs(root.hi.x)),
             std::max(std::abs(root.lo.y),std::abs(root.hi.y)),
             std::max(std::abs(root.lo.z),std::abs(root.hi.z))};
  radius_squared_=bound.dot(bound);
}
std::shared_ptr<Mesh> Mesh::loadSTL(const std::string& path,double scale) {
  if(!std::isfinite(scale)||scale<=0) throw std::invalid_argument("mesh scale must be positive");
  std::ifstream in(path,std::ios::binary);
  if(!in) throw std::runtime_error("cannot open STL: "+path);
  in.seekg(0,std::ios::end); auto bytes=in.tellg(); in.seekg(0);
  unsigned char header[84]{}; in.read(reinterpret_cast<char*>(header),84);
  uint32_t count=uint32_t(header[80])|(uint32_t(header[81])<<8)|(uint32_t(header[82])<<16)|(uint32_t(header[83])<<24);
  std::vector<Triangle> triangles;
  if(bytes>=84 && uint64_t(bytes)==84ull+50ull*count && count<=5000000) {
    triangles.reserve(count);
    for(uint32_t i=0;i<count;++i) {
      unsigned char record[50]; in.read(reinterpret_cast<char*>(record),50);
      auto vertex=[&](int off) { return Vec3(readFloat(record+off),readFloat(record+off+4),readFloat(record+off+8))*scale; };
      triangles.push_back({vertex(12),vertex(24),vertex(36)});
    }
  } else {
    in.clear(); in.seekg(0); std::string token; Vec3 vertices[3]; int n=0;
    while(in>>token) if(token=="vertex") {
      if(!(in>>vertices[n].x>>vertices[n].y>>vertices[n].z)) throw std::runtime_error("malformed ASCII STL");
      vertices[n]=vertices[n]*scale;
      if(++n==3) { triangles.push_back({vertices[0],vertices[1],vertices[2]}); n=0; }
      if(triangles.size()>5000000) throw std::runtime_error("STL too large");
    }
  }
  return std::make_shared<Mesh>(std::move(triangles));
}
int Mesh::build(int begin,int end) {
  Node n; n.begin=begin;n.end=end;
  n.lo={1e100,1e100,1e100};n.hi={-1e100,-1e100,-1e100};
  for(int i=begin;i<end;++i) for(Vec3 p:{triangles_[indices_[i]].a,triangles_[indices_[i]].b,triangles_[indices_[i]].c}) {
    n.lo={std::min(n.lo.x,p.x),std::min(n.lo.y,p.y),std::min(n.lo.z,p.z)};
    n.hi={std::max(n.hi.x,p.x),std::max(n.hi.y,p.y),std::max(n.hi.z,p.z)};
  }
  int index=nodes_.size();nodes_.push_back(n);
  if(end-begin>8) {
    Vec3 span=n.hi-n.lo;int axis=span.y>span.x?1:0;if(span.z>span[axis])axis=2;
    int mid=(begin+end)/2;
    std::nth_element(indices_.begin()+begin,indices_.begin()+mid,indices_.begin()+end,[&](int a,int b) {
      const auto& ta=triangles_[a];const auto& tb=triangles_[b];
      return ta.a[axis]+ta.b[axis]+ta.c[axis]<tb.a[axis]+tb.b[axis]+tb.c[axis];
    });
    int left=build(begin,mid),right=build(mid,end);
    nodes_[index].left=left;nodes_[index].right=right;
  }
  return index;
}
bool Mesh::intersect(const Ray& ray,double limit,Hit& hit) const {
  int stack[128],size=1;stack[0]=0;bool found=false;
  while(size) {
    const Node& n=nodes_[stack[--size]];
    if(!boxHit(ray,n.lo,n.hi,limit))continue;
    if(n.left>=0) { stack[size++]=n.left;stack[size++]=n.right;continue; }
    for(int j=n.begin;j<n.end;++j) {
      int id=indices_[j];const auto& t=triangles_[id];Vec3 e1=t.b-t.a,e2=t.c-t.a;
      Vec3 p=ray.direction.cross(e2);double det=e1.dot(p);
      if(std::abs(det)<1e-14)continue;
      double inv=1/det;Vec3 q=ray.origin-t.a;double u=q.dot(p)*inv;
      if(u<0||u>1)continue;
      Vec3 v=q.cross(e1);double b=ray.direction.dot(v)*inv;
      if(b<0||u+b>1)continue;
      double distance=e2.dot(v)*inv;
      if(distance>epsilon&&distance<limit) {
        limit=distance;hit.distance=distance;hit.point=ray.origin+ray.direction*distance;
        hit.normal=e1.cross(e2).unit();hit.primitive_id=id;found=true;
      }
    }
  }
  return found;
}
bool Scene::intersect(const Ray& ray,double limit,Hit& hit) const {
  bool found=false;
  for(const auto& o:objects) {
    if(o.shape!=Shape::Plane) {
      double radius2=o.shape==Shape::Mesh?(o.mesh?o.mesh->boundRadiusSquared():0):o.size.dot(o.size)*.25;
      Vec3 offset=o.pose.position-ray.origin;double projection=offset.dot(ray.direction);
      double closest2=offset.dot(offset)-(projection>0?projection*projection:0);
      if(closest2>radius2+1e-9)continue;
    }
    Ray local{o.pose.inverseRotate(ray.origin-o.pose.position),o.pose.inverseRotate(ray.direction)};
    Hit candidate;
    if(primitiveHit(o,local,limit,candidate)) {
      hit=candidate;limit=hit.distance;hit.point=o.pose.apply(hit.point);hit.normal=o.pose.rotate(hit.normal).unit();
      if(hit.normal.dot(ray.direction)>0)hit.normal=-hit.normal;
      hit.object_id=o.id;hit.material=o.material;found=true;
      hit.planar=o.shape==Shape::Plane||o.shape==Shape::Box||o.shape==Shape::Mesh;
    }
  }
  return found;
}
bool Scene::visible(Vec3 from,Vec3 to) const {
  Vec3 delta=to-from;double length=delta.norm();if(length<epsilon)return false;
  Hit hit;return !intersect({from,delta/length},length-5*epsilon,hit);
}

void Config::validate() {
  if(!bearings.empty()) num_beams=bearings.size();
  if(num_ranges==0 && std::isfinite(range_resolution) && range_resolution>0 && std::isfinite(max_range)) {
    double required=std::ceil(max_range/range_resolution);
    if(required>65535)throw std::invalid_argument("sampling grid exceeds 65536 range bins");
    num_ranges=int(required)+1;
  }
  if(num_beams<2||num_beams>2048||num_ranges<2||num_ranges>65536||width<2||height<2||width>4096||height>4096)
    throw std::invalid_argument("invalid sonar image dimensions");
  if(uint64_t(num_beams)*num_ranges>16000000)throw std::invalid_argument("native image exceeds 16 million cells");
  if(!std::isfinite(min_range+max_range+range_resolution+fov+vertical_fov)||min_range<0||max_range<=min_range||range_resolution<=0||
     fov<=0||fov>=pi||vertical_fov<=0||vertical_fov>=pi)
    throw std::invalid_argument("invalid sonar ranges/aperture");
  if(range_resolution*(num_ranges-1)+1e-9<max_range) {
    std::ostringstream message;
    message<<"sonar range grid covers "<<range_resolution*(num_ranges-1)
           <<" m (num_range_bins="<<num_ranges<<", range_resolution="<<range_resolution
           <<" m), but max_range="<<max_range
           <<" m; remove the incompatible sampling overlay or set num_range_bins=0 to derive the grid";
    throw std::invalid_argument(message.str());
  }
  if(elevation_samples<3||elevation_samples>4096||max_rays_per_beam<elevation_samples||max_rays_per_beam>16384||refinement_depth<0||refinement_depth>12||threads<1||threads>64)
    throw std::invalid_argument("invalid ray budget");
  for(double value:{source_power,noise_floor_power,gamma,frequency_hz})
    if(!std::isfinite(value)||value<=0)throw std::invalid_argument("power, frequency and gamma must be positive");
  for(double value:{absorption_db_per_m,angular_resolution,range_sigma,noise_rayleigh_db})
    if(!std::isfinite(value)||value<0)throw std::invalid_argument("negative or nonfinite acoustic parameter");
  if(!std::isfinite(display_min_db+display_max_db+gain_db+tvg_db_per_decade)||display_max_db<=display_min_db||
     horizontal_edge_gain<0||horizontal_edge_gain>1||vertical_edge_gain<0||vertical_edge_gain>1||
     noise_correlation<0||noise_correlation>1||gain_assist_alpha<=0||gain_assist_alpha>1)
    throw std::invalid_argument("invalid display/beam/noise parameters");
  if(bearings.empty())for(int i=0;i<num_beams;++i)bearings.push_back(-fov/2+fov*i/(num_beams-1));
  for(size_t i=0;i<bearings.size();++i)
    if(!std::isfinite(bearings[i])||std::abs(bearings[i])>fov/2+1e-6||(i&&bearings[i]<=bearings[i-1]))
      throw std::invalid_argument("bearings must be finite and strictly increasing inside FOV");
  for(double v:noise_mean_db)if(!std::isfinite(v))throw std::invalid_argument("nonfinite noise profile");
  for(double v:noise_std_db)if(!std::isfinite(v)||v<0)throw std::invalid_argument("invalid noise standard deviation");
  if(!std::isfinite(noise_profile_range_m)||noise_profile_range_m<0)
    throw std::invalid_argument("noise_profile_range_m must be finite and nonnegative");
  for(double v:noise_beam_bias_db)if(!std::isfinite(v))throw std::invalid_argument("nonfinite beam noise bias");
}
Renderer::Renderer(Config c):config_(std::move(c)) {
  config_.validate();
  double sigma=config_.angular_resolution/2.355;
  angular_weights_.resize(config_.num_beams);
  for(int col=0;col<config_.num_beams;++col) {
    double total=0;
    for(int j=0;j<config_.num_beams;++j) {
      double angle=config_.bearings[j]-config_.bearings[col];
      if((sigma<=0&&j!=col)||(sigma>0&&std::abs(angle)>3*sigma))continue;
      double width=j==0?config_.bearings[1]-config_.bearings[0]:
          (j==config_.num_beams-1?config_.bearings[j]-config_.bearings[j-1]:
           (config_.bearings[j+1]-config_.bearings[j-1])/2);
      double weight=sigma>0?std::exp(-.5*angle*angle/(sigma*sigma))*width:1;
      angular_weights_[col].push_back({j,weight});total+=weight;
    }
    for(auto& entry:angular_weights_[col])entry.second/=total;
  }
}
double Renderer::powerDb(double power) { return 10*std::log10(std::max(power,1e-20)); }
double Renderer::loss(double length) const {
  double r=std::max(length,0.1);
  return std::pow(10.,-config_.absorption_db_per_m*length/10)/(r*r);
}
double Renderer::beamGain(Vec3 d) const {
  const double theta=std::atan2(d.y,d.x),phi=std::atan2(d.z,std::hypot(d.x,d.y));
  if(d.x<=0||std::abs(theta)>config_.fov/2+1e-9||std::abs(phi)>config_.vertical_fov/2+1e-9)return 0;
  double a=2*theta/config_.fov,b=2*phi/config_.vertical_fov;
  return (1-(1-config_.horizontal_edge_gain)*a*a)*(1-(1-config_.vertical_edge_gain)*b*b);
}
double Renderer::scatter(const Material& m,Vec3 incoming,Vec3 outgoing,Vec3 n) const {
  double cosine=std::max(0.,-incoming.dot(n));
  if(outgoing.dot(n)<=0)return 0;
  double back=std::pow(std::max(0.,(-incoming).dot(outgoing)),m.backscatter_exponent);
  double diffuse=m.diffuse/pi*(1+m.backscatter*back)*std::pow(cosine,m.incidence_exponent);
  double specular=m.specular*(m.specular_exponent+2)/(2*pi)*
      std::pow(std::max(0.,reflect(incoming,n).dot(outgoing)),m.specular_exponent)*cosine;
  return diffuse+specular;
}
std::vector<PathSignal> Renderer::trace(const Scene& scene,const Pose& sensor,double theta,double phi) const {
  Vec3 local{std::cos(phi)*std::cos(theta),std::cos(phi)*std::sin(theta),std::sin(phi)};
  double receive=beamGain(local);if(receive==0)return {};
  Vec3 d=sensor.rotate(local);Hit a;
  if(!scene.intersect({sensor.position,d},config_.max_range,a))return {};
  double direct=config_.source_power*receive*receive*scatter(a.material,d,-d,a.normal)*std::pow(loss(a.distance),2);
  std::vector<PathSignal> signals;
  if(a.distance>=config_.min_range)signals.push_back({a.distance,direct,a.object_id,0,a.normal,a.planar});
  if(!config_.multipath)return signals;
  Vec3 reflected=reflect(d,a.normal).unit();Hit b;
  if(!scene.intersect({a.point+reflected*epsilon*4,reflected},2*config_.max_range-a.distance,b))return signals;
  // Use the physical hit-to-hit length; the numerical origin offset is not propagation distance.
  double ab=(b.point-a.point).norm(),ob=(b.point-sensor.position).norm();
  double at_return=scatter(a.material,-reflected,-d,a.normal);
  double wrange=(a.distance+ab+ob)/2;
  if(config_.wrt&&wrange>=config_.min_range&&wrange<=config_.max_range&&scene.visible(sensor.position,b.point)) {
    Vec3 obdir=(b.point-sensor.position)/ob;
    Vec3 bn=b.normal;if(bn.dot(obdir)>0)bn=-bn;
    double transmit=beamGain(sensor.inverseRotate(obdir));
    double p=config_.source_power*receive*transmit*scatter(b.material,obdir,-reflected,bn)*at_return*
        loss(ob)*loss(ab)*loss(a.distance);
    if(p>0)signals.push_back({wrange,p,b.object_id,1,{},false});
  }
  double prange=a.distance+ab;
  if(config_.prt&&prange>=config_.min_range&&prange<=config_.max_range) {
    double p=config_.source_power*receive*receive*scatter(a.material,d,reflected,a.normal)*
        scatter(b.material,reflected,-reflected,b.normal)*at_return*
        std::pow(loss(a.distance)*loss(ab),2);
    if(p>0)signals.push_back({prange,p,b.object_id,2,{},false});
  }
  return signals;
}

Frame Renderer::render(const Scene& scene,const Pose& sensor,uint64_t frame_index) {
  auto start=std::chrono::steady_clock::now();
  const auto& c=config_;Frame f;f.rows=c.num_ranges;f.cols=c.num_beams;
  size_t pixels=size_t(f.rows)*f.cols;
  f.direct.assign(pixels,0);f.wrt.assign(pixels,0);f.prt.assign(pixels,0);f.instances.assign(pixels,0);
  uint64_t ray_count=0;
  #pragma omp parallel for schedule(dynamic, 8) num_threads(c.threads) reduction(+:ray_count)
  for(int beam=0;beam<c.num_beams;++beam) {
    struct Sample { double phi;std::vector<PathSignal> paths; };
    std::vector<Sample> samples;samples.reserve(c.max_rays_per_beam);
    int used=0;
    auto cast=[&](double phi) { ++used;return Sample{phi,trace(scene,sensor,c.bearings[beam],phi)}; };
    auto needsRefinement=[&](const Sample& a,const Sample& b) {
      if(a.paths.size()!=b.paths.size())return true;
      for(size_t i=0;i<a.paths.size();++i) {
        const auto& pa=a.paths[i];const auto& pb=b.paths[i];
        if(pa.object_id!=pb.object_id||pa.kind!=pb.kind)return true;
        bool coplanar=false;
        if(pa.kind==0&&pa.planar&&pb.planar&&pa.normal.dot(pb.normal)>1-1e-8) {
          Vec3 n=sensor.inverseRotate(pa.normal);
          double xy=n.x*std::cos(c.bearings[beam])+n.y*std::sin(c.bearings[beam]);
          double ka=pa.range*(xy*std::cos(a.phi)+n.z*std::sin(a.phi));
          double kb=pb.range*(xy*std::cos(b.phi)+n.z*std::sin(b.phi));
          coplanar=std::abs(ka-kb)<1e-5;
        }
        if(!coplanar&&std::abs(pa.range-pb.range)>c.range_resolution)return true;
      }
      return false;
    };
    std::function<void(const Sample&,const Sample&,int)> refine;
    refine=[&](const Sample& a,const Sample& b,int depth) {
      if(depth<c.refinement_depth && used<c.max_rays_per_beam && needsRefinement(a,b)) {
        Sample middle=cast((a.phi+b.phi)/2);
        refine(a,middle,depth+1);samples.push_back(middle);refine(middle,b,depth+1);
      }
    };
    std::vector<Sample> initial;initial.reserve(c.elevation_samples);
    for(int j=0;j<c.elevation_samples;++j)initial.push_back(cast(-c.vertical_fov/2+c.vertical_fov*j/(c.elevation_samples-1)));
    for(int j=0;j<c.elevation_samples;++j) {
      samples.push_back(initial[j]);
      if(j+1<c.elevation_samples)refine(initial[j],initial[j+1],0);
    }
    ray_count+=used;
    std::sort(samples.begin(),samples.end(),[](const Sample& a,const Sample& b){return a.phi<b.phi;});
    // Integrate each continuous surface run once per range cell. A gap, a new object,
    // or another reflection path starts another run; disconnected returns are retained.
    std::vector<double> strongest(c.num_ranges,0);
    // Reconstruct only coplanar, contiguous first-hit runs. This fills native
    // range cells without casting one ray per cell on a distant flat floor.
    // A miss, object boundary, plane offset or normal discontinuity ends a run.
    // Curved surfaces and multipath still use adaptive casts below.
    std::vector<const Sample*> planar_run;
    auto directOf=[](const Sample& s)->const PathSignal* {
      for(const auto& p:s.paths)if(p.kind==0)return &p;
      return nullptr;
    };
    auto addDirect=[&](int bin,double power,uint16_t id) {
      if(bin<0||bin>=c.num_ranges)return;
      f.direct[size_t(bin)*c.num_beams+beam]+=power;
      if(power>strongest[bin]){strongest[bin]=power;f.instances[size_t(bin)*c.num_beams+beam]=id;}
    };
    auto flushPlane=[&]() {
      if(planar_run.empty())return;
      const auto& first=*directOf(*planar_run.front());
      double low=planar_run.front()->phi,high=planar_run.back()->phi;
      Vec3 n=sensor.inverseRotate(first.normal);
      double a=n.x*std::cos(c.bearings[beam])+n.y*std::sin(c.bearings[beam]),b=n.z;
      double magnitude=std::hypot(a,b),delta=std::atan2(b,a);
      double offset=first.range*(a*std::cos(low)+b*std::sin(low));
      double minimum=first.range,maximum=first.range;
      for(const auto* sample:planar_run) {
        double r=directOf(*sample)->range;minimum=std::min(minimum,r);maximum=std::max(maximum,r);
      }
      double extremum=delta+(offset<0?pi:0);
      extremum=std::atan2(std::sin(extremum),std::cos(extremum));
      if(extremum>=low&&extremum<=high&&magnitude>1e-12)minimum=std::min(minimum,std::abs(offset)/magnitude);
      Material mat;
      for(const auto& object:scene.objects)if(object.id==first.object_id){mat=object.material;break;}
      bool deposited=false;
      for(int j=std::max(0,int(std::ceil(minimum/c.range_resolution)));j<=std::min(c.num_ranges-1,int(std::floor(maximum/c.range_resolution)));++j) {
        double range=j*c.range_resolution;
        if(range<c.min_range||range>c.max_range||magnitude<1e-12||range<=0)continue;
        double cosine=offset/(range*magnitude);
        if(std::abs(cosine)>1+1e-9)continue;
        double angle=std::acos(clamp(cosine,-1,1)),power=0;double previous=1e100;
        for(double phi:{delta-angle,delta+angle}) {
          phi=std::atan2(std::sin(phi),std::cos(phi));
          if(phi<low-1e-8||phi>high+1e-8||std::abs(phi-previous)<1e-8)continue;
          previous=phi;
          Vec3 local{std::cos(phi)*std::cos(c.bearings[beam]),std::cos(phi)*std::sin(c.bearings[beam]),std::sin(phi)};
          Vec3 d=sensor.rotate(local);double gain=beamGain(local);
          power+=c.source_power*gain*gain*scatter(mat,d,-d,first.normal)*std::pow(loss(range),2);
        }
        if(power>0){addDirect(j,power,first.object_id);deposited=true;}
      }
      if(!deposited) {
        const auto& signal=*directOf(*planar_run[planar_run.size()/2]);
        double bin=signal.range/c.range_resolution;int lo=int(std::floor(bin));
        addDirect(lo,signal.power*(1-(bin-lo)),signal.object_id);
        addDirect(lo+1,signal.power*(bin-lo),signal.object_id);
      }
      planar_run.clear();
    };
    for(const auto& sample:samples) {
      const auto* p=directOf(sample);
      if(!p||!p->planar){flushPlane();continue;}
      if(!planar_run.empty()) {
        const auto& first=*directOf(*planar_run.front());
        Vec3 d0=sensor.rotate({std::cos(planar_run.front()->phi)*std::cos(c.bearings[beam]),std::cos(planar_run.front()->phi)*std::sin(c.bearings[beam]),std::sin(planar_run.front()->phi)});
        Vec3 d1=sensor.rotate({std::cos(sample.phi)*std::cos(c.bearings[beam]),std::cos(sample.phi)*std::sin(c.bearings[beam]),std::sin(sample.phi)});
        if(p->object_id!=first.object_id||p->normal.dot(first.normal)<1-1e-8||
           std::abs(d0.dot(first.normal)*first.range-d1.dot(first.normal)*p->range)>1e-5)flushPlane();
      }
      planar_run.push_back(&sample);
    }
    flushPlane();
    for(int kind=0;kind<3;++kind) {
      auto& output=kind==0?f.direct:(kind==1?f.wrt:f.prt);
      std::vector<PathSignal> run;int last_bin=-1;uint16_t last_id=0;
      auto flush=[&]() {
        if(run.empty())return;
        const auto& s=run[run.size()/2];double bin=s.range/c.range_resolution;int lo=int(std::floor(bin));
        for(int j=lo;j<=lo+1;++j) if(j>=0&&j<c.num_ranges) {
          double p=s.power*(j==lo?1-(bin-lo):bin-lo);
          output[size_t(j)*c.num_beams+beam]+=p;
          if(kind==0&&p>strongest[j]) { strongest[j]=p;f.instances[size_t(j)*c.num_beams+beam]=s.object_id; }
        }
        run.clear();
      };
      for(const auto& sample:samples) {
        const PathSignal* signal=nullptr;
        for(const auto& s:sample.paths)if(s.kind==kind&&!(kind==0&&s.planar)) {signal=&s;break;}
        int bin=signal?int(std::floor(signal->range/c.range_resolution)):-1;
        if(!signal||bin!=last_bin||signal->object_id!=last_id)flush();
        if(signal){run.push_back(*signal);last_bin=bin;last_id=signal->object_id;}
        else{last_bin=-1;last_id=0;}
      }
      flush();
    }
  }
  f.rays=ray_count;blur(f);
  std::vector<float> total(pixels);
  for(size_t i=0;i<pixels;++i)total[i]=f.direct[i]+f.wrt[i]+f.prt[i];
  f.raw=display(total,true,frame_index);scanConvert(f);
  f.render_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  return f;
}
void Renderer::blur(Frame& f) const {
  auto radial=gaussian(config_.range_sigma/config_.range_resolution);
  for(int axis=0;axis<2;++axis) {
    const auto& kernel=radial;
    if(axis==0&&config_.angular_resolution==0)continue;
    if(axis==1&&kernel.size()==1)continue;
    int radius=kernel.size()/2;
    std::vector<float> d(f.direct.size(),0),w(d.size(),0),p(d.size(),0);
    std::vector<uint16_t> ids(d.size(),0);
    #pragma omp parallel for schedule(static) num_threads(config_.threads)
    for(int row=0;row<f.rows;++row) for(int col=0;col<f.cols;++col) {
      size_t out=size_t(row)*f.cols+col;double strongest=0;
      auto integrate=[&](int rr,int cc,double weight) {
        if(rr<0||rr>=f.rows||cc<0||cc>=f.cols)return;
        size_t in=size_t(rr)*f.cols+cc;
        d[out]+=weight*f.direct[in];w[out]+=weight*f.wrt[in];p[out]+=weight*f.prt[in];
        if(weight*f.direct[in]>strongest){strongest=weight*f.direct[in];ids[out]=f.instances[in];}
      };
      if(axis==0)for(const auto& weight:angular_weights_[col])integrate(row,weight.first,weight.second);
      else for(int j=-radius;j<=radius;++j)integrate(row+j,col,kernel[j+radius]);
    }
    f.direct.swap(d);f.wrt.swap(w);f.prt.swap(p);f.instances.swap(ids);
  }
}
std::vector<uint8_t> Renderer::display(const std::vector<float>& power,bool noise,uint64_t index) {
  const auto& c=config_;
  if(power.size()!=size_t(c.num_ranges)*c.num_beams)throw std::invalid_argument("power image shape mismatch");
  std::vector<double> db(power.size());
  auto profile=[&](const std::vector<double>& v,int row,double fallback) {
    if(v.empty())return fallback;
    double fraction=c.noise_profile_range_m>0?
        row*c.range_resolution/c.noise_profile_range_m:double(row)/(c.num_ranges-1);
    double at=clamp(fraction,0,1)*(v.size()-1);
    size_t lo=size_t(at),hi=std::min(lo+1,v.size()-1);
    return v[lo]*(1-(at-lo))+v[hi]*(at-lo);
  };
  std::vector<double> beam_bias(c.num_beams,0);
  if(noise&&!c.noise_beam_bias_db.empty()) {
    const auto& values=c.noise_beam_bias_db;
    for(int b=0;b<c.num_beams;++b) {
      double at=double(b)*(values.size()-1)/(c.num_beams-1);
      size_t lo=size_t(at),hi=std::min(lo+1,values.size()-1);
      beam_bias[b]=values[lo]*(1-(at-lo))+values[hi]*(at-lo);
    }
  }
  #pragma omp parallel for schedule(static) num_threads(c.threads)
  for(int r=0;r<c.num_ranges;++r) {
    std::mt19937 gen(static_cast<uint32_t>(c.random_seed)^static_cast<uint32_t>(index*0x9e3779b9ULL)^uint32_t(r*2654435761U));
    std::normal_distribution<double> normal(0,1);std::uniform_real_distribution<double> uniform(1e-12,1-1e-12);
    double mean=profile(c.noise_mean_db,r,0),stddev=profile(c.noise_std_db,r,0),previous=0;
    double range=r*c.range_resolution;
    double tvg=std::pow(std::max(range,1.0),c.tvg_db_per_decade/10);
    for(int b=0;b<c.num_beams;++b) {
      size_t i=size_t(r)*c.num_beams+b;double n=0;
      if(noise) {
        double rayleigh=c.noise_rayleigh_db>0?c.noise_rayleigh_db*(std::sqrt(-2*std::log(uniform(gen)))-std::sqrt(pi/2)):0;
        n=(stddev>0?stddev*normal(gen):0)+rayleigh;
        if(b)n=std::sqrt(1-c.noise_correlation*c.noise_correlation)*n+c.noise_correlation*previous;
        previous=n;n+=mean+c.noise_rayleigh_db*std::sqrt(pi/2);
        // Empirical receive-direction interference: display only, independent
        // of vehicle speed, arming state, and the physical return buffers.
        n+=beam_bias[b];
      }
      // Receiver floor is added after propagation/TVG; geometric shadows do not erase it.
      double received=double(power[i])*tvg+(noise?c.noise_floor_power:0);
      db[i]=powerDb(received)+c.gain_db+n;
    }
  }
  if(c.gain_assist&&noise) {
    std::vector<double> sorted=db;size_t k=size_t(sorted.size()*0.994);
    std::nth_element(sorted.begin(),sorted.begin()+k,sorted.end());
    double desired=clamp(c.display_max_db-6-sorted[k],-30,30);
    assist_db_+=(desired-assist_db_)*c.gain_assist_alpha;
  }
  std::vector<uint8_t> out(power.size());
  #pragma omp parallel for schedule(static) num_threads(c.threads)
  for(int r=0;r<c.num_ranges;++r)for(int b=0;b<c.num_beams;++b) {
    size_t i=size_t(r)*c.num_beams+b;double range=r*c.range_resolution;
    double v=clamp((db[i]+(c.gain_assist&&noise?assist_db_:0)-c.display_min_db)/(c.display_max_db-c.display_min_db),0,1);
    out[i]=(range<c.min_range||range>c.max_range)?0:static_cast<uint8_t>(std::round(255*std::pow(v,c.gamma)));
  }
  return out;
}
bool Renderer::cartesianSample(int row,int col,double& range,double& bearing) const {
  const auto& c=config_;
  double x=c.max_range-double(row)/(c.height-1)*(c.max_range-c.min_range);
  double y=c.max_range*std::sin(c.fov/2)*(1-2.*col/(c.width-1));
  range=std::hypot(x,y);bearing=std::atan2(y,x);
  return range>=c.min_range&&range<=c.max_range&&bearing>=c.bearings.front()&&bearing<=c.bearings.back();
}
void Renderer::scanConvert(Frame& f) const {
  const auto& c=config_;f.cartesian.assign(size_t(c.width)*c.height,0);f.cartesian_instances.assign(f.cartesian.size(),0);
  for(int r=0;r<c.height;++r)for(int b=0;b<c.width;++b) {
    double range,bearing;if(!cartesianSample(r,b,range,bearing))continue;
    auto upper=std::upper_bound(c.bearings.begin(),c.bearings.end(),bearing);
    int lo=std::max(0,std::min(c.num_beams-2,int(upper-c.bearings.begin())-1));
    double beam=lo+(bearing-c.bearings[lo])/(c.bearings[lo+1]-c.bearings[lo]);
    double bin=range/c.range_resolution;int rr=std::min(c.num_ranges-2,int(bin));
    double fr=bin-rr,fb=beam-lo;
    auto at=[&](int row,int col){return f.raw[size_t(row)*c.num_beams+col];};
    double value=(1-fr)*((1-fb)*at(rr,lo)+fb*at(rr,lo+1))+fr*((1-fb)*at(rr+1,lo)+fb*at(rr+1,lo+1));
    size_t pixel=size_t(r)*c.width+b;f.cartesian[pixel]=uint8_t(std::round(clamp(value,0,255)));
    int nr=std::min(c.num_ranges-1,int(std::round(bin))),nb=std::min(c.num_beams-1,int(std::round(beam)));
    f.cartesian_instances[pixel]=f.instances[size_t(nr)*c.num_beams+nb];
  }
}
}  // namespace fls
