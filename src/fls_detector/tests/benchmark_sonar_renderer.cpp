#include "fls_detector/sonar_renderer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

int main(int argc,char** argv) {
  if(argc<2){std::cerr<<"usage: benchmark_sonar_renderer AUV.STL [frames=40] [multipath=0] [output.pgm]\n";return 2;}
  try {
    fls::Config config;config.multipath=argc>3&&std::atoi(argv[3]);
    fls::Renderer renderer(config);fls::Scene scene;
    fls::Object target;target.shape=fls::Shape::Mesh;target.mesh=fls::Mesh::loadSTL(argv[1]);target.pose.position={10,0,0};scene.objects.push_back(target);
    for(int i=0;i<16;++i) {
      fls::Object o;o.shape=i%3==0?fls::Shape::Sphere:(i%3==1?fls::Shape::Cylinder:fls::Shape::Box);
      double angle=(-55.+110.*i/15)*fls::pi/180,range=5.+(i%5)*3.;
      o.pose=fls::Pose::rpy({range*std::cos(angle),range*std::sin(angle),0},.05*i,.03*i,.1*i);
      o.size={.6,.7,.8};o.id=i+10;scene.objects.push_back(o);
    }
    // Include a physical floor to exercise both shadow and shallow multipath work.
    fls::Object floor;floor.shape=fls::Shape::Plane;floor.id=2;floor.pose.position.z=-1;scene.objects.push_back(floor);
    int frames=argc>2?std::max(5,std::atoi(argv[2])):40;std::vector<double> times;fls::Frame f;
    for(int i=0;i<frames+3;++i) {
      target.pose=fls::Pose::rpy({10,.1*std::sin(i*.1),0},0,0,i*.01);scene.objects[0]=target;
      f=renderer.render(scene,{},i);if(i>=3)times.push_back(f.render_ms);
    }
    std::sort(times.begin(),times.end());
    double p95=times[size_t(.95*(times.size()-1))];
    std::cout<<std::fixed<<std::setprecision(2)<<"frames="<<frames<<" mesh_triangles="<<target.mesh->triangleCount()
      <<" obstacles=16 floor=on multipath="<<config.multipath<<" mean_ms="
      <<std::accumulate(times.begin(),times.end(),0.)/times.size()<<" p95_ms="<<p95<<" max_ms="<<times.back()
      <<" rays_last="<<f.rays<<" budget_100ms="<<(p95<=100?"PASS":"FAIL")<<'\n';
    if(argc>4) {
      std::ofstream out(argv[4],std::ios::binary);out<<"P5\n"<<config.width<<' '<<config.height<<"\n255\n";
      out.write(reinterpret_cast<const char*>(f.cartesian.data()),f.cartesian.size());
    }
    return p95<=100?0:3;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
