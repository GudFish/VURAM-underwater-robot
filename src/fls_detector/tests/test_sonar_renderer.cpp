#include "fls_detector/sonar_renderer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {
void check(bool ok,const char* what) { if(!ok)throw std::runtime_error(what); }
void close(double actual,double expected,double tolerance,const char* what) {
  if(std::abs(actual-expected)>tolerance) {
    std::cerr<<what<<": "<<actual<<" != "<<expected<<'\n';throw std::runtime_error(what);
  }
}
fls::Object box(double x,double y,double z,fls::Vec3 size,uint16_t id) {
  fls::Object o;o.shape=fls::Shape::Box;o.pose.position={x,y,z};o.size=size;o.id=id;return o;
}
fls::Object plane(fls::Vec3 position,double roll,double pitch,double yaw,uint16_t id) {
  fls::Object o;o.shape=fls::Shape::Plane;o.pose=fls::Pose::rpy(position,roll,pitch,yaw);o.id=id;return o;
}
double sum(const std::vector<float>& v){return std::accumulate(v.begin(),v.end(),0.);}
}
int main(int argc,char** argv) {
  try {
    fls::Config c;c.num_beams=65;c.max_range=15;c.range_resolution=.02;c.threads=2;
    c.angular_resolution=0;c.noise_rayleigh_db=0;c.range_sigma=0;c.tvg_db_per_decade=0;
    auto transform=fls::Pose::rpy({2,3,4},.3,-.4,.7);
    fls::Vec3 v{5,6,7},back=transform.inverseRotate(transform.apply(v)-transform.position);
    close((back-v).norm(),0,1e-10,"full quaternion inverse");
    auto extrinsic=fls::Pose::rpy({.2,.1,-.3},0,.2,0);
    close((transform.compose(extrinsic).apply(v)-transform.apply(extrinsic.apply(v))).norm(),0,1e-10,"sensor extrinsic composition");

    fls::Config wide=c;wide.vertical_fov=90*fls::pi/180;
    fls::Scene vertical;vertical.objects.push_back(plane({8,0,0},0,fls::pi/2,0,1));
    fls::Renderer wide_renderer(wide),renderer(c);
    auto paths=wide_renderer.trace(vertical,{},0,std::atan2(6.,8.));
    check(paths.size()==1,"visible elevated return");close(paths[0].range,10,1e-9,"slant range 8,0,6");
    check(renderer.trace(vertical,{},0,std::atan2(6.,8.)).empty(),"20 degree vertical aperture");
    auto turned=fls::Pose::rpy({},0,-std::atan2(6.,8.),0);
    close(renderer.trace(vertical,turned,0,0).at(0).range,10,1e-9,"platform pitch rotates beam");
    check(renderer.trace(vertical,{},c.fov/2+.001,0).empty(),"azimuth aperture");

    fls::Scene occlusion;occlusion.objects={box(12,0,0,{1,1,1},1),box(8,0,0,{1,1,1},42)};
    paths=renderer.trace(occlusion,{},0,0);
    check(paths.size()==1&&paths[0].object_id==42,"front surface blocks target");close(paths[0].range,7.5,1e-9,"front face distance");
    fls::Scene floor;floor.objects={plane({0,0,-1},0,0,0,2),box(4,0,-.75,{1,1,.5},7)};
    double phi=std::atan2(-1.,12.);
    check(renderer.trace(floor,{},0,phi).at(0).object_id==2,"low obstacle leaves distant floor visible");
    floor.objects[1]=box(4,0,-.5,{1,1,1},7);
    check(renderer.trace(floor,{},0,phi).at(0).object_id==7,"taller obstacle casts longer geometric shadow");

    for(auto shape:{fls::Shape::Sphere,fls::Shape::Cylinder,fls::Shape::Box}) {
      fls::Scene s;auto o=box(5,0,0,{2,2,2},5);o.shape=shape;s.objects.push_back(o);
      fls::Hit hit;check(s.intersect({{0,0,0},{1,0,0}},10,hit),"analytic primitive intersection");
      close(hit.distance,4,1e-8,"analytic first return");
      check(!s.intersect({{0,3,0},{1,0,0}},10,hit),"analytic primitive miss");
    }
    close(fls::Renderer::powerDb(2)-fls::Renderer::powerDb(1),3.01029995664,1e-10,"power addition is 3 dB");
    fls::Scene flat;flat.objects.push_back(plane({8,0,0},0,fls::pi/2,0,1));
    c.elevation_samples=16;fls::Renderer r16(c);auto f16=r16.render(flat,{});
    c.elevation_samples=64;fls::Renderer r64(c);auto f64=r64.render(flat,{});
    close(sum(f16.direct)/sum(f64.direct),1,.06,"surface energy converges with elevation density");
    fls::Scene disjoint;disjoint.objects.push_back(box(8,0,-.6,{.1,1,.15},9));
    auto one=r64.render(disjoint,{});disjoint.objects.push_back(box(8,0,.6,{.1,1,.15},9));
    auto two=r64.render(disjoint,{});
    close(sum(two.direct)/sum(one.direct),2,.1,"disconnected surfaces in same bin retained");
    fls::Scene bare_floor;bare_floor.objects.push_back(plane({0,0,-1},0,0,0,2));
    auto smooth=r16.render(bare_floor,{});
    for(int j=int(std::ceil(6.0/c.range_resolution));j<int(14.5/c.range_resolution);++j)
      check(smooth.direct[size_t(j)*smooth.cols+smooth.cols/2]>0,"flat-floor reconstruction has no empty range stripes");

    fls::Config multi=c;multi.multipath=true;
    fls::Scene mirrors;
    auto mirror=plane({4,0,0},0,fls::pi/2,-fls::pi/4,2);
    mirror.material.diffuse=.1;mirror.material.specular=.8;mirrors.objects.push_back(mirror);
    fls::Object sphere;sphere.shape=fls::Shape::Sphere;sphere.pose.position={4,3,0};sphere.size={.4,.4,.4};sphere.id=1;
    mirrors.objects.push_back(sphere);fls::Renderer mr(multi);paths=mr.trace(mirrors,{},0,0);
    bool wrt=false,prt=false;
    for(const auto& p:paths) {
      if(p.kind==1){wrt=true;close(p.range,(4+2.8+std::hypot(4.,2.8))/2,1e-5,"WRT half path length");check(p.power>0,"WRT energy");}
      if(p.kind==2){prt=true;close(p.range,6.8,1e-5,"PRT half path length");check(p.power>0,"PRT energy");}
    }
    check(wrt&&prt,"both shallow reflection families");
    mirrors.objects.push_back(box(2,1.4,0,{.4,.4,1},13));
    auto blocked=mr.trace(mirrors,{},0,0);
    check(std::none_of(blocked.begin(),blocked.end(),[](const fls::PathSignal& s){return s.kind==1;}),"blocked WRT source connection rejected");
    check(std::any_of(blocked.begin(),blocked.end(),[](const fls::PathSignal& s){return s.kind==2;}),"PRT does not require direct source visibility");
    auto mf=mr.render(mirrors,{});
    check(sum(mf.prt)>0,"multipath component buffer");

    double range,bearing;
    check(!renderer.cartesianSample(0,0,range,bearing),"far corner outside circular range");
    auto noisy=c;noisy.noise_rayleigh_db=1.5;noisy.random_seed=12;
    fls::Renderer n1(noisy),n2(noisy);fls::Scene empty;
    auto a=n1.render(empty,{},10),b=n2.render(empty,{},10),next=n1.render(empty,{},11);
    check(a.raw==b.raw,"fixed seed/frame reproducibility");check(a.raw!=next.raw,"noise changes across frames");
    check(a.cartesian.front()==0,"outside fan remains black after noise");
    check(std::all_of(a.instances.begin(),a.instances.end(),[](uint16_t id){return id==0;}),"noise is not ground truth");

    auto directional=c;directional.num_beams=5;directional.max_range=1;directional.range_resolution=.1;
    directional.num_ranges=11;directional.gamma=1;directional.display_max_db=100;
    directional.noise_mean_db={20};directional.noise_rayleigh_db=0;
    fls::Renderer plain(directional);
    std::vector<float> power(size_t(plain.config().num_ranges)*5,100.f);
    auto plain_image=plain.display(power,true,4),plain_debug=plain.display(power,false,4);
    auto zero_bias=directional;zero_bias.noise_beam_bias_db={0};fls::Renderer zero(zero_bias);
    check(plain_image==zero.display(power,true,4),"empty beam bias preserves default display");
    directional.noise_beam_bias_db={0,20,0};fls::Renderer biased(directional);
    auto image=biased.display(power,true,4);
    for(int beam=0;beam<5;++beam) {
      double bias=20-std::abs(beam-2)*10;
      auto expected=std::round(255*(fls::Renderer::powerDb(101)+20+bias)/100);
      close(image[5*5+beam],expected,0,"beam bias interpolation and location");
    }
    check(plain_debug==biased.display(power,false,4),"beam bias excluded from noise-free display");
    auto constant_bias=directional;constant_bias.noise_beam_bias_db={10};fls::Renderer constant(constant_bias);
    auto constant_image=constant.display(power,true,4);
    for(int beam=0;beam<5;++beam)
      close(constant_image[5*5+beam],std::round(255*(fls::Renderer::powerDb(101)+30)/100),0,"one-element beam bias is constant");
    auto physical_bias=c;physical_bias.noise_beam_bias_db={0,20,0};
    fls::Renderer physical_plain(c),physical_biased(physical_bias);
    auto physical_before=physical_plain.render(flat,{},9),physical_after=physical_biased.render(flat,{},9);
    check(sum(physical_before.direct)>0,"beam bias test uses actual physical return");
    check(physical_before.direct==physical_after.direct&&physical_before.wrt==physical_after.wrt&&
          physical_before.prt==physical_after.prt&&physical_before.instances==physical_after.instances,
          "beam bias does not change physical powers or truth");
    check(physical_before.raw!=physical_after.raw,"beam bias affects displayed noise image");
    bool nonfinite_bias=false;
    try {auto bad=c;bad.noise_beam_bias_db={std::numeric_limits<double>::quiet_NaN()};fls::Renderer r(bad);}
    catch(const std::invalid_argument&){nonfinite_bias=true;}
    check(nonfinite_bias,"reject NaN beam noise bias");

    // Identical physical ranges must not move along the noise profile when
    // the selected display range changes. Same seed/grid makes this exact.
    auto metric=c;metric.num_beams=9;metric.num_ranges=0;metric.max_range=5.;metric.range_resolution=.5;
    metric.noise_profile_range_m=10.;metric.noise_mean_db={10,20,30};metric.noise_std_db={1,2,3};
    metric.noise_rayleigh_db=0.;metric.noise_beam_bias_db.clear();metric.gamma=1.;metric.display_max_db=100.;
    fls::Renderer short_noise(metric);
    auto short_image=short_noise.display(std::vector<float>(short_noise.config().num_ranges*9,0),true,12);
    for(double maximum:{10.,15.,30.}) {
      auto longer=metric;longer.max_range=maximum;fls::Renderer long_noise(longer);
      auto long_image=long_noise.display(std::vector<float>(long_noise.config().num_ranges*9,0),true,12);
      check(std::equal(short_image.begin(),short_image.end(),long_image.begin()),
            "same-distance noise is identical across selected ranges");
    }
    auto deterministic=metric;deterministic.max_range=15.;deterministic.noise_std_db={0};
    fls::Renderer mapped(deterministic);
    auto mapped_image=mapped.display(std::vector<float>(mapped.config().num_ranges*9,0),true);
    close(mapped_image[5*9+4],std::round(255*.15),0,"2.5 metre noise interpolates within physical 0..10 metre profile");
    close(mapped_image[20*9+4],mapped_image[30*9+4],0,"unmeasured distance holds 10 metre endpoint");
    auto finer=deterministic;finer.range_resolution=.25;fls::Renderer fine_noise(finer);
    auto fine_image=fine_noise.display(std::vector<float>(fine_noise.config().num_ranges*9,0),true);
    close(fine_image[10*9+4],mapped_image[5*9+4],0,"physical noise mapping survives range-grid changes");
    for(double invalid_range:{-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
      bool rejected=false;
      try {auto bad=metric;bad.noise_profile_range_m=invalid_range;fls::Renderer r(bad);}
      catch(const std::invalid_argument&){rejected=true;}
      check(rejected,"reject invalid physical noise profile range");
    }
    auto nonuniform=c;nonuniform.bearings={-.5,-.2,0,.1,.5};fls::Renderer nu(nonuniform);
    auto nf=nu.render(flat,{});check(nf.cols==5,"actual SDK bearings define native width");
    bool invalid=false;try {auto bad=c;bad.range_resolution=-1;fls::Renderer r(bad);}catch(const std::invalid_argument&){invalid=true;}
    check(invalid,"reject invalid sampling grid");
    auto tracking_grid=c;tracking_grid.max_range=15.;tracking_grid.num_ranges=0;
    fls::Renderer tracking_renderer(tracking_grid);
    check(tracking_renderer.config().num_ranges==751,"derive sufficient bins for 15 metre tracking");
    auto historical=tracking_grid;historical.num_ranges=494;historical.range_resolution=10./493.;
    bool insufficient=false;
    try {fls::Renderer r(historical);}
    catch(const std::invalid_argument& e) {
      std::string message=e.what();
      insufficient=message.find("covers 10 m")!=std::string::npos&&message.find("max_range=15")!=std::string::npos;
    }
    check(insufficient,"10 metre historical grid rejects 15 metres with useful diagnostic");
    historical.max_range=10.;fls::Renderer historical_renderer(historical);
    check(historical_renderer.config().num_ranges==494,"10 metre historical grid remains valid");
    if(argc>1) {
      auto mesh=fls::Mesh::loadSTL(argv[1]);fls::Object o;o.shape=fls::Shape::Mesh;o.mesh=mesh;o.pose.position={10,0,0};
      fls::Scene s;s.objects.push_back(o);fls::Hit hit;
      check(s.intersect({{0,0,0},{1,0,0}},15,hit),"real AUV mesh BVH hit");
      check(hit.distance>9&&hit.distance<11,"real mesh metre scale");
      std::cout<<"AUV mesh: "<<mesh->triangleCount()<<" valid triangles\n";
    }
    std::cout<<"PASS: geometry, 6DOF, aperture, occlusion, shadows, primitives, energy, resampling, WRT/PRT, masks, seed, beam noise bias, bearings, validation\n";
  }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
  return 0;
}
