// Reproduce display-domain calibration validation using the actual C++ renderer.
#include "fls_detector/sonar_renderer.hpp"
#include <fstream>
#include <iostream>
#include <string>

void pgm(const std::string& path, const std::vector<uint8_t>& data, int rows, int cols) {
  std::ofstream out(path,std::ios::binary);out<<"P5\n"<<cols<<" "<<rows<<"\n255\n";
  out.write(reinterpret_cast<const char*>(data.data()),data.size());
}
int main(int argc,char** argv) {
  if(argc!=4&&argc!=5) {std::cerr<<"usage: render_profile INPUT OUTPUT_PREFIX AUV_STL [MAX_RANGE]\n";return 2;}
  std::ifstream input(argv[1]);fls::Config c;
  input>>c.num_ranges>>c.num_beams>>c.range_resolution>>c.max_range;
  input>>c.noise_floor_power>>c.noise_rayleigh_db>>c.noise_correlation>>c.noise_profile_range_m;
  if(!input||c.num_ranges<2||c.num_ranges>4096||c.num_beams<2||c.num_beams>2048)return 3;
  c.noise_mean_db.resize(c.num_ranges);c.noise_std_db.resize(c.num_ranges);
  c.noise_beam_bias_db.resize(c.num_beams);
  for(auto& v:c.noise_mean_db)input>>v;
  for(auto& v:c.noise_std_db)input>>v;
  for(auto& v:c.noise_beam_bias_db)input>>v;
  if(!input)return 3;
  if(argc==5){c.max_range=std::stod(argv[4]);c.num_ranges=0;}
  fls::Renderer renderer(c);c=renderer.config();
  std::vector<float> empty(size_t(c.num_ranges)*c.num_beams,0);
  std::ofstream out(std::string(argv[2])+"_background.bin",std::ios::binary);
  if(!out)return 4;
  for(int i=0;i<64;++i) {
    auto data=renderer.display(empty,true,i);
    out.write(reinterpret_cast<const char*>(data.data()),data.size());
    if(i==0)pgm(std::string(argv[2])+"_background.pgm",data,c.num_ranges,c.num_beams);
  }
  fls::Scene scene;fls::Object target;target.shape=fls::Shape::Mesh;target.id=1;
  target.mesh=fls::Mesh::loadSTL(argv[3]);target.pose=fls::Pose::rpy({6,0,0},0,0,fls::pi/2);
  scene.objects.push_back(target);
  auto frame=renderer.render(scene,{},0);
  pgm(std::string(argv[2])+"_target_raw.pgm",frame.raw,frame.rows,frame.cols);
  pgm(std::string(argv[2])+"_target_fan.pgm",frame.cartesian,c.height,c.width);
  std::cout<<"rendered "<<frame.rows<<"x"<<frame.cols<<" with target; "<<frame.render_ms<<" ms\n";
}
