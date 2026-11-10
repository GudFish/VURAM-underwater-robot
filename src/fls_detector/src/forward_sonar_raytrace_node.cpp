#include "fls_detector/sonar_renderer.hpp"

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>
#include <XmlRpcValue.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
fls::Pose pose(const geometry_msgs::Pose& p) {
  return fls::Pose::quaternion({p.position.x,p.position.y,p.position.z},
      p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w);
}
template<class T> T parameter(ros::NodeHandle& nh,const std::string& name,T fallback) {
  T value;nh.param(name,value,fallback);return value;
}
double number(const XmlRpc::XmlRpcValue& v) {
  double result;
  if(v.getType()==XmlRpc::XmlRpcValue::TypeInt)result=int(v);
  else if(v.getType()==XmlRpc::XmlRpcValue::TypeDouble)result=double(v);
  else throw std::invalid_argument("scene coordinates must be numbers");
  if(!std::isfinite(result))throw std::invalid_argument("scene coordinates must be finite");
  return result;
}
fls::Vec3 vector3(const XmlRpc::XmlRpcValue& v) {
  if(v.getType()!=XmlRpc::XmlRpcValue::TypeArray||v.size()!=3)throw std::invalid_argument("expected [x,y,z]");
  return {number(v[0]),number(v[1]),number(v[2])};
}
bool finite(const geometry_msgs::Pose& p) {
  return std::isfinite(p.position.x)&&std::isfinite(p.position.y)&&std::isfinite(p.position.z);
}
fls::Material material(ros::NodeHandle& nh,const std::string& prefix,fls::Material m={}) {
  m.diffuse=parameter(nh,prefix+"/diffuse",m.diffuse);
  m.specular=parameter(nh,prefix+"/specular",m.specular);
  m.specular_exponent=parameter(nh,prefix+"/specular_exponent",m.specular_exponent);
  m.backscatter=parameter(nh,prefix+"/backscatter",m.backscatter);
  m.backscatter_exponent=parameter(nh,prefix+"/backscatter_exponent",m.backscatter_exponent);
  m.incidence_exponent=parameter(nh,prefix+"/incidence_exponent",m.incidence_exponent);
  for(double value:{m.diffuse,m.specular,m.backscatter,m.specular_exponent,m.backscatter_exponent,m.incidence_exponent})
    if(!std::isfinite(value)||value<0)throw std::invalid_argument("invalid material: "+prefix);
  return m;
}
template<class T> sensor_msgs::Image image(const std::vector<T>& data,int rows,int cols,
                                         const std::string& encoding,const std_msgs::Header& header) {
  sensor_msgs::Image msg;msg.header=header;msg.height=rows;msg.width=cols;
  msg.encoding=encoding;msg.step=cols*sizeof(T);
  const uint16_t endian=1;msg.is_bigendian=(*reinterpret_cast<const uint8_t*>(&endian)==0);
  msg.data.resize(data.size()*sizeof(T));std::memcpy(msg.data.data(),data.data(),msg.data.size());return msg;
}
class SonarNode {
 public:
  SonarNode():private_("~") {
    fls::Config c;
    c.min_range=parameter(private_,"min_range",c.min_range);c.max_range=parameter(private_,"max_range",c.max_range);
    c.num_beams=parameter(private_,"num_beams",c.num_beams);c.num_ranges=parameter(private_,"num_range_bins",c.num_ranges);
    c.range_resolution=parameter(private_,"range_resolution",c.range_resolution);
    c.width=parameter(private_,"image_width",c.width);c.height=parameter(private_,"image_height",c.height);
    c.fov=parameter(private_,"fov_deg",130.)*fls::pi/180;
    c.vertical_fov=parameter(private_,"vertical_fov_deg",20.)*fls::pi/180;
    int mode=parameter(private_,"mode",2);
    if(mode!=1&&mode!=2)throw std::invalid_argument("mode must be 1 (LF) or 2 (HF)");
    if(c.max_range>(mode==1?120.:40.))throw std::invalid_argument("range exceeds BHQ-750D mode limit");
    c.frequency_hz=mode==1?750000.:1200000.;
    c.angular_resolution=parameter(private_,"angular_resolution_deg",mode==1?1.:0.6)*fls::pi/180;
    c.range_sigma=parameter(private_,"range_sigma_m",(mode==1?0.004:0.0025)/2.355);
    std::vector<double> angles;
    if(private_.getParam("beam_angles_deg",angles))for(double a:angles)c.bearings.push_back(a*fls::pi/180);
    c.elevation_samples=parameter(private_,"elevation_samples",c.elevation_samples);
    c.max_rays_per_beam=parameter(private_,"max_rays_per_beam",c.max_rays_per_beam);
    c.refinement_depth=parameter(private_,"refinement_depth",c.refinement_depth);
    c.threads=parameter(private_,"threads",c.threads);
    c.absorption_db_per_m=parameter(private_,"absorption_db_per_m",c.absorption_db_per_m);
    c.source_power=parameter(private_,"source_power",c.source_power);
    c.tvg_db_per_decade=parameter(private_,"tvg_db_per_decade",c.tvg_db_per_decade);
    c.horizontal_edge_gain=parameter(private_,"horizontal_edge_gain",c.horizontal_edge_gain);
    c.vertical_edge_gain=parameter(private_,"vertical_edge_gain",c.vertical_edge_gain);
    c.noise_floor_power=parameter(private_,"noise_floor_power",c.noise_floor_power);
    c.noise_rayleigh_db=parameter(private_,"noise_rayleigh_db",c.noise_rayleigh_db);
    c.noise_correlation=parameter(private_,"noise_correlation",c.noise_correlation);
    c.noise_profile_range_m=parameter(private_,"noise_profile_range_m",c.noise_profile_range_m);
    private_.getParam("noise_mean_db",c.noise_mean_db);private_.getParam("noise_std_db",c.noise_std_db);
    private_.getParam("noise_beam_bias_db",c.noise_beam_bias_db);
    c.display_min_db=parameter(private_,"display_min_db",c.display_min_db);
    c.display_max_db=parameter(private_,"display_max_db",c.display_max_db);
    c.gain_db=parameter(private_,"gain_db",c.gain_db);
    int gamma=parameter(private_,"gamma",150);
    if(gamma<0||gamma>255)throw std::invalid_argument("gamma must be an SDK byte (0..255)");
    c.gamma=(gamma==0||gamma==255)?1.:gamma/255.;
    c.gain_assist=parameter(private_,"gain_assist",false);
    c.gain_assist_alpha=parameter(private_,"gain_assist_alpha",c.gain_assist_alpha);
    c.multipath=parameter(private_,"multipath",false);c.wrt=parameter(private_,"wrt",true);c.prt=parameter(private_,"prt",true);
    c.random_seed=parameter(private_,"random_seed",7);
    if(c.random_seed<0)c.random_seed=int(std::random_device{}()&0x7fffffff);
    renderer_.reset(new fls::Renderer(c));c=renderer_->config();
    rate_=parameter(private_,"publish_rate",10.);double sound=parameter(private_,"speed_of_sound",1510.);
    if(!std::isfinite(rate_)||rate_<=0||!std::isfinite(sound)||sound<=0)throw std::invalid_argument("invalid rate/sound speed");
    if(rate_>sound/(2*c.max_range))throw std::invalid_argument("publish rate exceeds round-trip acoustic limit");
    frame_=parameter(private_,"frame_id",std::string("sonar_frame"));
    publish_debug_=parameter(private_,"publish_ground_truth",true);
    waiting_=parameter(private_,"publish_waiting_image",true);
    obstacles_enabled_=parameter(private_,"enable_obstacle_echoes",true);
    std::vector<double> xyz{0,0,0},rpy{0,0,0};
    private_.getParam("sensor_xyz",xyz);private_.getParam("sensor_rpy_deg",rpy);
    if(xyz.size()!=3||rpy.size()!=3)throw std::invalid_argument("sensor extrinsics need three values");
    for(double v:xyz)if(!std::isfinite(v))throw std::invalid_argument("invalid sensor translation");
    for(double v:rpy)if(!std::isfinite(v))throw std::invalid_argument("invalid sensor rotation");
    extrinsic_=fls::Pose::rpy({xyz[0],xyz[1],xyz[2]},rpy[0]*fls::pi/180,rpy[1]*fls::pi/180,rpy[2]*fls::pi/180);
    target_.id=1;target_.material=material(private_,"materials/target");
    obstacle_material_=material(private_,"materials/obstacle");
    std::string resource=parameter(private_,"target_mesh_resource",std::string("package://vuram_description/meshes/AUV.STL"));
    if(resource.find("package://")==0) {
      size_t slash=resource.find('/',10);
      if(slash==std::string::npos)throw std::invalid_argument("invalid package URI");
      auto root=ros::package::getPath(resource.substr(10,slash-10));
      if(root.empty())throw std::runtime_error("mesh package not found");
      resource=root+resource.substr(slash);
    }
    target_.shape=fls::Shape::Mesh;
    target_.mesh=fls::Mesh::loadSTL(resource,parameter(private_,"target_mesh_scale",1.));
    ROS_INFO("[SonarRaytrace] cached BVH for %zu target triangles",target_.mesh->triangleCount());
    loadSurfaces();
    auto topic=parameter(private_,"image_topic",std::string("/forward_sonar/image"));
    cart_pub_=nh_.advertise<sensor_msgs::Image>(topic,1);
    raw_pub_=nh_.advertise<sensor_msgs::Image>(parameter(private_,"raw_image_topic",std::string("/forward_sonar/image_raw")),1);
    for(auto name:{"direct_raw","wrt_raw","prt_raw","instance_raw","instance_image"})
      debug_pub_[name]=nh_.advertise<sensor_msgs::Image>(std::string("/forward_sonar/")+name,1);
    info_pub_=nh_.advertise<std_msgs::String>("/forward_sonar/metadata",1,true);
    stats_pub_=nh_.advertise<std_msgs::Float64MultiArray>("/forward_sonar/render_stats",1);
    odom_sub_=nh_.subscribe(parameter(private_,"robot_odom_topic",std::string("/mavros/local_position/odom")),1,&SonarNode::odom,this);
    target_sub_=nh_.subscribe(parameter(private_,"target_pose_topic",std::string("/dynamic_target/pose")),1,&SonarNode::target,this);
    obstacle_sub_=nh_.subscribe(parameter(private_,"obstacle_marker_topic",std::string("/dynamic_obstacles/markers")),1,&SonarNode::obstacles,this);
    publishMetadata(gamma,sound);
    timer_=nh_.createTimer(ros::Duration(1/rate_),&SonarNode::tick,this);
    ROS_INFO("[SonarRaytrace] %dx%d native -> %dx%d Cartesian, vertical=%.1f deg, multipath=%s",
        c.num_beams,c.num_ranges,c.width,c.height,c.vertical_fov*180/fls::pi,c.multipath?"on":"off");
  }
 private:
  void loadSurfaces() {
    for(auto name:{"floor","water_surface"}) {
      if(!parameter(private_,std::string(name)+"/enabled",false))continue;
      double z;
      if(!private_.getParam(std::string(name)+"/z",z)||!std::isfinite(z))throw std::invalid_argument(std::string(name)+" requires finite world z");
      fls::Object o;o.id=std::string(name)=="floor"?2:3;o.shape=fls::Shape::Plane;o.pose.position.z=z;
      o.material=material(private_,std::string("materials/")+name);static_.push_back(o);
    }
    XmlRpc::XmlRpcValue walls;
    if(private_.getParam("walls",walls)) {
      if(walls.getType()!=XmlRpc::XmlRpcValue::TypeArray||walls.size()>1000)throw std::invalid_argument("walls must be an array (max 1000)");
      for(int i=0;i<walls.size();++i) {
        fls::Object o;o.shape=fls::Shape::Box;o.id=uint16_t(4+i);
        o.size=vector3(walls[i]["size"]);fls::Vec3 p=vector3(walls[i]["position"]),rpy;
        if(walls[i].hasMember("rpy_deg"))rpy=vector3(walls[i]["rpy_deg"])*fls::pi/180;
        if(o.size.x<=0||o.size.y<=0||o.size.z<=0)throw std::invalid_argument("wall dimensions must be positive");
        o.pose=fls::Pose::rpy(p,rpy.x,rpy.y,rpy.z);o.material=material(private_,"materials/wall");static_.push_back(o);
      }
    }
  }
  bool matching(const std::string& frame) const { return frame.empty()||world_.empty()||frame==world_; }
  void odom(const nav_msgs::Odometry::ConstPtr& msg) {
    try {
      if(!finite(msg->pose.pose))throw std::invalid_argument("nonfinite odometry");
      robot_=pose(msg->pose.pose);world_=msg->header.frame_id.empty()?"map":msg->header.frame_id;have_robot_=true;
    } catch(const std::exception& e){ROS_WARN_THROTTLE(3,"[SonarRaytrace] %s",e.what());}
  }
  void target(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    target_frame_=msg->header.frame_id;
    try {if(!finite(msg->pose))throw std::invalid_argument("nonfinite target");target_.pose=pose(msg->pose);have_target_=true;}
    catch(const std::exception& e){have_target_=false;ROS_WARN_THROTTLE(3,"[SonarRaytrace] %s",e.what());}
  }
  void obstacles(const visualization_msgs::MarkerArray::ConstPtr& msg) {
    for(const auto& m:msg->markers) {
      auto key=std::make_pair(m.ns,m.id);
      if(m.action==visualization_msgs::Marker::DELETEALL){markers_.clear();expires_.clear();continue;}
      if(m.action==visualization_msgs::Marker::DELETE){markers_.erase(key);expires_.erase(key);continue;}
      if(m.action==visualization_msgs::Marker::ADD) {
        markers_[key]=m;
        expires_[key]=m.lifetime.isZero()?ros::Time(0):ros::Time::now()+m.lifetime;
      }
    }
  }
  void publishMetadata(int gamma,double sound) {
    const auto& c=renderer_->config();std::ostringstream out;out<<std::setprecision(12);
    out<<"{\"version\":1,\"backend\":\"raytrace\",\"device\":\"BHQ-750D\",\"calibrated\":false,"
       <<"\"n_beams\":"<<c.num_beams<<",\"n_ranges\":"<<c.num_ranges<<",\"range_resolution_m\":"<<c.range_resolution
       <<",\"range_m\":"<<c.max_range<<",\"min_range_m\":"<<c.min_range<<",\"frequency_hz\":"<<c.frequency_hz
       <<",\"cartesian_fov_deg\":"<<c.fov*180/fls::pi
       <<",\"vertical_fov_deg\":"<<c.vertical_fov*180/fls::pi<<",\"angular_resolution_deg\":"<<c.angular_resolution*180/fls::pi
       <<",\"gamma_byte\":"<<gamma<<",\"gain_db\":"<<c.gain_db<<",\"gain_assist\":"<<(c.gain_assist?"true":"false")
       <<",\"noise_profile_range_m\":"<<c.noise_profile_range_m
       <<",\"speed_of_sound_mps\":"<<sound<<",\"random_seed\":"<<c.random_seed
       <<",\"multipath\":"<<(c.multipath?"true":"false")<<",\"beam_angles_deg\":[";
    for(size_t i=0;i<c.bearings.size();++i){if(i)out<<',';out<<c.bearings[i]*180/fls::pi;}
    out<<"]}";std_msgs::String msg;msg.data=out.str();info_pub_.publish(msg);
  }
  void tick(const ros::TimerEvent&) {
    if(!have_robot_&&!waiting_)return;
    auto start=std::chrono::steady_clock::now();ros::Time stamp=ros::Time::now();
    fls::Scene scene;scene.objects=static_;
    if(have_robot_) {
      if(have_target_&&matching(target_frame_))scene.objects.push_back(target_);
      else if(have_target_)ROS_WARN_THROTTLE(3,"[SonarRaytrace] target frame differs from odometry; transform inputs to %s",world_.c_str());
      if(obstacles_enabled_)for(auto it=markers_.begin();it!=markers_.end();) {
        const auto& m=it->second;
        auto expiration=expires_.find(it->first);
        if(expiration!=expires_.end()&&!expiration->second.isZero()&&stamp>expiration->second) {
          expires_.erase(expiration);it=markers_.erase(it);continue;
        }
        ++it;
        if(!matching(m.header.frame_id)||!finite(m.pose))continue;
        fls::Object o;
        if(m.type==visualization_msgs::Marker::SPHERE)o.shape=fls::Shape::Sphere;
        else if(m.type==visualization_msgs::Marker::CUBE)o.shape=fls::Shape::Box;
        else if(m.type==visualization_msgs::Marker::CYLINDER)o.shape=fls::Shape::Cylinder;
        else continue;
        if(!(m.scale.x>0&&m.scale.y>0&&m.scale.z>0)||!std::isfinite(m.scale.x+m.scale.y+m.scale.z))continue;
        if(!marker_ids_.count({m.ns,m.id})) {
          if(next_id_>65535){ROS_ERROR_THROTTLE(3,"[SonarRaytrace] instance ID space exhausted");continue;}
          marker_ids_[{m.ns,m.id}]=uint16_t(next_id_++);
        }
        o.id=marker_ids_.at({m.ns,m.id});o.size={m.scale.x,m.scale.y,m.scale.z};o.material=obstacle_material_;
        try{o.pose=pose(m.pose);scene.objects.push_back(o);}catch(const std::exception&){}
      }
    } else {scene.objects.clear();ROS_WARN_THROTTLE(3,"[SonarRaytrace] waiting for odometry");}
    fls::Pose sensor=robot_.compose(extrinsic_);
    fls::Frame f=renderer_->render(scene,sensor,index_++);const auto& c=renderer_->config();
    std_msgs::Header header;header.stamp=stamp;header.frame_id=frame_;
    raw_pub_.publish(image(f.raw,f.rows,f.cols,"mono8",header));
    cart_pub_.publish(image(f.cartesian,c.height,c.width,"mono8",header));
    if(publish_debug_) {
      debug_pub_["direct_raw"].publish(image(f.direct,f.rows,f.cols,"32FC1",header));
      debug_pub_["wrt_raw"].publish(image(f.wrt,f.rows,f.cols,"32FC1",header));
      debug_pub_["prt_raw"].publish(image(f.prt,f.rows,f.cols,"32FC1",header));
      debug_pub_["instance_raw"].publish(image(f.instances,f.rows,f.cols,"16UC1",header));
      debug_pub_["instance_image"].publish(image(f.cartesian_instances,c.height,c.width,"16UC1",header));
    }
    if(have_robot_&&world_!=frame_) {
      geometry_msgs::TransformStamped tf;tf.header.stamp=stamp;tf.header.frame_id=world_;tf.child_frame_id=frame_;
      tf.transform.translation.x=sensor.position.x;tf.transform.translation.y=sensor.position.y;tf.transform.translation.z=sensor.position.z;
      const auto& r=sensor.rotation;tf2::Quaternion q;
      tf2::Matrix3x3(r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8]).getRotation(q);
      tf.transform.rotation.x=q.x();tf.transform.rotation.y=q.y();tf.transform.rotation.z=q.z();tf.transform.rotation.w=q.w();tf_pub_.sendTransform(tf);
    }
    double total=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    std_msgs::Float64MultiArray stats;stats.data={f.render_ms,total,double(f.rays)};stats_pub_.publish(stats);
    if(total>1000/rate_)ROS_WARN_THROTTLE(3,"[SonarRaytrace] frame %.1f ms exceeds %.1f ms budget; model unchanged",total,1000/rate_);
  }
  ros::NodeHandle nh_,private_;ros::Subscriber odom_sub_,target_sub_,obstacle_sub_;ros::Timer timer_;
  ros::Publisher cart_pub_,raw_pub_,info_pub_,stats_pub_;std::map<std::string,ros::Publisher> debug_pub_;
  tf2_ros::TransformBroadcaster tf_pub_;
  std::unique_ptr<fls::Renderer> renderer_;fls::Pose robot_,extrinsic_;fls::Object target_;
  fls::Material obstacle_material_;std::vector<fls::Object> static_;
  std::map<std::pair<std::string,int>,visualization_msgs::Marker> markers_;
  std::map<std::pair<std::string,int>,ros::Time> expires_;
  std::map<std::pair<std::string,int>,uint16_t> marker_ids_;
  int next_id_=1024;uint64_t index_=0;std::string frame_,world_,target_frame_;
  bool have_robot_=false,have_target_=false,publish_debug_=true,waiting_=true,obstacles_enabled_=true;
  double rate_=10;
};
}  // namespace
int main(int argc,char** argv) {
  ros::init(argc,argv,"forward_sonar_sim");
  try{SonarNode node;ros::spin();}
  catch(const std::exception& e){ROS_FATAL("[SonarRaytrace] %s",e.what());return 1;}
  return 0;
}
