#include "Registration.h"


struct PointDistance
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // This class should include an auto-differentiable cost function. 
  // To rotate a point given an axis-angle rotation, use
  // the Ceres function:
  // AngleAxisRotatePoint(...) (see ceres/rotation.h)
  // Similarly to the Bundle Adjustment case initialize the struct variables with the source and  the target point.
  // You have to optimize only the 6-dimensional array (rx, ry, rz, tx ,ty, tz).
  // WARNING: When dealing with the AutoDiffCostFunction template parameters,
  // pay attention to the order of the template parameters
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  PointDistance(const Eigen::Vector3d& source, const Eigen::Vector3d& target)
    : source_(source), target_(target) {}

  template <typename T>
  bool operator()(const T* const transform, T* residual) const
  {
        // Convert source_ from Eigen::Vector3d to an array of T
        T source[3];
        source[0] = T(source_[0]);
        source[1] = T(source_[1]);
        source[2] = T(source_[2]);

        T p[3];
        ceres::AngleAxisRotatePoint(transform, source, p);
        p[0] += transform[3]; // tx 
        p[1] += transform[4]; // ty
        p[2] += transform[5]; // tz   

        residual[0] = p[0] - T(target_[0]);
        residual[1] = p[1] - T(target_[1]);
        residual[2] = p[2] - T(target_[2]);
        return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& source, const Eigen::Vector3d& target)
  {
    return (new ceres::AutoDiffCostFunction<PointDistance, 3, 6>(
        new PointDistance(source, target)));
  }

  Eigen::Vector3d source_;
  Eigen::Vector3d target_;

};


Registration::Registration(std::string cloud_source_filename, std::string cloud_target_filename)
{
  std::cout << "Loading source point cloud..." << std::endl;
  open3d::io::ReadPointCloud(cloud_source_filename, source_ );
  std::cout << "Loading target point cloud..." << std::endl;
  open3d::io::ReadPointCloud(cloud_target_filename, target_ );
  Eigen::Vector3d gray_color;
  source_for_icp_ = source_;
}


Registration::Registration(open3d::geometry::PointCloud cloud_source, open3d::geometry::PointCloud cloud_target)
{
  source_ = cloud_source;
  target_ = cloud_target;
  source_for_icp_ = source_;
}


void Registration::draw_registration_result()
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  //different color
  Eigen::Vector3d color_s;
  Eigen::Vector3d color_t;
  color_s<<1, 0.706, 0;
  color_t<<0, 0.651, 0.929;

  target_clone.PaintUniformColor(color_t);
  source_clone.PaintUniformColor(color_s);
  source_clone.Transform(transformation_);

  auto src_pointer =  std::make_shared<open3d::geometry::PointCloud>(source_clone);
  auto target_pointer =  std::make_shared<open3d::geometry::PointCloud>(target_clone);
  open3d::visualization::DrawGeometries({src_pointer, target_pointer});
  return;
}



void Registration::execute_icp_registration(double threshold, int max_iteration, double relative_rmse, std::string mode)
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //ICP main loop
  //Check convergence criteria and the current iteration.
  //If mode=="svd" use get_svd_icp_transformation if mode=="lm" use get_lm_icp_transformation.
  //Remember to update transformation_ class variable, you can use source_for_icp_ to store transformed 3d points.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    std::cout << "Starting ICP registration..." << std::endl;
    double prev_rmse = std::numeric_limits<double>::max();
    for (int i = 0; i < max_iteration; ++i) {
        std::cout << "Iteration: " << i << std::endl;
        auto [source_indices, target_indices, rmse] = find_closest_point(threshold);
        std::cout << "RMSE: " << rmse << std::endl;

        Eigen::Matrix4d transformation;
        if (mode == "svd") {
            transformation = get_svd_icp_transformation(source_indices, target_indices);
        } else if (mode == "lm") {
            transformation = get_lm_icp_registration(source_indices, target_indices);
        }

        source_for_icp_.Transform(transformation);
        transformation_ = transformation * transformation_;

        if (fabs(prev_rmse - rmse) < relative_rmse) {
            std::cout << "Convergence reached." << std::endl; 
            break;
        }
        prev_rmse = rmse;
    }





  return;
}


std::tuple<std::vector<size_t>, std::vector<size_t>, double> Registration::find_closest_point(double threshold)
{ ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find source and target indices: for each source point find the closest one in the target and discard if their 
  //distance is bigger than threshold
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  std::vector<size_t> target_indices;
  std::vector<size_t> source_indices;
  // my code here 
  open3d::geometry::KDTreeFlann target_kd_tree(target_);

  int num_source_points = source_for_icp_.points_.size();
  Eigen::Vector3d source_point;
  double mse = 0.0;

  std::vector<int> idx(1);
  std::vector<double> dist2(1);


  for (size_t i = 0; i < num_source_points; ++i) {
    source_point = source_for_icp_.points_[i];
    target_kd_tree.SearchKNN(source_point, 1, idx, dist2);
    if (dist2[0] < threshold * threshold) {
      source_indices.push_back(i);
      target_indices.push_back(idx[0]);
      mse = mse * i / (i + 1) + dist2[0] / (i + 1);
    }
  }



  return {source_indices, target_indices, sqrt(mse)};
}

Eigen::Matrix4d Registration::get_svd_icp_transformation(std::vector<size_t> source_indices, std::vector<size_t> target_indices){
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find point clouds centroids and subtract them. 
  //Use SVD (Eigen::JacobiSVD<Eigen::MatrixXd>) to find best rotation and translation matrix.
  //Use source_indices and target_indices to extract point to compute the 3x3 matrix to be decomposed.
  //Remember to manage the special reflection case.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);

  int num_points = source_indices.size();
  Eigen::MatrixXd source_mat(num_points, 3);
  Eigen::MatrixXd target_mat(num_points, 3);

  for (int i = 0; i < num_points; ++i) {
    source_mat.row(i) = source_for_icp_.points_[source_indices[i]];
    target_mat.row(i) = target_.points_[target_indices[i]];
  }


  Eigen::Vector3d source_centroid = source_mat.colwise().mean();
  Eigen::Vector3d target_centroid = target_mat.colwise().mean();
  source_mat.rowwise() -= source_centroid.transpose();
  target_mat.rowwise() -= target_centroid.transpose();

  Eigen::MatrixXd H = source_mat.transpose() * target_mat;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();


  if (R.determinant() < 0) {
    Eigen::Matrix3d V = svd.matrixV();
    V.col(2) *= -1;
    R = V * svd.matrixU().transpose();
  }
  Eigen::Vector3d t = target_centroid - R * source_centroid;

  transformation.block<3, 3>(0, 0) = R;
  transformation.block<3, 1>(0, 3) = t;



  return transformation;
}

Eigen::Matrix4d Registration::get_lm_icp_registration(std::vector<size_t> source_indices, std::vector<size_t> target_indices)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Use LM (Ceres) to find best rotation and translation matrix. 
  //Remember to convert the euler angles in a rotation matrix, store it coupled with the final translation on:
  //Eigen::Matrix4d transformation.
  //The first three elements of std::vector<double> transformation_arr represent the euler angles, the last ones
  //the translation.
  //use source_indices and target_indices to extract point to compute the matrix to be decomposed.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 4;
  options.max_num_iterations = 100;

  ceres::Problem problem;




  std::vector<double> transformation_arr(6, 0.0);
  int num_points = source_indices.size();
  // For each point....
  for( int i = 0; i < num_points; i++ )
  {
    
    Eigen::Vector3d source_point = source_for_icp_.points_[source_indices[i]];
    Eigen::Vector3d target_point = target_.points_[target_indices[i]];

    ceres::CostFunction* cost_function = PointDistance::Create(source_point, target_point);
    problem.AddResidualBlock(cost_function, nullptr, transformation_arr.data());



  }



  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  Eigen::Matrix3d R;
  ceres::AngleAxisToRotationMatrix(transformation_arr.data(), R.data());

  transformation.block<3, 3>(0, 0) = R;
  transformation.block<3, 1>(0, 3) = Eigen::Vector3d(transformation_arr[3], transformation_arr[4], transformation_arr[5]);


  return transformation;
}


void Registration::set_transformation(Eigen::Matrix4d init_transformation)
{
  transformation_=init_transformation;
}


Eigen::Matrix4d  Registration::get_transformation()
{
  return transformation_;
}

double Registration::compute_rmse()
{
  open3d::geometry::KDTreeFlann target_kd_tree(target_);
  open3d::geometry::PointCloud source_clone = source_;
  source_clone.Transform(transformation_);
  int num_source_points  = source_clone.points_.size();
  Eigen::Vector3d source_point;
  std::vector<int> idx(1);
  std::vector<double> dist2(1);
  double mse;
  for(size_t i=0; i < num_source_points; ++i) {
    source_point = source_clone.points_[i];
    target_kd_tree.SearchKNN(source_point, 1, idx, dist2);
    mse = mse * i/(i+1) + dist2[0]/(i+1);
  }
  return sqrt(mse);
}

void Registration::write_tranformation_matrix(std::string filename)
{
  std::ofstream outfile (filename);
  if (outfile.is_open())
  {
    outfile << transformation_;
    outfile.close();
  }
}

void Registration::save_merged_cloud(std::string filename)
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  source_clone.Transform(transformation_);
  open3d::geometry::PointCloud merged = target_clone+source_clone;
  open3d::io::WritePointCloud(filename, merged );
}

