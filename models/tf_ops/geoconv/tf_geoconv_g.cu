/* Geometric Convolution
 * Original author: Shiyi Lan
 * All Rights Reserved. 2019.
 */

#define get_square_euclidean_dist(x,y,z) \
        ((x)*(x)+(y)*(y)+(z)*(z))
#define _CUDA_NUM_THREADS 512
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)
// blockDim.x * gridDim.x是1
static int _GET_BLOCKS(const int N) {
  return (N + _CUDA_NUM_THREADS - 1) / _CUDA_NUM_THREADS;
}

__global__ void Normalization(const int top_count, float* aggre_feat, 
    const float* norm_buffer, const int num_batchs, const int num_points, 
    const int num_channels) {

  CUDA_KERNEL_LOOP(index, top_count) {
    const int base = index * num_channels;
    for (int i = 0; i < num_channels; ++i)
      aggre_feat[base + i] /= norm_buffer[index] + 1;  //通过除以所有的边缘距离权重之和求得最终的边缘权重
  }
}



__global__ void AggregateKernel(const int num_pairs, const int num_batchs, const int num_points, const int num_channels,
    const float* feat, const float* xyz, float* aggre_feat, float* norm_buffer, const float std_square_dist,
    const float square_decay_dist, const int delta) {

  CUDA_KERNEL_LOOP(index, num_pairs) {//循环所有的线程，每一个线程做一对pair，因此index可以看作pair的下标
    const int p0 = index % num_points;
    const int p1 = index / num_points % num_points;
    if (p0 == p1) continue;
    const int b  = index / (num_points * num_points);

    const int pos0 = (b * num_points + p0) * 3;
    const int pos1 = (b * num_points + p1) * 3;
    const float x0 = xyz[pos0], y0 = xyz[pos0+1], z0 = xyz[pos0+2];
    const float x1 = xyz[pos1], y1 = xyz[pos1+1], z1 = xyz[pos1+2];
    const float dx = x0 - x1, dy = y0 - y1, dz = z0 - z1;

    const float square_dist = get_square_euclidean_dist(dx, dy, dz);
    const float dist = sqrt(square_dist);
    if (dist < 1e-4) continue;
    float dist_weight = 0;
    if (square_dist < square_decay_dist) {
      if (square_dist <= std_square_dist)
        dist_weight = 1;
      else
        dist_weight = max(1 - (square_dist - std_square_dist) / (square_decay_dist - std_square_dist), 0.0);

      const float weights[3] = {abs(dx)/dist, abs(dy)/dist, abs(dz)/dist};

      int act[3];
      act[0] = (dx > 0) ? 1 : 0;
      act[1] = (dy > 0) ? 1 : 0;
      act[2] = (dz > 0) ? 1 : 0;

      atomicAdd(norm_buffer + b * num_points + p1, dist_weight);  //计算距离权重的分母，在normalize里面会用到

      for (int i = 0; i < 3; ++i) {
        int dir = (i<<1)  + act[i];
        int p1_idx = (b * num_points + p1) * num_channels;  //注意这个在三个方向内是不变的
        int p0_idx = ((b * num_points + p0) * 6 + dir) * num_channels;  //边缘点的特征会随着三个方向而改变，
                                                                        //并通过下面的channel的循环把每个方向对各个channel的贡献都加上去
                                                                        //也就是把原本的选中的三个方向的特征(6*chennels中的其中3*channel)通过加权进行聚合，得到(1*channel)的特征
        float weight = weights[i] * dist_weight;
        //三个方向的循环
        for (int c = 0; c < num_channels; ++c)
        //每个方向num_channels维的数据的循环
          if (!delta)
            //由于传入的feat已经被flatten了，导致我们必须计算出他的准确的float的位置，而aggre_feat
            atomicAdd(aggre_feat + p1_idx + c, feat[p0_idx + c] * weight);
          else
            atomicAdd(aggre_feat + p1_idx + c, (feat[p0_idx + c] - feat[((b * num_points + p1) + dir) * num_channels]) * weight);
      }
    }
  }
}


__global__ void AggregateGradKernel(const int num_pairs, const int num_batchs, const int num_points, const int num_channels,
    const float* top_feat_grad, const float* xyz, float* bottom_feat_grad, float* norm_buffer, const float std_square_dist,
    const float square_decay_dist, const int delta) {

  CUDA_KERNEL_LOOP(index, num_pairs) {
    const int p0 = index % num_points;
    const int p1 = index / num_points % num_points;  //index除以n，也就是一个数据有多少个点，这样可以使得P0和P1都遍历一遍1到n
    if (p0 == p1) continue;
    const int b  = index / (num_points * num_points);  //计算到第几张图了

    const int pos0 = (b * num_points + p0) * 3;  //取出该张图对应的p0的存储的起始位置
    const int pos1 = (b * num_points + p1) * 3;  //取出该张图对应的p1的存储的起始位置
    const float x0 = xyz[pos0], y0 = xyz[pos0+1], z0 = xyz[pos0+2];  //根据存储起始位置以及偏移取出xyz
    const float x1 = xyz[pos1], y1 = xyz[pos1+1], z1 = xyz[pos1+2];
    const float dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;  //计算xyz三个方向的坐标差

    const float square_dist = get_square_euclidean_dist(dx, dy, dz);   //计算两个点的平方距离
    const float dist = sqrt(square_dist);  //计算两个点的标准距离
    if (dist < 1e-4) continue;  //如果距离过小，那么久不管了
    float dist_weight = 0;
    if (square_dist < square_decay_dist) {  //如果该点落在decay_redius之内
      if (square_dist <= std_square_dist)  //如果该点落在std_redius之内，那么权重就为1
        dist_weight = 1;
      else  //如果落在两个半径之内，那么就取：q到p的距离平方-内半径平方 / 内外半径平方差
        dist_weight = max(1 - (square_dist - std_square_dist) / (square_decay_dist - std_square_dist), .0);
      //使用weights记录三个方向的cos值
      const float weights[3] = {abs(dx)/dist, abs(dy)/dist, abs(dz)/dist};  //计算三个方向的cos值

      int act[3];
      act[0] = (dx > 0) ? 1 : 0;
      act[1] = (dy > 0) ? 1 : 0;
      act[2] = (dz > 0) ? 1 : 0;  //根据dx dy dz的正负号确定选择哪个方向
      //在norm_buffer这个batchsize*n的float内存中的 第b张图的p1位置加上上面通过判断语句计算出来的dist_weight
      atomicAdd(norm_buffer + b * num_points + p1, dist_weight);

      for (int i = 0; i < 3; ++i) {
        int dir = (i<<1)  + act[i];
        int p0_idx = (b * num_points + p0) * num_channels;
        int p1_idx = ((b * num_points + p1) * 6 + dir) * num_channels;
        float weight = weights[i] * dist_weight;
        for (int c = 0; c < num_channels; ++c)
          atomicAdd(bottom_feat_grad + p1_idx + c, top_feat_grad[p0_idx + c] * weight);
      }
    }
  }
}



void aggregateLauncher(int b, int n, int c, const float* feat, const float* xyz, float* out, float* norm_buffer, const float radius, const float decay_radius, const int delta=0) {
  const int num_pair = b * n * n;
  const int top_count = b * n;
  cudaMemset(norm_buffer, 0, sizeof(float) * b * n);  // 给normbuffer指向的前b*n*sizeof（float）个位置置0
  cudaMemset(out, 0, sizeof(float) * b * n * c);
  AggregateKernel<<<_GET_BLOCKS(num_pair), _CUDA_NUM_THREADS>>>(num_pair, b, n, c, feat, xyz, out, norm_buffer, radius * radius, decay_radius * decay_radius, delta);
  Normalization<<<_GET_BLOCKS(top_count), _CUDA_NUM_THREADS>>>(top_count, out, norm_buffer, b, n, c);
}



void aggregategradLauncher(const int b, const int n, const int c, const float* feat, const float* xyz, const float* out, float* norm_buffer, float* grad, const float radius, const float decay_radius, const int delta=0) {
  const int num_pair = b * n * n;
  const int top_count = b * n;
  cudaMemset(norm_buffer, 0, sizeof(float) * b * n);
  cudaMemset(grad, 0, sizeof(float) * b * n * c * 6);
  // 每一对都单独使用一个线程来处理，grid和block都采用1维的形式将线程分为多个block，每个block的大小维num_threads个，
  AggregateGradKernel<<<_GET_BLOCKS(num_pair), _CUDA_NUM_THREADS>>>(num_pair, b, n, c, out, xyz, grad, norm_buffer, radius * radius, decay_radius * decay_radius, delta);
  Normalization<<<_GET_BLOCKS(top_count), _CUDA_NUM_THREADS>>>(top_count, grad, norm_buffer, b, n, c * 6);
}


