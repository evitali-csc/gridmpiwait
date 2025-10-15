#include <chrono>
#include <cmath>
#include <iostream>
#include <mpi.h>
// #include <spdlog/spdlog.h>
#include <sstream>
#include <vector>
#include <numeric>
#include <algorithm>

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/device_vector.h>
#endif

using Timer = std::chrono::high_resolution_clock;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int size, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Assuming a "cube" topology, we need size to be 2^3n, i.e. 8, 64, 512, 4096

  // Compute 3D position
  int nx = static_cast<int>(std::cbrt(static_cast<double>(size)));
  int ny = static_cast<int>(
      std::sqrt(static_cast<double>(size) / static_cast<double>(nx)));
  int nz = size / nx / ny;

  if (size != nx * ny * nz)
    throw std::runtime_error("Invalid number of ranks");

//  if (rank == 0)
//    spdlog::info("size={} [{}x{}x{}]", size, nx, ny, nz);

  // Compute position and list of neighbors
  int x = rank / (ny * nz);
  int r0 = rank % (ny * nz);
  int y = r0 / nz;
  int z = r0 % nz;
//  spdlog::info("rank={}, pos=({}, {}, {})", rank, x, y, z);

  std::vector<int> neighbours;
  auto add = [&](int xi, int yi, int zi) {
    int idx = xi * ny * nz + yi * nz + zi;
    neighbours.push_back(idx);
  };

  // Add neighbours via "facet"
//  if (x > 0)
    add((x - 1 + nx)%nx, y, z);
//  if (x < nx - 1)
    add((x + 1)%nx, y, z);
//  if (y > 0)
    add(x, (y - 1 + ny)%ny, z);
//  if (y < ny - 1)
    add(x, (y + 1)%ny, z);
//  if (z > 0)
    add(x, y, (z - 1 + nz)%nz);
//  if (z < nz - 1)
    add(x, y, (z + 1)%nz);

  // Add neighbours via "edge"
//  if (x > 0 and y > 0)
    add((x - 1 + nx)%nx, (y - 1 + ny)%ny, z);
//  if (x > 0 and z > 0)
    add((x - 1 +nx)%nx, y, (z - 1 +nz)%nz);
//  if (y > 0 and z > 0)
    add(x, (y - 1 +ny)%ny, (z - 1 + nz)%nz);

//  if (x > 0 and y < ny - 1)
    add((x - 1 + nx)%nx, (y + 1)%ny, z);
//  if (x > 0 and z < nz - 1)
    add((x - 1 + nx)%nx, y, (z + 1)%nz);

//  if (y > 0 and x < nx - 1)
    add((x + 1)%nx, (y - 1 + ny)%ny, z);
//  if (y > 0 and z < nz - 1)
    add(x, (y - 1 + ny)%ny, (z + 1)%nz);

//  if (z > 0 and x < nx - 1)
    add((x + 1)%nx, y, (z - 1 + nz)%nz);
//  if (z > 0 and y < ny - 1)
    add(x, (y + 1)%ny, (z - 1 + nz)%nz);

//  if (x < nx - 1 and y < ny - 1)
    add((x + 1)%nx, (y + 1)%ny, z);
//  if (x < nx - 1 and z < nz - 1)
    add((x + 1)%nx, y, (z + 1)%nz);
//  if (y < ny - 1 and z < nz - 1)
    add(x, (y + 1)%ny, (z + 1)%nz);

//  std::stringstream s;
//  for (auto q : neighbours)
//    s << q << ", ";
//  spdlog::info("rank={}, nbr=[{}]", rank, s.str());

#if defined(__CUDACC__) || defined(__HIPCC__)
  using Vector = thrust::device_vector<double>;
#else
  using Vector = std::vector<double>;
#endif

  int datasize = 1000000;
  std::vector<MPI_Request> requests(2 * neighbours.size(), MPI_REQUEST_NULL);
  std::vector<Vector> sendbuf(neighbours.size(), Vector(datasize, 3.14));
  std::vector<Vector> recvbuf(neighbours.size(), Vector(datasize, 2.87));

  std::vector<double> times(1000);
  std::vector<double> times2(1000);
  std::vector<double> times_s(1000);

  for (int j = 0; j < 1000; ++j) {

    using TimerT = std::chrono::duration<double, std::ratio<1>>;
    auto t0 = Timer::now();
    for (std::size_t i = 0; i < neighbours.size(); ++i) {
#if defined(__CUDACC__) || defined(__HIPCC__)
      double *rbuf = thrust::raw_pointer_cast(recvbuf[i].data());
      double *sbuf = thrust::raw_pointer_cast(sendbuf[i].data());
#else
      double *rbuf = recvbuf[i].data();
      double *sbuf = sendbuf[i].data();
#endif
      MPI_Irecv(rbuf, recvbuf[i].size(), MPI_DOUBLE, neighbours[i], MPI_ANY_TAG,
                MPI_COMM_WORLD, &requests[i + neighbours.size()]);
      MPI_Isend(sbuf, sendbuf[i].size(), MPI_DOUBLE, neighbours[i], 0,
                MPI_COMM_WORLD, &requests[i]);
    }
    auto t1 = Timer::now();
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
    auto t2 = Timer::now();

    TimerT dt1 = t1 - t0;
    TimerT dt2 = t2 - t1;

    times[j] = dt1.count();
    times2[j] = dt2.count();
    times_s[j] = times[j] + times2[j];

//    spdlog::info("[{}] [t1] = {}", rank, static_cast<double>(dt1.count()));
//    spdlog::info("[{}] [t2] = {}", rank, static_cast<double>(dt2.count()));
  }


  auto st1 = std::accumulate(times.begin()+5, times.end(), 0.);
  auto st2 = std::accumulate(times2.begin()+5, times2.end(), 0.);
  auto st = std::accumulate(times_s.begin()+5, times_s.end(), 0.);
  auto mt1 = std::min_element(times.begin()+5, times.end());

  auto mt2 = std::min_element(times2.begin()+5, times2.end());
  auto mt = std::min_element(times_s.begin()+5, times_s.end());
  auto mat1 = std::max_element(times.begin()+5, times.end());
  auto mat2 = std::max_element(times2.begin()+5, times2.end());
  auto mat = std::max_element(times_s.begin()+5, times_s.end());

  printf("%d %e %e %e %e %e %e %e %e %e\n", rank, st1, st2, st, *mt1, *mt2, *mt, *mat1, *mat2, *mat);

  MPI_Finalize();
  return 0;
}
