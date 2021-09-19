/// \ingroup base
/// \class ttk::Stairs
/// \author Julien Tierny <julien.tierny@lip6.fr>
/// \date October 2014.
///
/// \brief TTK processing package for scalar field smoothing.
///
/// This class is a dummy example for the development of TTK classes. It
/// smooths an input scalar field by averaging the scalar values on the link
/// of each vertex.
///
/// \param dataType Data type of the input scalar field (char, float,
/// etc.).
///
/// \sa ttkStairs.cpp %for a usage example.

#ifndef _STAIRS_H
#define _STAIRS_H

// base code includes
#include <Triangulation.h>

namespace ttk {

  class Stairs : virtual public Debug {

  public:
    Stairs();

    ~Stairs();

    int setDimensionNumber(const int &dimensionNumber) {
      dimensionNumber_ = dimensionNumber;
      return 0;
    }

    int setInputDataPointer(void *data) {
      inputData_ = data;
      return 0;
    }

    int setOutputDataPointer(void *data) {
      outputData_ = data;
      return 0;
    }

    int setMaskDataPointer(void *mask) {
      mask_ = (char *)mask;
      return 0;
    }

    inline int preconditionTriangulation(AbstractTriangulation *triangulation) {

      // Pre-condition functions.
      if(triangulation) {
        triangulation->preconditionVertexNeighbors();
      }

      return 0;
    }

    template <class dataType, class TriangulationType = AbstractTriangulation>
    int smooth(const TriangulationType *triangulation,
               const dataType &rangeMin,
               const dataType &rangeMax,
               const double &eps) const;

  protected:
    int dimensionNumber_;
    void *inputData_, *outputData_;
    char *mask_;
  };

} // namespace ttk

// template functions
template <class dataType, class TriangulationType>
int ttk::Stairs::smooth(const TriangulationType *triangulation,
                        const dataType &rangeMin,
                        const dataType &rangeMax,
                        const double &eps) const {

  Timer timer;

#ifndef TTK_ENABLE_KAMIKAZE
  if(!triangulation)
    return -1;
  if(!dimensionNumber_)
    return -2;
  if(!inputData_)
    return -3;
  if(!outputData_)
    return -4;
#endif

  SimplexId vertexNumber = triangulation->getNumberOfVertices();

  dataType *const outputData = static_cast<dataType *>(this->outputData_);
  dataType *inputData = (dataType *)inputData_;

  // init the output
  // init the output
  for(SimplexId i = 0; i < vertexNumber; i++) {
    for(int j = 0; j < dimensionNumber_; j++) {
      outputData[dimensionNumber_ * i + j]
        = inputData[dimensionNumber_ * i + j];
    }
  }
  if(eps == 0) {
    this->printMsg(
      "Complete", 1.0, timer.getElapsedTime(), this->threadNumber_);
    return 0;
  }

  std::cout << "Range " << (double)rangeMin << " " << (double)rangeMax
            << std::endl;
  std::cout << "Epsilon " << eps << std::endl;
  dataType delta = eps * (rangeMax - rangeMin);
  std::vector<dataType> table(floor(1. / (2 * eps)) + 1);
  std::cout << "Delta " << (double)delta << std::endl;
  std::cout << "Dimension " << dimensionNumber_ << std::endl;

  std::cout << "Table" << std::endl;
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(size_t i = 0; i < table.size(); i++) {
    table[i] = rangeMin + i * 2 * delta;
    std::cout << " " << table[i];
  }
  std::cout << std::endl;

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(SimplexId i = 0; i < vertexNumber; i++) {

    dataType v = inputData[i];
    auto it = std::upper_bound(table.begin(), table.end(), v + delta);
    if(debugLevel_ > 3) {
      std::cout << "v " << v << " ,  v' " << *(it - 1) << std::endl;
    }
    outputData[i] = (dataType)(*(it - 1));
    // outputData
  }

  std::cout << "test " << (double)(delta * table.size() + rangeMin)
            << std::endl;
  this->printMsg("Complete", 1.0, timer.getElapsedTime(), this->threadNumber_);
  return 0;
}

#endif // _STAIRS_H
