/// \ingroup base
/// \class ttk::PersistenceDiagram
/// \author Guillaume Favelier <guillaume.favelier@lip6.fr>
/// \author Julien Tierny <julien.tierny@lip6.fr>
/// \date September 2016.
///
/// \brief TTK processing package for the computation of persistence diagrams.
///
/// This package computes the persistence diagram of the extremum-saddle pairs
/// of an input scalar field. The X-coordinate of each pair corresponds to its
/// birth, while its smallest and highest Y-coordinates correspond to its birth
/// and death respectively.
///
/// In practice, each extremity of a persistence pair is represented by its
/// vertexId and critical type. Based on that, the persistence of the pair
/// and its 2D embedding can easily be obtained.
///
/// Persistence diagrams are useful and stable concise representations of the
/// topological features of a data-set. It is useful to fine-tune persistence
/// thresholds for topological simplification or for fast similarity
/// estimations for instance.
///
/// \b Related \b publication \n
/// "Computational Topology: An Introduction" \n
/// Herbert Edelsbrunner and John Harer \n
/// American Mathematical Society, 2010
///
///  Two backends can be chosen for the computation:
///
///  1) FTM
/// \b Related \b publication \n
/// "Task-based Augmented Contour Trees with Fibonacci Heaps"
/// Charles Gueunet, Pierre Fortin, Julien Jomier, Julien Tierny
/// IEEE Transactions on Parallel and Distributed Systems, 2019
///
///  2) Progressive Approach
/// \b Related \b publication \n
/// "A Progressive Approach to Scalar Field Topology" \n
/// Jules Vidal, Pierre Guillou, Julien Tierny\n
/// IEEE Transactions on Visualization and Computer Graphics, 2021
///
/// \sa ttkPersistenceDiagram.cpp %for a usage example.

#pragma once

// base code includes
#include <DiscreteGradient.h>
#include <FTMTreePP.h>
#include <ProgressiveTopology.h>
#include <Triangulation.h>

namespace ttk {

  /**
   * @brief Persistence pair type (with persistence in double)
   */
  struct PersistencePair {
    /** first (lower) vertex id */
    ttk::SimplexId birth{};
    /** first vertex type */
    ttk::CriticalType birthType{};
    /** second (higher) vertex id */
    ttk::SimplexId death{};
    /** second vertex type */
    ttk::CriticalType deathType{};
    /** persistence value (scalars[second] - scalars[first]) */
    double persistence{};
    /** pair type (min-saddle: 0, saddle-saddle: 1, saddle-max: 2) */
    ttk::SimplexId pairType{};

    PersistencePair() = default;
    PersistencePair(const SimplexId b,
                    const CriticalType bType,
                    const SimplexId d,
                    const CriticalType dType,
                    const double pers,
                    const SimplexId pType)
      : birth{b}, birthType{bType}, death{d}, deathType{dType},
        persistence{pers}, pairType{pType} {
    }
  };

  /**
   * Compute the persistence diagram of a function on a triangulation.
   * TTK assumes that the input dataset is made of only one connected component.
   */
  class PersistenceDiagram : virtual public Debug {

  public:
    enum class BACKEND { FTM = 0, PROGRESSIVE_TOPOLOGY = 1, ADAPTIVE = 2 };

    PersistenceDiagram();

    inline void setComputeSaddleConnectors(bool state) {
      ComputeSaddleConnectors = state;
    }

    ttk::CriticalType getNodeType(ftm::FTMTree_MT *tree,
                                  ftm::TreeType treeType,
                                  const SimplexId vertexId) const;

    void sortPersistenceDiagram(std::vector<PersistencePair> &diagram,
                                const SimplexId *const offsets) const;

    template <typename scalarType>
    int computeCTPersistenceDiagram(
      ftm::FTMTreePP &tree,
      const std::vector<
        std::tuple<ttk::SimplexId, ttk::SimplexId, scalarType, bool>> &pairs,
      std::vector<PersistencePair> &diagram,
      const scalarType *scalars) const;

    /**
     * @pre For this function to behave correctly in the absence of
     * the VTK wrapper, ttk::preconditionOrderArray() needs to be
     * called to fill the @p inputOffsets buffer prior to any
     * computation (the VTK wrapper already includes a mecanism to
     * automatically generate such a preconditioned buffer).
     * @see examples/c++/main.cpp for an example use.
     */
    template <typename scalarType, class triangulationType>
    int execute(std::vector<PersistencePair> &CTDiagram,
                const scalarType *inputScalars,
                const SimplexId *inputOffsets,
                const triangulationType *triangulation);

    template <typename scalarType, class triangulationType>
    int executeFTM(std::vector<PersistencePair> &CTDiagram,
                   const scalarType *inputScalars,
                   const SimplexId *inputOffsets,
                   const triangulationType *triangulation);

    template <typename scalarType, class triangulationType>
    int executeProgressiveTopology(std::vector<PersistencePair> &CTDiagram,
                                   const scalarType *inputScalars,
                                   const SimplexId *inputOffsets,
                                   const triangulationType *triangulation);

    template <class triangulationType>
    void checkProgressivityRequirement(const triangulationType *triangulation);

    inline void
      preconditionTriangulation(AbstractTriangulation *triangulation) {
      if(triangulation) {
        triangulation->preconditionBoundaryVertices();
        contourTree_.setDebugLevel(debugLevel_);
        contourTree_.setThreadNumber(threadNumber_);
        contourTree_.preconditionTriangulation(triangulation);
        if(this->ComputeSaddleConnectors) {
          dcg_.setDebugLevel(debugLevel_);
          dcg_.setThreadNumber(threadNumber_);
          dcg_.preconditionTriangulation(triangulation);
        }
      }
    }
    double getEpsilon() const {
      return progT_.getEpsilon();
    }
    inline void setOutputMonotonyOffsets(void *data) {
      outputMonotonyOffsets_ = data;
    }
    inline void setOutputOffsets(void *data) {
      outputOffsets_ = data;
    }
    inline void setOutputScalars(void *data) {
      outputScalars_ = data;
    }
    inline void setDeltaAdaptive(double data) {
      progT_.setDelta(data);
    }

  protected:
    bool ComputeSaddleConnectors{false};
    ftm::FTMTreePP contourTree_{};
    dcg::DiscreteGradient dcg_{};

    // int BackEnd{0};
    BACKEND BackEnd{BACKEND::FTM};
    // progressivity
    ttk::ProgressiveTopology progT_{};
    void *outputScalars_{};
    void *outputOffsets_{};
    void *outputMonotonyOffsets_{};

    int StartingResolutionLevel{0};
    int StoppingResolutionLevel{-1};
    bool IsResumable{false};
    double TimeLimit{};

    // adaptive
    bool UseAdaptive{};
    double Epsilon{};
    int NbBuckets{};
  };
} // namespace ttk

template <typename scalarType>
int ttk::PersistenceDiagram::computeCTPersistenceDiagram(
  ftm::FTMTreePP &tree,
  const std::vector<
    std::tuple<ttk::SimplexId, ttk::SimplexId, scalarType, bool>> &pairs,
  std::vector<PersistencePair> &diagram,
  const scalarType *scalars) const {

  const ttk::SimplexId numberOfPairs = pairs.size();
  diagram.resize(numberOfPairs);
  for(ttk::SimplexId i = 0; i < numberOfPairs; ++i) {
    const ttk::SimplexId v0 = std::get<0>(pairs[i]);
    const ttk::SimplexId v1 = std::get<1>(pairs[i]);
    const auto persistenceValue = static_cast<double>(std::get<2>(pairs[i]));
    const bool type = std::get<3>(pairs[i]);

    if(type == true) {
      diagram[i] = PersistencePair{
        v0,
        getNodeType(tree.getJoinTree(), ftm::TreeType::Join, v0),
        v1,
        getNodeType(tree.getJoinTree(), ftm::TreeType::Join, v1),
        persistenceValue,
        0};
    } else {
      diagram[i] = PersistencePair{
        v1,
        getNodeType(tree.getSplitTree(), ftm::TreeType::Split, v1),
        v0,
        getNodeType(tree.getSplitTree(), ftm::TreeType::Split, v0),
        persistenceValue,
        2};
    }
  }

  return 0;
}

template <typename scalarType, class triangulationType>
int ttk::PersistenceDiagram::execute(std::vector<PersistencePair> &CTDiagram,
                                     const scalarType *inputScalars,
                                     const SimplexId *inputOffsets,
                                     const triangulationType *triangulation) {

  printMsg(ttk::debug::Separator::L1);

  checkProgressivityRequirement(triangulation);

  switch(BackEnd) {

    case BACKEND::PROGRESSIVE_TOPOLOGY:
      executeProgressiveTopology(
        CTDiagram, inputScalars, inputOffsets, triangulation);
      break;

    case BACKEND::FTM:
      executeFTM(CTDiagram, inputScalars, inputOffsets, triangulation);
      break;

    default:
      printErr("No method was selected");
  }

  // finally sort the diagram
  sortPersistenceDiagram(CTDiagram, inputOffsets);

  printMsg(ttk::debug::Separator::L1);

  return 0;
}

template <typename scalarType, class triangulationType>
int ttk::PersistenceDiagram::executeProgressiveTopology(
  std::vector<PersistencePair> &CTDiagram,
  const scalarType *inputScalars,
  const SimplexId *inputOffsets,
  const triangulationType *triangulation) {

  progT_.setDebugLevel(debugLevel_);
  progT_.setThreadNumber(threadNumber_);
  progT_.setupTriangulation((ttk::ImplicitTriangulation *)triangulation);
  progT_.setStartingResolutionLevel(StartingResolutionLevel);
  progT_.setStoppingResolutionLevel(StoppingResolutionLevel);
  // progT_.setStoppingDecimationLevel(StoppingDecimationLevel);
  // progT_.setStartingDecimationLevel(StartingDecimationLevel);
  progT_.setTimeLimit(TimeLimit);
  progT_.setIsResumable(IsResumable);
  progT_.setPreallocateMemory(true);

  std::vector<ProgressiveTopology::PersistencePair> resultDiagram{};
  if(UseAdaptive) {
    if(!outputScalars_ or !outputOffsets_) {
      printErr("Missing output field");
    }
    progT_.setNbBuckets(NbBuckets);
    progT_.setEpsilon(Epsilon);
    int *test;
    progT_.computeProgressivePDAdaptive(
      resultDiagram, inputScalars, (scalarType *)outputScalars_,
      (SimplexId *)outputOffsets_, (int *)outputMonotonyOffsets_, inputOffsets);
  } else {
    progT_.computeProgressivePD(resultDiagram, inputScalars, inputOffsets);
  }

  const scalarType* scalars = UseAdaptive ? (scalarType*)outputScalars_ : inputScalars;
  // create the final diagram
  for(const auto &p : resultDiagram) {
    if(p.pairType == 0) {
      CTDiagram.emplace_back(
        p.birth, CriticalType::Local_minimum, p.death, CriticalType::Saddle1,
        scalars[p.death] - scalars[p.birth], p.pairType);
    } else if(p.pairType == 2) {
      CTDiagram.emplace_back(
        p.birth, CriticalType::Saddle2, p.death, CriticalType::Local_maximum,
        scalars[p.death] - scalars[p.birth], p.pairType);
    } else if(p.pairType == -1) {
      CTDiagram.emplace_back(p.birth, CriticalType::Local_minimum, p.death,
                             CriticalType::Local_maximum,
                             scalars[p.death] - scalars[p.birth],
                             p.pairType);
    }
  }

  return 0;
}

template <typename scalarType, class triangulationType>
int ttk::PersistenceDiagram::executeFTM(
  std::vector<PersistencePair> &CTDiagram,
  const scalarType *inputScalars,
  const SimplexId *inputOffsets,
  const triangulationType *triangulation) {

  contourTree_.setVertexScalars(inputScalars);
  contourTree_.setTreeType(ftm::TreeType::Join_Split);
  contourTree_.setVertexSoSoffsets(inputOffsets);
  contourTree_.setSegmentation(false);
  contourTree_.build<scalarType>(triangulation);

  // get persistence pairs
  std::vector<std::tuple<ttk::SimplexId, ttk::SimplexId, scalarType>> JTPairs;
  std::vector<std::tuple<ttk::SimplexId, ttk::SimplexId, scalarType>> STPairs;
  contourTree_.computePersistencePairs<scalarType>(JTPairs, true);
  contourTree_.computePersistencePairs<scalarType>(STPairs, false);

  // merge pairs
  const auto JTSize = JTPairs.size();
  const auto STSize = STPairs.size();
  std::vector<std::tuple<ttk::SimplexId, ttk::SimplexId, scalarType, bool>>
    CTPairs(JTSize + STSize);
  for(size_t i = 0; i < JTSize; ++i) {
    const auto &x = JTPairs[i];
    CTPairs[i]
      = std::make_tuple(std::get<0>(x), std::get<1>(x), std::get<2>(x), true);
  }
  for(size_t i = 0; i < STSize; ++i) {
    const auto &x = STPairs[i];
    CTPairs[JTSize + i]
      = std::make_tuple(std::get<0>(x), std::get<1>(x), std::get<2>(x), false);
  }

  // remove the last pair which is present two times (global extrema pair)
  if(!CTPairs.empty()) {
    auto cmp =
      [](
        const std::tuple<ttk::SimplexId, ttk::SimplexId, scalarType, bool> &a,
        const std::tuple<ttk::SimplexId, ttk::SimplexId, scalarType, bool> &b) {
        return std::get<2>(a) < std::get<2>(b);
      };

    std::sort(CTPairs.begin(), CTPairs.end(), cmp);
    CTPairs.erase(CTPairs.end() - 1);
  }

  // get persistence diagrams
  computeCTPersistenceDiagram<scalarType>(
    contourTree_, CTPairs, CTDiagram, inputScalars);

  // get the saddle-saddle pairs
  std::vector<std::tuple<SimplexId, SimplexId, scalarType>>
    pl_saddleSaddlePairs;
  if(triangulation->getDimensionality() == 3 and ComputeSaddleConnectors) {
    dcg_.setInputScalarField(inputScalars);
    dcg_.setInputOffsets(inputOffsets);
    dcg_.computeSaddleSaddlePersistencePairs<scalarType>(
      pl_saddleSaddlePairs, *triangulation);

    // add saddle-saddle pairs to the diagram
    for(const auto &i : pl_saddleSaddlePairs) {
      const ttk::SimplexId v0 = std::get<0>(i);
      const ttk::SimplexId v1 = std::get<1>(i);
      const auto persistenceValue = static_cast<double>(std::get<2>(i));

      CTDiagram.emplace_back(v0, ttk::CriticalType::Saddle1, v1,
                             ttk::CriticalType::Saddle2, persistenceValue, 1);
    }
  }
  return 0;
}

template <class triangulationType>
void ttk::PersistenceDiagram::checkProgressivityRequirement(
  const triangulationType *triangulation) {
  if(BackEnd == BACKEND::PROGRESSIVE_TOPOLOGY or BackEnd == BACKEND::ADAPTIVE) {
    if(!std::is_same<triangulationType, ttk::ImplicitTriangulation>::value) {

      printWrn("Explicit triangulation detected.");
      printWrn("Defaulting to the FTM backend.");

      BackEnd = BACKEND::FTM;
    }
  }
}
