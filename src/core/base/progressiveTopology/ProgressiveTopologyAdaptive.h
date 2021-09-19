
template <typename scalarType>
int ttk::ProgressiveTopology::executeCPProgressiveAdaptive(
  const scalarType *scalars,
  scalarType *fakeScalars,
  SimplexId *outputOffsets,
  int *outputMonotonyOffsets,
  const SimplexId *offsetsOrder) {

  Timer timer;

  SimplexId *const vertsOrder = static_cast<SimplexId *>(outputOffsets);


  decimationLevel_ = startingDecimationLevel_;
  multiresTriangulation_.setTriangulation(triangulation_);
  const SimplexId vertexNumber = multiresTriangulation_.getVertexNumber();

  int *monotonyOffsets = static_cast<SimplexId *>(outputMonotonyOffsets);


#ifdef TTK_ENABLE_KAMIKAZE
  if(vertexNumber == 0) {
    this->printErr("No points in triangulation");
    return 1;
  }
#endif // TTK_ENABLE_KAMIKAZE

  double tm_allocation = timer.getElapsedTime();

  const auto dim = multiresTriangulation_.getDimensionality();
  const size_t maxNeigh = dim == 3 ? 14 : (dim == 2 ? 6 : 0);

  std::vector<std::vector<SimplexId>> saddleCCMin(vertexNumber),
    saddleCCMax(vertexNumber);
  std::vector<std::vector<SimplexId>> vertexRepresentativesMin(vertexNumber),
    vertexRepresentativesMax(vertexNumber);

  std::vector<std::vector<std::pair<polarity, polarity>>> vertexLinkPolarity(
    vertexNumber);

  std::vector<polarity> isNew(vertexNumber, 255);
  std::vector<polarity> toPropageMin(vertexNumber, 0),
    toPropageMax(vertexNumber, 0);
  std::vector<polarity> isUpToDateMin(vertexNumber, 0),
    isUpToDateMax(vertexNumber, 0);

  // index in vertexLinkByBoundaryType
  std::vector<uint8_t> vertexLink(vertexNumber);
  VLBoundaryType vertexLinkByBoundaryType{};
  std::vector<DynamicTree> link(vertexNumber);
  std::vector<polarity> toProcess(vertexNumber, 0), toReprocess{};

  std::vector<SimplexId> offsetsVec(vertexNumber);
  std::iota(offsetsVec.begin(), offsetsVec.end(), 0);
  const SimplexId *const offsets = offsetsVec.data();

  // std::vector<std::tuple<SimplexId, double, int>> globalErrorList{};

  if(this->startingDecimationLevel_ > this->stoppingDecimationLevel_
     || this->isResumable_) {
    // only needed for progressive computation
    toReprocess.resize(vertexNumber, 0);
  }

  // lock vertex thread access for firstPropage
  std::vector<Lock> vertLockMin(vertexNumber), vertLockMax(vertexNumber);

  // pre-allocate memory
  if(preallocateMemory_) {
    double tm_prealloc = timer.getElapsedTime();
    printMsg("Pre-allocating data structures", 0, 0, threadNumber_,
             ttk::debug::LineMode::REPLACE);
    for(SimplexId i = 0; i < vertexNumber; ++i) {
      // for(int d = 0; d < vertexLinkPolarity.size(); d++) {
      //   vertexLinkPolarity[d][i].reserve(maxNeigh);
      // }
      vertexLinkPolarity[i].reserve(maxNeigh);
      link[i].alloc(maxNeigh);
    }
    printMsg("Pre-allocating data structures", 1,
             timer.getElapsedTime() - tm_prealloc, threadNumber_);
  }

  tm_allocation = timer.getElapsedTime() - tm_allocation;
  printMsg("Total memory allocation", 1, tm_allocation, threadNumber_);

  // computation of implicit link
  std::vector<SimplexId> boundReps{};
  multiresTriangulation_.findBoundaryRepresentatives(boundReps);

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(size_t i = 0; i < boundReps.size(); i++) {
    if(boundReps[i] != -1) {
      buildVertexLinkByBoundary(boundReps[i], vertexLinkByBoundaryType);
    }
  }

  if(debugLevel_ > 4) {
    std::cout << "boundary representatives : ";
    for(auto bb : boundReps) {
      std::cout << ", " << bb;
    }
    std::cout << std::endl;
  }

  scalarType *deltaField;
  scalarType *deltaFieldDynamic;

  multiresTriangulation_.setDecimationLevel(decimationLevel_);


  initGlobalPolarityAdaptive(isNew, vertexLinkPolarity, toProcess, fakeScalars,
                             offsets, monotonyOffsets);


  double delta = epsilon_ * delta_;
  totalVertices_ += multiresTriangulation_.getDecimatedVertexNumber();

  while(decimationLevel_ > stoppingDecimationLevel_) {
    Timer tmIter{};
    decimationLevel_--;
    multiresTriangulation_.setDecimationLevel(decimationLevel_);
    double progress = (double)(startingDecimationLevel_ - decimationLevel_)
                      / (startingDecimationLevel_);
    this->printMsg("decimation level: " + std::to_string(decimationLevel_),
                   progress, timer.getElapsedTime() - tm_allocation,
                   threadNumber_);

    int ret = updateGlobalPolarityAdaptive(
      delta, isNew, vertexLinkPolarity, toProcess,
      toReprocess, /*globalErrorList,*/
      scalars, fakeScalars, offsets, monotonyOffsets);
    if(ret == -1) {
      std::cout << "Found ERROR - aborting" << std::endl;
      return -1;
    }
  } // end while

  computeCriticalPointsAdaptive(
    vertexLinkPolarity, toPropageMin, toPropageMax, toProcess, toReprocess,
    link, vertexLink, vertexLinkByBoundaryType, saddleCCMin, saddleCCMax,
    scalars, fakeScalars, offsets, monotonyOffsets);

  updatePropagationCPAdaptive(
    toPropageMin, toPropageMax, vertexRepresentativesMin,
    vertexRepresentativesMax, saddleCCMin, saddleCCMax, vertLockMin,
    vertLockMax, isUpToDateMin, isUpToDateMax, fakeScalars, offsets,
    monotonyOffsets);

  computePersistencePairsFromSaddlesAdaptive(
    CTDiagram_, scalars, fakeScalars, offsets, monotonyOffsets,
    vertexRepresentativesMin, vertexRepresentativesMax, toPropageMin,
    toPropageMax);

  CTDiagram_.emplace_back(this->globalMin_, this->globalMax_, -1);


  std::cout << "# epsilon " << epsilon_ << " in "
            << timer.getElapsedTime() - tm_allocation << " s." << std::endl;


  this->printMsg("Complete", 1.0, timer.getElapsedTime() - tm_allocation,
                 this->threadNumber_);

  // finally sort the diagram
  std::cout << "SORTING" << std::endl;
  sortPersistenceDiagramAdaptive(
    CTDiagram_, scalars, fakeScalars, offsets, monotonyOffsets);
  std::cout << "Final epsilon " << epsilon_ << std::endl;

  if(0) {
    std::vector<SimplexId> sortedVertices{};
    sortVertices(vertexNumber, sortedVertices, vertsOrder, fakeScalars, offsets,
                 monotonyOffsets);
  } else {
    int min_mono_offset = -pow(2, stoppingDecimationLevel_);
    int max_mono_offset = pow(2, stoppingDecimationLevel_);
    int delta_mono_offsets = max_mono_offset - min_mono_offset;

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
    for(SimplexId i = 0; i < vertexNumber; i++) {
      vertsOrder[i]
        = (monotonyOffsets[i] - min_mono_offset) * vertexNumber + offsets[i];
    }
  }

  return 0;
}

template <typename ScalarType, typename OffsetType>
void ttk::ProgressiveTopology::computePersistencePairsFromSaddlesAdaptive(
  std::vector<PersistencePair> &CTDiagram,
  const ScalarType *const scalars,
  const ScalarType *const fakeScalars,
  const OffsetType *const offsets,
  const int *const monotonyOffsets,
  std::vector<std::vector<SimplexId>> &vertexRepresentativesMin,
  std::vector<std::vector<SimplexId>> &vertexRepresentativesMax,
  const std::vector<polarity> &toPropageMin,
  const std::vector<polarity> &toPropageMax) const {

  Timer timer{};
  // CTDiagram.clear();
  std::vector<triplet> tripletsMax{}, tripletsMin{};
  const SimplexId nbDecVert = multiresTriangulation_.getDecimatedVertexNumber();

  for(SimplexId localId = 0; localId < nbDecVert; localId++) {
    SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(localId);
    if(toPropageMin[globalId]) {
      getTripletsFromSaddles(globalId, tripletsMin, vertexRepresentativesMin);
    }
    if(toPropageMax[globalId]) {
      getTripletsFromSaddles(globalId, tripletsMax, vertexRepresentativesMax);
    }
  }

  // std::cout << "TRIPLETS " << timer.getElapsedTime() << std::endl;
  double tm_pairs = timer.getElapsedTime();

  sortTripletsAdaptive(
    tripletsMax, scalars, fakeScalars, offsets, monotonyOffsets, true);
  sortTripletsAdaptive(
    tripletsMin, scalars, fakeScalars, offsets, monotonyOffsets, false);

  const auto tm_sort = timer.getElapsedTime();
  // std::cout << "TRIPLETS SORT " << tm_sort - tm_pairs << std::endl;

  typename std::remove_reference<decltype(CTDiagram)>::type CTDiagramMin{},
    CTDiagramMax{};

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel sections num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
  {
#ifdef TTK_ENABLE_OPENMP
#pragma omp section
#endif // TTK_ENABLE_OPENMP
    tripletsToPersistencePairsAdaptive(CTDiagramMin, vertexRepresentativesMax,
                                       tripletsMax, scalars, fakeScalars,
                                       offsets, monotonyOffsets, true);
#ifdef TTK_ENABLE_OPENMP
#pragma omp section
#endif // TTK_ENABLE_OPENMP
    tripletsToPersistencePairsAdaptive(CTDiagramMax, vertexRepresentativesMin,
                                       tripletsMin, scalars, fakeScalars,
                                       offsets, monotonyOffsets, false);
  }
  CTDiagram = std::move(CTDiagramMin);
  CTDiagram.insert(CTDiagram.end(), CTDiagramMax.begin(), CTDiagramMax.end());

  if(debugLevel_ > 3) {
    std::cout << "PAIRS " << timer.getElapsedTime() - tm_sort << std::endl;
  }
}

template <typename ScalarType, typename OffsetType>
void ttk::ProgressiveTopology::sortTripletsAdaptive(
  std::vector<triplet> &triplets,
  const ScalarType *const scalars,
  const ScalarType *const fakeScalars,
  const OffsetType *const offsets,
  const int *const monotonyOffsets,
  const bool splitTree) const {
  if(triplets.empty())
    return;

  // const auto lt = [=](const SimplexId a, const SimplexId b) -> bool {
  //   return (scalars[a] < scalars[b])
  //          || (scalars[a] == scalars[b] && offsets[a] < offsets[b]);
  // };
  const auto lt = [=](const SimplexId a, const SimplexId b) -> bool {
    return ((fakeScalars[a] < fakeScalars[b])
            || (fakeScalars[a] == fakeScalars[b]
                && ((monotonyOffsets[a] < monotonyOffsets[b])
                    || (monotonyOffsets[a] == monotonyOffsets[b]
                        && offsets[a] < offsets[b]))));
  };

  // Sorting step
  const auto cmp = [=](const triplet &t1, const triplet &t2) {
    const SimplexId s1 = std::get<0>(t1);
    const SimplexId s2 = std::get<0>(t2);
    const SimplexId m1 = std::get<2>(t1);
    const SimplexId m2 = std::get<2>(t2);
    if(s1 != s2)
      return lt(s1, s2) != splitTree;
    else // s1 == s2
      return lt(m1, m2) == splitTree;
  };

  PARALLEL_SORT(triplets.begin(), triplets.end(), cmp);
}

template <typename scalarType, typename offsetType>
void ttk::ProgressiveTopology::tripletsToPersistencePairsAdaptive(
  std::vector<PersistencePair> &pairs,
  std::vector<std::vector<SimplexId>> &vertexRepresentatives,
  std::vector<triplet> &triplets,
  const scalarType *const scalars,
  const scalarType *const fakeScalars,
  const offsetType *const offsets,
  const int *const monotonyOffsets,

  const bool splitTree) const {
  Timer tm;
  if(triplets.empty())
    return;
  size_t numberOfPairs = 0;

  // // accelerate getRep lookup?
  // std::vector<SimplexId> firstRep(vertexRepresentatives.size());

  // #ifdef TTK_ENABLE_OPENMP
  // #pragma omp parallel for num_threads(threadNumber_)
  // #endif // TTK_ENABLE_OPENMP
  //   for(size_t i = 0; i < firstRep.size(); ++i) {
  //     if(!vertexRepresentatives[i].empty()) {
  //       firstRep[i] = vertexRepresentatives[i][0];
  //     }
  //   }

  const auto lt = [=](const SimplexId a, const SimplexId b) -> bool {
    return ((fakeScalars[a] < fakeScalars[b])
            || (fakeScalars[a] == fakeScalars[b]
                && ((monotonyOffsets[a] < monotonyOffsets[b])
                    || (monotonyOffsets[a] == monotonyOffsets[b]
                        && offsets[a] < offsets[b]))));
  };

  const auto getRep = [&](SimplexId v) -> SimplexId {
    auto r = vertexRepresentatives[v][0];
    while(r != v) {
      v = r;
      r = vertexRepresentatives[v][0];
    }
    return r;
  };

  for(const auto &t : triplets) {
    SimplexId r1 = getRep(std::get<1>(t));
    SimplexId r2 = getRep(std::get<2>(t));
    if(r1 != r2) {
      SimplexId s = std::get<0>(t);
      numberOfPairs++;

      // Add pair
      if(splitTree) {
        // r1 = min(r1, r2), r2 = max(r1, r2)
        if(lt(r2, r1)) {
          std::swap(r1, r2);
        }
        // pair saddle-max: s -> min(r1, r2);
        pairs.emplace_back(s, r1, 2);
        // fakeScalars[r1] - fakeScalars[s], 2);

      } else {
        // r1 = max(r1, r2), r2 = min(r1, r2)
        if(lt(r1, r2)) {
          std::swap(r1, r2);
        }
        // pair min-saddle: max(r1, r2) -> s;
        pairs.emplace_back(r1, s, 0);
        // fakeScalars[s] - fakeScalars[r1], 0);
      }

      vertexRepresentatives[std::get<1>(t)][0] = r2;
      vertexRepresentatives[r1][0] = r2;
      // vertexRepresentatives[std::get<1>(t)][0] = r2;
      // vertexRepresentatives[r1][0] = r2;
    }
  }

  if(debugLevel_ > 3) {
    std::string prefix = splitTree ? "[sad-max]" : "[min-sad]";
    std::cout << prefix << "  found all pairs in " << tm.getElapsedTime()
              << " s." << std::endl;
  }
}

template <typename scalarType, typename offsetType>
void ttk::ProgressiveTopology::sortVertices(
  const SimplexId vertexNumber,
  std::vector<SimplexId> &sortedVertices,
  SimplexId *vertsOrder,
  const scalarType *const fakeScalars,
  const offsetType *const offsets,
  const int *const monotonyOffsets) {

  Timer tm;

  sortedVertices.resize(vertexNumber);

  // fill with numbers from 0 to vertexNumber - 1
  std::iota(sortedVertices.begin(), sortedVertices.end(), 0);

  // sort vertices in ascending order following scalarfield / offsets
  PARALLEL_SORT(sortedVertices.begin(), sortedVertices.end(),
                [&](const SimplexId a, const SimplexId b) {
                  return ((fakeScalars[a] < fakeScalars[b])
                          || (fakeScalars[a] == fakeScalars[b]
                              && ((monotonyOffsets[a] < monotonyOffsets[b])
                                  || (monotonyOffsets[a] == monotonyOffsets[b]
                                      && offsets[a] < offsets[b]))));
                });

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(size_t i = 0; i < sortedVertices.size(); ++i) {
    vertsOrder[sortedVertices[i]] = i;
  }

  if(debugLevel_ > 2) {
    std::cout << "SORT " << tm.getElapsedTime() << std::endl;
  }
}

template <typename scalarType, typename offsetType>
bool ttk::ProgressiveTopology::getDeltaFromOldPoint(
  const SimplexId vertexId,
  const std::vector<polarity> &isNew,
  std::vector<polarity> &toProcess,
  std::vector<polarity> &toReprocess,
  std::vector<std::tuple<SimplexId, scalarType, int>> &nonMonotonicList,
  std::vector<std::pair<polarity, polarity>> &vlp,
  const scalarType *const scalarField,
  scalarType *fakeScalars,
  scalarType *deltaField,
  scalarType *deltaFieldDynamic,
  const offsetType *const offsets,
  int *monotonyOffsets) const {

  bool hasMonotonyChanged = false;
  const SimplexId neighborNumber
    = multiresTriangulation_.getVertexNeighborNumber(vertexId);
  for(SimplexId i = 0; i < neighborNumber; i++) {
    SimplexId neighborId = -1;
    multiresTriangulation_.getVertexNeighbor(vertexId, i, neighborId);

    // check for monotony changes
    // const bool lower = vertsOrder_[neighborId] < vertsOrder_[vertexId];
    const bool lowerStatic
      = (scalarField[neighborId] < fakeScalars[vertexId])
        || ((scalarField[neighborId] == fakeScalars[vertexId])
            && (offsets[neighborId] < offsets[vertexId]));
    const bool lowerDynamic
      = ((fakeScalars[neighborId] < fakeScalars[vertexId])
         || (fakeScalars[neighborId] == fakeScalars[vertexId]
             && ((monotonyOffsets[neighborId] < monotonyOffsets[vertexId])
                 || (monotonyOffsets[neighborId] == monotonyOffsets[vertexId]
                     && offsets[neighborId] < offsets[vertexId]))));
    // const bool lowerDynamic
    //   = (fakeScalars[neighborId] < fakeScalars[vertexId])
    //     || ((fakeScalars[neighborId] == fakeScalars[vertexId])
    //         && (offsets[neighborId] < offsets[vertexId]));
    const polarity isUpperDynamic = lowerDynamic ? 0 : 255;
    const polarity isUpperStatic = lowerStatic ? 0 : 255;
    const polarity isUpperOld = vlp[i].first;

    scalarType deltaStatic = 0;
    if(isUpperStatic != isUpperOld) { // change of monotony
      scalarType replacementValueStatic = (scalarType)scalarField[vertexId];
      deltaStatic = abs((scalarType)scalarField[neighborId]
                        - replacementValueStatic); // depends on epsilon
      if(isNew[neighborId]) {
        deltaField[neighborId] = deltaStatic;
        int dl = multiresTriangulation_.getDecimationLevel();
        nonMonotonicList.push_back({neighborId, deltaStatic, dl});
        // std::cout << "new delta for " << neighborId << " : " << deltaStatic
        //           << std::endl;
        // std::cout << "    values: " << scalarField[neighborId] << " , "
        //           << scalarField[vertexId] << std::endl;
      }
    }

    if(isUpperStatic != isUpperOld) { // change of monotony

      scalarType replacementValueDynamic
        = (scalarType)fakeScalars[vertexId]; // depends on epsilon
      scalarType deltaDynamic
        = abs((scalarType)fakeScalars[neighborId] - replacementValueDynamic);
      // if(isNew[neighborId]) {
      //   deltaFieldDynamic[neighborId] = deltaDynamic;
      //   // std::cout << decimationLevel_ << " - " << delta << std::endl;
      // }

      //=====================
      // SimplexId oldNeighNumber = 0;
      // SimplexId nnumber
      //   = multiresTriangulation_.getVertexNeighborNumber(neighborId);
      // for(int iii = 0; iii < nnumber; iii++) {
      //   SimplexId neighborId2 = -1;
      //   multiresTriangulation_.getVertexNeighbor(neighborId, iii,
      //   neighborId2); if(!isNew[neighborId2]) {
      //     oldNeighNumber++;
      //   }
      // }
      //=====================
      scalarType delta = useStaticDelta_ ? deltaStatic : deltaDynamic;
      // scalarType replacementValue = useStaticReplacement_ ?
      // replacementValueStatic
      //                                                 :
      //                                                 replacementValueDynamic;
      if(delta > -2) {
        hasMonotonyChanged = true;

        toReprocess[vertexId] = 255;
        if(isNew[neighborId]) {
          // toProcess[neighborId] = 255;
        } else {
          toReprocess[neighborId] = 255;
        }
        // const SimplexId neighborNumberNew
        //   = multiresTriangulation_.getVertexNeighborNumber(neighborId);
        // for(SimplexId j = 0; j < neighborNumberNew; j++) {
        //   SimplexId neighborIdNew = -1;
        //   multiresTriangulation_.getVertexNeighbor(
        //     neighborId, j, neighborIdNew);
        //   if(isNew[neighborIdNew])
        //     toProcess[neighborIdNew] = 255;
        // }

        vlp[i].second = 255;
      } else {
        std::cout << "Non-monotonic vertex ignored in preprocessing step: THIS "
                     "SHOULD NOT HAPPEN"
                  << std::endl;
        fakeScalars[neighborId] = replacementValueDynamic;
        // check if new value is actually interpolant
        // scalarType v0 = fakeScalars[vertexId];
        // scalarType v1 = fakeScalars[oldNeighbor];
        // int oldOffset = monotonyOffsets[neighborId];
        // // change monotony offset
        // // The monotony must be preserved, meaning that
        // // if we had f(vertexId)>f(oldNeighbor)
        // // we should have f(vertexId)>f(neighborId)
        // // which is enforced when
        // // monotonyOffsets[vertexId]>monotonyOffsets[neighborId]
        // if(isUpperOld) { // we should enforce f(vertexId)<f(neighborId)
        //   if(offsets[vertexId] > offsets[neighborId]) {
        //     monotonyOffsets[neighborId]
        //       = monotonyOffsets[vertexId] + pow(2, decimationLevel_);
        //   } else {
        //     monotonyOffsets[neighborId] = monotonyOffsets[vertexId];
        //   }
        // } else { // we should enforce f(vertexId)>f(neighborId)
        //   if(offsets[vertexId] < offsets[neighborId]) {
        //     monotonyOffsets[neighborId]
        //       = monotonyOffsets[vertexId] - pow(2, decimationLevel_);
        //   } else {
        //     monotonyOffsets[neighborId] = monotonyOffsets[vertexId];
        //   }
        // }
        // bool v0higher
        //   = ((fakeScalars[neighborId] < fakeScalars[vertexId])
        //      || (fakeScalars[neighborId] == fakeScalars[vertexId]
        //          && ((monotonyOffsets[neighborId] <
        //          monotonyOffsets[vertexId])
        //              || (monotonyOffsets[neighborId]
        //                    == monotonyOffsets[vertexId]
        //                  && offsets[neighborId] < offsets[vertexId]))));
        // bool v1higher
        //   = ((fakeScalars[neighborId] < fakeScalars[oldNeighbor])
        //      || (fakeScalars[neighborId] == fakeScalars[oldNeighbor]
        //          && ((monotonyOffsets[neighborId]
        //               < monotonyOffsets[oldNeighbor])
        //              || (monotonyOffsets[neighborId]
        //                    == monotonyOffsets[oldNeighbor]
        //                  && offsets[neighborId] < offsets[oldNeighbor]))));
        // bool interpolant = (v0higher and !v1higher) or (!v0higher and
        // v1higher);

        // if(!interpolant) {
        //   std::cout << "ERREUR - not interpolant" << std::endl;
        //   std::cout << "  " << vertexId << " " << neighborId << " "
        //             << oldNeighbor << std::endl;
        //   std::cout << " fake " << v0 << " " << replacementValueDynamic <<
        //   "
        //   "
        //             << v1 << std::endl;
        //   std::cout << " real " << scalarField[vertexId] << " "
        //             << scalarField[neighborId] << " "
        //             << scalarField[oldNeighbor] << std::endl;
        //   std::cout << "      " << offsets[vertexId] << " "
        //             << offsets[neighborId] << " " << offsets[oldNeighbor]
        //             << std::endl;
        //   std::cout << "      " << monotonyOffsets[vertexId] << " "
        //             << monotonyOffsets[neighborId] << " "
        //             << monotonyOffsets[oldNeighbor] << std::endl;
        //   std::cout << "  old offset " << oldOffset << std::endl;
        //   // std::cout << "  " << scalarField[vertexId] << " "
        //   //           << replacementValueStatic << " " <<
        //   //           scalarField[oldNeighbor]
        //   //           << std::endl;
        //   // std::cout << "  " << (v0 <= replacementValueDynamic) << " "
        //   //           << (replacementValueDynamic <= v1) << std::endl;
        // }
      }
    }
  }
  return hasMonotonyChanged;
}

template <typename scalarType, typename offsetType>
int ttk::ProgressiveTopology::getMonotonyChangeByOldPointCPAdaptive(
  const SimplexId vertexId,
  double eps,
  const std::vector<polarity> &isNew,
  std::vector<polarity> &toProcess,
  std::vector<polarity> &toReprocess,
  std::vector<std::pair<polarity, polarity>> &vlp,
  const scalarType *const scalarField,
  scalarType *fakeScalars,
  scalarType *deltaField,
  scalarType *deltaFieldDynamic,
  // std::vector<std::pair<SimplexId, double>> &localErrorList,
  const offsetType *const offsets,
  int *monotonyOffsets) const {

  int hasMonotonyChanged = 0;
  const SimplexId neighborNumber
    = multiresTriangulation_.getVertexNeighborNumber(vertexId);
  for(SimplexId i = 0; i < neighborNumber; i++) {
    SimplexId neighborId = -1;
    multiresTriangulation_.getVertexNeighbor(vertexId, i, neighborId);

    const bool lowerDynamic
      = ((fakeScalars[neighborId] < fakeScalars[vertexId])
         || (fakeScalars[neighborId] == fakeScalars[vertexId]
             && ((monotonyOffsets[neighborId] < monotonyOffsets[vertexId])
                 || (monotonyOffsets[neighborId] == monotonyOffsets[vertexId]
                     && offsets[neighborId] < offsets[vertexId]))));

    const polarity isUpperDynamic = lowerDynamic ? 0 : 255;
    
    const polarity isUpperOld = vlp[i].first;

    

    if(isUpperDynamic != isUpperOld) { // change of monotony
      SimplexId oldNeighbor = -1;
      int oldDecimation = pow(2, decimationLevel_ + 1);
      multiresTriangulation_.getVertexNeighborAtDecimation(
        vertexId, i, oldNeighbor, oldDecimation);

      double replacementValueDynamic
        = (0.5 * (double)fakeScalars[oldNeighbor]
           + .5 * (double)fakeScalars[vertexId]); // depends on epsilon
      double deltaDynamic
        = abs((double)fakeScalars[neighborId] - replacementValueDynamic);

      SimplexId oldNeighNumber = 0;
      SimplexId nnumber
        = multiresTriangulation_.getVertexNeighborNumber(neighborId);
      for(int iii = 0; iii < nnumber; iii++) {
        SimplexId neighborId2 = -1;
        multiresTriangulation_.getVertexNeighbor(neighborId, iii, neighborId2);
        if(!isNew[neighborId2]) {
          oldNeighNumber++;
        }
      }

      if(deltaDynamic > eps or !isNew[neighborId] or oldNeighNumber > 2) {
        hasMonotonyChanged = 1;

        toReprocess[vertexId] = 255;
        if(isNew[neighborId]) {
          toProcess[neighborId] = 255;
        } else {
          toReprocess[neighborId] = 255;
        }
        const SimplexId neighborNumberNew
          = multiresTriangulation_.getVertexNeighborNumber(neighborId);
        for(SimplexId j = 0; j < neighborNumberNew; j++) {
          SimplexId neighborIdNew = -1;
          multiresTriangulation_.getVertexNeighbor(
            neighborId, j, neighborIdNew);
          if(isNew[neighborIdNew])
            toProcess[neighborIdNew] = 255;
        }
        vlp[i].second = 255;
      } else {
        fakeScalars[neighborId] = replacementValueDynamic;

        // corrects rounding error when we process an integer scalar field
        if(fakeScalars[neighborId] == fakeScalars[oldNeighbor]) {
          fakeScalars[neighborId] = fakeScalars[vertexId];
        }

        if(isUpperOld) { // we should enforce f(vertexId)<f(neighborId)
          if(offsets[vertexId] > offsets[neighborId]) {
            monotonyOffsets[neighborId]
              = monotonyOffsets[vertexId] + pow(2, decimationLevel_);
            if(monotonyOffsets[vertexId] == monotonyOffsets[oldNeighbor]
               and fakeScalars[vertexId] == fakeScalars[oldNeighbor]) {
              std::cout << "THIS IS AN ISSUE" << std::endl;
            }
          } else {
            monotonyOffsets[neighborId] = monotonyOffsets[vertexId];
          }
        } else { // we should enforce f(vertexId)>f(neighborId)
          if(offsets[vertexId] < offsets[neighborId]) {
            monotonyOffsets[neighborId]
              = monotonyOffsets[vertexId] - pow(2, decimationLevel_);
            if(monotonyOffsets[vertexId] == monotonyOffsets[oldNeighbor]
               and fakeScalars[vertexId] == fakeScalars[oldNeighbor]) {
              std::cout << "THIS IS AN ISSUE" << std::endl;
            }
          } else {
            monotonyOffsets[neighborId] = monotonyOffsets[vertexId];
          }
        }
      }
    }
  }
  return hasMonotonyChanged;
}

template <typename scalarType, typename offsetType>
ttk::SimplexId ttk::ProgressiveTopology::propageFromSaddlesAdaptive(
  const SimplexId vertexId,
  std::vector<Lock> &vertLock,
  std::vector<polarity> &toPropage,
  std::vector<std::vector<SimplexId>> &vertexRepresentatives,
  std::vector<std::vector<SimplexId>> &saddleCC,
  std::vector<polarity> &isUpdated,
  std::vector<SimplexId> &globalExtremum,
  const bool splitTree,
  const scalarType *fakeScalars,
  const offsetType *const offsets,
  const int *const monotonyOffsets) const {

  auto &toProp = toPropage[vertexId];
  auto &reps = vertexRepresentatives[vertexId];
  auto &updated = isUpdated[vertexId];

  if(updated) {
    return reps[0];
  }

  const auto gt = [=](const SimplexId v1, const SimplexId v2) {
    return ((fakeScalars[v1] > fakeScalars[v2])
            || (fakeScalars[v1] == fakeScalars[v2]
                && ((monotonyOffsets[v1] > monotonyOffsets[v2])
                    || (monotonyOffsets[v1] == monotonyOffsets[v2]
                        && offsets[v1] > offsets[v2]))))
           == splitTree;
  };

  if(this->threadNumber_ > 1) {
    vertLock[vertexId].lock();
  }
  if(saddleCC[vertexId].size()
     and !toProp) { // tis a saddle point, should have to propage on it
    printErr("ERRRROR");
  }
  if(toProp) { // SADDLE POINT
    if(debugLevel_ > 5)
      printMsg("to saddle " + std::to_string(vertexId) + " "
               + std::to_string(saddleCC[vertexId].size()));
    // for(auto ccid : saddleCC[vertexId]) {
    //   SimplexId ccV = -1;
    //   multiresTriangulation_.getVertexNeighbor(vertexId, ccid, ccV);
    //   std::cout << " " << ccid << " (" << ccV << ") , ";
    // }
    // std::cout << std::endl;
    const auto &CC = saddleCC[vertexId];
    reps.clear();
    reps.reserve(CC.size());
    for(size_t r = 0; r < CC.size(); r++) {
      SimplexId neighborId = -1;
      SimplexId localId = CC[r];
      multiresTriangulation_.getVertexNeighbor(vertexId, localId, neighborId);
      // printMsg("       CC " + std::to_string(CC[r]) + " "
      //          + std::to_string(neighborId) + " ("
      //          + std::to_string(fakeScalars[neighborId]) + " , "
      //          + std::to_string(offsets[neighborId]) + ")");
      SimplexId ret = propageFromSaddlesAdaptive(
        neighborId, vertLock, toPropage, vertexRepresentatives, saddleCC,
        isUpdated, globalExtremum, splitTree, fakeScalars, offsets,
        monotonyOffsets);
      reps.emplace_back(ret);
    }

    if(reps.size() > 1) {
      // sort & remove duplicate elements
      std::sort(reps.begin(), reps.end(), gt);
      const auto last = std::unique(reps.begin(), reps.end());
      reps.erase(last, reps.end());
    }

    updated = 255;
    if(this->threadNumber_ > 1) {
      vertLock[vertexId].unlock();
    }

    return reps[0];

  } else {
    if(debugLevel_ > 5)
      printMsg("to non saddle " + std::to_string(vertexId) + " "
               + std::to_string(saddleCC[vertexId].size()));

    SimplexId ret = vertexId;
    SimplexId neighborNumber
      = multiresTriangulation_.getVertexNeighborNumber(vertexId);
    SimplexId maxNeighbor = vertexId;
    // std::cout << "neigh number" << neighborNumber << std::endl;
    for(SimplexId i = 0; i < neighborNumber; i++) {
      SimplexId neighborId = -1;
      multiresTriangulation_.getVertexNeighbor(vertexId, i, neighborId);
      if(gt(neighborId, maxNeighbor)) {
        maxNeighbor = neighborId;
      }
    }
    if(maxNeighbor != vertexId) { // not an extremum
      ret = propageFromSaddlesAdaptive(maxNeighbor, vertLock, toPropage,
                                       vertexRepresentatives, saddleCC,
                                       isUpdated, globalExtremum, splitTree,
                                       fakeScalars, offsets, monotonyOffsets);

    } else { // needed to find the globalExtremum per thread
#ifdef TTK_ENABLE_OPENMP
      const auto tid = omp_get_thread_num();
#else
      const auto tid = 0;
#endif // TTK_ENABLE_OPENMP
      if(gt(vertexId, globalExtremum[tid])) {
        globalExtremum[tid] = vertexId;
      }
    }
    reps.resize(1);
    reps[0] = ret;
    updated = 255;
    if(this->threadNumber_ > 1) {
      vertLock[vertexId].unlock();
    }
    return ret;
  }
}

template <typename scalarType, typename offsetType>
void ttk::ProgressiveTopology::initPropagationAdaptive(
  std::vector<polarity> &toPropageMin,
  std::vector<polarity> &toPropageMax,
  std::vector<std::vector<SimplexId>> &vertexRepresentativesMin,
  std::vector<std::vector<SimplexId>> &vertexRepresentativesMax,
  std::vector<std::vector<SimplexId>> &saddleCCMin,
  std::vector<std::vector<SimplexId>> &saddleCCMax,
  std::vector<Lock> &vertLockMin,
  std::vector<Lock> &vertLockMax,
  std::vector<polarity> &isUpdatedMin,
  std::vector<polarity> &isUpdatedMax,
  const scalarType *fakeScalars,
  const offsetType *const offsets,
  const int *monotonyOffsets) {

  Timer timer{};
  const size_t nDecVerts = multiresTriangulation_.getDecimatedVertexNumber();

  std::vector<SimplexId> globalMaxThr(threadNumber_, 0);
  std::vector<SimplexId> globalMinThr(threadNumber_, 0);
  if(debugLevel_ > 3) {
    int minn = 0, maxn = 0;
    for(size_t i = 0; i < nDecVerts; i++) {
      SimplexId v = multiresTriangulation_.localToGlobalVertexId(i);
      if(toPropageMin[v])
        minn++;
      if(toPropageMax[v])
        maxn++;
    }
    printMsg("Propaging on " + std::to_string(minn) + " and "
             + std::to_string(maxn));
  }

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(size_t i = 0; i < nDecVerts; i++) {
    SimplexId v = multiresTriangulation_.localToGlobalVertexId(i);
    if(toPropageMin[v]) {
      propageFromSaddlesAdaptive(v, vertLockMin, toPropageMin,
                                 vertexRepresentativesMin, saddleCCMin,
                                 isUpdatedMin, globalMinThr, false, fakeScalars,
                                 offsets, monotonyOffsets);
    }
    if(toPropageMax[v]) {
      propageFromSaddlesAdaptive(v, vertLockMax, toPropageMax,
                                 vertexRepresentativesMax, saddleCCMax,
                                 isUpdatedMax, globalMaxThr, true, fakeScalars,
                                 offsets, monotonyOffsets);
    }
  }

  const auto lt = [=](const SimplexId a, const SimplexId b) -> bool {
    return ((fakeScalars[a] < fakeScalars[b])
            || (fakeScalars[a] == fakeScalars[b]
                && ((monotonyOffsets[a] < monotonyOffsets[b])
                    || (monotonyOffsets[a] == monotonyOffsets[b]
                        && offsets[a] < offsets[b]))));
  };

  globalMin_ = *std::min_element(globalMinThr.begin(), globalMinThr.end(), lt);
  globalMax_ = *std::max_element(globalMaxThr.begin(), globalMaxThr.end(), lt);

  printMsg("Propagation Init", 1, timer.getElapsedTime(), threadNumber_);
}

template <typename scalarType, typename offsetType>
void ttk::ProgressiveTopology::updatePropagationCPAdaptive(
  std::vector<polarity> &toPropageMin,
  std::vector<polarity> &toPropageMax,
  std::vector<std::vector<SimplexId>> &vertexRepresentativesMin,
  std::vector<std::vector<SimplexId>> &vertexRepresentativesMax,
  std::vector<std::vector<SimplexId>> &saddleCCMin,
  std::vector<std::vector<SimplexId>> &saddleCCMax,
  std::vector<Lock> &vertLockMin,
  std::vector<Lock> &vertLockMax,
  std::vector<polarity> &isUpdatedMin,
  std::vector<polarity> &isUpdatedMax,
  const scalarType *fakeScalars,
  const offsetType *const offsets,
  const int *const monotonyOffsets) {

  Timer tm{};
  const size_t nDecVerts = multiresTriangulation_.getDecimatedVertexNumber();

  if(debugLevel_ > 5) {
    const auto pred = [](const polarity a) { return a > 0; };
    const auto numberOfCandidatesToPropageMax
      = std::count_if(toPropageMax.begin(), toPropageMax.end(), pred);
    std::cout << " sad-max we have " << numberOfCandidatesToPropageMax
              << " vertices to propage from outta " << nDecVerts << std::endl;
    const auto numberOfCandidatesToPropageMin
      = std::count_if(toPropageMin.begin(), toPropageMin.end(), pred);
    std::cout << " min-sad we have " << numberOfCandidatesToPropageMin
              << " vertices to propage from outta " << nDecVerts << std::endl;
  }

  std::vector<SimplexId> globalMaxThr(threadNumber_, 0);
  std::vector<SimplexId> globalMinThr(threadNumber_, 0);

  // reset updated flag
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(size_t i = 0; i < nDecVerts; i++) {
    SimplexId v = multiresTriangulation_.localToGlobalVertexId(i);
    isUpdatedMin[v] = 0;
    isUpdatedMax[v] = 0;
  }

  // propage along split tree
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(size_t i = 0; i < nDecVerts; i++) {
    SimplexId v = multiresTriangulation_.localToGlobalVertexId(i);
    if(toPropageMin[v]) {
      // printMsg("propage min on " + std::to_string(v));
      // // std::cout << "propage min on " << v << " ";
      // for(auto cc : saddleCCMin[v]) {
      //   std::cout << cc << " ";
      // }
      // std::cout << std::endl;
      propageFromSaddlesAdaptive(v, vertLockMin, toPropageMin,
                                 vertexRepresentativesMin, saddleCCMin,
                                 isUpdatedMin, globalMinThr, false, fakeScalars,
                                 offsets, monotonyOffsets);
    }
    if(toPropageMax[v]) {
      // std::cout << "propage max on " << v << " ";
      // for(auto cc : saddleCCMax[v]) {
      //   std::cout << cc << " ";
      // }
      // std::cout << std::endl;
      // printMsg("propage max on " + std::to_string(v));
      propageFromSaddlesAdaptive(v, vertLockMax, toPropageMax,
                                 vertexRepresentativesMax, saddleCCMax,
                                 isUpdatedMax, globalMaxThr, true, fakeScalars,
                                 offsets, monotonyOffsets);
    }
  }

  const auto lt = [=](const SimplexId a, const SimplexId b) -> bool {
    return ((fakeScalars[a] < fakeScalars[b])
            || (fakeScalars[a] == fakeScalars[b]
                && ((monotonyOffsets[a] < monotonyOffsets[b])
                    || (monotonyOffsets[a] == monotonyOffsets[b]
                        && offsets[a] < offsets[b]))));
  };

  globalMin_ = *std::min_element(globalMinThr.begin(), globalMinThr.end(), lt);
  globalMax_ = *std::max_element(globalMaxThr.begin(), globalMaxThr.end(), lt);

  if(globalMin_ == 0 and globalMax_ == 0) {
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
    for(size_t i = 0; i < nDecVerts; i++) {
      SimplexId v = multiresTriangulation_.localToGlobalVertexId(i);

#ifdef TTK_ENABLE_OPENMP
      const auto tid = omp_get_thread_num();
#else
      const auto tid = 0;
#endif // TTK_ENABLE_OPENMP

      if(lt(globalMaxThr[tid], v)) {
        globalMaxThr[tid] = v;
      }
      if(lt(v, globalMinThr[tid])) {
        globalMinThr[tid] = v;
      }
    }
    globalMin_
      = *std::min_element(globalMinThr.begin(), globalMinThr.end(), lt);
    globalMax_
      = *std::max_element(globalMaxThr.begin(), globalMaxThr.end(), lt);
     printMsg("Explicitely found global extremas");
  }
  if(debugLevel_ > 3) {
    printMsg("Propagation Update", 1, tm.getElapsedTime(), threadNumber_);
  }
}

template <typename scalarType, typename offsetType>
void ttk::ProgressiveTopology::buildVertexLinkPolarityAdaptive(
  const SimplexId vertexId,
  std::vector<std::pair<polarity, polarity>> &vlp,
  const scalarType *fakeScalars,
  const offsetType *const offsets,
  const int *const monotonyOffsets) const {

  const SimplexId neighborNumber
    = multiresTriangulation_.getVertexNeighborNumber(vertexId);
  vlp.resize(neighborNumber);

  for(SimplexId i = 0; i < neighborNumber; i++) {
    SimplexId neighborId0 = 0;
    multiresTriangulation_.getVertexNeighbor(vertexId, i, neighborId0);

    // const bool lower0 = vertsOrder_[neighborId0] < vertsOrder_[vertexId];
    const bool lower0
      = ((fakeScalars[neighborId0] < fakeScalars[vertexId])
         || (fakeScalars[neighborId0] == fakeScalars[vertexId]
             && ((monotonyOffsets[neighborId0] < monotonyOffsets[vertexId])
                 || (monotonyOffsets[neighborId0] == monotonyOffsets[vertexId]
                     && offsets[neighborId0] < offsets[vertexId]))));

    const polarity isUpper0 = static_cast<polarity>(!lower0) * 255;
    vlp[i] = std::make_pair(isUpper0, 0);
  }
}

template <typename scalarType, typename offsetType>
bool ttk::ProgressiveTopology::printPolarity(
  std::vector<polarity> &isNew,
  const SimplexId vertexId,
  std::vector<std::vector<std::pair<polarity, polarity>>> &vertexLinkPolarity,
  const scalarType *scalars,
  const scalarType *fakeScalars,
  const offsetType *const offsets,
  const int *const monotonyOffsets,
  bool verbose) {

  bool error = false;
  std::stringstream mymsg;
  std::vector<std::pair<polarity, polarity>> &vlp
    = vertexLinkPolarity[vertexId];
  mymsg << "POLARITY PRINT"
        << "\n";
  mymsg << "vertex " << vertexId << " has "
        << multiresTriangulation_.getVertexNeighborNumber(vertexId)
        << " neighbors"
        << "\n";
  mymsg << "\tself  f:" << fakeScalars[vertexId] << " s:" << scalars[vertexId]
        << " o:" << offsets[vertexId] << " m:" << monotonyOffsets[vertexId]
        << "  isnew: " << (int)isNew[vertexId] << "\n";
  if(vlp.empty()) {
    if(verbose) {
      std::cout << "\tpolarity not initialized for " << vertexId << std::endl;
      std::cout << mymsg.str() << std::endl;
    }
    return false;
  }
  for(size_t i = 0;
      i < multiresTriangulation_.getVertexNeighborNumber(vertexId); i++) {
    SimplexId nId = -1;
    multiresTriangulation_.getVertexNeighbor(vertexId, i, nId);
    // find reverse pol
    std::vector<std::pair<polarity, polarity>> &vlp2 = vertexLinkPolarity[nId];
    bool init = false;
    polarity rpol;
    if(!vlp2.empty()) {
      for(size_t j = 0; j < multiresTriangulation_.getVertexNeighborNumber(nId);
          j++) {
        SimplexId nId2 = -1;
        multiresTriangulation_.getVertexNeighbor(nId, j, nId2);
        if(nId2 == vertexId) {
          rpol = vlp2[j].first;
          init = true;
        }
      }
    }

    const bool lower
      = ((fakeScalars[nId] < fakeScalars[vertexId])
         || (fakeScalars[nId] == fakeScalars[vertexId]
             && ((monotonyOffsets[nId] < monotonyOffsets[vertexId])
                 || (monotonyOffsets[nId] == monotonyOffsets[vertexId]
                     && offsets[nId] < offsets[vertexId]))));

    const polarity isUpper = lower ? 0 : 255;

    mymsg << " " << i << "th: " << nId << " f:" << fakeScalars[nId]
          << " s:" << scalars[nId] << " o:" << offsets[nId]
          << " m:" << monotonyOffsets[nId] << "  , pol:" << (bool)vlp[i].first
          << "(" << (bool)vlp[i].second << ")"
          << " rpol:" << (bool)rpol << "  true pol:" << (bool)isUpper
          << " init " << init << "  isnew: " << (int)isNew[nId] << "\n";
    if((rpol == isUpper and !vlp2.empty())
       or (isUpper != vlp[i].first and !vlp[i].second)) {
      mymsg << "POLARITY ERROR "
            << "\n";
      error = true;
    }
  }
  if(error or verbose) {
    std::cout << mymsg.str() << std::endl;
  }
  return error;
}

template <typename scalarType, typename offsetType>
bool ttk::ProgressiveTopology::flagPolarityChanges(
  const SimplexId vertexId,
  std::vector<std::pair<polarity, polarity>> &vlp,
  const scalarType *fakeScalars,
  const offsetType *const offsets,
  const int *const monotonyOffsets) const {

  bool hasChanged = false;
  for(size_t i = 0; i < vlp.size(); i++) {
    SimplexId nId = -1;
    multiresTriangulation_.getVertexNeighbor(vertexId, i, nId);
    const bool lower
      = ((fakeScalars[nId] < fakeScalars[vertexId])
         || (fakeScalars[nId] == fakeScalars[vertexId]
             && ((monotonyOffsets[nId] < monotonyOffsets[vertexId])
                 || (monotonyOffsets[nId] == monotonyOffsets[vertexId]
                     && offsets[nId] < offsets[vertexId]))));

    const polarity isUpper = lower ? 0 : 255;
    if(vlp[i].first != isUpper) {
      vlp[i].second = 255;
      hasChanged = true;
    }
  }
  return hasChanged;
}

template <typename scalarType, typename offsetType>
bool ttk::ProgressiveTopology::signalPolarityChange(
  const SimplexId vertexId,
  const SimplexId neighborId,
  std::vector<std::pair<polarity, polarity>> &vlp,
  const scalarType *fakeScalars,
  const offsetType *const offsets,
  const int *const monotonyOffsets) const {

  bool hasChanged = false;
  for(size_t i = 0; i < vlp.size(); i++) {
    SimplexId nId = -1;
    multiresTriangulation_.getVertexNeighbor(vertexId, i, nId);
    if(nId == neighborId) {
      const bool lower
        = ((fakeScalars[nId] < fakeScalars[vertexId])
           || (fakeScalars[nId] == fakeScalars[vertexId]
               && ((monotonyOffsets[nId] < monotonyOffsets[vertexId])
                   || (monotonyOffsets[nId] == monotonyOffsets[vertexId]
                       && offsets[nId] < offsets[vertexId]))));

      const polarity isUpper = lower ? 0 : 255;
      if(vlp[i].first != isUpper) {
        vlp[i].second = 255;
        hasChanged = true;
      }
      return hasChanged;
    }
  }
  return hasChanged;
}

template <typename scalarType, typename offsetType>
bool ttk::ProgressiveTopology::updateLinkPolarityAdaptive(
  const SimplexId vertexId,
  std::vector<std::pair<polarity, polarity>> &vlp,
  const scalarType *fakeScalars,
  const offsetType *const offsets,
  const int *const monotonyOffsets) const {

  bool hasChanged = false;
  // std::cout << " updating polarity of vertex " << vertexId << ", size "
  //           << vlp.size() << std::endl;
  if(vlp.empty()) {
    buildVertexLinkPolarityAdaptive(
      vertexId, vlp, fakeScalars, offsets, monotonyOffsets);
    return true;
  }
  for(size_t i = 0; i < vlp.size(); i++) {
    SimplexId neighborId = -1;
    multiresTriangulation_.getVertexNeighbor(vertexId, i, neighborId);

    const bool lower
      = ((fakeScalars[neighborId] < fakeScalars[vertexId])
         || (fakeScalars[neighborId] == fakeScalars[vertexId]
             && ((monotonyOffsets[neighborId] < monotonyOffsets[vertexId])
                 || (monotonyOffsets[neighborId] == monotonyOffsets[vertexId]
                     && offsets[neighborId] < offsets[vertexId]))));

    const polarity isUpper = lower ? 0 : 255;
    if(vlp[i].first != isUpper) {
      vlp[i] = std::make_pair(isUpper, 0);
      hasChanged = true;
    }
  }
  return hasChanged;
}

template <typename scalarType, typename offsetType>
void ttk::ProgressiveTopology::getCriticalTypeAdaptive(
  const SimplexId &vertexId,
  std::vector<std::pair<polarity, polarity>> &vlp,
  uint8_t &vertexLink,
  DynamicTree &link,
  VLBoundaryType &vlbt,
  const scalarType *fakeScalars,
  const offsetType *const offsets,
  const int *const monotonyOffsets) const {

  if(vlp.empty()) {
    buildVertexLinkPolarityAdaptive(
      vertexId, vlp, fakeScalars, offsets, monotonyOffsets);
  }
  const SimplexId neighborNumber
    = multiresTriangulation_.getVertexNeighborNumber(vertexId);
  link.alloc(neighborNumber);

  // associate vertex link boundary
  vertexLink = multiresTriangulation_.getVertexBoundaryIndex(vertexId);

  // update the link polarity for old points that are processed for
  // the first time
  const auto &vl = vlbt[vertexLink];

  for(size_t edgeId = 0; edgeId < vl.size(); edgeId++) {
    const SimplexId n0 = vl[edgeId].first;
    const SimplexId n1 = vl[edgeId].second;
    if(vlp[n0].first == vlp[n1].first) {
      // the smallest id (n0) becomes the parent of n1
      link.insertEdge(n1, n0);
    }
  }
}

template <typename scalarType, typename offsetType>
void ttk::ProgressiveTopology::initCriticalPointsAdaptive(
  std::vector<polarity> &isNew,
  std::vector<std::vector<std::pair<polarity, polarity>>> &vertexLinkPolarity,
  std::vector<polarity> &toPropageMin,
  std::vector<polarity> &toPropageMax,
  std::vector<polarity> &toProcess,
  std::vector<DynamicTree> &link,
  std::vector<uint8_t> &vertexLink,
  VLBoundaryType &vertexLinkByBoundaryType,
  std::vector<std::vector<SimplexId>> &saddleCCMin,
  std::vector<std::vector<SimplexId>> &saddleCCMax,
  const scalarType *fakeScalars,
  const offsetType *const offsets,
  const int *const monotonyOffsets) const {

  Timer timer{};
  const size_t nDecVerts = multiresTriangulation_.getDecimatedVertexNumber();

  // computes the critical types of all points
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(size_t i = 0; i < nDecVerts; i++) {
    SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(i);
    buildVertexLinkPolarityAdaptive(globalId, vertexLinkPolarity[globalId],
                                    fakeScalars, offsets, monotonyOffsets);
    getCriticalTypeAdaptive(globalId, vertexLinkPolarity[globalId],
                            vertexLink[globalId], link[globalId],
                            vertexLinkByBoundaryType, fakeScalars, offsets,
                            monotonyOffsets);
    getValencesFromLink(globalId, vertexLinkPolarity[globalId], link[globalId],
                        toPropageMin, toPropageMax, saddleCCMin, saddleCCMax);
    toProcess[globalId] = 255;
    isNew[globalId] = 0;
  }

  // std::cout << "initial critical types in " << timer.getElapsedTime() << "
  // s."
  //           << std::endl;
  printMsg("CriticalPoints Init", 1, timer.getElapsedTime(), threadNumber_);
}

template <typename scalarType, typename offsetType>
void ttk::ProgressiveTopology::initGlobalPolarityAdaptive(
  std::vector<polarity> &isNew,
  std::vector<std::vector<std::pair<polarity, polarity>>> &vertexLinkPolarity,
  std::vector<polarity> &toProcess,
  const scalarType *fakeScalars,
  const offsetType *const offsets,
  const int *const monotonyOffsets) const {

  Timer timer{};
  const size_t nDecVerts = multiresTriangulation_.getDecimatedVertexNumber();

  // computes the critical types of all points
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(size_t i = 0; i < nDecVerts; i++) {
    SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(i);
    buildVertexLinkPolarityAdaptive(globalId, vertexLinkPolarity[globalId],
                                    fakeScalars, offsets, monotonyOffsets);
    toProcess[globalId] = 255;
    isNew[globalId] = 0;
  }

  // std::cout << "initial critical types in " << timer.getElapsedTime() << "
  // s."
  //           << std::endl;
  printMsg("Polarity Init", 1, timer.getElapsedTime(), threadNumber_);
}

// template <typename scalarType, typename offsetType>
// bool ttk::ProgressiveTopology::getInitialErrorField(
//   scalarType *deltaField,
//   const scalarType *scalarField,
//   const offsetType *offsets) {

//   int decimationLevel = decimationLevel_;
//   std::vector<polarity> isNew(multiresTriangulation_.getVertexNumber(),
//   255);

//   while(decimationLevel >= stoppingDecimationLevel_) {
//     multiresTriangulation_.setDecimationLevel(decimationLevel);
//     SimplexId vertDecNumber =
//     multiresTriangulation_.getDecimatedVertexNumber();

//     // get monotony changes
// #ifdef TTK_ENABLE_OPENMP
// #pragma omp parallel for num_threads(threadNumber_)
// #endif // TTK_ENABLE_OPENMP
//     for(size_t v = 0; v < vertDecNumber; v++) {
//       SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(v);
//       if(!isNew[globalId]) {
//         deltaField[globalId] = getDelta(globalId, isNew);
//       }
//     }

//     // build vertex link polarity
// #ifdef TTK_ENABLE_OPENMP
// #pragma omp parallel for num_threads(threadNumber_)
// #endif // TTK_ENABLE_OPENMP
//     for(size_t v = 0; v < vertDecNumber; v++) {
//       SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(v);
//       if(isNew[globalId]) {
//         if(decimationLevel > stoppingDecimationLevel_) {
//           buildVertexLinkPolarity(
//             globalId, vertexLinkPolarity[globalId], scalarField, offsets);
//         }
//         isNew[globalId] = 0;
//       }
//     }
//     decimationLevel--;
//   } // end while

//   multiresTriangulation_.setDecimationLevel(decimationLevel_);
// }

template <typename scalarType, typename offsetType>
bool ttk::ProgressiveTopology::buildNonMonotonicList(
  std::vector<polarity> &isNew,
  std::vector<std::vector<std::pair<polarity, polarity>>> &vertexLinkPolarity,
  std::vector<polarity> &toProcess,
  std::vector<polarity> &toReprocess,
  std::vector<std::tuple<SimplexId, scalarType, int>> &nonMonotonicList,
  const scalarType *const scalarField,
  scalarType *fakeScalars,
  scalarType *deltaField,
  scalarType *deltaFieldDynamic,
  const offsetType *const offsets,
  int *monotonyOffsets) {
  std::cout << "building non monotonic list" << std::endl;

  int decimationLevel = startingDecimationLevel_;
  multiresTriangulation_.setDecimationLevel(decimationLevel);

  SimplexId nDecVerts = multiresTriangulation_.getDecimatedVertexNumber();

  // init link polarity
  std::cout << "Init link polarity" << std::endl;
  for(SimplexId localId = 0; localId < nDecVerts; localId++) {
    SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(localId);
    buildVertexLinkPolarityAdaptive(globalId, vertexLinkPolarity[globalId],
                                    fakeScalars, offsets, monotonyOffsets);
    isNew[globalId] = 0;
  }

  while(decimationLevel > stoppingDecimationLevel_) {
    decimationLevel--;
    multiresTriangulation_.setDecimationLevel(decimationLevel);
    std::cout << "decimation level "
              << multiresTriangulation_.getDecimationLevel() << std::endl;
    nDecVerts = multiresTriangulation_.getDecimatedVertexNumber();
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
    for(SimplexId localId = 0; localId < nDecVerts; localId++) {
      SimplexId globalId
        = multiresTriangulation_.localToGlobalVertexId(localId);
      if(!isNew[globalId]) {
        getDeltaFromOldPoint(globalId, isNew, toProcess, toReprocess,
                             nonMonotonicList, vertexLinkPolarity[globalId],
                             scalarField, fakeScalars, deltaField,
                             deltaFieldDynamic, offsets, monotonyOffsets);
      }
    }
    if(decimationLevel > stoppingDecimationLevel_) {
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
      for(SimplexId localId = 0; localId < nDecVerts; localId++) {
        SimplexId globalId
          = multiresTriangulation_.localToGlobalVertexId(localId);
        if(isNew[globalId]) {
          buildVertexLinkPolarityAdaptive(
            globalId, vertexLinkPolarity[globalId], fakeScalars, offsets,
            monotonyOffsets);
          isNew[globalId] = 0;
        } else { // old vertex
          if(toReprocess[globalId]) { // link polarity has changed.
            updateLinkPolarityPonctual(vertexLinkPolarity[globalId]);
            toReprocess[globalId] = 0;
          }
        }
      }
    }
  }

  std::fill(isNew.begin(), isNew.end(), 255);
  // sort list
  auto cmp = [](const std::tuple<SimplexId, scalarType, int> &t0,
                const std::tuple<SimplexId, scalarType, int> &t1) {
    return std::get<1>(t0) > std::get<1>(t1);
  };
  std::sort(nonMonotonicList.begin(), nonMonotonicList.end(), cmp);

  // print list
  std::cout << "output list : (" << nonMonotonicList.size() << ")" << std::endl;
  if(debugLevel_ > 4) {
    for(auto t : nonMonotonicList) {
      std::cout << std::get<0>(t) << " (" << std::get<1>(t) << " , "
                << std::get<2>(t) << ")" << std::endl;
    }
  }

  return true;
}

template <typename scalarType, typename offsetType>
bool ttk::ProgressiveTopology::jumpToLevelAdaptive(
  int level,
  double eps,
  std::vector<polarity> &isNew,
  std::vector<std::vector<std::pair<polarity, polarity>>> &vertexLinkPolarity,
  std::vector<polarity> &toPropageMin,
  std::vector<polarity> &toPropageMax,
  std::vector<polarity> &toProcess,
  std::vector<polarity> &toReprocess,
  std::vector<DynamicTree> &link,
  std::vector<uint8_t> &vertexLink,
  VLBoundaryType &vertexLinkByBoundaryType,
  std::vector<std::vector<SimplexId>> &saddleCCMin,
  std::vector<std::vector<SimplexId>> &saddleCCMax,
  const scalarType *const scalars,
  scalarType *fakeScalars,
  scalarType *deltaField,
  scalarType *deltaFieldDynamic,
  const offsetType *const offsets,
  int *monotonyOffsets) {

  // no jump
  if(decimationLevel_ <= level) {
    return false;
  }

  if(decimationLevel_ == startingDecimationLevel_) {
    initCriticalPoints(isNew, vertexLinkPolarity, toPropageMin, toPropageMax,
                       toProcess, link, vertexLink, vertexLinkByBoundaryType,
                       saddleCCMin, saddleCCMax, fakeScalars, offsets,
                       monotonyOffsets);
  }
  while(decimationLevel_ > level) {
    Timer tmIter{};
    decimationLevel_--;
    multiresTriangulation_.setDecimationLevel(decimationLevel_);
    double progress = (double)(startingDecimationLevel_ - decimationLevel_)
                      / (startingDecimationLevel_);
    this->printMsg(
      "JUMPING decimation level: " + std::to_string(decimationLevel_), progress,
      0, threadNumber_);
    // std::cout << "\ndecimation level " << decimationLevel_ << std::endl;
    std::vector<std::pair<SimplexId, double>> globalErrorList;
    // updateCriticalPoints(eps, isNew, vertexLinkPolarity, toPropageMin,
    //                      toPropageMax, toProcess, toReprocess, link,
    //                      vertexLink, vertexLinkByBoundaryType, saddleCCMin,
    //                      saddleCCMax, globalErrorList, scalars, fakeScalars,
    //                      deltaField, deltaFieldDynamic, offsets,
    //                      monotonyOffsets);
  }
  std::cout << "done jumping" << std::endl;
  return true;
}

template <typename scalarType, typename offsetType>
void ttk::ProgressiveTopology::resetVertexValueAdaptive(
  SimplexId vertexId,
  std::vector<polarity> &isNew,
  std::vector<std::vector<std::pair<polarity, polarity>>> &vertexLinkPolarity,
  std::vector<polarity> &toPropageMin,
  std::vector<polarity> &toPropageMax,
  std::vector<polarity> &toProcess,
  std::vector<polarity> &toReprocess,
  std::vector<DynamicTree> &link,
  std::vector<uint8_t> &vertexLink,
  VLBoundaryType &vertexLinkByBoundaryType,
  std::vector<std::vector<SimplexId>> &saddleCCMin,
  std::vector<std::vector<SimplexId>> &saddleCCMax,
  const scalarType *const scalars,
  scalarType *fakeScalars,
  scalarType *deltaField,
  scalarType *deltaFieldDynamic,
  const offsetType *const offsets,
  int *monotonyOffsets) {

  if(debugLevel_ > 4) {
    std::cout << "Adding vertex " << vertexId << std::endl;
  }
  printPolarity(isNew, vertexId, vertexLinkPolarity, scalars, fakeScalars,
                offsets, monotonyOffsets);

  fakeScalars[vertexId] = scalars[vertexId];
  deltaFieldDynamic[vertexId] = vertexId;

  updateLinkPolarityAdaptive(vertexId, vertexLinkPolarity[vertexId],
                             fakeScalars, offsets, monotonyOffsets);
  int nneigh = multiresTriangulation_.getVertexNeighborNumber(vertexId);

  std::vector<SimplexId> verticesToUpdate(nneigh + 1, -1);
  verticesToUpdate[nneigh] = vertexId;

  for(int ineigh = 0; ineigh < nneigh; ineigh++) {
    SimplexId neighborId = -1;
    multiresTriangulation_.getVertexNeighbor(vertexId, ineigh, neighborId);
    bool toUpdate;
    if(toProcess[neighborId]) {
      toUpdate = signalPolarityChange(neighborId, vertexId,
                                      vertexLinkPolarity[neighborId],
                                      fakeScalars, offsets, monotonyOffsets);
    } else {
      toUpdate
        = updateLinkPolarityAdaptive(neighborId, vertexLinkPolarity[neighborId],
                                     fakeScalars, offsets, monotonyOffsets);
    }
    if(toUpdate) {
      verticesToUpdate[ineigh] = neighborId;
      std::cout << " vertex " << neighborId << " to update " << std::endl;
    }
  }

  for(size_t i = 0; i < verticesToUpdate.size(); i++) {
    SimplexId v = verticesToUpdate[i];
    if(v != -1) {
      if(toProcess[v]) { // was already processed : need to reprocess
        updateDynamicLink(link[v], vertexLinkPolarity[v],
                          vertexLinkByBoundaryType[vertexLink[v]]);
      } else { // first processing
        if(vertexLinkPolarity[v].empty()) {
          buildVertexLinkPolarityAdaptive(
            v, vertexLinkPolarity[v], fakeScalars, offsets, monotonyOffsets);
        }
        getCriticalTypeAdaptive(v, vertexLinkPolarity[v], vertexLink[v],
                                link[v], vertexLinkByBoundaryType, fakeScalars,
                                offsets, monotonyOffsets);
        toProcess[v] = 255; // mark as processed
      }
      getValencesFromLink(v, vertexLinkPolarity[v], link[v], toPropageMin,
                          toPropageMax, saddleCCMin, saddleCCMax);
    }
  }
  printPolarity(isNew, vertexId, vertexLinkPolarity, scalars, fakeScalars,
                offsets, monotonyOffsets);
}

// template <typename scalarType, typename offsetType>
// bool ttk::ProgressiveTopology::preprocessError(
//   std::vector<polarity> &isNew,
//   const scalarType *const scalarField,
//   scalarType *fakeScalars,
//   scalarType *deltaField,
//   scalarType *deltaFieldDynamic,
//   const offsetType *const offsets,
//   int *monotonyOffsets) {
//   Timer tm_preprocess;
//   // std::cout << "building non monotonic list" << std::endl;

//   int decimationLevel = startingDecimationLevel_;
//   multiresTriangulation_.setDecimationLevel(decimationLevel);

//   scalarType gmin = 0, gmax = 0;

//   SimplexId nDecVerts = multiresTriangulation_.getDecimatedVertexNumber();
//   std::vector<double> maxError(
//     startingDecimationLevel_ - stoppingDecimationLevel_, 0);
//   std::vector<std::vector<double>> epsilon_i(
//     multiresTriangulation_.getVertexNumber());
//   for(SimplexId globalId = 0;
//       globalId < multiresTriangulation_.getVertexNumber(); globalId++) {
//     if(scalarField[globalId] > gmax) {
//       gmax = scalarField[globalId];
//     }
//     if(scalarField[globalId] < gmin) {
//       gmin = scalarField[globalId];
//     }
//   }
//   std::cout << "gmax " << gmax << " " << gmin << std::endl;
//   for(SimplexId localId = 0; localId < nDecVerts; localId++) {
//     SimplexId globalId =
//     multiresTriangulation_.localToGlobalVertexId(localId); isNew[globalId]
//     = 0;
//   }

//   auto cmp = [](scalarType a, scalarType b) { return abs(a) > abs(b); };

//   while(decimationLevel > stoppingDecimationLevel_) {
//     decimationLevel--;
//     multiresTriangulation_.setDecimationLevel(decimationLevel);
//     // std::cout << "decimation level "
//     //           << multiresTriangulation_.getDecimationLevel() <<
//     std::endl; nDecVerts =
//     multiresTriangulation_.getDecimatedVertexNumber();

//     std::vector<std::vector<SimplexId>> vertexList(threadNumber_);
//     std::vector<std::vector<float>> baryCentrics(threadNumber_);
//     if(multiresTriangulation_.getDimensionality() == 3) {
//       for(auto ttid = 0; ttid < threadNumber_; ttid++) {
//         vertexList[ttid].resize(8);
//         baryCentrics[ttid].resize(4);
//       }
//     } else {
//       for(auto ttid = 0; ttid < threadNumber_; ttid++) {
//         vertexList[ttid].resize(4);
//         baryCentrics[ttid].resize(3);
//       }
//     }

//     // #ifdef TTK_ENABLE_OPENMP
//     // #pragma omp parallel for num_threads(threadNumber_)
//     // #endif // TTK_ENABLE_OPENMP
//     for(SimplexId localId = 0; localId < nDecVerts; localId++) {
//       // #ifdef TTK_ENABLE_OPENMP
//       //       const auto tid = omp_get_thread_num();
//       // #else
//       const auto tid = 0;
//       // #endif // TTK_ENABLE_OPENMP
//       SimplexId globalId
//         = multiresTriangulation_.localToGlobalVertexId(localId);
//       if(isNew[globalId]) {
//         if(debugLevel_ > 4) {
//           std::cout << "vertex " << globalId << " "
//                     << (double)scalarField[globalId] << std::endl;
//         }
//         deltaField[globalId] = 0;
//         // if(scalarField[globalId] > gmax) {
//         //   gmax = scalarField[globalId];
//         // }
//         // if(scalarField[globalId] < gmin) {
//         //   gmin = scalarField[globalId];
//         // }
//         for(int dl = startingDecimationLevel_; dl > decimationLevel; dl--)
//         {
//           // std::cout << "   dl :" << dl << std::endl;
//           multiresTriangulation_.findEnclosingVoxel(
//             globalId, pow(2, dl), vertexList[tid], baryCentrics[tid]);

//           if(debugLevel_ > 4) {
//             std::cout << "      vbounds: ";
//             for(SimplexId v : vertexList[tid]) {
//               std::cout << " " << v;
//             }
//             std::cout << std::endl;
//             std::cout << "      baryC: ";
//             for(float f : baryCentrics[tid]) {
//               std::cout << " " << f;
//             }
//             std::cout << std::endl;
//           }

//           // std::vector<scalarType> bounds(0);
//           // for(SimplexId v : vertexList[tid]) {
//           //   if(v != -1) {
//           //     bounds.push_back(scalarField[globalId] - scalarField[v]);
//           //   }
//           // }
//           double interp = 0;
//           double error = 0;
//           for(int ivert = 0; ivert < baryCentrics[tid].size(); ivert++) {
//             SimplexId vId = vertexList[tid][ivert];
//             if(vId != -1) {
//               interp += baryCentrics[tid][ivert] *
//               (double)(scalarField[vId]);
//             }
//           }
//           error = (double)(scalarField[globalId]) - interp;
//           if(abs(error) > (double)(gmax - gmin)) {
//             std::cout << " vertex " << globalId << " dl " << dl <<
//             std::endl; std::cout << "      vbounds: "; for(SimplexId v :
//             vertexList[tid]) {
//               std::cout << " " << v;
//             }
//             std::cout << std::endl;
//             std::cout << "      bounds: ";
//             for(SimplexId v : vertexList[tid]) {
//               std::cout << " " << scalarField[v];
//             }
//             std::cout << std::endl;
//             std::cout << "      baryC: ";
//             for(float f : baryCentrics[tid]) {
//               std::cout << " " << f;
//             }
//             std::cout << std::endl;
//             std::cout << "  val " << scalarField[globalId] << "  interp "
//                       << interp << std::endl;

//             std::cout << "ERROR in error" << std::endl;
//           }

//           if(debugLevel_ > 4) {
//             std::cout << "     interp " << (double)interp << ",  val "
//                       << (double)scalarField[globalId] << std::endl;
//             // std::cout << "      bounds: ";
//             // for(scalarType b : bounds) {
//             //   std::cout << " " << (double)b;
//             // }
//             // std::cout << std::endl;
//           }

//           // std::sort(bounds.begin(), bounds.end(), cmp);

//           // if(debugLevel_ > 4) {
//           //   std::cout << "      sorted bounds: ";
//           //   for(scalarType b : bounds) {
//           //     std::cout << " " << (double)b;
//           //   }
//           //   std::cout << std::endl;
//           // }

//           // epsilon_i[globalId].push_back(bounds[0]);
//           epsilon_i[globalId].push_back(error);
//           // if(abs(bounds[0]) > abs(deltaField[globalId])) {
//           // deltaField[globalId] = bounds[0];
//           deltaField[globalId] = error;
//           // }
//           // if(abs(bounds[0]) > abs(maxError[startingDecimationLevel_ -
//           dl])) { if(abs(error) > abs(maxError[startingDecimationLevel_ -
//           dl])) {
//             // std::cout<<"found bigger error "<<(double)bounds[0]<<" for
//             dl
//             // "<<dl<<std::endl;
//             maxError[startingDecimationLevel_ - dl] = error;
//             // std::cout<<" vertex "<<globalId<<" dl "<<dl<<std::endl;
//             // std::cout << "      vbounds: ";
//             // for(SimplexId v : vertexList[tid]) {
//             //   std::cout << " " << v;
//             // }
//             // std::cout << std::endl;
//             // std::cout << "      bounds: ";
//             // for(SimplexId v : vertexList[tid]) {
//             //   std::cout << " " << (double)scalarField[v];
//             // }
//             // std::cout << std::endl;
//             // std::cout << "      baryC: ";
//             // for(float f : baryCentrics[tid]) {
//             //   std::cout << " " << f;
//             // }
//             // std::cout << std::endl;
//             // std::cout<<"  val "<<scalarField[globalId]<<"  interp
//             // "<<interp<<std::endl;

//             // std::cout<<"ERROR in error"<<std::endl;
//             // return false;
//           }
//         }
//         isNew[globalId] = 0;

//         if(debugLevel_ > 4) {
//           std::cout << "\n      epsilon i: ";
//           for(scalarType e : epsilon_i[globalId]) {
//             std::cout << " " << (double)e;
//           }
//           std::cout << std::endl;
//         }
//       }
//     }
//   }

//   std::cout << "   Max Error: ";
//   for(scalarType e : maxError) {
//     std::cout << " " << (double)e;
//   }
//   std::cout << std::endl;
//   std::cout << "   Max Error %: ";
//   for(scalarType e : maxError) {
//     std::cout << " " << (double)e / (double)(gmax - gmin) * 100;
//   }
//   std::cout << std::endl;

//   for(int dl = startingDecimationLevel_; dl > stoppingDecimationLevel_;
//   dl--)
//   {
//     std::cout << "     DL " << dl << std::endl;
//     SimplexId ten = 0;
//     SimplexId five = 0;
//     SimplexId twenty = 0;
//     for(SimplexId globalId = 0;
//         globalId < multiresTriangulation_.getVertexNumber(); globalId++) {
//       if(epsilon_i[globalId].size() > startingDecimationLevel_ - dl) {
//         double percentage
//           = abs((double)epsilon_i[globalId][startingDecimationLevel_ - dl])
//             / (double)(gmax - gmin) * 100;
//         if(percentage < 5.0) {
//           five++;
//         }
//         if(percentage < 10.0) {
//           ten++;
//         }
//         if(percentage < 20.0) {
//           twenty++;
//         }
//       }
//     }
//     std::cout << std::setprecision(2) << std::fixed;
//     std::cout << "      "
//               << (double)five / multiresTriangulation_.getVertexNumber() *
//               100
//               << "% of vertices have a <5% error" << std::endl;
//     std::cout << "      "
//               << (double)ten / multiresTriangulation_.getVertexNumber() *
//               100
//               << "% of vertices have a <10% error" << std::endl;
//     std::cout << "      "
//               << (double)twenty / multiresTriangulation_.getVertexNumber()
//               * 100
//               << "% of vertices have a <20% error" << std::endl;
//   }

//   printMsg("Preprocess complete", 1, tm_preprocess.getElapsedTime(),
//            this->threadNumber_);

//   return true;
// }

template <typename scalarType, typename offsetType>
void ttk::ProgressiveTopology::processBucketAdaptive(
  double eps,
  size_t &start_index,
  std::vector<std::tuple<SimplexId, double, int>> &globalErrorList,
  std::vector<polarity> &isNew,
  std::vector<std::vector<std::pair<polarity, polarity>>> &vertexLinkPolarity,
  std::vector<polarity> &toPropageMin,
  std::vector<polarity> &toPropageMax,
  std::vector<polarity> &toProcess,
  std::vector<polarity> &toReprocess,
  std::vector<DynamicTree> &link,
  std::vector<uint8_t> &vertexLink,
  VLBoundaryType &vertexLinkByBoundaryType,
  std::vector<std::vector<SimplexId>> &saddleCCMin,
  std::vector<std::vector<SimplexId>> &saddleCCMax,
  const scalarType *const scalars,
  scalarType *fakeScalars,
  scalarType *deltaField,
  scalarType *deltaFieldDynamic,
  const offsetType *const offsets,
  int *monotonyOffsets) {

  std::cout << "Concerned by new epsilon " << eps << std::endl;
  std::tuple<SimplexId, double, int> val
    = std::make_tuple<SimplexId, double, int>(0, (double)eps, 0);
  auto cmp = [](const std::tuple<SimplexId, double, int> &t0,
                const std::tuple<SimplexId, double, int> &t1) {
    return std::get<1>(t0) > std::get<1>(t1);
  };
  auto it = std::lower_bound(
    globalErrorList.begin(), globalErrorList.end(), val, cmp);

  // first pass: flag vertices to reprocess
  // reprocess the obvious ones
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(auto it2 = globalErrorList.begin() + start_index; it2 < it; it2++) {
    SimplexId vertexId = std::get<0>(*it2);
    if(debugLevel_ > 4) {
      std::cout << " adding vertex " << vertexId
                << " - error: " << std::get<1>(*it2) << std::endl;
    }
    auto &vlp = vertexLinkPolarity[vertexId];

    fakeScalars[vertexId] = scalars[vertexId];
    bool hasChanged = false;
    if(vlp.empty()) { // need to build the link polarity and flag each one of
                      // its neighbors
      buildVertexLinkPolarityAdaptive(vertexId, vertexLinkPolarity[vertexId],
                                      fakeScalars, offsets, monotonyOffsets);
      for(size_t i = 0; i < vlp.size(); i++) {
        SimplexId neighborId = -1;
        multiresTriangulation_.getVertexNeighbor(vertexId, i, neighborId);
        toReprocess[neighborId] = 255;
      }
      toReprocess[vertexId] = 255;

    } else {
      for(size_t i = 0; i < vlp.size(); i++) {
        SimplexId neighborId = -1;
        multiresTriangulation_.getVertexNeighbor(vertexId, i, neighborId);
        const bool lower
          = ((fakeScalars[neighborId] < fakeScalars[vertexId])
             || (fakeScalars[neighborId] == fakeScalars[vertexId]
                 && ((monotonyOffsets[neighborId] < monotonyOffsets[vertexId])
                     || (monotonyOffsets[neighborId]
                           == monotonyOffsets[vertexId]
                         && offsets[neighborId] < offsets[vertexId]))));

        const polarity isUpper = lower ? 0 : 255;
        if(vlp[i].first != isUpper) {
          toReprocess[neighborId] = 255;
          hasChanged = true;
        }
      }
      if(hasChanged) { // immediatly reprocess this vertex
        toReprocess[vertexId] = 255;
      }
    }
  }
  start_index = it - globalErrorList.begin();

  std::cout << "finished unflattening vertices" << std::endl;

  // clean the interpolating vertices

  // process not unflattened vertices that were flagged
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(int i = 0; i < multiresTriangulation_.getDecimatedVertexNumber(); i++) {
    SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(i);
    if(toReprocess[globalId]) {
      if(debugLevel_ > 4) {
        std::cout << "Processing " << globalId << "... ";
      }
      if(toProcess[globalId]) { // was already processed : need to reprocess
        if(debugLevel_ > 4) {
          std::cout << "not first processing... ";
        }
        flagPolarityChanges(globalId, vertexLinkPolarity[globalId], fakeScalars,
                            offsets, monotonyOffsets);
        if(debugLevel_ > 4) {
          std::cout << "flagged... ";
        }
        updateDynamicLink(link[globalId], vertexLinkPolarity[globalId],
                          vertexLinkByBoundaryType[vertexLink[globalId]]);
        if(debugLevel_ > 4) {
          std::cout << "updated... ";
        }
      } else { // first processing
        if(debugLevel_ > 4) {
          std::cout << "first processing... ";
        }
        if(vertexLinkPolarity[globalId].empty()) {
          buildVertexLinkPolarityAdaptive(
            globalId, vertexLinkPolarity[globalId], fakeScalars, offsets,
            monotonyOffsets);
          if(debugLevel_ > 4) {
            std::cout << "built polarity... ";
          }
        } else {
          updateLinkPolarityAdaptive(globalId, vertexLinkPolarity[globalId],
                                     fakeScalars, offsets, monotonyOffsets);
          if(debugLevel_ > 4) {
            std::cout << "updated polarity... ";
          }
        }
        getCriticalTypeAdaptive(globalId, vertexLinkPolarity[globalId],
                                vertexLink[globalId], link[globalId],
                                vertexLinkByBoundaryType, fakeScalars, offsets,
                                monotonyOffsets);
        if(debugLevel_ > 4) {
          std::cout << "got criticaltype... ";
        }
        toProcess[globalId] = 255; // mark as processed
      }
      if(debugLevel_ > 4) {
        std::cout << "valence... ";
      }
      getValencesFromLink(globalId, vertexLinkPolarity[globalId],
                          link[globalId], toPropageMin, toPropageMax,
                          saddleCCMin, saddleCCMax);
      toReprocess[globalId] = 0;
      if(debugLevel_ > 4) {
        std::cout << "done" << std::endl;
      }
    }
  } // end for openmp

  if(debugLevel_ > 3) {
    std::cout << "polarity test" << std::endl;
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
    for(int i = 0; i < multiresTriangulation_.getDecimatedVertexNumber(); i++) {
      SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(i);
      if(vertexLinkPolarity[globalId].empty()) {
        std::cout << "vertex " << globalId << " has no polarity. Process "
                  << (bool)toProcess[globalId] << "  reprocess "
                  << (bool)toReprocess[globalId] << std::endl;
      }
      // printPolarity(
      //     globalId, vertexLinkPolarity, fakeScalars, offsets,
      //     monotonyOffsets);
    }
  }

  std::cout << "Finished processing all flagged vertices" << std::endl;
}

template <typename scalarType, typename offsetType>
bool ttk::ProgressiveTopology::processHierarchicalBucketAdaptive(
  double eps,
  size_t &start_index,
  std::vector<std::tuple<SimplexId, double, int>> &globalErrorList,
  std::vector<polarity> &isNew,
  std::vector<std::vector<std::vector<std::pair<polarity, polarity>>>>
    &vertexLinkPolarity,
  std::vector<polarity> &toPropageMin,
  std::vector<polarity> &toPropageMax,
  std::vector<polarity> &toProcess,
  std::vector<polarity> &toReprocess,
  std::vector<polarity> &isUpdated,
  std::vector<polarity> &toUpdatePolarity,
  std::vector<DynamicTree> &link,
  std::vector<uint8_t> &vertexLink,
  VLBoundaryType &vertexLinkByBoundaryType,
  std::vector<std::vector<SimplexId>> &saddleCCMin,
  std::vector<std::vector<SimplexId>> &saddleCCMax,
  const scalarType *const scalarField,
  scalarType *fakeScalars,
  scalarType *deltaField,
  scalarType *deltaFieldDynamic,
  const offsetType *const offsets,
  int *monotonyOffsets) {

  Timer tm{};

  // std::cout << "Concerned by new epsilon " << eps << std::endl;

  std::tuple<SimplexId, double, int> targetEpsilonTuple
    = std::make_tuple<SimplexId, double, int>(0, (double)eps * delta_, 0);
  auto cmp = [](const std::tuple<SimplexId, double, int> &t0,
                const std::tuple<SimplexId, double, int> &t1) {
    return std::get<1>(t0) > std::get<1>(t1);
  };

  // // std::cout << "\nBucket " << std::endl;
  // // for(auto it2 = globalErrorList.begin() + start_index; it2 < itBucketEnd;
  // //     it2++) {
  // //   std::cout << "T " << std::get<0>(*it2) << " " << std::get<1>(*it2) <<
  // " "
  // //             << std::get<2>(*it2) << std::endl;
  // // }

  auto cmpDecimation = [](const std::tuple<SimplexId, double, int> &t0,
                          const std::tuple<SimplexId, double, int> &t1) {
    return std::get<2>(t0) > std::get<2>(t1);
  };
  // PARALLEL_SORT(globalErrorList.begin() + start_index, itBucketEnd,
  // cmpDecimation);

  if(debugLevel_ > 5) {
    std::cout << "\nSorted Bucket " << std::endl;
    for(auto it2 = globalErrorList.begin() + start_index;
        it2 < globalErrorList.end(); it2++) {
      std::cout << "T " << std::get<0>(*it2) << " " << std::get<1>(*it2) << " "
                << std::get<2>(*it2) << std::endl;
    }
  }
  // std::cout << "\nError bucket in " << tm.getElapsedTime() << std::endl;
  double tm_elapsed = 0;
  if(debugLevel_ > 3) {
    tm_elapsed = tm.getElapsedTime();
  }

  auto nDecVerts = multiresTriangulation_.getDecimatedVertexNumber();

  // std::vector<polarity> isUpdated(multiresTriangulation_.getVertexNumber(),
  // 0); std::vector<polarity> toUpdatePolarity(
  //   multiresTriangulation_.getVertexNumber(), 0);

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(SimplexId localId = 0; localId < nDecVerts; localId++) {
    SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(localId);
    isNew[globalId] = 255;
    isUpdated[globalId] = 0;
    toUpdatePolarity[globalId] = 0;
    // toProcess[globalId] = 0;
  }
  if(debugLevel_ > 3) {
    std::cout << "Init arrays in " << tm.getElapsedTime() - tm_elapsed
              << std::endl;
    tm_elapsed = tm.getElapsedTime();
  }

  decimationLevel_ = startingDecimationLevel_ + 1;
  // multiresTriangulation_.setDecimationLevel(decimationLevel_);
  // nDecVerts = multiresTriangulation_.getDecimatedVertexNumber();
  // polarity isNewPolarity = 255;

  // for(SimplexId localId = 0; localId < nDecVerts; localId++) {
  //   SimplexId globalId =
  //   multiresTriangulation_.localToGlobalVertexId(localId); isNew[globalId] =
  //   0;
  // }

  double current_eps = 0;
  double delta = 0;

  while(decimationLevel_ > stoppingDecimationLevel_) {
    // Timer tmIter{};
    decimationLevel_--;
    multiresTriangulation_.setDecimationLevel(decimationLevel_);
    nDecVerts = multiresTriangulation_.getDecimatedVertexNumber();
    this->printMsg("decimation level: " + std::to_string(decimationLevel_), 0,
                   tm.getElapsedTime(), threadNumber_);

    Timer timer;

    std::tuple<SimplexId, double, int> targetDecimationTuple
      = std::make_tuple<SimplexId, double, int>(
        0, (double)0, decimationLevel_ - 1);
    auto itBucketEnd = std::lower_bound(globalErrorList.begin() + start_index,
                                        /*itBucketEnd*/ globalErrorList.end(),
                                        targetDecimationTuple, cmpDecimation);

    if(debugLevel_ > 3) {
      std::cout << "\ndecimation bucket in " << timer.getElapsedTime()
                << std::endl;
      tm_elapsed = timer.getElapsedTime();
    }

    if(decimationLevel_ == stoppingDecimationLevel_) {
      if(debugLevel_ > 3) {
        std::cout << "Last decimation, processing bucket of error > " << eps
                  << std::endl;
      }
      itBucketEnd
        = std::lower_bound(globalErrorList.begin() + start_index,
                           globalErrorList.end(), targetEpsilonTuple, cmp);

      // std::cout << "\nBucket Last " << std::endl;
      // for(auto it2 = globalErrorList.begin() + start_index;
      //     it2 < globalErrorList.end(); it2++) {
      //   std::cout << "T " << std::get<0>(*it2) << " " << std::get<1>(*it2)
      //             << " " << std::get<2>(*it2) << std::endl;
      // }
      current_eps = eps;
      delta = current_eps * delta_;
    }

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
    for(SimplexId localId = 0; localId < nDecVerts; localId++) {
      SimplexId globalId
        = multiresTriangulation_.localToGlobalVertexId(localId);
      if(!isNew[globalId]) {
        if(isUpdated[globalId]) {
          const SimplexId neighborNumberNew
            = multiresTriangulation_.getVertexNeighborNumber(globalId);
          for(SimplexId j = 0; j < neighborNumberNew; j++) {
            SimplexId neighborIdNew = -1;
            multiresTriangulation_.getVertexNeighbor(
              globalId, j, neighborIdNew);
            if(isNew[neighborIdNew]) {
              isUpdated[neighborIdNew] = 255;
            }
          }
        }
      }
    }
    if(debugLevel_ > 3) {
      std::cout << "first loop in " << timer.getElapsedTime() - tm_elapsed
                << std::endl;
      tm_elapsed = timer.getElapsedTime();
    }

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
    for(auto it2 = globalErrorList.begin() + start_index; it2 < itBucketEnd;
        it2++) {

      SimplexId vertexId = std::get<0>(*it2);
      isUpdated[vertexId] = 255;
    }
    if(debugLevel_ > 3) {
      std::cout << "second loop in " << timer.getElapsedTime() - tm_elapsed
                << std::endl;
      tm_elapsed = timer.getElapsedTime();
    }

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
    for(SimplexId localId = 0; localId < nDecVerts; localId++) {
      SimplexId globalId
        = multiresTriangulation_.localToGlobalVertexId(localId);
      if(isNew[globalId]) {
        if(isUpdated[globalId]) {
          // std::cout << "  reseting " << globalId << std::endl;
          fakeScalars[globalId] = scalarField[globalId];
          monotonyOffsets[globalId] = 0;
          toUpdatePolarity[globalId] = 255;

          toPropageMin[globalId] = 0;
          toPropageMax[globalId] = 0;
          saddleCCMax[globalId].clear();
          saddleCCMin[globalId].clear();

          const SimplexId neighborNumber
            = multiresTriangulation_.getVertexNeighborNumber(globalId);
          for(SimplexId i = 0; i < neighborNumber; i++) {
            SimplexId neighborId = -1;
            multiresTriangulation_.getVertexNeighbor(globalId, i, neighborId);
            if(isNew[neighborId]) {
              toUpdatePolarity[neighborId] = 255;
              isUpdated[neighborId] = 255; // needed
            }
          }
        }
      }
    }
    if(debugLevel_ > 3) {
      std::cout << "third loop in " << timer.getElapsedTime() - tm_elapsed
                << std::endl;
      tm_elapsed = timer.getElapsedTime();
    }

    std::vector<std::vector<std::pair<SimplexId, double>>> localErrorList(
      threadNumber_);
    if(preallocateMemoryErrorList_) {
      printMsg("Preallocating error list");
      for(int i = 0; i < localErrorList.size(); i++) {
        localErrorList[i].reserve(100000);
      }
    }

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
    for(SimplexId localId = 0; localId < nDecVerts; localId++) {
      SimplexId globalId
        = multiresTriangulation_.localToGlobalVertexId(localId);
      if(!isNew[globalId]) {
        if(isUpdated[globalId]) {
          copyPolarity(
            vertexLinkPolarity[decimationLevel_ + 1 - stoppingDecimationLevel_]
                              [globalId],
            vertexLinkPolarity[decimationLevel_ - stoppingDecimationLevel_]
                              [globalId]);
        }
        getMonotonyChangeByOldPointForFlattenedVertices(
          globalId, delta, isNew, isUpdated, toProcess, toReprocess,
          toUpdatePolarity,
          vertexLinkPolarity[decimationLevel_ - stoppingDecimationLevel_]
                            [globalId],
          scalarField, fakeScalars, deltaField, deltaFieldDynamic,
          localErrorList[omp_get_thread_num()], offsets, monotonyOffsets);
      }
    }
    if(debugLevel_ > 3) {
      std::cout << "fourth loop in " << timer.getElapsedTime() - tm_elapsed
                << std::endl;
      tm_elapsed = timer.getElapsedTime();
    }

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
    for(int i = 0; i < multiresTriangulation_.getDecimatedVertexNumber(); i++) {
      SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(i);
      if(toUpdatePolarity[globalId]) {
        if(isNew[globalId]) { // new point
          updateLinkPolarityAdaptive(
            globalId,
            vertexLinkPolarity[decimationLevel_ - stoppingDecimationLevel_]
                              [globalId],
            fakeScalars, offsets, monotonyOffsets);
          // isUpdated[globalId] = 255;

        } else { // old point
          updateLinkPolarityPonctual(
            vertexLinkPolarity[decimationLevel_ - stoppingDecimationLevel_]
                              [globalId]);
        }
        toUpdatePolarity[globalId] = 0;
        // if(decimationLevel_ > stoppingDecimationLevel_) {
        //   // copyPolarity(
        //   vertexLinkPolarity[decimationLevel_ - stoppingDecimationLevel_]
        //                     [globalId],
        //   vertexLinkPolarity[decimationLevel_ - 1 -
        //   stoppingDecimationLevel_]
        //                     [globalId]);
        // }
      }
      if(isNew[globalId]) {
        isNew[globalId] = 0;
      }
    } // end for openmp
    if(debugLevel_ > 3) {
      std::cout << "fifth loop in " << timer.getElapsedTime() - tm_elapsed
                << std::endl;
      tm_elapsed = timer.getElapsedTime();
    }

    // start_index = itBucketEnd - globalErrorList.begin();
    globalErrorList.erase(globalErrorList.begin(), itBucketEnd);
    std::cout << "erased, new size " << globalErrorList.size() << std::endl;

    if(debugLevel_ > 5) {
      std::cout << "\nBucket after dec " << decimationLevel_ << std::endl;
      for(auto it2 = globalErrorList.begin() + start_index;
          it2 < globalErrorList.end(); it2++) {
        std::cout << "T " << std::get<0>(*it2) << " " << std::get<1>(*it2)
                  << " " << std::get<2>(*it2) << std::endl;
      }
    }

    double terrorlist = 0;
    if(debugLevel_ > 3) {
      terrorlist = tm.getElapsedTime();
    }
    size_t globalErrorListSize = globalErrorList.size();
    size_t globalErrorListNewSize = 0;
    std::vector<size_t> localErrorListSizes(localErrorList.size());

    for(int ithread = 0; ithread < localErrorList.size(); ithread++) {
      std::cout << "thread " << ithread << " size "
                << localErrorList[ithread].size() << std::endl;
    }

    for(int i = 0; i < localErrorList.size(); i++) {
      localErrorListSizes[i] = globalErrorListNewSize;
      globalErrorListNewSize += localErrorList[i].size();
    }
    globalErrorListNewSize += globalErrorListSize;
    globalErrorList.resize(globalErrorListNewSize);

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
    for(int i = 0; i < localErrorList.size(); i++) {
      for(int j = 0; j < localErrorList[i].size(); j++) {
        globalErrorList[globalErrorListSize + localErrorListSizes[i] + j]
          = std::make_tuple(localErrorList[i][j].first,
                            localErrorList[i][j].second, decimationLevel_);
      }
    }

    if(debugLevel_ > 3) {
      std::cout << "global size " << globalErrorListNewSize << " in "
                << tm.getElapsedTime() - terrorlist << std::endl;
    }
    if(decimationLevel_ == stoppingDecimationLevel_) {
      if(debugLevel_ > 5) {
        std::cout << "\nBucket after aggregation on last res "
                  << decimationLevel_ << std::endl;
        for(auto it2 = globalErrorList.begin() + start_index;
            it2 < globalErrorList.end(); it2++) {
          std::cout << "T " << std::get<0>(*it2) << " " << std::get<1>(*it2)
                    << " " << std::get<2>(*it2) << std::endl;
        }
      }
    }

    if(debugLevel_ > 4) {
      std::cout << "testing polarity" << std::endl;
      for(int i = 0; i < multiresTriangulation_.getDecimatedVertexNumber();
          i++) {
        SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(i);
        bool error = printPolarity(
          isNew, globalId,
          vertexLinkPolarity[decimationLevel_ - stoppingDecimationLevel_],
          scalarField, fakeScalars, offsets, monotonyOffsets);
        if(error) {
          return true;
        }
      }
    }
    // if(decimationLevel_ > stoppingDecimationLevel_) {
    //   for(int i = 0; i < multiresTriangulation_.getDecimatedVertexNumber();
    //       i++) {
    //     SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(i);
    //     if(scalarField[globalId] != fakeScalars[globalId]) {
    //       std::cout << " vertex " << globalId << " is still folded "
    //                 << fakeScalars[globalId] << " " << scalarField[globalId]
    //                 << std::endl;
    //     }
    //   }
    // }
  } // end while

  return false;
}

template <typename scalarType, typename offsetType>
bool ttk::ProgressiveTopology::getMonotonyChangeByOldPointForFlattenedVertices(
  const SimplexId vertexId,
  double eps,
  const std::vector<polarity> &isNew,
  std::vector<polarity> &isUpdated,
  std::vector<polarity> &toProcess,
  std::vector<polarity> &toReprocess,
  std::vector<polarity> &toUpdatePolarity,
  std::vector<std::pair<polarity, polarity>> &vlp,
  const scalarType *const scalarField,
  scalarType *fakeScalars,
  scalarType *deltaField,
  scalarType *deltaFieldDynamic,
  std::vector<std::pair<SimplexId, double>> &localErrorList,
  const offsetType *const offsets,
  int *monotonyOffsets) const {

  int hasMonotonyChanged = 0;
  const SimplexId neighborNumber
    = multiresTriangulation_.getVertexNeighborNumber(vertexId);

  // std::cout << "old updated vertex " << vertexId << std::endl;

  for(SimplexId i = 0; i < neighborNumber; i++) {
    SimplexId neighborId = -1;
    multiresTriangulation_.getVertexNeighbor(vertexId, i, neighborId);

    // toUpdatePolarity[neighborId] = 255;

    if(!isUpdated[neighborId]) {
      // std::cout << "  neighbor " << neighborId << "  not to be updated"
      //           << std::endl;
      continue;
    }

    // check for monotony changes
    const bool lowerDynamic
      = ((fakeScalars[neighborId] < fakeScalars[vertexId])
         || (fakeScalars[neighborId] == fakeScalars[vertexId]
             && ((monotonyOffsets[neighborId] < monotonyOffsets[vertexId])
                 || (monotonyOffsets[neighborId] == monotonyOffsets[vertexId]
                     && offsets[neighborId] < offsets[vertexId]))));

    const polarity isUpperDynamic = lowerDynamic ? 0 : 255;
    // const polarity isUpperStatic = lowerStatic ? 0 : 255;
    const polarity isUpperOld = vlp[i].first;

    if(isUpperDynamic != isUpperOld) { // change of monotony
      // std::cout << "  monotony change on " << neighborId << std::endl;
      SimplexId oldNeighbor = -1;
      int oldDecimation = pow(2, decimationLevel_ + 1);
      multiresTriangulation_.getVertexNeighborAtDecimation(
        vertexId, i, oldNeighbor, oldDecimation);

      double replacementValueDynamic
        = (0.5 * (double)fakeScalars[oldNeighbor]
           + .5 * (double)fakeScalars[vertexId]); // depends on epsilon
      double deltaDynamic
        = abs((double)fakeScalars[neighborId] - replacementValueDynamic);

      SimplexId oldNeighNumber = 0;

      //===================== meant to check irregularities at the boundary
      SimplexId nnumber
        = multiresTriangulation_.getVertexNeighborNumber(neighborId);
      for(int iii = 0; iii < nnumber; iii++) {
        SimplexId neighborId2 = -1;
        multiresTriangulation_.getVertexNeighbor(neighborId, iii, neighborId2);
        if(!isNew[neighborId2]) {
          oldNeighNumber++;
        }
      }
      //==================================================================

      if(deltaDynamic > eps or !isNew[neighborId] or oldNeighNumber > 2) {
        hasMonotonyChanged = 1;

        // UNFOLDING
        // fakeScalars[neighborId] = scalarField[neighborId];
        // monotonyOffsets[neighborId] = 0;

        // std::cout << "  unflattened " << neighborId << std::endl;

        toReprocess[vertexId] = 255;
        toUpdatePolarity[vertexId] = 255;
        isUpdated[vertexId] = 255;

        toReprocess[neighborId] = 255;
        toUpdatePolarity[neighborId] = 255;

        const SimplexId neighborNumberNew
          = multiresTriangulation_.getVertexNeighborNumber(neighborId);
        for(SimplexId j = 0; j < neighborNumberNew; j++) {
          SimplexId neighborIdNew = -1;
          multiresTriangulation_.getVertexNeighbor(
            neighborId, j, neighborIdNew);
          if(isNew[neighborIdNew]) {
            toReprocess[neighborIdNew] = 255;
            toUpdatePolarity[neighborIdNew] = 255;
          }
        }
        vlp[i].second = 255;
      } else {
        // std::cout << "  reinterpolated " << neighborId;
        // std::cout << (double)fakeScalars[neighborId] << " -> "
        //           << replacementValueDynamic << std::endl;
        // std::cout << vertexId << " " << neighborId << " " << oldNeighbor
        //           << std::endl;
        fakeScalars[neighborId] = replacementValueDynamic;

        const SimplexId neighborNumberNew
          = multiresTriangulation_.getVertexNeighborNumber(neighborId);
        // for(SimplexId j = 0; j < neighborNumberNew; j++) {
        //   SimplexId neighborIdNew = -1;
        //   multiresTriangulation_.getVertexNeighbor(
        //     neighborId, j, neighborIdNew);
        //   if(isNew[neighborIdNew]) {
        //     toUpdatePolarity[neighborIdNew] = 255;
        //   }
        // }
        // toReprocess[neighborId] = 255;

        // corrects rounding error when we process an integer scalar field
        if(fakeScalars[neighborId] == fakeScalars[oldNeighbor]) {
          fakeScalars[neighborId] = fakeScalars[vertexId];
        }
        localErrorList.push_back({neighborId, (double)deltaDynamic});

        scalarType v0 = fakeScalars[vertexId];
        scalarType v1 = fakeScalars[oldNeighbor];
        int oldOffset = monotonyOffsets[neighborId];
        // change monotony offset
        // The monotony must be preserved, meaning that
        // if we had f(vertexId)>f(oldNeighbor)
        // we should have f(vertexId)>f(neighborId)
        // which is enforced when
        // monotonyOffsets[vertexId]>monotonyOffsets[neighborId]
        if(isUpperOld) { // we should enforce f(vertexId)<f(neighborId)
          if(offsets[vertexId] > offsets[neighborId]) {
            monotonyOffsets[neighborId]
              = monotonyOffsets[vertexId] + pow(2, decimationLevel_);
            if(monotonyOffsets[vertexId] == monotonyOffsets[oldNeighbor]
               and fakeScalars[vertexId] == fakeScalars[oldNeighbor]) {
              std::cout << "THIS IS AN ISSUE" << std::endl;
            }
          } else {
            monotonyOffsets[neighborId] = monotonyOffsets[vertexId];
          }
        } else { // we should enforce f(vertexId)>f(neighborId)
          if(offsets[vertexId] < offsets[neighborId]) {
            monotonyOffsets[neighborId]
              = monotonyOffsets[vertexId] - pow(2, decimationLevel_);
            if(monotonyOffsets[vertexId] == monotonyOffsets[oldNeighbor]
               and fakeScalars[vertexId] == fakeScalars[oldNeighbor]) {
              std::cout << "THIS IS AN ISSUE" << std::endl;
            }
          } else {
            monotonyOffsets[neighborId] = monotonyOffsets[vertexId];
          }
        }
        bool v0higher
          = ((fakeScalars[neighborId] < fakeScalars[vertexId])
             || (fakeScalars[neighborId] == fakeScalars[vertexId]
                 && ((monotonyOffsets[neighborId] < monotonyOffsets[vertexId])
                     || (monotonyOffsets[neighborId]
                           == monotonyOffsets[vertexId]
                         && offsets[neighborId] < offsets[vertexId]))));
        bool v1higher
          = ((fakeScalars[neighborId] < fakeScalars[oldNeighbor])
             || (fakeScalars[neighborId] == fakeScalars[oldNeighbor]
                 && ((monotonyOffsets[neighborId]
                      < monotonyOffsets[oldNeighbor])
                     || (monotonyOffsets[neighborId]
                           == monotonyOffsets[oldNeighbor]
                         && offsets[neighborId] < offsets[oldNeighbor]))));
        bool interpolant = (v0higher and !v1higher) or (!v0higher and v1higher);

        if(!interpolant) {
          std::cout << "ERREUR - not interpolant" << std::endl;
          std::cout << "  " << vertexId << " " << neighborId << " "
                    << oldNeighbor << std::endl;
          std::cout << " fake " << (double)v0 << " "
                    << (double)fakeScalars[neighborId] << " ("
                    << replacementValueDynamic << ") " << (double)v1
                    << std::endl;
          std::cout << " real " << (double)scalarField[vertexId] << " "
                    << (double)scalarField[neighborId] << " "
                    << (double)scalarField[oldNeighbor] << std::endl;
          std::cout << "      " << offsets[vertexId] << " "
                    << offsets[neighborId] << " " << offsets[oldNeighbor]
                    << std::endl;
          std::cout << "      " << monotonyOffsets[vertexId] << " "
                    << monotonyOffsets[neighborId] << " "
                    << monotonyOffsets[oldNeighbor] << std::endl;
          std::cout << "  old offset " << oldOffset << std::endl;
          return -1;
          // std::cout << "  " << scalarField[vertexId] << " "
          //           << replacementValueStatic << " " <<
          //           scalarField[oldNeighbor]
          //           << std::endl;
          // std::cout << "  " << (v0 <= replacementValueDynamic) << " "
          //           << (replacementValueDynamic <= v1) << std::endl;
        }
      }
    }
  }
  return hasMonotonyChanged;
}

// template <typename scalarType, typename offsetType>
// void ttk::ProgressiveTopology::processHBucket(
//   double eps,
//   size_t &start_index,
//   std::vector<std::pair<SimplexId, double>> globalErrorList,
//   std::vector<polarity> &isNew,
//   std::vector<std::vector<std::pair<polarity, polarity>>>
//   &vertexLinkPolarity, std::vector<polarity> &toPropageMin,
//   std::vector<polarity> &toPropageMax,
//   std::vector<polarity> &toProcess,
//   std::vector<polarity> &toReprocess,
//   std::vector<DynamicTree> &link,
//   std::vector<uint8_t> &vertexLink,
//   VLBoundaryType &vertexLinkByBoundaryType,
//   std::vector<std::vector<SimplexId>> &saddleCCMin,
//   std::vector<std::vector<SimplexId>> &saddleCCMax,
//   const scalarType *const scalars,
//   scalarType *fakeScalars,
//   scalarType *deltaField,
//   scalarType *deltaFieldDynamic,
//   const offsetType *const offsets,
//   int *monotonyOffsets) {

//   std::cout << "Concerned by new epsilon " << eps << std::endl;
//   std::pair<SimplexId, double> val
//     = std::make_pair<SimplexId, double>(0, (double)eps);
//   auto cmp = [](const std::tuple<SimplexId, double> &t0,
//                 const std::tuple<SimplexId, double> &t1) {
//     return std::get<1>(t0) > std::get<1>(t1);
//   };
//   auto it = std::lower_bound(
//     globalErrorList.begin(), globalErrorList.end(), val, cmp);

//   // first pass: flag vertices to reprocess
//   // reprocess the obvious ones
// #ifdef TTK_ENABLE_OPENMP
// #pragma omp parallel for num_threads(threadNumber_)
// #endif
//   for(auto it2 = globalErrorList.begin() + start_index; it2 < it; it2++) {
//     SimplexId vertexId = (*it2).first;
//     if(debugLevel_ > 4) {
//       std::cout << " adding vertex " << vertexId
//                 << " - error: " << (*it2).second << std::endl;
//     }
//     auto &vlp = vertexLinkPolarity[vertexId];

//     fakeScalars[vertexId] = scalars[vertexId];
//     bool hasChanged = false;
//     if(vlp.empty()) { // need to build the link polarity and flag each one
//     of
//                       // its neighbors
//       buildVertexLinkPolarity(vertexId, vertexLinkPolarity[vertexId],
//                               fakeScalars, offsets, monotonyOffsets);
//       for(size_t i = 0; i < vlp.size(); i++) {
//         SimplexId neighborId = -1;
//         multiresTriangulation_.getVertexNeighbor(vertexId, i, neighborId);
//         toReprocess[neighborId] = 255;
//       }
//       toReprocess[vertexId] = 255;

//     } else {
//       for(size_t i = 0; i < vlp.size(); i++) {
//         SimplexId neighborId = -1;
//         multiresTriangulation_.getVertexNeighbor(vertexId, i, neighborId);
//         const bool lower
//           = ((fakeScalars[neighborId] < fakeScalars[vertexId])
//              || (fakeScalars[neighborId] == fakeScalars[vertexId]
//                  && ((monotonyOffsets[neighborId] <
//                  monotonyOffsets[vertexId])
//                      || (monotonyOffsets[neighborId]
//                            == monotonyOffsets[vertexId]
//                          && offsets[neighborId] < offsets[vertexId]))));

//         const polarity isUpper = lower ? 0 : 255;
//         if(vlp[i].first != isUpper) {
//           toReprocess[neighborId] = 255;
//           hasChanged = true;
//         }
//       }
//       if(hasChanged) { // immediatly reprocess this vertex
//         toReprocess[vertexId] = 255;
//       }
//     }
//   }
//   start_index = it - globalErrorList.begin();

//   std::cout << "finished unflattening vertices" << std::endl;
//   // process not unflattened vertices that were flagged
// #ifdef TTK_ENABLE_OPENMP
// #pragma omp parallel for num_threads(threadNumber_)
// #endif
//   for(int i = 0; i < multiresTriangulation_.getDecimatedVertexNumber();
//   i++)
//   {
//     SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(i);
//     if(toReprocess[globalId]) {
//       if(debugLevel_ > 4) {
//         std::cout << "Processing " << globalId << "... ";
//       }
//       if(toProcess[globalId]) { // was already processed : need to
//       reprocess
//         if(debugLevel_ > 4) {
//           std::cout << "not first processing... ";
//         }
//         flagPolarityChanges(globalId, vertexLinkPolarity[globalId],
//         fakeScalars,
//                             offsets, monotonyOffsets);
//         if(debugLevel_ > 4) {
//           std::cout << "flagged... ";
//         }
//         updateDynamicLink(link[globalId], vertexLinkPolarity[globalId],
//                            vertexLinkByBoundaryType[vertexLink[globalId]]);
//         if(debugLevel_ > 4) {
//           std::cout << "updated... ";
//         }
//       } else { // first processing
//         if(debugLevel_ > 4) {
//           std::cout << "first processing... ";
//         }
//         if(vertexLinkPolarity[globalId].empty()) {
//           buildVertexLinkPolarity(globalId, vertexLinkPolarity[globalId],
//                                   fakeScalars, offsets, monotonyOffsets);
//           if(debugLevel_ > 4) {
//             std::cout << "built polarity... ";
//           }
//         } else {
//           updateLinkPolarity(globalId, vertexLinkPolarity[globalId],
//                              fakeScalars, offsets, monotonyOffsets);
//           if(debugLevel_ > 4) {
//             std::cout << "updated polarity... ";
//           }
//         }
//         getCriticalType(globalId, vertexLinkPolarity[globalId],
//                         vertexLink[globalId], link[globalId],
//                         vertexLinkByBoundaryType, fakeScalars, offsets,
//                         monotonyOffsets);
//         if(debugLevel_ > 4) {
//           std::cout << "got criticaltype... ";
//         }
//         toProcess[globalId] = 255; // mark as processed
//       }
//       if(debugLevel_ > 4) {
//         std::cout << "valence... ";
//       }
//       getValencesFromLink(globalId, vertexLinkPolarity[globalId],
//                           link[globalId], toPropageMin, toPropageMax,
//                           saddleCCMin, saddleCCMax);
//       toReprocess[globalId] = 0;
//       if(debugLevel_ > 4) {
//         std::cout << "done" << std::endl;
//       }
//     }
//   } // end for openmp

//   if(debugLevel_ > 3) {
//     std::cout << "polarity test" << std::endl;
// #ifdef TTK_ENABLE_OPENMP
// #pragma omp parallel for num_threads(threadNumber_)
// #endif
//     for(int i = 0; i < multiresTriangulation_.getDecimatedVertexNumber();
//     i++) {
//       SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(i);
//       if(vertexLinkPolarity[globalId].empty()) {
//         std::cout << "vertex " << globalId << " has no polarity. Process "
//                   << (bool)toProcess[globalId] << "  reprocess "
//                   << (bool)toReprocess[globalId] << std::endl;
//       }
//       // printPolarity(
//       //     globalId, vertexLinkPolarity, fakeScalars, offsets,
//       //     monotonyOffsets);
//     }
//   }

//   std::cout << "Finished processing all flagged vertices" << std::endl;
// }

// template <typename scalarType, typename offsetType>
// bool ttk::ProgressiveTopology::unfoldVertex(
//   const SimplexId vertexId,
//   double eps,
//   const std::vector<polarity> &isNew,
//   std::vector<polarity> &toProcess,
//   std::vector<polarity> &toReprocess,
//   std::vector<std::pair<polarity, polarity>> &vlp,
//   const scalarType *const scalarField,
//   scalarType *fakeScalars,
//   const offsetType *const offsets,
//   int *monotonyOffsets) {

//   fakeScalars[vertexId] = scalars[vertexId];
//   bool hasChanged = false;
//   if(vlp.empty()) { // need to build the link polarity and flag each one of
//                     // its neighbors
//     buildVertexLinkPolarity(vertexId, vertexLinkPolarity[vertexId],
//     fakeScalars,
//                             offsets, monotonyOffsets);
//     for(size_t i = 0; i < vlp.size(); i++) {
//       SimplexId neighborId = -1;
//       multiresTriangulation_.getVertexNeighbor(vertexId, i, neighborId);
//       toReprocess[neighborId] = 255;
//     }
//     toReprocess[vertexId] = 255;

//   } else {
//     for(size_t i = 0; i < vlp.size(); i++) {
//       SimplexId neighborId = -1;
//       multiresTriangulation_.getVertexNeighbor(vertexId, i, neighborId);
//       const bool lower
//         = ((fakeScalars[neighborId] < fakeScalars[vertexId])
//            || (fakeScalars[neighborId] == fakeScalars[vertexId]
//                && ((monotonyOffsets[neighborId] <
//                monotonyOffsets[vertexId])
//                    || (monotonyOffsets[neighborId] ==
//                    monotonyOffsets[vertexId]
//                        && offsets[neighborId] < offsets[vertexId]))));

//       const polarity isUpper = lower ? 0 : 255;
//       if(vlp[i].first != isUpper) {
//         toReprocess[neighborId] = 255;
//         hasChanged = true;
//       }
//     }
//     if(hasChanged) { // immediatly reprocess this vertex
//       toReprocess[vertexId] = 255;
//     }
//   }
// }

// template <typename scalarType, typename offsetType>
// bool ttk::ProgressiveTopology::reinterpolateNeighbor(
//   const SimplexId vertexId,
//   const int localNeighborId,
//   int currentDecimationLevel,
//   double eps,
//   const std::vector<polarity> &isNew,
//   std::vector<polarity> &toProcess,
//   std::vector<polarity> &toReprocess,
//   std::vector<std::pair<polarity, polarity>> &vlp,
//   const scalarType *const scalarField,
//   scalarType *fakeScalars,
//   const offsetType *const offsets,
//   int *monotonyOffsets) {

//   SimplexId neighborId=-1;
//   SimplexId neighborIdDeeper=-1;
//   multiresTriangulation_.getVertexNeighbor(vertexId,i,neighborId);
//   multiresTriangulation_.getVertexNeighborAtDecimation(vertexId,i,neighborId,currentDecimationLevel-1);
//   fakeScalars[vertexId] = scalars[vertexId];
//   bool hasChanged = false;
//   if(vlp.empty()) { // need to build the link polarity and flag each one of
//                     // its neighbors
//     buildVertexLinkPolarity(vertexId, vertexLinkPolarity[vertexId],
//     fakeScalars,
//                             offsets, monotonyOffsets);
//     for(size_t i = 0; i < vlp.size(); i++) {
//       SimplexId neighborId = -1;
//       multiresTriangulation_.getVertexNeighbor(vertexId, i, neighborId);
//       toReprocess[neighborId] = 255;
//     }
//     toReprocess[vertexId] = 255;

//   } else {
//     for(size_t i = 0; i < vlp.size(); i++) {
//       SimplexId neighborId = -1;
//       multiresTriangulation_.getVertexNeighbor(vertexId, i, neighborId);
//       const bool lower
//         = ((fakeScalars[neighborId] < fakeScalars[vertexId])
//            || (fakeScalars[neighborId] == fakeScalars[vertexId]
//                && ((monotonyOffsets[neighborId] <
//                monotonyOffsets[vertexId])
//                    || (monotonyOffsets[neighborId] ==
//                    monotonyOffsets[vertexId]
//                        && offsets[neighborId] < offsets[vertexId]))));

//       const polarity isUpper = lower ? 0 : 255;
//       if(vlp[i].first != isUpper) {
//         toReprocess[neighborId] = 255;
//         hasChanged = true;
//       }
//     }
//     if(hasChanged) { // immediatly reprocess this vertex
//       toReprocess[vertexId] = 255;
//     }
//   }
// }
//

template <typename scalarType>
void ttk::ProgressiveTopology::generateAdaptiveMeshVisualisation(
  std::vector<std::pair<SimplexId, SimplexId>> &edgeList,
  const scalarType *scalars,
  const scalarType *fakeScalars) {

  multiresTriangulation_.setDecimationLevel(startingDecimationLevel_);
  for(int decimationLevel = startingDecimationLevel_;
      decimationLevel >= stoppingDecimationLevel_; decimationLevel--) {
    multiresTriangulation_.setDecimationLevel(decimationLevel);
    SimplexId nDecVerts = multiresTriangulation_.getDecimatedVertexNumber();

    for(size_t localId = 0; localId < nDecVerts; localId++) {
      SimplexId globalId
        = multiresTriangulation_.localToGlobalVertexId(localId);
      if(abs((double)fakeScalars[globalId] - (double)scalars[globalId]) < 1e-6
         or 1) {
        for(int i = 0;
            i < multiresTriangulation_.getVertexNeighborNumber(globalId); i++) {
          SimplexId neighborId = -1;
          multiresTriangulation_.getVertexNeighbor(globalId, i, neighborId);
          if(abs((double)fakeScalars[neighborId] - (double)scalars[neighborId])
             < 1e-5) {
            edgeList.push_back({globalId, neighborId});
            // std::cout << "Added edge " << globalId << "-" << neighborId;
            // std::cout << "  (" << scalars[globalId] << "-"
            //           << fakeScalars[neighborId] << ")";
            // std::cout << "  (" << scalars[globalId] << "-"
            //           << fakeScalars[neighborId] << ")" << std::endl;
          }
        }
      }
    }
  }
  std::cout << "mesh generated" << std::endl;
}

template <typename ScalarType, typename offsetType>
int ttk::ProgressiveTopology::updateGlobalPolarityAdaptive(
  double eps,
  std::vector<polarity> &isNew,
  // std::vector<std::vector<std::vector<std::pair<polarity, polarity>>>>
  std::vector<std::vector<std::pair<polarity, polarity>>> &vertexLinkPolarity,
  std::vector<polarity> &toProcess,
  std::vector<polarity> &toReprocess,
  // std::vector<std::tuple<SimplexId, double, int>> &globalErrorList,
  const ScalarType *const scalarField,
  ScalarType *fakeScalars,
  const offsetType *const offsets,
  int *monotonyOffsets) {

  Timer tm;
  const auto nDecVerts = multiresTriangulation_.getDecimatedVertexNumber();

  // std::vector<std::vector<std::pair<SimplexId, double>>> localErrorList(
  //   threadNumber_);
  // if(preallocateMemoryErrorList_) {
  // #ifdef TTK_ENABLE_OPENMP
  // #pragma omp parallel for num_threads(threadNumber_)
  // #endif // TTK_ENABLE_OPENMP
  //   for(int i = 0; i < localErrorList.size(); i++) {
  //     localErrorList[i].reserve(100000);
  //   }
  // }

  ScalarType *deltaDummy{};

#ifdef TTK_ENABLE_OPENMP
  const auto tid = omp_get_thread_num();
#else
  const auto tid = 0;
#endif // TTK_ENABLE_OPENMP

  double tmono = tm.getElapsedTime();

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(SimplexId localId = 0; localId < nDecVerts; localId++) {
    SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(localId);
    if(!isNew[globalId]) {
      // copyPolarity(
      //   vertexLinkPolarity[decimationLevel_ + 1 - stoppingDecimationLevel_]
      //                     [globalId],
      //   vertexLinkPolarity[decimationLevel_ - stoppingDecimationLevel_]
      //                     [globalId]);
      // int monotonyChanged = getMonotonyChangeByOldPointCPAdaptive(
      //   globalId, eps, isNew, toProcess, toReprocess,
      //   vertexLinkPolarity[decimationLevel_ - stoppingDecimationLevel_]
      //                     [globalId],
      //   scalarField, fakeScalars, deltaDummy, deltaDummy,
      //   localErrorList[omp_get_thread_num()], offsets, monotonyOffsets);
      int monotonyChanged = getMonotonyChangeByOldPointCPAdaptive(
        globalId, eps, isNew, toProcess, toReprocess,
        vertexLinkPolarity[globalId], scalarField, fakeScalars, deltaDummy,
        deltaDummy, /*localErrorList[omp_get_thread_num()],*/ offsets,
        monotonyOffsets);
      // if(monotonyChanged == -1) {
      //   return -1;
      // }
    }
  }
  if(debugLevel_ > 3) {
    std::cout << "MONONTONY " << tm.getElapsedTime() - tmono << std::endl;
  }

  // double terrorlist = 0;
  // if(debugLevel_ > 3) {
  // terrorlist = tm.getElapsedTime();
  // }
  // size_t globalErrorListSize = globalErrorList.size();
  // size_t globalErrorListNewSize = 0;
  // std::vector<size_t> localErrorListSizes(localErrorList.size());

  // for(int i = 0; i < localErrorList.size(); i++) {
  //   localErrorListSizes[i] = globalErrorListNewSize;
  //   globalErrorListNewSize += localErrorList[i].size();
  // }
  // globalErrorListNewSize += globalErrorListSize;
  // globalErrorList.resize(globalErrorListNewSize);

  // #ifdef TTK_ENABLE_OPENMP
  // #pragma omp parallel for num_threads(threadNumber_)
  // #endif // TTK_ENABLE_OPENMP
  // for(int i = 0; i < localErrorList.size(); i++) {
  //   for(int j = 0; j < localErrorList[i].size(); j++) {
  //     globalErrorList[globalErrorListSize + localErrorListSizes[i] + j]
  //       = std::make_tuple(localErrorList[i][j].first,
  //                         localErrorList[i][j].second, decimationLevel_);
  //   }
  // }

  // if(debugLevel_ > 3) {
  //   std::cout << "global size " << globalErrorListNewSize << " in "
  //             << tm.getElapsedTime() - terrorlist << std::endl;
  // }

  double t_critical = tm.getElapsedTime();
// second Loop  process or reprocess
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(int i = 0; i < multiresTriangulation_.getDecimatedVertexNumber(); i++) {
    SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(i);
    if(isNew[globalId]) { // new point
      if(decimationLevel_ > stoppingDecimationLevel_ || isResumable_) {
        buildVertexLinkPolarityAdaptive(globalId, vertexLinkPolarity[globalId],
                                        fakeScalars, offsets, monotonyOffsets);
        // buildVertexLinkPolarityAdaptive(
        //   globalId,
        //   vertexLinkPolarity[decimationLevel_ - stoppingDecimationLevel_]
        //                     [globalId],
        //   fakeScalars, offsets, monotonyOffsets);
      }
      // if(!toProcess[globalId]){
      //   nbNewTIs_++;
      // }
      isNew[globalId] = 0;

    } else { // old point
      if(toReprocess[globalId]) {
        updateLinkPolarityPonctual(vertexLinkPolarity[globalId]);
        // updateLinkPolarityPonctual(
        //   vertexLinkPolarity[decimationLevel_ - stoppingDecimationLevel_]
        //                     [globalId]);
        toProcess[globalId] = 255; // mark for processing
        toReprocess[globalId] = 0;
      }
      // else{
      //   nbOldTIs_++;
      // }
    }
  } // end for openmp

  // std::cout << "POLARITY UPDATE " << tm.getElapsedTime() - t_critical
  //           << std::endl;
  // if(debugLevel_ > 4) {
  //   for(int i = 0; i < multiresTriangulation_.getDecimatedVertexNumber();
  //   i++) {
  //     SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(i);
  //     printPolarity(
  //       isNew, globalId,
  //       vertexLinkPolarity[decimationLevel_ - stoppingDecimationLevel_],
  //       scalarField, fakeScalars, offsets, monotonyOffsets);
  //   }
  // }
  // for(int i = 0; i < multiresTriangulation_.getDecimatedVertexNumber();
  // i++)
  // {
  //   SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(i);
  //   if(isNew[globalId]) { // new point
  //     isNew[globalId] = 0;
  //   }
  // }
  // printPolarity(isNew, 3128,
  //               vertexLinkPolarity[decimationLevel_ -
  //               stoppingDecimationLevel_], fakeScalars, offsets,
  //               monotonyOffsets, true);
  if(debugLevel_ > 3) {
    printMsg(
      "Polarity Update", 1, tm.getElapsedTime() - t_critical, threadNumber_);
  }
  return 0;
}

template <typename ScalarType, typename offsetType>
void ttk::ProgressiveTopology::computeCriticalPointsAdaptive(
  std::vector<std::vector<std::pair<polarity, polarity>>> &vertexLinkPolarity,
  std::vector<polarity> &toPropageMin,
  std::vector<polarity> &toPropageMax,
  std::vector<polarity> &toProcess,
  std::vector<polarity> &toReprocess,
  std::vector<DynamicTree> &link,
  std::vector<uint8_t> &vertexLink,
  VLBoundaryType &vertexLinkByBoundaryType,
  std::vector<std::vector<SimplexId>> &saddleCCMin,
  std::vector<std::vector<SimplexId>> &saddleCCMax,
  const ScalarType *const scalarField,
  ScalarType *fakeScalars,
  const offsetType *const offsets,
  int *monotonyOffsets) {

  Timer tm;
  const auto nDecVerts = multiresTriangulation_.getDecimatedVertexNumber();
  // SimplexId nbComputations = 0;

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(int i = 0; i < multiresTriangulation_.getDecimatedVertexNumber(); i++) {
    SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(i);

    if(toProcess[globalId]) {
      // nbComputations++;
      getCriticalTypeAdaptive(globalId, vertexLinkPolarity[globalId],
                              vertexLink[globalId], link[globalId],
                              vertexLinkByBoundaryType, fakeScalars, offsets,
                              monotonyOffsets);
      getValencesFromLink(globalId, vertexLinkPolarity[globalId],
                          link[globalId], toPropageMin, toPropageMax,
                          saddleCCMin, saddleCCMax);
    }
  } // end for openmp

  if(debugLevel_ > 3) {
    printMsg(
      "Critical Points Computation", 1, tm.getElapsedTime(), threadNumber_);
  }
  // std::cout << "computed " << nbComputations << " / "
  //           << multiresTriangulation_.getVertexNumber() << std::endl;
}

template <typename ScalarType, typename offsetType>
void ttk::ProgressiveTopology::reComputeCriticalPointsAdaptive(
  std::vector<std::vector<std::pair<polarity, polarity>>> &vertexLinkPolarity,
  std::vector<polarity> &toPropageMin,
  std::vector<polarity> &toPropageMax,
  std::vector<polarity> &toProcess,
  std::vector<polarity> &toReprocess,
  std::vector<DynamicTree> &link,
  std::vector<uint8_t> &vertexLink,
  VLBoundaryType &vertexLinkByBoundaryType,
  std::vector<std::vector<SimplexId>> &saddleCCMin,
  std::vector<std::vector<SimplexId>> &saddleCCMax,
  const ScalarType *const scalarField,
  ScalarType *fakeScalars,
  const offsetType *const offsets,
  int *monotonyOffsets) {

  Timer tm;
  const auto nDecVerts = multiresTriangulation_.getDecimatedVertexNumber();

  // SimplexId nbComputations = 0;
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(int i = 0; i < multiresTriangulation_.getDecimatedVertexNumber(); i++) {
    SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(i);

    if(toReprocess[globalId]) {
      // nbComputations++;
      if(!toProcess[globalId]) {
        getCriticalTypeAdaptive(globalId, vertexLinkPolarity[globalId],
                                vertexLink[globalId], link[globalId],
                                vertexLinkByBoundaryType, fakeScalars, offsets,
                                monotonyOffsets);
      } else {

        // if(globalId == 3128) {
        //   std::cout << "flag pol changes " << std::endl;
        // }
        // flagPolarityChanges(globalId, vertexLinkPolarity[globalId],
        // fakeScalars,
        //                     offsets, monotonyOffsets);
        resetCriticalType(globalId, link[globalId],
                          vertexLinkPolarity[globalId],
                          vertexLinkByBoundaryType[vertexLink[globalId]]);
      }
      getValencesFromLink(globalId, vertexLinkPolarity[globalId],
                          link[globalId], toPropageMin, toPropageMax,
                          saddleCCMin, saddleCCMax);
      toReprocess[globalId] = 0;
      toProcess[globalId] = 255;
    }
  } // end for openmp

  if(debugLevel_ > 3) {
    printMsg(
      "Critical Points Computation", 1, tm.getElapsedTime(), threadNumber_);
  }
  // std::cout << "computed " << nbComputations << " / "
  //           << multiresTriangulation_.getVertexNumber() << std::endl;
}

template <typename ScalarType, typename offsetType>
void ttk::ProgressiveTopology::updateCriticalPointsAdaptive(
  std::vector<std::vector<std::pair<polarity, polarity>>> &vertexLinkPolarity,
  std::vector<polarity> &toPropageMin,
  std::vector<polarity> &toPropageMax,
  std::vector<polarity> &toProcess,
  std::vector<polarity> &toReprocess,
  std::vector<DynamicTree> &link,
  std::vector<uint8_t> &vertexLink,
  VLBoundaryType &vertexLinkByBoundaryType,
  std::vector<std::vector<SimplexId>> &saddleCCMin,
  std::vector<std::vector<SimplexId>> &saddleCCMax,
  const ScalarType *const scalarField,
  ScalarType *fakeScalars,
  const offsetType *const offsets,
  int *monotonyOffsets) {

  Timer tm;
  const auto nDecVerts = multiresTriangulation_.getDecimatedVertexNumber();

  // SimplexId nbComputations = 0;
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(int i = 0; i < multiresTriangulation_.getDecimatedVertexNumber(); i++) {
    SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(i);

    if(toReprocess[globalId]) {
      // nbComputations++;
      if(!toProcess[globalId]) {
        getCriticalTypeAdaptive(globalId, vertexLinkPolarity[globalId],
                                vertexLink[globalId], link[globalId],
                                vertexLinkByBoundaryType, fakeScalars, offsets,
                                monotonyOffsets);
        toProcess[globalId] = 255;
      } else {
        updateDynamicLink(link[globalId], vertexLinkPolarity[globalId],
                          vertexLinkByBoundaryType[vertexLink[globalId]]);
      }
      getValencesFromLink(globalId, vertexLinkPolarity[globalId],
                          link[globalId], toPropageMin, toPropageMax,
                          saddleCCMin, saddleCCMax);
      toReprocess[globalId] = 0;
    }
  } // end for openmp

  if(debugLevel_ > 3) {
    printMsg("Critical Points Update", 1, tm.getElapsedTime(), threadNumber_);
  }
  // std::cout << "computed " << nbComputations << " / "
  //           << multiresTriangulation_.getVertexNumber() << std::endl;
}

template <typename scalarType, typename offsetType>
bool ttk::ProgressiveTopology::processBucketOnLastLevelAdaptive(
  double eps,
  size_t &start_index,
  std::vector<std::tuple<SimplexId, double, int>> &globalErrorList,
  std::vector<polarity> &isNew,
  std::vector<std::vector<std::vector<std::pair<polarity, polarity>>>>
    &vertexLinkPolarity,
  std::vector<polarity> &toProcess,
  std::vector<polarity> &toReprocess,
  const scalarType *const scalarField,
  scalarType *fakeScalars,
  scalarType *deltaField,
  scalarType *deltaFieldDynamic,
  const offsetType *const offsets,
  int *monotonyOffsets) {

  Timer tm{};

  printMsg("Processing bucket on last level "
           + std::to_string(decimationLevel_));
  std::tuple<SimplexId, double, int> targetEpsilonTuple
    = std::make_tuple<SimplexId, double, int>(0, (double)eps, 0);

  auto cmp = [](const std::tuple<SimplexId, double, int> &t0,
                const std::tuple<SimplexId, double, int> &t1) {
    return std::get<1>(t0) > std::get<1>(t1);
  };

  auto nDecVerts = multiresTriangulation_.getDecimatedVertexNumber();

  std::cout << "test0" << std::endl;
  auto itBucketEnd
    = std::lower_bound(globalErrorList.begin() + start_index,
                       globalErrorList.end(), targetEpsilonTuple, cmp);
  std::cout << "test1" << std::endl;

  double tm_elapsed = 0;
  if(debugLevel_ > 3) {
    tm_elapsed = tm.getElapsedTime();
    std::cout << " bucket in " << tm.getElapsedTime() << std::endl;
  }

  // printPolarity(isNew, 390727,
  //               vertexLinkPolarity[decimationLevel_ -
  //               stoppingDecimationLevel_], scalarField, fakeScalars, offsets,
  //               monotonyOffsets, true);
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(auto it2 = globalErrorList.begin() + start_index; it2 < itBucketEnd;
      it2++) {

    SimplexId vertexId = std::get<0>(*it2);

    // isUpdated[vertexId] = 255;
    toReprocess[vertexId] = 255;

    fakeScalars[vertexId] = scalarField[vertexId];
    // monotonyOffsets[vertexId] = 0;

    auto &vlp = vertexLinkPolarity[decimationLevel_ - stoppingDecimationLevel_]
                                  [vertexId];
    if(vlp.empty()) {
      const SimplexId neighborNumber
        = multiresTriangulation_.getVertexNeighborNumber(vertexId);
      for(SimplexId i = 0; i < neighborNumber; i++) {
        SimplexId neighborId = -1;
        multiresTriangulation_.getVertexNeighbor(vertexId, i, neighborId);
        // isUpdated[neighborId] = 255; // needed
        toReprocess[neighborId] = 255;
        // std::cout << decimationLevel_ << " " << vertexId << " flags "
        //           << neighborId << " was processed ? "
        //           << (int)toProcess[neighborId] << std::endl;
      }
    } else {
      bool hasChanged = false;
      for(size_t i = 0; i < vlp.size(); i++) {
        SimplexId neighborId = -1;
        multiresTriangulation_.getVertexNeighbor(vertexId, i, neighborId);
        const bool lower
          = ((fakeScalars[neighborId] < fakeScalars[vertexId])
             || (fakeScalars[neighborId] == fakeScalars[vertexId]
                 && ((monotonyOffsets[neighborId] < monotonyOffsets[vertexId])
                     || (monotonyOffsets[neighborId]
                           == monotonyOffsets[vertexId]
                         && offsets[neighborId] < offsets[vertexId]))));

        const polarity isUpper = lower ? 0 : 255;
        if(vlp[i].first != isUpper) {
          toReprocess[neighborId] = 255;
          // std::cout << decimationLevel_ << " " << vertexId << " flags "
          //           << neighborId << " was processed ? "
          //           << (int)toProcess[neighborId] << std::endl;
          hasChanged = true;
        }
      }
      if(hasChanged) { // immediatly reprocess this vertex
        toReprocess[vertexId] = 255;
      }
    }
  }
  if(debugLevel_ > 3) {
    std::cout << " first loop in " << tm.getElapsedTime() << std::endl;
    tm_elapsed = tm.getElapsedTime();
  }
  // printPolarity(isNew, 390727,
  //               vertexLinkPolarity[decimationLevel_ -
  //               stoppingDecimationLevel_], scalarField, fakeScalars, offsets,
  //               monotonyOffsets, true);

  SimplexId nbRepro = 0;
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(int i = 0; i < multiresTriangulation_.getDecimatedVertexNumber(); i++) {
    SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(i);

    if(toReprocess[globalId]) {
      nbRepro++;
      if(toProcess[globalId]) { // already processed: flag its polarity changes
        flagPolarityChanges(
          globalId,
          vertexLinkPolarity[decimationLevel_ - stoppingDecimationLevel_]
                            [globalId],
          fakeScalars, offsets, monotonyOffsets);
      } else { // never processed, simply build or update its polarity
        updateLinkPolarityAdaptive(
          globalId,
          vertexLinkPolarity[decimationLevel_ - stoppingDecimationLevel_]
                            [globalId],
          fakeScalars, offsets, monotonyOffsets);
      }
    }
  } // end for openmp
  std::cout << "nbrepro " << nbRepro << " outta "
            << multiresTriangulation_.getDecimatedVertexNumber() << " "
            << (double)nbRepro
                 / multiresTriangulation_.getDecimatedVertexNumber() * 100
            << std::endl;
  if(debugLevel_ > 3) {
    std::cout << " second loop in " << tm.getElapsedTime() << std::endl;
    tm_elapsed = tm.getElapsedTime();
  }
  // printPolarity(isNew, 390727,
  //               vertexLinkPolarity[decimationLevel_ -
  //               stoppingDecimationLevel_], scalarField, fakeScalars, offsets,
  //               monotonyOffsets, true);

  // start_index = itBucketEnd - globalErrorList.begin();
  globalErrorList.erase(globalErrorList.begin(), itBucketEnd);

  if(debugLevel_ > 4) {
    std::cout << "testing polarity" << std::endl;
    for(int i = 0; i < multiresTriangulation_.getDecimatedVertexNumber(); i++) {
      SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(i);
      bool error = printPolarity(
        isNew, globalId,
        vertexLinkPolarity[decimationLevel_ - stoppingDecimationLevel_],
        scalarField, fakeScalars, offsets, monotonyOffsets);
      if(error) {
        return true;
      }
    }
  }
  // for(int i = 0; i < multiresTriangulation_.getDecimatedVertexNumber(); i++)
  // {
  //   SimplexId globalId = multiresTriangulation_.localToGlobalVertexId(i);
  //   if(scalarField[globalId] != fakeScalars[globalId]) {
  //     std::cout << " vertex " << globalId << " is still folded "
  //               << fakeScalars[globalId] << " " << scalarField[globalId]
  //               << std::endl;
  //   }
  // }

  return false;
}

template <typename scalarType>
int ttk::ProgressiveTopology::sortPersistenceDiagramAdaptive(
  std::vector<PersistencePair> &diagram,
  const scalarType *const scalars,
  scalarType *fakeScalars,
  const SimplexId *const offsets,
  int *monotonyOffsets) const {
  auto cmp = [scalars, fakeScalars, offsets, monotonyOffsets](
               const PersistencePair &pA, const PersistencePair &pB) {
    const ttk::SimplexId a = pA.birth;
    const ttk::SimplexId b = pB.birth;

    return ((fakeScalars[a] < fakeScalars[b])
            || (fakeScalars[a] == fakeScalars[b]
                && ((monotonyOffsets[a] < monotonyOffsets[b])
                    || (monotonyOffsets[a] == monotonyOffsets[b]
                        && offsets[a] < offsets[b]))));
  };

  std::sort(diagram.begin(), diagram.end(), cmp);

  return 0;
}

template <typename scalarType>
int ttk::ProgressiveTopology::computeProgressivePDAdaptive(
  std::vector<PersistencePair> &CTDiagram,
  const scalarType *scalars,
  scalarType *const fakeScalars,
  SimplexId *const outputOffsets,
  int *const outputMonotonyOffsets,
  const SimplexId *offsets) {

  int ret = -1;
  printMsg("Adaptive Persistence Diagram computation");
  ret = executeCPProgressiveAdaptive(
    scalars, fakeScalars, outputOffsets, outputMonotonyOffsets, offsets);
  CTDiagram = std::move(CTDiagram_);
  CTDiagram_.clear();

  return ret;
}
