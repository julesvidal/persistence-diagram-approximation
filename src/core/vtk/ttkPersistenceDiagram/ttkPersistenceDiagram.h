// \ingroup vtk
/// \class ttkPersistenceDiagram
/// \author Guillaume Favelier <guillaume.favelier@lip6.fr>
/// \author Julien Tierny <julien.tierny@lip6.fr>
/// \date September 2016.
///
/// \brief TTK VTK-filter for the computation of persistence diagrams.
///
/// This filter computes the persistence diagram of the extremum-saddle pairs
/// of an input scalar field. The X-coordinate of each pair corresponds to its
/// birth, while its smallest and highest Y-coordinates correspond to its birth
/// and death respectively.
///
/// In practice, the diagram is represented by a vtkUnstructuredGrid. Each
/// vertex of this mesh represent a critical point of the input data. It is
/// associated with point data (vertexId, critical type). Each vertical edge
/// of this mesh represent a persistence pair. It is associated with cell data
/// (persistence of the pair, critical index of the extremum of the pair).
///
/// Persistence diagrams are useful and stable concise representations of the
/// topological features of a data-set. It is useful to fine-tune persistence
/// thresholds for topological simplification or for fast similarity
/// estimations for instance.
///
/// \param Input Input scalar field, either 2D or 3D, regular grid or
/// triangulation (vtkDataSet)
/// \param Output Output persistence diagram (vtkUnstructuredGrid)
///
/// This filter can be used as any other VTK filter (for instance, by using the
/// sequence of calls SetInputData(), Update(), GetOutput()).
///
/// See the related ParaView example state files for usage examples within a
/// VTK pipeline.
///
/// \b Related \b publication \n
/// "Computational Topology: An Introduction" \n
/// Herbert Edelsbrunner and John Harer \n
/// American Mathematical Society, 2010
///
/// Two backends are available for the computation:
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
/// \sa ttkFTMTreePP
/// \sa ttkPersistenceCurve
/// \sa ttkScalarFieldCriticalPoints
/// \sa ttkTopologicalSimplification
/// \sa ttk::PersistenceDiagram

#pragma once

// VTK includes
#include <vtkDataArray.h>
#include <vtkUnstructuredGrid.h>

// VTK Module
#include <ttkPersistenceDiagramModule.h>

// ttk code includes
#include <PersistenceDiagram.h>
#include <ttkAlgorithm.h>
#include <ttkMacros.h>

class TTKPERSISTENCEDIAGRAM_EXPORT ttkPersistenceDiagram
  : public ttkAlgorithm,
    protected ttk::PersistenceDiagram {

public:
  static ttkPersistenceDiagram *New();

  vtkTypeMacro(ttkPersistenceDiagram, ttkAlgorithm);

  vtkSetMacro(ForceInputOffsetScalarField, bool);
  vtkGetMacro(ForceInputOffsetScalarField, bool);

  vtkSetMacro(ComputeSaddleConnectors, bool);
  vtkGetMacro(ComputeSaddleConnectors, bool);

  vtkSetMacro(ShowInsideDomain, bool);
  vtkGetMacro(ShowInsideDomain, bool);

  ttkSetEnumMacro(BackEnd, BACKEND);
  vtkGetEnumMacro(BackEnd, BACKEND);

  vtkGetMacro(StartingResolutionLevel, int);
  vtkSetMacro(StartingResolutionLevel, int);

  vtkGetMacro(StoppingResolutionLevel, int);
  vtkSetMacro(StoppingResolutionLevel, int);

  vtkGetMacro(IsResumable, bool);
  vtkSetMacro(IsResumable, bool);

  vtkGetMacro(TimeLimit, double);
  vtkSetMacro(TimeLimit, double);

  vtkGetMacro(UseAdaptive, bool);
  vtkSetMacro(UseAdaptive, bool);

  vtkGetMacro(Epsilon, double);
  vtkSetMacro(Epsilon, double);

  vtkGetMacro(NbBuckets, int);
  vtkSetMacro(NbBuckets, int);

protected:
  ttkPersistenceDiagram();

  int RequestData(vtkInformation *request,
                  vtkInformationVector **inputVector,
                  vtkInformationVector *outputVector) override;

  int FillInputPortInformation(int port, vtkInformation *info) override;
  int FillOutputPortInformation(int port, vtkInformation *info) override;

private:
  template <typename scalarType, typename triangulationType>
  int dispatch(vtkUnstructuredGrid *outputCTPersistenceDiagram,
               vtkUnstructuredGrid *outputBounds,
               vtkDataArray *const inputScalarsArray,
               const scalarType *const inputScalars,
               vtkDataArray *const outputScalarsArray,
               scalarType *outputScalars,
               SimplexId *outputOffsets,
               int *outputMonotonyOffsets,
               const SimplexId *const inputOrder,
               const triangulationType *triangulation);

  template <typename scalarType, typename triangulationType>
  int setPersistenceDiagram(vtkUnstructuredGrid *outputCTPersistenceDiagram,
                            const std::vector<ttk::PersistencePair> &diagram,
                            vtkDataArray *inputScalarsArray,
                            const scalarType *const inputScalars,
                            const triangulationType *triangulation) const;

  template <typename scalarType, typename triangulationType>
  int drawBottleneckBounds(vtkUnstructuredGrid *outputBounds,
                           const std::vector<ttk::PersistencePair> &diagram,
                           vtkDataArray *inputScalarsArray,
                           const scalarType *const outputScalars,
                           const scalarType *const inputScalars,
                           const triangulationType *triangulation) const;

  bool ForceInputOffsetScalarField{false};
  bool ShowInsideDomain{false};
};
