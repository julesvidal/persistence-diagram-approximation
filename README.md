Fast Approximation of Persistence Diagrams with Guarantees
=========================================================

This repository contains the proposed implementation described in the publication:


Fast Approximation of Persistence Diagrams with Guarantees

Jules Vidal and Julien Tierny

IEEE Symposium on Large Data Analysis and Visualization (LDAV), 2021

The paper is available on arxiv: https://arxiv.org/abs/2108.05766

## Examine the code

The implementation code is based on The Topology ToolKit
(https://topology-tool-kit.github.io/).

Most of the added code is available in
the archive under the src/core/base/progressiveTopology directory and is
located in the file ProgressiveTopologyAdaptive.h

## Build & Install

The present build instructions have been tested onto an vanilla installation
of Ubuntu 20.04 Linux distribution. 
Other distributions may need specific instructions.

The installation necessitated to build either ParaView 5.8.1 (or later) of VTK 9.
We propose in these instructions to build ParaView.
The instructions are identical to the installation tutorial of TTK's website.

The first step is to install the software dependencies. Copy the
following shell statement into a terminal (omit the $ character):

```bash
$ sudo apt update
$ sudo apt-get install cmake-qt-gui libboost-system-dev libpython3.8-dev libxt-dev build-essential python-numpy
$ sudo apt-get install qt5-default qttools5-dev libqt5x11extras5-dev libqt5svg5-dev qtxmlpatterns5-dev-tools 
```

This might take some time, depending on the number of already
installed packages on the target system.

Now, move the compressed archive to your working directory and
decompress it.

```bash
$ unzip implementation.zip
```

### Build ParaView

Download ParaView's source code:
https://www.paraview.org/paraview-downloads/download.php?submit=Download&version=v5.8&type=source&os=Sources&downloadFile=ParaView-v5.8.1.tar.xz

Extract it in your working directory and create the build folder:

```bash
$ tar xJf ParaView-v5.8.1.tar.xz
$ cd ParaView-v5.8.1
$ mkdir build && cd build
```

Now launch the following command:
```bash
$ cmake-gui ../
```

The configuration window opens. Click on the "Configure" button to proceed.
Once the configuration is finished, please tick the "Advanced" check box and
set the following variables as follows (required for TTK's installation):

. CMAKE_BUILD_TYPE=Release
. PARAVIEW_USE_PYTHON=ON
. PARAVIEW_INSTALL_DEVELOPMENT_FILES=ON
. PARAVIEW_PYTHON_VERSION=3

Next, click on the "Generate" button and close the configuration window when
the generation is completed.

Run the compilation, replace 4 with the number of available cores on your
system (it may take some time):

```bash
$ make -j 4 4
```

Once ParaView is built, install it on your system:

```bash
$ sudo make install
```

### Build our code

Now you can build our source code. Note that, in order to speed up the
compilation process, we stripped TTK off the modules
that were not required for our implementation.

Move to the root of your working directory and create the build folder.
Then run cmake:

```bash
$ cd ../../
$ mkdir build && cd build
& cmake-gui ./src/
```
Click "Configure".
Be sure to select the following option in order to unlock the full performance
of TTK:

. TTK_ENABLE_KAMIKAZE=ON

Next, click on the "Generate" button and close the configuration window when
the generation is completed.


To build the proposed implementation and install it, please use the following
commands:

```bash
$ make -j 4
$ sudo make install
```

## Fetching the data

Download the compressed archive containing the data, available at this link:

https://nuage.lip6.fr/s/qZpXrbnTbxSnYzq/download

Decompress the archive and move the resulting `data` folder into the
`persistence-diagram-approximation` folder.

```bash
$ unzip data.zip
$ mv data/ persistence-diagram-approximation/
```


## Reproducing the paper results

The *data* folder contains all the datasets used in this submission.
The *scripts* folder contains Shell and Python scripts used to generate the
Table 2 of the paper.

To reproduce those results, please move to the *scripts* folder and run the
benchmarks: 

```bash
$ cd scripts/
$ bash benchs.sh
```

If a dataset is too large in practice for the memory of your computer, or
takes too long to process, you can filter the list of datasets used in the
benchmarks.
Just edit the `data_list` file in the script folder, replace 1 by 0 to filter
a dataset out of the benchmarks.

Once the computation is done, you may run the Python scripts to generate the
table, both as a .csv and Latex files.

```bash
$ python3 generateTable2.py
```

The files `table2.csv` and `table2.tex` are created in the same folder.


## Using the standalone executables

The above installation has provided you with the standalone command
**ttkPersistenceDiagramCmd** for the computation of a persistence diagram.

It can be used directly to try out our implementation of our
method on your own data.

The input data must be a scalar field defined on a regular grid, in the .vti format.

Type the command without any argument to get see how to use it:

```bash
[CMD] [ERROR] Missing mandatory argument:
[CMD] [ERROR]    -i <{Input data-sets (*.vti, *vtu, *vtp)}>
[CMD] [ERROR]
[CMD] [ERROR] Usage:
[CMD] [ERROR]   ttkPersistenceDiagramCmd
[CMD] [ERROR] Argument(s):
[CMD] [ERROR]    [-d <Global debug level (default: 3)>]
[CMD] [ERROR]    [-t <Global thread number (default: 12)>]
[CMD] [ERROR]    -i <{Input data-sets (*.vti, *vtu, *vtp)}>
[CMD] [ERROR]    [-a <{Input array names}>]
[CMD] [ERROR]    [-o <Output file prefix (no extension) (default: `output')>]
[CMD] [ERROR]    [-B <Method (0:FTM, 1: Hierarchical) (default: 0)>]
[CMD] [ERROR]    [-S <Starting Resolution Level for the hierarchy (-1: finest level) (default: 0)>]
[CMD] [ERROR]    [-E <Stopping Resolution Level for for the hierarchy (-1: finest level) (default: -1)>]
[CMD] [ERROR]    [-T <Time limit for progressive method (default: 0.000000)>]
[CMD] [ERROR]    [-A <Use Approximate approach (default: 0)>]
[CMD] [ERROR]    [-e <Epsilon (default: 0.000000)>]
[CMD] [ERROR] Option(s):
[CMD] [ERROR]    [-l: List available arrays (default: 0)]
```

For instance, the following command uses the approach by Vidal et al.;

```bash
$ ttkPersistenceDiagramCmd -i input.vti -t 1 -B 1 -A 0
```
The following command uses our approximate approach with an epsilon of 0.05:

```bash
$ ttkPersistenceDiagramCmd -i input.vti -t 1 -B 1 -A 1 -e 0.05
```

And the following uses the default approach of TTK:
```bash 
$ ttkPersistenceDiagramCmd -i input.vti -t 1 -B 0
```

The command creates 3 outputs (the prefix "output" can be changed with the -o
parameter):

- The file 'output_port_0.vtu' contains the computed persistence diagram.
- The file 'output_port_1.vti' contains the related approximation of the
  scalar field (named "Cropped").
- The file 'output_port_2.vtu' contains the visual indications about the
  uncertainty in the diagrams.

These files can be visualized with ParaView:
```bash
$ paraview output_port_0.vtu
```
