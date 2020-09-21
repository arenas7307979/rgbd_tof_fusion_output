Sophus
======
Commit: 05f9645077c4e59ca9aa05bb0d0eaf0dd83d2b26
------------------------------------------------
Ref: https://github.com/strasdat/Sophus
---------------------------------------
Overview
--------

This is a c++ implementation of Lie groups commonly used for 2d and 3d
geometric problems (i.e. for Computer Vision or Robotics applications).
Among others, this package includes the special orthogonal groups SO(2) and
SO(3) to present rotations in 2d and 3d as well as the special Euclidean group
SE(2) and SE(3) to represent rigid body transformations (i.e. rotations and
translations) in 2d and 3d.

API documentation: https://strasdat.github.io/Sophus/

Cross platform support
----------------------

Sophus compiles with clang and gcc on Linux and OS X as well as msvc on Windows.
The specific compiler and operating system versions which are supported are
the ones which are used in the Continuous Integration (CI): See TravisCI_ and
AppVeyor_ for details.

However, it should work (with no to minor modification) on many other
modern configurations as long they support c++11, CMake, and Eigen 3.X.
