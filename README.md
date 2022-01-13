# JRC Changes to original repo
This will list the changes added to the project by people at JRC

## Install and run

The code has several dependencies and uses C, CUDA and JAVA languages. Unfortunately, there was no project configuration, so we used Maven to build the java code and addeed a CMake build system on top of the C/CUDA code.

First let's install CUDA and the examples

    $ sudo apt-get install maven openjdk-8-jdk default-jdk intel-mkl jblas
    $ git clone https://github.com/NVIDIA/cuda-samples.git

Be sure to clone the examples in the same directory of the source code, because we define their location in the CMakeFiles.txt as:

> include_directories(${CMAKE_SOURCE_DIR}"/cuda-samples/Common/")

Compile java source

    $ mvn package

Compile CUDA code

    $ mkdir build && cd build
    $ cmake -DCMAKE_BUILD_TYPE=Release ../c_src/ && make 
    $ cp libcudaconverge.so ../


Run (from main source directory, not previous build dir)

    $ mkdir results
    $ mvn exec:java -Dexec.mainClass="CRC_Prediction.MainProgram" -Dexec.args="-outputDir results/"
# PUR-IRL
Pop-Up Restaurant for Inverse Reinforcement Learning (PUR-IRL)

This Java/C++ code is the accompanying software for the paper 
[The Unreasonable Effectiveness of Inverse Reinforcement Learning in Advancing Cancer Research] (AAAI-2020), with authors John Kalantari Ph.D., Heidi Nelson M.D., Nicholas Chia Ph.D.


License
-------

Copyright (C) 2019 Mayo Foundation for Medical Education and Research

PUR-IRL is licensed under the terms of GPLv3 for open source use, or alternatively under the terms of the Mayo Clinic Commercial License for commercial use.

You may use PUR-IRL according to either of these licenses as is most appropriate for your project on a case-by-case basis.

You should have received a copy of the GNU General Public License along with PUR-IRL.  If not, see <https://www.gnu.org/licenses/>.

For a license for commercial use, please contact Mayo Clinic Ventures at mayoclinicventures@mayo.edu.
