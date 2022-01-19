# JRC Changes to original repo
This will list the changes added to the project by people at JRC

## Install and run

The code has several dependencies and uses C, CUDA and JAVA languages. Unfortunately, there was no project configuration, so we used Maven to build the java code and addeed a CMake build system on top of the C/CUDA code.

(1) First let's install CUDA and the examples

    $ sudo apt-get install maven openjdk-8-jdk default-jdk intel-mkl jblas
    $ git clone https://github.com/NVIDIA/cuda-samples.git

Be sure to clone the examples in the same directory of the source code, because we define their location in the CMakeFiles.txt as:

> include_directories(${CMAKE_SOURCE_DIR}"/cuda-samples/Common/")

(2) Compile java code first using maven:

    $ mvn package

(3) Compile accelerated native code in c_src directory
First generate the list of all classpaths used in the maven build

    $ mvn dependency:build-classpath -Dmdep.outputFile=cp.txt

This will create a cp.txt files with all the external dependencies of our project

Now generate the C interface that glues Java code and the native one and build it with cmake:

    $ javah -d c_src/ -cp target/classes/:$(cat cp.txt) CRC_Prediction.InferenceAlgoCancer
    $ mkdir build && cd build
    $ cmake -DCMAKE_BUILD_TYPE=Release ../c_src/ && make 
    $ cp *.so ../

This will build and copy both the cudaconverge.so and mklconverge.so libraries.

If you only want to activate one of the two accelerators, you can do it by passing
either -DBUILD_CUDA=OFF or -DBUILD_MKL=OFF as a cmake configuration option

If the location of cuda samples is not in the default directory, 
you can set it with DCUDA_SAMPLES_BASE_PATH as in the following example:

    $ cmake -DCMAKE_BUILD_TYPE=Release ../c_src/ -DCUDA_SAMPLES_BASE_PATH=~/projects/jrc/genomic/cuda-samples/Common/

Run code with MKL support (from main source directory, not previous build dir)

    $ mkdir results
    $ mvn exec:java -Dexec.mainClass="CRC_Prediction.MainProgram" -Dexec.args="-outputDir results/"
    
For GPU support run instead:

    $ mvn exec:java -Dexec.mainClass="CRC_Prediction.MainProgram" -Dexec.args="-outputDir results/ -numGPUs 1"

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
