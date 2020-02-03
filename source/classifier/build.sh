TARGETS="arm-linux-gnueabihf x86_64-pc-linux-gnu"

BUILD_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

for TARGET in $TARGETS
do
  mkdir -p ${BUILD_DIR}/${TARGET}
  cd ${BUILD_DIR}/${TARGET}
  cmake -DCMAKE_C_COMPILER="/usr/bin/${TARGET}-gcc" -DCMAKE_CXX_COMPILER="/usr/bin/${TARGET}-g++" ../CMakeLists.txt

  cd ${BUILD_DIR}
  make
  mv spp ${TARGET}/spp
  rm -r CMakeFiles/
done
