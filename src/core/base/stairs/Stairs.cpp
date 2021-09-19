#include <Stairs.h>

using namespace std;
using namespace ttk;

Stairs::Stairs() {
  inputData_ = nullptr;
  outputData_ = nullptr;
  dimensionNumber_ = 1;
  mask_ = nullptr;
  setDebugMsgPrefix("Stairs");
}

Stairs::~Stairs() {
}
