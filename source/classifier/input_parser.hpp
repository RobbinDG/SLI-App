#pragma once

#include "RCNN.hpp"

void errorUsage(char** argv);

void parseAndExecute(RCNN& net, int argc, char** argv);