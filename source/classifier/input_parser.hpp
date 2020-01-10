#pragma once

#include "RCNN.hpp"
#include "environments/ExecEnvironment.hpp"
#include <memory>

void errorUsage(char** argv);

std::unique_ptr<spp::envs::ExecEnvironment> parse(int argc, char** argv);