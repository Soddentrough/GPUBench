#include "test_harness.h"
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  std::string filter = "";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--list" || arg == "-l") {
      std::cout << "Registered test cases:\n";
      for (const auto &t : ::gpubench::test::TestRegistry::instance().tests()) {
        std::cout << "  - " << t.suite << "." << t.name << "\n";
      }
      return 0;
    } else if (arg.rfind("--filter=", 0) == 0) {
      filter = arg.substr(9);
    } else if ((arg == "-f" || arg == "--filter") && i + 1 < argc) {
      filter = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "GPUBench Automated Test Suite\n";
      std::cout << "Usage: " << argv[0] << " [options]\n";
      std::cout << "Options:\n";
      std::cout << "  -f, --filter <str>   Run only tests whose suite or name contains <str>\n";
      std::cout << "  -l, --list           List all registered test cases\n";
      std::cout << "  -h, --help           Display this help message\n";
      return 0;
    } else if (!arg.empty() && arg[0] != '-') {
      filter = arg;
    }
  }

  return ::gpubench::test::TestRegistry::instance().runAll(filter);
}
