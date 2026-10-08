#pragma once

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace gpubench::test {

struct TestCase {
  std::string suite;
  std::string name;
  std::function<void()> func;
};

class TestRegistry {
public:
  static TestRegistry &instance() {
    static TestRegistry reg;
    return reg;
  }

  void addTest(const std::string &suite, const std::string &name, std::function<void()> func) {
    m_tests.push_back({suite, name, std::move(func)});
  }

  const std::vector<TestCase> &tests() const { return m_tests; }

  int runAll(const std::string &filter = "") {
    int passed = 0;
    int failed = 0;
    int skipped = 0;

    std::cout << "\n╭─ GPUBench Test Runner ──────────────────────────────────────────────────────╮\n";
    std::cout << "│ Running registered automated unit tests                                     │\n";
    std::cout << "╰─────────────────────────────────────────────────────────────────────────────╯\n\n";

    for (const auto &t : m_tests) {
      std::string fullName = t.suite + "." + t.name;
      if (!filter.empty() && fullName.find(filter) == std::string::npos && t.suite.find(filter) == std::string::npos) {
        ++skipped;
        continue;
      }

      std::cout << "  • [" << std::left << std::setw(24) << t.suite << "] " 
                << std::setw(42) << t.name << " ... " << std::flush;
      try {
        t.func();
        std::cout << "\033[32m✔ PASS\033[0m\n";
        ++passed;
      } catch (const std::exception &e) {
        std::cout << "\033[31m✘ FAIL\033[0m\n";
        std::cerr << "      \033[31m[ERROR]\033[0m " << e.what() << "\n";
        ++failed;
      } catch (...) {
        std::cout << "\033[31m✘ FAIL (Unknown Exception)\033[0m\n";
        ++failed;
      }
    }

    std::cout << "\n───────────────────────────────────────────────────────────────────────────────\n";
    std::cout << " Test Summary: ";
    if (failed == 0) {
      std::cout << "\033[32mAll " << passed << " tests passed\033[0m";
    } else {
      std::cout << "\033[31m" << failed << " failed\033[0m, \033[32m" << passed << " passed\033[0m";
    }
    if (skipped > 0) {
      std::cout << " (" << skipped << " filtered out)";
    }
    std::cout << "\n───────────────────────────────────────────────────────────────────────────────\n\n";

    return (failed == 0) ? 0 : 1;
  }

private:
  std::vector<TestCase> m_tests;
};

struct TestRegistrar {
  TestRegistrar(const std::string &suite, const std::string &name, std::function<void()> func) {
    TestRegistry::instance().addTest(suite, name, std::move(func));
  }
};

class AssertionFailure : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

} // namespace gpubench::test

#define TEST_CASE(suite, name) \
  static void suite##_##name##_impl(); \
  static ::gpubench::test::TestRegistrar suite##_##name##_reg(#suite, #name, suite##_##name##_impl); \
  static void suite##_##name##_impl()

#define ASSERT_TRUE(cond) \
  do { \
    if (!(cond)) { \
      std::ostringstream oss; \
      oss << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__; \
      throw ::gpubench::test::AssertionFailure(oss.str()); \
    } \
  } while (0)

#define ASSERT_FALSE(cond) \
  do { \
    if (cond) { \
      std::ostringstream oss; \
      oss << "Assertion failed: !(" #cond ") at " << __FILE__ << ":" << __LINE__; \
      throw ::gpubench::test::AssertionFailure(oss.str()); \
    } \
  } while (0)

#define ASSERT_EQ(a, b) \
  do { \
    auto va = (a); \
    auto vb = (b); \
    if (!(va == vb)) { \
      std::ostringstream oss; \
      oss << "Assertion failed: (" #a " == " #b ") [" << va << " != " << vb << "] at " << __FILE__ << ":" << __LINE__; \
      throw ::gpubench::test::AssertionFailure(oss.str()); \
    } \
  } while (0)

#define ASSERT_NE(a, b) \
  do { \
    auto va = (a); \
    auto vb = (b); \
    if (va == vb) { \
      std::ostringstream oss; \
      oss << "Assertion failed: (" #a " != " #b ") [" << va << " == " << vb << "] at " << __FILE__ << ":" << __LINE__; \
      throw ::gpubench::test::AssertionFailure(oss.str()); \
    } \
  } while (0)

#define ASSERT_NEAR(a, b, eps) \
  do { \
    double va = static_cast<double>(a); \
    double vb = static_cast<double>(b); \
    double diff = std::abs(va - vb); \
    if (diff > (eps)) { \
      std::ostringstream oss; \
      oss << "Assertion failed: |" #a " - " #b "| <= " #eps " [|" << va << " - " << vb << "| = " << diff << " > " << (eps) << "] at " << __FILE__ << ":" << __LINE__; \
      throw ::gpubench::test::AssertionFailure(oss.str()); \
    } \
  } while (0)

#define ASSERT_GT(a, b) \
  do { \
    auto va = (a); \
    auto vb = (b); \
    if (!(va > vb)) { \
      std::ostringstream oss; \
      oss << "Assertion failed: (" #a " > " #b ") [" << va << " <= " << vb << "] at " << __FILE__ << ":" << __LINE__; \
      throw ::gpubench::test::AssertionFailure(oss.str()); \
    } \
  } while (0)

#define ASSERT_GE(a, b) \
  do { \
    auto va = (a); \
    auto vb = (b); \
    if (!(va >= vb)) { \
      std::ostringstream oss; \
      oss << "Assertion failed: (" #a " >= " #b ") [" << va << " < " << vb << "] at " << __FILE__ << ":" << __LINE__; \
      throw ::gpubench::test::AssertionFailure(oss.str()); \
    } \
  } while (0)

#define ASSERT_LT(a, b) \
  do { \
    auto va = (a); \
    auto vb = (b); \
    if (!(va < vb)) { \
      std::ostringstream oss; \
      oss << "Assertion failed: (" #a " < " #b ") [" << va << " >= " << vb << "] at " << __FILE__ << ":" << __LINE__; \
      throw ::gpubench::test::AssertionFailure(oss.str()); \
    } \
  } while (0)

#define ASSERT_LE(a, b) \
  do { \
    auto va = (a); \
    auto vb = (b); \
    if (!(va <= vb)) { \
      std::ostringstream oss; \
      oss << "Assertion failed: (" #a " <= " #b ") [" << va << " > " << vb << "] at " << __FILE__ << ":" << __LINE__; \
      throw ::gpubench::test::AssertionFailure(oss.str()); \
    } \
  } while (0)
