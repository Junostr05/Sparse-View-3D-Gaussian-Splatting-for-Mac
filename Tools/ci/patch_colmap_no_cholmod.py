#!/usr/bin/env python3
"""Removes COLMAP's CHOLMOD dependency (SuiteSparse's supernodal module is GPL) so that a minimal,
permissively licensed build is possible. COLMAP only uses CHOLMOD as a fast path inside
SparseCholeskyWithFallbackSolver and already falls back to Eigen's SimplicialLDLT; the patch keeps the
class API and always uses the simplicial solver. Verified against COLMAP 4.2.1 (asserts on mismatch).

Usage: patch_colmap_no_cholmod.py COLMAP_SOURCE_DIR
"""
import re
import sys
from pathlib import Path


def edit(path, fn):
    text = path.read_text()
    new = fn(text)
    assert new != text, "patch had no effect on %s" % path
    path.write_text(new)


def main():
    root = Path(sys.argv[1])
    optim = root / "src/colmap/optim"

    def header(text):
        assert "#include <Eigen/CholmodSupport>" in text
        text = text.replace("#include <Eigen/CholmodSupport>\n", "")
        text, n = re.subn(r"\n\s*Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> supernodal_;", "", text)
        assert n == 1, "supernodal_ member not found"
        return text

    edit(optim / "sparse_cholesky.h", header)

    def source(text):
        marker = "namespace colmap {"
        head = text[: text.index(marker)]
        body = """namespace colmap {

// CHOLMOD (GPL supernodal module) is not used in this build; the solver always uses Eigen's simplicial
// LDLT, which the original implementation already used as its fallback.

void SparseCholeskyWithFallbackSolver::AnalyzePattern(
    const Eigen::SparseMatrix<double>& A) {
  ldlt_.analyzePattern(A);
  use_ldlt_ = true;
}

bool SparseCholeskyWithFallbackSolver::Factorize(
    const Eigen::SparseMatrix<double>& A) {
  ldlt_.factorize(A);
  return ldlt_.info() == Eigen::Success;
}

bool SparseCholeskyWithFallbackSolver::Compute(
    const Eigen::SparseMatrix<double>& A) {
  AnalyzePattern(A);
  return Factorize(A);
}

bool SparseCholeskyWithFallbackSolver::Solve(const Eigen::VectorXd& b,
                                             Eigen::VectorXd* x) const {
  THROW_CHECK_NOTNULL(x);
  x->noalias() = ldlt_.solve(b);
  return ldlt_.info() == Eigen::Success;
}

}  // namespace colmap
"""
        return head + body

    edit(optim / "sparse_cholesky.cc", source)

    def optim_cmake(text):
        assert "        CHOLMOD::CHOLMOD\n" in text
        return text.replace("        CHOLMOD::CHOLMOD\n", "")

    edit(optim / "CMakeLists.txt", optim_cmake)

    def deps(text):
        assert "find_package(CHOLMOD REQUIRED)" in text
        return text.replace("find_package(CHOLMOD REQUIRED)", "# CHOLMOD removed by Tools/ci/patch_colmap_no_cholmod.py")

    edit(root / "cmake/FindDependencies.cmake", deps)
    print("patched", root)


if __name__ == "__main__":
    main()
