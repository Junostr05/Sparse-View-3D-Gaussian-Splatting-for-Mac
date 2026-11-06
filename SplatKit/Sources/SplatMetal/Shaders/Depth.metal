// Depth.metal — Phase 4: depth loss kernels (Pearson correlation and L1 against a per-view depth prior).
//
// Concatenated after Densify.metal (see `MetalContext.shaderFiles`), so everything in Common.metal is
// visible here. Uniform structs for these kernels are declared at the top of this file and mirrored in
// GPUDepthLoss.swift. Kernels are added by Phase 4 Task 15; this file is empty until then.
