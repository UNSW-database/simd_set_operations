use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use bindgen;
use bindgen::CargoCallbacks;

fn main() {
    // --- 1. Compile qfilter (existing C++ logic) ---
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        println!("Compiling with qfilter");
        cc::Build::new()
            .file("ffi/qfilter/qfilter.cpp")
            .cpp(true)
            .flag("-msse3")
            .flag("-mavx")
            .flag("-mavx2")
            .opt_level(3)
            .compile("qfilter");
        println!("cargo:rerun-if-changed=ffi/qfilter/qfilter.cpp");

        let bindings = bindgen::Builder::default()
            .header("ffi/qfilter/qfilter.h")
            .parse_callbacks(Box::new(CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate bindings");

        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("qfilter_c.rs"))
            .expect("Failed to write bindings");
    }

    // --- 2. Compile CUDA Kernel (.cu file) ---
    cc::Build::new()
        .cuda(true)
        .file("../Cuda/main.cu")  // Update path if needed
        .flag("-gencode=arch=compute_61,code=sm_61") // For GTX 1080 Pascal
		.flag("-ccbin=/usr/bin/gcc-12")
        .flag("-O2")
        .define("BUILDING_RUST_LIB", None)
        .compile("cuda_kernels");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");

    // --- 3. Rerun triggers for all sources ---
    println!("cargo:rerun-if-changed=ffi/qfilter/qfilter.cpp");
    println!("cargo:rerun-if-changed=ffi/qfilter/qfilter.h");
    println!("cargo:rerun-if-changed=src/kernel.cu");
}

