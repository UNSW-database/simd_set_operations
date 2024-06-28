use std::env;
use std::path::PathBuf;

fn main() {
    if cfg!(target_os = "linux") {
        cc::Build::new()
            .file("ffi/qfilter/qfilter.cpp")
            .cpp(true)
            .flag("-mssse3")
            .flag("-mavx")
            .flag("-mavx2")
            .opt_level(3)
            .compile("qfilter");
        println!("cargo::rerun-if-changed=ffi/qfilter/qfilter.cpp");
    }
    let bindings = bindgen::Builder::default()
        .header("ffi/qfilter/qfilter.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.
        write_to_file(out_path.join("qfilter_c.rs"))
        .expect("Failed to write bindings");
}
