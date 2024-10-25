use std::env;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

// working directory is crate top-level (setops)

const FILENAMES: &'static [&'static str] = &[
    "zipper",
];

fn main() -> ExitCode {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let out_dir_str = out_dir.to_str().unwrap();

    println!("cargo::rustc-link-search={}", out_dir_str);
    println!("cargo::rustc-link-arg=-Wl,-T,link.ld");

    // build files
    for filename in FILENAMES {
        let in_file = format!("src/intersect/{}.S", filename);
        let o_out_file = format!("{}/{}.o", out_dir_str, filename);
        let a_out_file = format!("{}/{}.a", out_dir_str, filename);

        println!("cargo::rerun-if-changed={}", in_file);

        let as_output = Command::new("nasm").args(["-f", "elf64", "-o", o_out_file.as_str(), in_file.as_str()]).output();
        match as_output {
            Ok(_) => {},
            Err(e) => {
                println!("cargo::warning={}", e);
                return ExitCode::from(1);
            },
        };

        let ar_output = Command::new("ar").args(["r", a_out_file.as_str(), o_out_file.as_str()]).output();
        match ar_output {
            Ok(_) => {},
            Err(e) => {
                println!("cargo::warning={}", e);
                return ExitCode::from(1);
            },
        };

        println!("cargo::rustc-link-lib=static:+verbatim={}.a", filename);
    }

    return ExitCode::from(0);
}
