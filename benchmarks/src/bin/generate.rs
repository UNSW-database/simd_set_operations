use benchmarks::{schema::*, generators};
use clap::Parser;
use std::{path::PathBuf, fs, io::{self, Write}};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(default_value = "experiment.toml")]
    experiment: PathBuf,
    #[arg(default_value = "datasets/")]
    datasets: PathBuf,
    #[arg(long, action)]
    clean: bool,
}

fn main() {
    let cli = Cli::parse();
    let result = if cli.clean {
        cli.clean().map_err(|e| e.to_string())
    }
    else {
        cli.generate()
    };
    println!("Done");

    if let Err(err) = result {
        println!("{}", err);
    }
}

impl Cli {
    fn clean(&self) -> io::Result<()> {
        for entry in fs::read_dir(self.datasets.join("2set"))? {
            fs::remove_file(entry?.path())?;
        }
        Ok(())
    }

    fn generate(&self) -> Result<(), String> {
        let experiment_toml = fs::read_to_string(&self.experiment)
            .map_err(|e| e.to_string())?;
        let experiments: Experiment = toml::from_str(&experiment_toml)
            .map_err(|e| e.to_string())?;

        for dataset in &experiments.dataset {
            match dataset {
                DatasetInfo::TwoSet(info) => generate_twoset(&self.datasets, info)?,
                DatasetInfo::KSet(_) => todo!(),
            }
        }
        Ok(())
    }
}

fn generate_twoset(datasets: &PathBuf, info: &TwoSetDatasetInfo) -> Result<(), String> {
    // Create directories
    let twoset = datasets.join("2set");
    fs::create_dir_all(&twoset).map_err(|e| e.to_string())?;

    let id = benchmarks::dataset_id(info);
    let dataset_path = twoset.join(&id);
    let info_path = twoset.join(id.clone() + ".info");

    // Check info file
    if let Ok(info_file) = fs::File::open(&info_path) {
        let existing_info: TwoSetDatasetInfo =
            ciborium::from_reader(info_file)
            .map_err(|e| e.to_string())?;

        if existing_info == *info {
            println!("skipping {}", id);
            return Ok(());
        }
        else {
            println!("rebuilding {}", id);
        }
    }
    else {
        println!("building {}", id);
    }
    // Build dataset
    let dataset_file = fs::File::options()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&dataset_path)
        .map_err(|e| format!(
            "failed to open file {}:\n{}",
            dataset_path.to_str().unwrap_or("<unknown>"),
            e.to_string()
        ))?;

    ciborium::into_writer(&build_twoset(info), dataset_file)
        .map_err(|e| e.to_string())?;

    // Write new info file
    let info_file = fs::File::options()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&info_path)
        .map_err(|e| format!(
            "failed to open file {}:\n{}",
            info_path.to_str().unwrap_or("<unknown>"),
            e.to_string()
        ))?;

    ciborium::into_writer(info, info_file)
        .map_err(|e| e.to_string())?;

    Ok(())
}

fn build_twoset(info: &TwoSetDatasetInfo) -> TwoSetFile {
    let begin = match info.vary {
        Parameter::Selectivity => info.props.selectivity,
        Parameter::Density => info.props.density,
        Parameter::Size => info.props.size,
        Parameter::Skew => info.props.skew,
    };

    let xvalues = (begin..=info.to).step_by(info.step as usize);
    let inputs = xvalues.map(|x| {
        print!("[x: {:4}] ", x);
        let pairs = (0..info.count)
            .map(|i| build_twoset_pair(info, x, i))
            .collect();
        println!();
        TwoSetInput { x, pairs: pairs }
    }
    ).collect();

    TwoSetFile {
        info: info.clone(),
        xvalues: inputs,
    }
}

fn build_twoset_pair(info: &TwoSetDatasetInfo, x: u32, i: usize) -> (Vec<i32>, Vec<i32>) {
    print!("{} ", i);
    let _ = io::stdout().flush();
    let mut props = info.props.clone();
    let prop = match info.vary {
        Parameter::Selectivity => &mut props.selectivity,
        Parameter::Density     => &mut props.density,
        Parameter::Size        => &mut props.size,
        Parameter::Skew        => &mut props.skew,
    };
    *prop = x;
    generators::gen_twoset(&props)
}
