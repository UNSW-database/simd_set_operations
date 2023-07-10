use benchmarks::schema::*;

fn main() {
    if let Err(err) = generate() {
        println!("{}", err);
    }
}

fn generate() -> Result<(), String> {
    let experiment_toml = std::fs::read_to_string("experiment.toml")
        .map_err(|e| e.to_string())?;
    let experiments: Experiments = toml::from_str(&experiment_toml)
        .map_err(|e| e.to_string())?;

    dbg!(experiments);

    Ok(())
}
