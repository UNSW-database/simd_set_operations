use benchmark::{fmt_open_err, path_str, schema::*};
use clap::Parser;
use colored::Colorize;
use plotters::{prelude::*, coord::Shift};
use std::{path::PathBuf, fs::{self, File}, collections::HashMap};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(default_value = "results.json")]
    results: PathBuf,
    #[arg(default_value = "plots/")]
    plots: PathBuf,
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = plot_experiments(&cli) {
        let msg = format!("error: {}", e);
        println!("{}", msg.red().bold());
    }
}

fn plot_experiments(cli: &Cli) -> Result<(), String> {
    // Load results
    let results_json = File::open(&cli.results)
        .map_err(|e| fmt_open_err(e, &cli.results))?;

    let results: Results = serde_json::from_reader(&results_json)
        .map_err(|e| format!(
            "invalid toml file {}: {}",
            path_str(&cli.results), e
        ))?;

    fs::create_dir_all(&cli.plots)
        .map_err(|e| format!(
            "unable to create directory {}: {}",
            path_str(&cli.plots), e.to_string()
        ))?;
    
    for experiment in results.experiments {
        let plot_path = cli.plots
            .join(format!("{}.svg", experiment.name));

        println!("{}", path_str(&plot_path));

        let root = SVGBackend::new(&plot_path, (640, 480))
            .into_drawing_area();

        root.fill(&WHITE)
            .map_err(|e| format!(
                "unable to fill bg with white for {}: {}",
                &experiment.name, e.to_string()
            ))?;

        plot_experiment(&root, &experiment, &results.datasets)?;

        root.present()
            .map_err(|e| format!(
                "unable to present {}: {}",
                &experiment.name, e.to_string()
            ))?;
    }
    Ok(())
}

fn plot_experiment<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    experiment: &ExperimentEntry,
    datasets: &HashMap<DatasetId, DatasetResults>) -> Result<(), String>
{
    let dataset = datasets.get(&experiment.dataset)
        .ok_or_else(|| format!(
            "dataset {} not found in results", &experiment.dataset
        ))?;
    
    let max_time = *dataset.algos.iter().map(
        |(_, a)| a.iter().map(
            |r| r.times.iter().max().unwrap()
        ).max().unwrap()
    ).max().unwrap() as f64 / 1000.0;

    let min_time = *dataset.algos.iter().map(
        |(_, a)| a.iter().map(
            |r| r.times.iter().min().unwrap()
        ).min().unwrap()
    ).min().unwrap() as f64 / 1000.0;

    let mut chart = ChartBuilder::on(root)
        .caption(&experiment.title, ("sans-serif", 20).into_font())
        .x_label_area_size(40)
        .y_label_area_size(50)
        .margin(10)
        .build_cartesian_2d(0..dataset.info.to, min_time..max_time)
        .map_err(|e| format!(
            "unable to create chart for {}: {}",
            &experiment.name, e.to_string()
        ))?;
        
    for (i, algorithm_name) in experiment.algorithms.iter().enumerate() {

        let algorithm = dataset.algos.get(algorithm_name)
            .ok_or_else(|| format!(
                "algorithm {} not found in results for dataset {}",
                algorithm_name, &dataset.info.name
            ))?;

        let color = Palette99::pick(i);

        chart
            .draw_series(LineSeries::new(
                algorithm.iter().map(|r| (
                    r.x,
                    // Average runs for each x value
                    r.times.iter().sum::<u64>() as f64 / r.times.len() as f64
                    / 1000.0 // microseconds
                )),
                color.stroke_width(2),
            ))
            .map_err(|e| format!(
                "unable to draw series {} for {}: {}",
                algorithm_name, &experiment.name, e.to_string()
            ))?
            .label(algorithm_name)
            .legend(move |(x, y)| Rectangle::new(
                [(x, y - 5), (x + 10, y + 5)],
                color.filled()
            ));
    }

    chart
        .configure_mesh()
        .x_desc(format_xlabel(dataset.info.vary))
        .y_desc("Time (μs)")
        .x_label_formatter(&|&x| format_x(x, dataset.info.vary))
        .y_label_formatter(&|&x| format_time(x))
        .max_light_lines(4)
        .draw()
        .map_err(|e| format!(
            "unable to draw mesh {}: {}",
            experiment.name, e.to_string()
        ))?;

    chart.configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE)
        .position(SeriesLabelPosition::UpperRight)
        .draw()
        .map_err(|e| format!(
            "unable to draw series labels {}: {}",
            experiment.name, e.to_string()
        ))?;

    Ok(())
}

fn format_x(x: u32, vary: Parameter) -> String {
    match vary {
        Parameter::Density | Parameter::Selectivity =>
            format!("{:.2}", x as f64 / 1000.0),
        Parameter::Size => format_size(x),
        Parameter::Skew => if x == 0 {
            String::new()
        } else {
            format!("1:{}", 1 << (x - 1))
        }
    }
}

fn format_size(size: u32) -> String {
    match size {
        0..=9   => format!("{size}"),
        10..=19 => format!("{}KiB", 1 << (size - 10)),
        20..=29 => format!("{}MiB", 1 << (size - 20)),
        30..=39 => format!("{}GiB", 1 << (size - 30)),
        _ => size.to_string(),
    }
}

fn format_time(micros: f64) -> String {
    format!("{:.0}", micros)
}

fn format_xlabel(parameter: Parameter) -> &'static str {
    match parameter {
        Parameter::Density => "density",
        Parameter::Selectivity => "selectivity",
        Parameter::Size => "size",
        Parameter::Skew => "skew",
    }
}
